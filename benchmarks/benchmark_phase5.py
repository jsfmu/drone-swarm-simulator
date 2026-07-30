"""Phase 5 benchmark harness: optimization/scaling/parallel-execution evidence.

Unlike ``benchmark_simulation.py``/``benchmark_avoidance.py`` (fixed workloads,
a handful of flags), this is the single bounded, parametrized entry point
Phase 5's acceptance criteria ask for -- one script covering local execution,
distributed execution at multiple worker counts, executor choice, profiling,
checkpoint save/load cost, and memory, all with machine-readable JSON/CSV
output alongside the printed table. It reuses ``benchmark_simulation.py``'s
world-scaling helper and ``benchmark_avoidance.py``'s memory-measurement
helpers rather than duplicating them.

Usage:
    python benchmarks/benchmark_phase5.py
    python benchmarks/benchmark_phase5.py --drones 1000 10000 --mode local --policy local_avoidance
    python benchmarks/benchmark_phase5.py --mode distributed --workers 1 2 4 --executor threaded
    python benchmarks/benchmark_phase5.py --mode distributed --workers 4 --executor process
    python benchmarks/benchmark_phase5.py --checkpoint-bench --drones 10000
    python benchmarks/benchmark_phase5.py --profile --drones 10000 --json-out benchmarks/phase5_results/run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

import numpy as np  # noqa: E402

from benchmark_avoidance import peak_rss_bytes  # noqa: E402 (reused, not duplicated)
from benchmark_simulation import CELL_SIZE, COLLISION_RADIUS, NEAR_MISS_RADIUS, world_side_for  # noqa: E402
from drone_sim.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.coordinator import DistributedConfig, DistributedCoordinator  # noqa: E402
from drone_sim.movement import (  # noqa: E402
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementSystem,
    RandomMovementAlgorithm,
    ScriptedMovementAlgorithm,
)
from drone_sim.simulation import Simulation, TickProfile  # noqa: E402
from drone_sim.state import World  # noqa: E402

DEFAULT_DRONES = [1_000, 10_000, 50_000, 100_000]
DEFAULT_TICKS = 10
DEFAULT_WARMUP = 3

_POLICY_FACTORIES = {
    "random": lambda: None,  # None -> MovementSystem() default registry (Random+Scripted)
    "scripted": lambda: None,
    "goal_directed": GoalDirectedMovementAlgorithm,
    "local_avoidance": LocalAvoidanceMovementAlgorithm,
}


def build_config(n: int, seed: int = 0) -> SimulationConfig:
    """Same ~64-cells/drone world scaling as every other benchmark in this
    project, so Phase 5 numbers stay comparable to the Phase 1-4 baselines."""
    side = world_side_for(n)
    return SimulationConfig(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(side, side, side),
        collision_radius=COLLISION_RADIUS,
        near_miss_radius=NEAR_MISS_RADIUS,
        cell_size=CELL_SIZE,
        dt=1.0,
        max_speed=5.0,
        seed=seed,
    )


def _movement_system_for(policy: str) -> Optional[MovementSystem]:
    if policy in ("random", "scripted", None):
        return None
    algo = _POLICY_FACTORIES[policy]()
    return MovementSystem(policies={algo.policy_id: algo})


def _world_with_goals_if_needed(config: SimulationConfig, policy: str) -> World:
    world = World.create(config)
    if policy in ("goal_directed", "local_avoidance"):
        center = (config.bounds_min_arr + config.bounds_max_arr) / 2.0
        world.state.goal_positions = (
            2.0 * center[None, :] - world.state.positions.astype(np.float64)
        ).astype(np.float32)
        algo_id = _POLICY_FACTORIES[policy].policy_id
        world.state.movement_policy_ids[:] = algo_id
    return world


@dataclass
class LocalRunResult:
    drones: int
    policy: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    ticks_per_s: float
    peak_rss_mb: Optional[float]
    stage_ms: Optional[dict] = None


def run_local(n: int, policy: str, ticks: int, warmup: int, profile: bool) -> LocalRunResult:
    cfg = build_config(n)
    movement = _movement_system_for(policy)
    world = _world_with_goals_if_needed(cfg, policy)
    sim = Simulation(cfg, movement=movement, world=world)

    for _ in range(warmup):
        sim.step()

    times_ms = []
    stage_totals = None
    for _ in range(ticks):
        if profile:
            tp = TickProfile()
            t0 = time.perf_counter()
            sim.step(profile=tp)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            if stage_totals is None:
                stage_totals = {k: 0.0 for k in (
                    "pre_grid_ns", "pre_pairs_ns", "prediction_ns", "context_ns",
                    "movement_ns", "boundary_ns", "post_grid_ns", "post_pairs_ns",
                    "detection_ns", "resolution_ns",
                )}
            for k in stage_totals:
                stage_totals[k] += getattr(tp, k) / 1e6  # ns -> ms
        else:
            t0 = time.perf_counter()
            sim.step()
            times_ms.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(times_ms)
    stage_mean = {k: v / ticks for k, v in stage_totals.items()} if stage_totals else None
    rss = peak_rss_bytes()
    return LocalRunResult(
        drones=n, policy=policy,
        mean_ms=float(arr.mean()), median_ms=float(np.median(arr)), p95_ms=float(np.percentile(arr, 95)),
        ticks_per_s=float(1000.0 / arr.mean()) if arr.mean() > 0 else float("inf"),
        peak_rss_mb=(rss / 1e6) if rss is not None else None,
        stage_ms=stage_mean,
    )


@dataclass
class DistributedRunResult:
    drones: int
    workers: int
    executor: str
    mean_ms: float
    ticks_per_s: float
    collisions_last_tick: int
    peak_rss_mb: Optional[float]


def run_distributed(n: int, workers: int, executor: str, ticks: int, warmup: int) -> DistributedRunResult:
    cfg = build_config(n)
    movement = _movement_system_for("goal_directed")  # deterministic, RNG-free -> meaningful cross-worker-count agreement
    world = _world_with_goals_if_needed(cfg, "goal_directed")

    dist_cfg = DistributedConfig(
        num_workers=workers,
        use_threads=(executor == "threaded"),
        use_processes=(executor == "process"),
    )
    coord = DistributedCoordinator(cfg, dist_cfg, movement=movement, world=world)
    try:
        for _ in range(warmup):
            coord.step()

        times_ms = []
        result = None
        for _ in range(ticks):
            t0 = time.perf_counter()
            result = coord.step()
            times_ms.append((time.perf_counter() - t0) * 1e3)
    finally:
        coord.shutdown()  # no-op unless executor == "process" -- never leak worker processes

    arr = np.asarray(times_ms)
    rss = peak_rss_bytes()
    return DistributedRunResult(
        drones=n, workers=workers, executor=executor,
        mean_ms=float(arr.mean()), ticks_per_s=float(1000.0 / arr.mean()) if arr.mean() > 0 else float("inf"),
        collisions_last_tick=result.num_collisions if result is not None else 0,
        peak_rss_mb=(rss / 1e6) if rss is not None else None,
    )


@dataclass
class CheckpointBenchResult:
    drones: int
    save_ms: float
    load_ms: float
    file_bytes: int
    resume_matches_uninterrupted: bool


def run_checkpoint_bench(n: int, ticks_before: int, ticks_after: int, tmp_dir: Path) -> CheckpointBenchResult:
    """Save/load cost, plus the deterministic-resume equivalence check
    (Phase 5 acceptance criterion #8): running N ticks -> checkpoint -> M more
    ticks must match loading that checkpoint fresh and running M ticks."""
    cfg = build_config(n)
    movement = _movement_system_for("goal_directed")
    world = _world_with_goals_if_needed(cfg, "goal_directed")

    sim = Simulation(cfg, movement=movement, world=world)
    for _ in range(ticks_before):
        sim.step()

    ckpt_path = tmp_dir / f"phase5_bench_{n}.ckpt.json"
    t0 = time.perf_counter()
    save_checkpoint(sim, ckpt_path)
    save_ms = (time.perf_counter() - t0) * 1e3
    file_bytes = ckpt_path.stat().st_size

    # Continue the original, uninterrupted simulation.
    for _ in range(ticks_after):
        sim.step()
    uninterrupted_positions = sim.world.state.positions.copy()
    uninterrupted_tick = sim.clock.tick

    t0 = time.perf_counter()
    resumed_sim = load_checkpoint(ckpt_path, movement=movement)
    load_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(ticks_after):
        resumed_sim.step()

    matches = (
        resumed_sim.clock.tick == uninterrupted_tick
        and np.allclose(resumed_sim.world.state.positions, uninterrupted_positions)
    )
    ckpt_path.unlink(missing_ok=True)
    return CheckpointBenchResult(
        drones=n, save_ms=save_ms, load_ms=load_ms, file_bytes=file_bytes,
        resume_matches_uninterrupted=bool(matches),
    )


def _print_local_table(results: list[LocalRunResult]) -> None:
    header = f"{'policy':>16} | {'drones':>8} | {'mean ms':>10} | {'median ms':>10} | {'p95 ms':>10} | {'ticks/s':>9} | {'peak RSS MB':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        rss = f"{r.peak_rss_mb:.2f}" if r.peak_rss_mb is not None else "n/a"
        print(
            f"{r.policy:>16} | {r.drones:>8,d} | {r.mean_ms:>10.3f} | {r.median_ms:>10.3f} | "
            f"{r.p95_ms:>10.3f} | {r.ticks_per_s:>9.2f} | {rss:>12}"
        )


def _print_distributed_table(results: list[DistributedRunResult]) -> None:
    header = f"{'drones':>8} | {'workers':>7} | {'executor':>9} | {'mean ms':>10} | {'ticks/s':>9} | {'peak RSS MB':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        rss = f"{r.peak_rss_mb:.2f}" if r.peak_rss_mb is not None else "n/a"
        print(
            f"{r.drones:>8,d} | {r.workers:>7,d} | {r.executor:>9} | {r.mean_ms:>10.3f} | "
            f"{r.ticks_per_s:>9.2f} | {rss:>12}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 optimization/scaling/parallel-execution benchmark")
    ap.add_argument("--drones", type=int, nargs="+", default=DEFAULT_DRONES)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--policy", choices=list(_POLICY_FACTORIES), default="goal_directed")
    ap.add_argument("--mode", choices=["local", "distributed", "both"], default="local")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--executor", choices=["sequential", "threaded", "process"], default="sequential")
    ap.add_argument("--profile", action="store_true", help="record per-stage TickProfile timings (local mode only)")
    ap.add_argument("--checkpoint-bench", action="store_true", help="also run the checkpoint save/load/resume benchmark")
    ap.add_argument("--json-out", type=str, default=None, help="write machine-readable results to this path")
    ap.add_argument("--csv-out", type=str, default=None, help="write a CSV summary to this path")
    args = ap.parse_args()

    print(f"numpy {np.__version__}")
    print(f"drones={args.drones} ticks={args.ticks} warmup={args.warmup} policy={args.policy} mode={args.mode}\n")

    local_results: list[LocalRunResult] = []
    dist_results: list[DistributedRunResult] = []
    ckpt_results: list[CheckpointBenchResult] = []

    if args.mode in ("local", "both"):
        print("-- Local execution --")
        for n in args.drones:
            r = run_local(n, args.policy, args.ticks, args.warmup, profile=args.profile)
            local_results.append(r)
        _print_local_table(local_results)
        if args.profile:
            print("\nPer-stage mean ms/tick (last drone count only, to keep output short):")
            last = local_results[-1]
            if last.stage_ms:
                for k, v in last.stage_ms.items():
                    print(f"  {k:>16}: {v:8.3f} ms")

    if args.mode in ("distributed", "both"):
        print("\n-- Distributed execution --")
        for n in args.drones:
            for w in args.workers:
                r = run_distributed(n, w, args.executor, args.ticks, args.warmup)
                dist_results.append(r)
        _print_distributed_table(dist_results)

    if args.checkpoint_bench:
        print("\n-- Checkpoint save/load/resume --")
        tmp_dir = Path(args.json_out).resolve().parent if args.json_out else _BENCH_DIR / "phase5_results"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for n in args.drones:
            r = run_checkpoint_bench(n, ticks_before=5, ticks_after=5, tmp_dir=tmp_dir)
            ckpt_results.append(r)
            print(
                f"  drones={n:>8,d}  save={r.save_ms:7.3f} ms  load={r.load_ms:7.3f} ms  "
                f"file={r.file_bytes/1e6:6.3f} MB  resume_matches_uninterrupted={r.resume_matches_uninterrupted}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args),
            "local_results": [asdict(r) for r in local_results],
            "distributed_results": [asdict(r) for r in dist_results],
            "checkpoint_results": [asdict(r) for r in ckpt_results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {out_path}")

    if args.csv_out:
        import csv

        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "drones", "policy_or_workers", "executor", "mean_ms", "ticks_per_s", "peak_rss_mb"])
            for r in local_results:
                writer.writerow(["local", r.drones, r.policy, "", f"{r.mean_ms:.4f}", f"{r.ticks_per_s:.4f}", r.peak_rss_mb])
            for r in dist_results:
                writer.writerow(["distributed", r.drones, r.workers, r.executor, f"{r.mean_ms:.4f}", f"{r.ticks_per_s:.4f}", r.peak_rss_mb])
        print(f"Wrote CSV results to {out_path}")


if __name__ == "__main__":
    main()
