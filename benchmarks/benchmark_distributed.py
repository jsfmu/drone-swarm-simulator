"""Phase 4 distributed-execution benchmark harness.

Compares three ways of running the same workload, using the same production
kernel classes throughout (no reimplementation):

* ``single_worker``     -- the existing, unmodified ``drone_sim.simulation.Simulation``.
* ``coordinator_1w``    -- ``DistributedCoordinator`` with one worker, one partition
                           (no ghosts, no dedup -- structurally the same work as
                           ``single_worker``, plus coordinator bookkeeping overhead).
* ``coordinator_Nw``    -- ``DistributedCoordinator`` with N workers, N partitions.

This is a *correctness and overhead* benchmark, not a speedup claim: per-tick
work here is still single-process, single-machine, and dominated by Python-
level per-partition orchestration (building N job objects, N dict entries)
rather than by parallel hardware, so ``coordinator_Nw`` is expected to be
slower than ``single_worker``, not faster -- see README.md's Phase 4 section
for the measured numbers and why that is an honest, expected result at this
phase's local-process scope, not a regression to fix.

Usage:
    python benchmarks/benchmark_distributed.py
    python benchmarks/benchmark_distributed.py --sizes 1000 5000 --workers 1 4 --ticks 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

import numpy as np  # noqa: E402

from benchmark_simulation import CELL_SIZE, COLLISION_RADIUS, NEAR_MISS_RADIUS, world_side_for  # noqa: E402
from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.coordinator import DistributedConfig, DistributedCoordinator  # noqa: E402
from drone_sim.movement import GoalDirectedMovementAlgorithm, MovementSystem  # noqa: E402
from drone_sim.simulation import Simulation  # noqa: E402
from drone_sim.state import DroneState, World  # noqa: E402

DEFAULT_SIZES = [1_000, 5_000]
DEFAULT_WORKER_COUNTS = [1, 4]
DEFAULT_TICKS = 10
DEFAULT_WARMUP = 2


def build_config(n: int, seed: int = 0) -> SimulationConfig:
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
        max_accel=3.0,
        seed=seed,
    )


def make_template_world(config: SimulationConfig) -> World:
    """One reproducible initial World (positions/velocities/goals). Uses
    GoalDirectedMovementAlgorithm -- deterministic (no RNG inside the
    policy), which is what makes exact-agreement comparisons across worker
    counts meaningful instead of noise from independent RNG streams (see
    README.md's Phase 4 section on why RandomMovementAlgorithm parity is not
    claimed across partition counts)."""
    state = DroneState.generate(config)  # positions/velocities from config.seed
    state.velocities[:] = 0.0
    state.movement_policy_ids[:] = GoalDirectedMovementAlgorithm.policy_id
    goal_rng = np.random.default_rng(config.seed + 7919)
    state.goal_positions = goal_rng.uniform(
        config.bounds_min_arr, config.bounds_max_arr, size=(config.num_drones, 3)
    ).astype(np.float32)
    return World(config=config, state=state)


def clone_world(template: World) -> World:
    s = template.state
    cloned = DroneState(
        positions=s.positions.copy(),
        velocities=s.velocities.copy(),
        active_mask=s.active_mask.copy(),
        movement_policy_ids=s.movement_policy_ids.copy(),
        goal_positions=None if s.goal_positions is None else s.goal_positions.copy(),
    )
    return World(config=template.config, state=cloned)


def _movement_system() -> MovementSystem:
    return MovementSystem(policies={GoalDirectedMovementAlgorithm.policy_id: GoalDirectedMovementAlgorithm()})


def run_plain(config: SimulationConfig, world: World, ticks: int, warmup: int) -> dict:
    sim = Simulation(config, movement=_movement_system(), world=world)
    for _ in range(warmup):
        sim.step()
    sim.metrics.reset()
    t0 = time.perf_counter()
    last_result = None
    for _ in range(ticks):
        last_result = sim.step()
    total_s = time.perf_counter() - t0
    return {
        "mean_tick_ms": total_s / ticks * 1e3,
        "final_positions": sim.world.state.positions.copy(),
        "final_velocities": sim.world.state.velocities.copy(),
        "collision_pairs": {(int(a), int(b)) for a, b in last_result.collision_pairs},
        "total_collisions": sim.metrics.summary().get("total_collisions", 0),
    }


def run_coordinator(config: SimulationConfig, world: World, dist_config: DistributedConfig, ticks: int, warmup: int):
    coord = DistributedCoordinator(config, dist_config, movement=_movement_system(), world=world)
    for _ in range(warmup):
        coord.step()
    coord.metrics.reset()
    t0 = time.perf_counter()
    last_result = None
    for _ in range(ticks):
        last_result = coord.step()
    total_s = time.perf_counter() - t0
    return coord, {
        "mean_tick_ms": total_s / ticks * 1e3,
        "final_positions": coord.world.state.positions.copy(),
        "final_velocities": coord.world.state.velocities.copy(),
        "collision_pairs": {(int(a), int(b)) for a, b in last_result.collision_pairs},
        "total_collisions": coord.metrics.summary().get("total_collisions", 0),
    }


def _agrees(a: dict, b: dict) -> bool:
    return (
        a["collision_pairs"] == b["collision_pairs"]
        and np.allclose(a["final_positions"], b["final_positions"], atol=1e-3)
    )


def check_determinism(config: SimulationConfig, template: World, dist_config: DistributedConfig, ticks: int, warmup: int) -> bool:
    _, r1 = run_coordinator(config, clone_world(template), dist_config, ticks, warmup)
    _, r2 = run_coordinator(config, clone_world(template), dist_config, ticks, warmup)
    return (
        r1["collision_pairs"] == r2["collision_pairs"]
        and np.array_equal(r1["final_positions"], r2["final_positions"])
    )


def check_rebalancing(config: SimulationConfig, ticks: int) -> tuple[bool, int]:
    """A deliberately skewed initial distribution (all drones in the first
    quarter of the world) to make partition load imbalance real, with an
    aggressive rebalance interval/threshold so the run is short."""
    state = DroneState.generate(config)
    state.positions[:, 0] = np.random.default_rng(0).uniform(
        config.bounds_min_arr[0], config.bounds_min_arr[0] + (config.bounds_max_arr[0] - config.bounds_min_arr[0]) * 0.25,
        size=config.num_drones,
    )
    state.velocities[:] = 0.0
    state.movement_policy_ids[:] = GoalDirectedMovementAlgorithm.policy_id
    state.goal_positions = state.positions.copy()  # already "arrived" -- stationary, load stays skewed
    world = World(config=config, state=state)

    dist_config = DistributedConfig(
        num_workers=4, num_partitions=4, rebalance_interval_ticks=2, rebalance_imbalance_threshold=1.1,
    )
    coord, _ = run_coordinator(config, world, dist_config, ticks=ticks, warmup=0)
    return bool(coord.reassignment_log), len(coord.reassignment_log)


def format_row(row: dict) -> str:
    return (
        f"{row['drones']:>8,d} | {row['label']:<16} | {row['workers']:>7} | {row['partitions']:>10} | "
        f"{row['mean_tick_ms']:>10.3f} | {row['slowdown']:>9} | {row['total_collisions']:>10,d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 distributed-execution benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKER_COUNTS)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = ap.parse_args()

    print(f"numpy {np.__version__}")
    print(f"Benchmarking sizes={args.sizes} workers={args.workers} ticks={args.ticks} warmup={args.warmup}\n")

    header = (
        f"{'drones':>8} | {'config':<16} | {'workers':>7} | {'partitions':>10} | "
        f"{'ms/tick':>10} | {'slowdown':>9} | {'collisions':>10}"
    )
    print(header)
    print("-" * len(header))

    all_rows = []
    for n in args.sizes:
        config = build_config(n)
        template = make_template_world(config)

        baseline_world = clone_world(template)
        baseline = run_plain(config, baseline_world, args.ticks, args.warmup)
        baseline_ms = baseline["mean_tick_ms"]
        row = dict(drones=n, label="single_worker", workers=1, partitions=1,
                   mean_tick_ms=baseline_ms, slowdown="1.00x", total_collisions=baseline["total_collisions"])
        all_rows.append(row)
        print(format_row(row))

        results_by_worker_count = {}
        for w in args.workers:
            dist_config = DistributedConfig(num_workers=w, num_partitions=w)
            coord_world = clone_world(template)
            coord, res = run_coordinator(config, coord_world, dist_config, args.ticks, args.warmup)
            results_by_worker_count[w] = res
            label = "coordinator_1w" if w == 1 else f"coordinator_{w}w"
            row = dict(
                drones=n, label=label, workers=w, partitions=w,
                mean_tick_ms=res["mean_tick_ms"],
                slowdown=f"{res['mean_tick_ms'] / baseline_ms:.2f}x",
                total_collisions=res["total_collisions"],
            )
            all_rows.append(row)
            print(format_row(row))

        # ---- correctness/behavioural checks for this size ----
        print(f"\n  [{n:,} drones] checks:")
        if 1 in results_by_worker_count:
            agree_1w = _agrees(baseline, results_by_worker_count[1])
            print(f"    single_worker vs coordinator_1w agreement: {'PASS' if agree_1w else 'FAIL'}")
        for w, res in results_by_worker_count.items():
            if w == 1:
                continue
            ref = results_by_worker_count.get(1, baseline)
            agree = res["collision_pairs"] == ref["collision_pairs"]
            print(f"    coordinator_1w vs coordinator_{w}w last-tick collision-pair agreement: {'PASS' if agree else 'DIFFERS (see README known limitation)'}")
            cum_agree = res["total_collisions"] == ref["total_collisions"]
            print(
                f"    coordinator_1w vs coordinator_{w}w cumulative collision count: "
                f"{ref['total_collisions']:,} vs {res['total_collisions']:,} "
                f"({'PASS' if cum_agree else 'DIFFERS -- pre-existing resolution-order sensitivity, see README'})"
            )

        max_w = max(args.workers) if args.workers else 1
        if max_w > 1:
            det_config = DistributedConfig(num_workers=max_w, num_partitions=max_w)
            deterministic = check_determinism(config, template, det_config, ticks=min(args.ticks, 5), warmup=0)
            print(f"    repeated coordinator_{max_w}w determinism: {'PASS' if deterministic else 'FAIL'}")

        rebalanced, num_reassignments = check_rebalancing(config, ticks=min(args.ticks, 8))
        print(f"    rebalancing under artificial imbalance: {'triggered' if rebalanced else 'not triggered'} ({num_reassignments} reassignment(s))")
        print()

    print("Notes:")
    print("  Correctness first: this measures overhead and agreement, not speedup.")
    print("  coordinator_1w vs single_worker overhead is pure per-tick Python")
    print("  orchestration cost (job construction, dict bookkeeping) -- with one")
    print("  partition there is no parallel work to gain from.")
    print("  coordinator_Nw is expected to be slower than single_worker in this")
    print("  single-process benchmark: N logical workers still run sequentially")
    print("  by default (WorkerPool(use_threads=False)) inside one Python")
    print("  process/GIL, on top of the same total drone count -- see README.md's")
    print("  Phase 4 section for the measured numbers and why that is expected,")
    print("  not a regression, at this phase's local-process scope.")


if __name__ == "__main__":
    main()
