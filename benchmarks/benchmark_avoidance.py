"""Phase 2 avoidance benchmark harness.

``benchmark_simulation.py`` measures the Phase 1 tick path (Random/Scripted
policies), which never registers a ``requires_context`` policy — so it never
exercises the pre-movement spatial hash, trajectory prediction,
``NeighborFeatureBuilder``/``MovementContext`` construction, or the extra
post-movement grid rebuild that ``LocalAvoidanceMovementAlgorithm`` triggers
inside ``SimulationEngine.step``. This script measures that complete Phase 2
pipeline, using the same production ``Simulation``/``SimulationEngine`` code
(no reimplementation), comparing:

* ``GoalDirectedMovementAlgorithm`` — no avoidance, the no-avoidance baseline.
* ``LocalAvoidanceMovementAlgorithm`` — the full context-aware tick flow.

Fair comparison: for each (drone count, seed) one reproducible initial World
(positions, velocities, active mask, goal positions) is generated once, then
deep-copied for each policy with only ``movement_policy_ids`` differing.
Setup/copy/import time is excluded from all tick timing. Warm-up ticks run on
a world that is discarded afterward; measured ticks always start from a
fresh, untouched copy of the same initial state.

Usage:
    python benchmarks/benchmark_avoidance.py
    python benchmarks/benchmark_avoidance.py --sizes 1000 10000 --ticks 20 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

import numpy as np  # noqa: E402

# Reused, unmodified, from the existing Phase 1 benchmark — same world-scaling
# approach so drone density stays comparable across the two benchmarks.
from benchmark_simulation import (  # noqa: E402
    CELL_SIZE,
    COLLISION_RADIUS,
    NEAR_MISS_RADIUS,
    world_side_for,
)
from drone_sim.collisions import CollisionDetectionEngine  # noqa: E402
from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.movement import (  # noqa: E402
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementSystem,
    NeighborFeatureBuilder,
)
from drone_sim.simulation import Simulation, TickProfile  # noqa: E402
from drone_sim.spatial_hash import SpatialHashGrid  # noqa: E402
from drone_sim.state import DroneState, World  # noqa: E402
from drone_sim.trajectory import TrajectoryPredictionService  # noqa: E402
from drone_sim.validation import CollisionEventAccumulator  # noqa: E402 (reused, not reimplemented)

DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_TICKS = 10
DEFAULT_WARMUP = 2
DEFAULT_SEEDS = [42, 43, 44]

# Avoidance-relevant tunables layered on top of the reused world-size scaling.
MAX_SPEED = 5.0
MAX_ACCEL = 3.0
AVOIDANCE_MAX_ACCEL = 3.0
AVOIDANCE_STRENGTH = 1.0
PREDICTION_HORIZON = 5.0
GOAL_TOLERANCE = 1.0

POLICIES: Dict[str, Tuple[int, Callable]] = {
    "goal_directed": (GoalDirectedMovementAlgorithm.policy_id, GoalDirectedMovementAlgorithm),
    "local_avoidance": (LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm),
}


def build_config(n: int, seed: int) -> SimulationConfig:
    """Same world-scaling approach as benchmark_simulation.py's build_config,
    with Phase 2 movement/avoidance tunables added."""
    side = world_side_for(n)
    return SimulationConfig(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(side, side, side),
        collision_radius=COLLISION_RADIUS,
        near_miss_radius=NEAR_MISS_RADIUS,
        cell_size=CELL_SIZE,
        dt=1.0,
        max_speed=MAX_SPEED,
        max_accel=MAX_ACCEL,
        avoidance_max_accel=AVOIDANCE_MAX_ACCEL,
        avoidance_strength=AVOIDANCE_STRENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        goal_tolerance=GOAL_TOLERANCE,
        seed=seed,
    )


def make_template_world(config: SimulationConfig) -> World:
    """One reproducible initial World (positions/velocities/goals), generated
    once per (size, seed) and shared as the common starting point for BOTH
    policies via deep copies — never mutated itself."""
    state = DroneState.generate(config)  # reproducible from config.seed
    # A separate RNG stream (offset from config.seed) so goal placement is
    # reproducible but independent of position/velocity generation.
    goal_rng = np.random.default_rng(config.seed + 7919)
    state.goal_positions = goal_rng.uniform(
        config.bounds_min_arr, config.bounds_max_arr, size=(config.num_drones, 3)
    ).astype(np.float32)
    return World(config=config, state=state)


def clone_world_for_policy(template: World, policy_id: int) -> World:
    """Deep-copy so the two policies never share mutable state; only
    movement_policy_ids differs. Copying is never included in tick timing."""
    s = template.state
    cloned = DroneState(
        positions=s.positions.copy(),
        velocities=s.velocities.copy(),
        active_mask=s.active_mask.copy(),
        movement_policy_ids=np.full(s.num_drones, policy_id, dtype=np.int32),
        goal_positions=None if s.goal_positions is None else s.goal_positions.copy(),
    )
    return World(config=template.config, state=cloned)


@dataclass
class RunResult:
    policy: str
    drones: int
    seed: int
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    ticks_per_second: float = 0.0
    mean_candidate_pairs: float = 0.0
    unique_collision_events: float = 0.0
    collision_pair_ticks: float = 0.0
    average_collision_pairs_per_tick: float = 0.0
    average_collision_duration_ticks: float = 0.0
    avg_drone_speed: float = 0.0
    stationary_percentage: float = 0.0
    destination_completion_rate: float = 0.0


def run_policy_benchmark(
    config: SimulationConfig, template: World, policy_name: str, policy_id: int, policy_factory, ticks: int, warmup: int
) -> RunResult:
    """Run ``ticks`` measured steps of ``policy_factory()`` against a fresh
    copy of ``template``, after warming up on a separate, discarded copy.

    Metric bookkeeping (collision-event dedup, speed, arrival tracking) reads
    ``sim.step()``'s already-computed return value and ``sim.world.state``
    AFTER the timed call returns — it is not part of the timed region. The
    mean/median/p95/ticks-per-second/candidate-pair figures instead come
    straight from ``sim.metrics.summary()`` (the same production
    ``MetricsCollector`` bookkeeping ``benchmark_simulation.py`` already uses),
    not a reimplementation.
    """
    # Warm-up on a separate, throwaway world so first-tick allocation/cache
    # costs never contaminate the measured initial conditions.
    warmup_world = clone_world_for_policy(template, policy_id)
    warmup_sim = Simulation(config, movement=MovementSystem(policies={policy_id: policy_factory()}), world=warmup_world)
    for _ in range(warmup):
        warmup_sim.step()

    # Fresh identical copy for the measured run.
    world = clone_world_for_policy(template, policy_id)
    sim = Simulation(config, movement=MovementSystem(policies={policy_id: policy_factory()}), world=world)

    collision_tracker = CollisionEventAccumulator()
    speed_sum = 0.0
    speed_count = 0
    n = world.state.num_drones
    arrival_tick = np.full(n, -1, dtype=np.int64)

    for tick in range(ticks):
        result = sim.step()  # profile=None: zero timing overhead beyond Phase 1's own cost

        coll_pairs = {(int(a), int(b)) for a, b in result.collision_pairs}
        collision_tracker.record_tick(coll_pairs)

        state = sim.world.state
        active_idx = np.nonzero(state.active_mask)[0]
        speed = np.linalg.norm(state.velocities[active_idx], axis=1)
        speed_sum += float(speed.sum())
        speed_count += speed.size

        dist_to_goal = np.linalg.norm(
            state.goal_positions[active_idx] - state.positions[active_idx], axis=1
        )
        arrived_now = dist_to_goal <= config.goal_tolerance
        not_yet = arrival_tick[active_idx] < 0
        newly = active_idx[arrived_now & not_yet]
        arrival_tick[newly] = tick

    summary = sim.metrics.summary()
    collision_pair_ticks = collision_tracker.pair_ticks
    unique_collision_events = collision_tracker.unique_events

    final_speed = np.linalg.norm(
        sim.world.state.velocities[sim.world.state.active_mask], axis=1
    )
    stationary_percentage = float((final_speed < 1e-3).mean()) if final_speed.size else 0.0

    return RunResult(
        policy=policy_name,
        drones=config.num_drones,
        seed=config.seed,
        mean_ms=summary["mean_tick_ms"],
        median_ms=summary["median_tick_ms"],
        p95_ms=summary["p95_tick_ms"],
        ticks_per_second=summary["ticks_per_second"],
        mean_candidate_pairs=summary["mean_candidate_pairs"],
        unique_collision_events=float(unique_collision_events),
        collision_pair_ticks=float(collision_pair_ticks),
        average_collision_pairs_per_tick=(collision_pair_ticks / ticks) if ticks else 0.0,
        average_collision_duration_ticks=(
            collision_pair_ticks / unique_collision_events if unique_collision_events else 0.0
        ),
        avg_drone_speed=(speed_sum / speed_count) if speed_count else 0.0,
        stationary_percentage=stationary_percentage,
        destination_completion_rate=float((arrival_tick >= 0).mean()),
    )


def run_stage_profile(
    config: SimulationConfig, template: World, policy_id: int, policy_factory, ticks: int, warmup: int
) -> TickProfile:
    """Separate profiled pass (fresh world, same warm-up discipline) that
    returns the MEAN per-stage timing across ``ticks`` measured steps.

    Profiling is opt-in and adds a real (documented) measurement tax — see
    ``TickProfile``'s docstring — so this pass's ``total_ns`` is not expected
    to exactly equal the un-profiled comparison run's mean tick time.
    """
    warmup_world = clone_world_for_policy(template, policy_id)
    warmup_sim = Simulation(config, movement=MovementSystem(policies={policy_id: policy_factory()}), world=warmup_world)
    for _ in range(warmup):
        warmup_sim.step()

    world = clone_world_for_policy(template, policy_id)
    sim = Simulation(config, movement=MovementSystem(policies={policy_id: policy_factory()}), world=world)

    totals = TickProfile()
    fields = [
        "pre_grid_ns", "pre_pairs_ns", "prediction_ns", "context_ns",
        "movement_ns", "boundary_ns", "post_grid_ns", "post_pairs_ns",
        "detection_ns", "resolution_ns", "total_ns",
    ]
    skipped = False
    for _ in range(ticks):
        profile = TickProfile()
        sim.step(profile=profile)
        skipped = skipped or profile.context_stages_skipped
        for name in fields:
            setattr(totals, name, getattr(totals, name) + getattr(profile, name))

    mean = TickProfile(context_stages_skipped=skipped)
    for name in fields:
        setattr(mean, name, getattr(totals, name) / ticks if ticks else 0.0)
    return mean


@dataclass
class MemoryFootprint:
    drone_state_bytes: int = 0
    candidate_pair_bytes: int = 0
    context_bytes: int = 0
    prediction_bytes: int = 0
    total_tracked_bytes: int = 0


def measure_memory_footprint(config: SimulationConfig, template: World) -> MemoryFootprint:
    """One untimed, extra invocation of the SAME production classes used by
    SimulationEngine's context-building stages (SpatialHashGrid,
    TrajectoryPredictionService, NeighborFeatureBuilder), purely to obtain
    array handles for byte-size introspection. Not part of any timed
    comparison; deterministic and side-effect-free, so it cannot skew results.
    """
    world = clone_world_for_policy(template, LocalAvoidanceMovementAlgorithm.policy_id)
    state = world.state

    grid = SpatialHashGrid(config)
    grid.build(state.positions, state.active_indices())
    pairs = grid.candidate_pairs()

    predictor = TrajectoryPredictionService(config)
    prediction = predictor.predict(state, pairs)

    context = NeighborFeatureBuilder().build(state, prediction, state.goal_positions)

    drone_state_bytes = (
        state.positions.nbytes
        + state.velocities.nbytes
        + state.active_mask.nbytes
        + state.movement_policy_ids.nbytes
        + (state.goal_positions.nbytes if state.goal_positions is not None else 0)
    )
    candidate_pair_bytes = pairs.nbytes
    context_bytes = (
        context.neighbor_features.nbytes
        + context.neighbor_valid_mask.nbytes
        + (context.goal_vectors.nbytes if context.goal_vectors is not None else 0)
    )
    prediction_bytes = (
        prediction.pairs.nbytes
        + prediction.time_to_closest_approach.nbytes
        + prediction.predicted_separation.nbytes
        + prediction.risk.nbytes
    )
    total = drone_state_bytes + candidate_pair_bytes + context_bytes + prediction_bytes

    return MemoryFootprint(
        drone_state_bytes=drone_state_bytes,
        candidate_pair_bytes=candidate_pair_bytes,
        context_bytes=context_bytes,
        prediction_bytes=prediction_bytes,
        total_tracked_bytes=total,
    )


def peak_rss_bytes() -> Optional[int]:
    """Best-effort peak process RSS using only the standard library — no new
    dependency. Returns None if unavailable on this platform, rather than a
    fabricated number."""
    system = platform.system()
    if system in ("Linux", "Darwin"):
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return usage * 1024 if system == "Linux" else usage  # Linux: KB, macOS: bytes
        except Exception:
            return None
    if system == "Windows":
        try:
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
            ]
            ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None
        except Exception:
            return None
    return None


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def format_main_row(policy: str, drones: int, runs: List[RunResult], mem_mb: Optional[float]) -> str:
    mean_ms, std_ms = _mean_std([r.mean_ms for r in runs])
    median_ms, _ = _mean_std([r.median_ms for r in runs])
    p95_ms, _ = _mean_std([r.p95_ms for r in runs])
    tps, _ = _mean_std([r.ticks_per_second for r in runs])
    cand, _ = _mean_std([r.mean_candidate_pairs for r in runs])
    cand_per_drone = cand / drones if drones else 0.0
    uce, _ = _mean_std([r.unique_collision_events for r in runs])
    cpt, _ = _mean_std([r.collision_pair_ticks for r in runs])

    mem_str = f"{mem_mb:>10.2f}" if mem_mb is not None else f"{'n/a':>10}"
    return (
        f"{policy:>16} | {drones:>8,d} | {mean_ms:>7.2f}+/-{std_ms:<5.2f} | {median_ms:>8.2f} | "
        f"{p95_ms:>8.2f} | {tps:>8.2f} | {cand:>10,.0f} | {cand_per_drone:>7.2f} | "
        f"{uce:>7.2f} | {cpt:>7.2f} | {mem_str}"
    )


def format_stage_row(policy: str, drones: int, profile: TickProfile) -> str:
    def ms(ns: float) -> float:
        return ns / 1e6

    return (
        f"{policy:>16} | {drones:>8,d} | {ms(profile.pre_grid_ns):>9.3f} | {ms(profile.pre_pairs_ns):>9.3f} | "
        f"{ms(profile.prediction_ns):>9.3f} | {ms(profile.context_ns):>9.3f} | {ms(profile.movement_ns):>9.3f} | "
        f"{ms(profile.boundary_ns):>9.3f} | {ms(profile.post_grid_ns):>9.3f} | {ms(profile.post_pairs_ns):>9.3f} | "
        f"{ms(profile.detection_ns):>9.3f} | {ms(profile.resolution_ns):>9.3f} | {ms(profile.total_ns):>9.3f}"
        + ("  (context stages skipped: requires_context=False)" if profile.context_stages_skipped else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Drone simulator Phase 2 avoidance benchmark")
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="measured ticks per size/seed/policy")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warmup ticks (discarded world)")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="drone counts to benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="deterministic seeds to average over")
    args = ap.parse_args()

    print(f"numpy {np.__version__}")
    print(
        f"Benchmarking sizes={args.sizes} ticks={args.ticks} warmup={args.warmup} seeds={args.seeds}\n"
    )

    header = (
        f"{'policy':>16} | {'drones':>8} | {'mean ms':>13} | {'med ms':>8} | {'p95 ms':>8} | "
        f"{'ticks/s':>8} | {'cand/t':>10} | {'cand/dr':>7} | {'uniq col':>7} | {'col-pr-t':>7} | {'mem MB':>10}"
    )

    all_main_results: Dict[Tuple[str, int], List[RunResult]] = {}
    stage_results: Dict[Tuple[str, int], TickProfile] = {}
    memory_results: Dict[int, MemoryFootprint] = {}
    largest_completed = 0
    failure: Optional[str] = None

    tracemalloc.start()

    for n in args.sizes:
        try:
            for seed in args.seeds:
                config = build_config(n, seed)
                template = make_template_world(config)
                for policy_name, (policy_id, policy_factory) in POLICIES.items():
                    r = run_policy_benchmark(config, template, policy_name, policy_id, policy_factory, args.ticks, args.warmup)
                    all_main_results.setdefault((policy_name, n), []).append(r)

            # Stage timing + memory: first seed only, to keep runtime bounded.
            first_config = build_config(n, args.seeds[0])
            first_template = make_template_world(first_config)
            for policy_name, (policy_id, policy_factory) in POLICIES.items():
                stage_results[(policy_name, n)] = run_stage_profile(
                    first_config, first_template, policy_id, policy_factory, args.ticks, args.warmup
                )
            memory_results[n] = measure_memory_footprint(first_config, first_template)
            largest_completed = n
        except MemoryError as exc:
            failure = f"MemoryError at {n:,} drones: {exc}"
            print(f"\n!! {failure}\n")
            break
        except Exception as exc:  # noqa: BLE001 - report any failure honestly, don't hide it
            failure = f"{type(exc).__name__} at {n:,} drones: {exc}"
            print(f"\n!! {failure}\n")
            break

    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = peak_rss_bytes()

    print(header)
    print("-" * len(header))
    for n in args.sizes:
        mem = memory_results.get(n)
        mem_mb = (mem.total_tracked_bytes / (1024 * 1024)) if mem else None
        for policy_name in POLICIES:
            runs = all_main_results.get((policy_name, n))
            if not runs:
                continue
            print(format_main_row(policy_name, n, runs, mem_mb))

    slowdown_notes = []
    for n in args.sizes:
        gd = all_main_results.get(("goal_directed", n))
        la = all_main_results.get(("local_avoidance", n))
        if gd and la:
            gd_mean = float(np.mean([r.mean_ms for r in gd]))
            la_mean = float(np.mean([r.mean_ms for r in la]))
            ratio = (la_mean / gd_mean) if gd_mean > 0 else float("inf")
            slowdown_notes.append((n, gd_mean, la_mean, ratio))

    print("\nSlowdown ratio (local_avoidance_mean_ms / goal_directed_mean_ms):")
    for n, gd_mean, la_mean, ratio in slowdown_notes:
        print(f"  {n:>8,d} drones: {la_mean:>9.3f} ms / {gd_mean:>9.3f} ms = {ratio:>6.2f}x")

    print("\nPhase 2 stage timings (mean ms/tick; avoidance policy plus goal-directed for skip verification):")
    stage_header = (
        f"{'policy':>16} | {'drones':>8} | {'pre_grid':>9} | {'pre_prs':>9} | {'predict':>9} | "
        f"{'context':>9} | {'movement':>9} | {'boundary':>9} | {'post_grd':>9} | {'post_prs':>9} | "
        f"{'detect':>9} | {'resolve':>9} | {'total':>9}"
    )
    print(stage_header)
    print("-" * len(stage_header))
    for n in args.sizes:
        for policy_name in POLICIES:
            key = (policy_name, n)
            if key in stage_results:
                print(format_stage_row(policy_name, n, stage_results[key]))

    print("\nMemory (approximate tracked NumPy-array bytes, one representative tick, local_avoidance world):")
    for n in args.sizes:
        mem = memory_results.get(n)
        if mem is None:
            continue
        print(
            f"  {n:>8,d} drones: DroneState={mem.drone_state_bytes/1e6:>8.2f} MB | "
            f"candidate_pairs={mem.candidate_pair_bytes/1e6:>7.2f} MB | "
            f"MovementContext={mem.context_bytes/1e6:>7.2f} MB | "
            f"PredictionResult={mem.prediction_bytes/1e6:>7.2f} MB | "
            f"total_tracked={mem.total_tracked_bytes/1e6:>8.2f} MB"
        )

    print(
        f"\ntracemalloc peak (Python/NumPy allocations traced during this whole benchmark run, "
        f"NOT total process RSS): {traced_peak/1e6:.2f} MB"
    )
    if rss is not None:
        print(f"Peak process working-set size (OS-reported, whole process, whole run): {rss/1e6:.2f} MB")
    else:
        print("Peak process RSS: not available on this platform without adding a new dependency.")

    if failure:
        print(f"\nLargest population that completed successfully: {largest_completed:,} drones")
        print(f"Failure beyond that point: {failure}")
    else:
        print(f"\nAll requested sizes completed successfully up to {args.sizes[-1]:,} drones.")

    print("\nNotes:")
    print("  World scaling reused unmodified from benchmark_simulation.py (same ~64 cells/drone density).")
    print(
        "  Stage-timing pass adds one extra grid.candidate_pairs() call (post_pairs measured separately "
        "from detection) versus the un-profiled comparison run above - see TickProfile's docstring. "
        "Stage-timing and memory measurements use only the first seed to keep runtime bounded."
    )
    print("  goal_directed correctly reports pre_grid/pre_pairs/prediction/context as skipped (requires_context=False).")


if __name__ == "__main__":
    main()
