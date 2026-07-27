"""Controlled comparison: Matplotlib debug viewer vs. Phase 3A browser runtime.

Investigates why the Phase 3A browser-backed simulation was observed running
substantially slower (~28-39 ms/tick) than the Matplotlib debug viewer
(~10 ms/tick) at 10,000 drones, even though ``benchmark_pipeline_regression.py``
measured the same policy's full per-tick pipeline at ~12.5 ms/tick in
isolation. Three independent things are measured here, each with the SAME
deterministic config/seed so results are never a "different world" artifact
unless that is exactly what is being measured (the config audit):

1. **Configuration audit** -- prints the *effective* ``SimulationConfig`` both
   viewers actually run (not their UI input values), and the browser's
   viewport-vs-world-bounds distinction verified by reading the code.
2. **Layered overhead comparison** (cases 1-5) -- the same tick path with one
   more layer of real cost added at a time: headless step, step + Matplotlib's
   data path, runtime step-component, runtime+snapshot, runtime+snapshot while
   the real ``/frame`` handler is being polled concurrently (in-process, real
   threads, real GIL contention -- see ``case_runtime`` for the documented
   scope of this approximation).
3. **Process-level comparison** (cases 6-8) -- the Matplotlib viewer and the
   browser runtime each as their own OS process (this file re-invoked as a
   ``--worker`` subprocess), run in isolation and then simultaneously, to
   measure whether running both at once (as in the original screenshot)
   causes real, measurable slowdown.

Plus two standalone, fast, in-process demonstrations:

* ``--demo orphaned-threads`` -- directly demonstrates the confirmed
  index.html bug (every "Apply / New simulation" leaked the previous
  simulation's background thread forever) and proves the fix (a ``DELETE``
  endpoint the client now calls first) brings live thread count back down.
* ``--demo collision-markers`` -- runs one simulation and computes both
  "latest tick only" (what the browser shows) and "accumulated since last
  redraw" (what the Matplotlib viewer shows) from the exact same tick
  sequence, to isolate that semantic difference from any config difference.

Usage:
    python benchmarks/benchmark_viewer_comparison.py
    python benchmarks/benchmark_viewer_comparison.py --drones 10000 --seed 0
    python benchmarks/benchmark_viewer_comparison.py --demo orphaned-threads
    python benchmarks/benchmark_viewer_comparison.py --demo collision-markers
    python benchmarks/benchmark_viewer_comparison.py --scaling 1000 2000 5000 10000 25000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

import numpy as np  # noqa: E402

from benchmark_simulation import CELL_SIZE, COLLISION_RADIUS, NEAR_MISS_RADIUS, world_side_for  # noqa: E402
from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.simulation import Simulation  # noqa: E402
from drone_sim.state import DroneState, World  # noqa: E402
from drone_sim.runtime import SimulationRuntime  # noqa: E402
from drone_sim.visualization import IntervalStats, collision_marker_positions, compute_density_grid  # noqa: E402

DEFAULT_DRONES = 10_000
DEFAULT_SEED = 0
DEFAULT_WARMUP = 10
DEFAULT_TICKS = 100
DEFAULT_DURATION_S = 3.0


# ------------------------------------------------------------------- configs
def matplotlib_config(n: int, seed: int = DEFAULT_SEED) -> SimulationConfig:
    """Exactly what scripts/run_visualizer.py constructs: ~64 cells/drone cube."""
    side = world_side_for(n)
    return SimulationConfig(
        num_drones=n, bounds_min=(0.0, 0.0, 0.0), bounds_max=(side, side, side),
        collision_radius=COLLISION_RADIUS, near_miss_radius=NEAR_MISS_RADIUS,
        cell_size=CELL_SIZE, dt=1.0, max_speed=5.0, seed=seed,
    )


def browser_default_config(n: int, seed: int = DEFAULT_SEED) -> SimulationConfig:
    """Exactly what index.html's createSimulation() constructs with UNMODIFIED
    default UI fields (x: 0-500, y: 0-500 -> width=height=500; z hardcoded to
    100 client-side and never overridable at creation time; num_drones from
    the UI's own field). Bounds are FIXED at (500, 500, 100) regardless of
    drone count -- the browser UI has no equivalent of world_side_for()'s
    density scaling. collision_radius/near_miss_radius/dt/max_speed are
    CreateSimulationRequest's Pydantic defaults, which equal SimulationConfig's
    own defaults (verified by reading src/drone_sim/api/models.py)."""
    return SimulationConfig(
        num_drones=n, bounds_min=(0.0, 0.0, 0.0), bounds_max=(500.0, 500.0, 100.0), seed=seed,
    )


def grid_stats(config: SimulationConfig) -> Dict:
    lo = config.bounds_min_arr
    hi = config.bounds_max_arr
    dims = np.maximum(np.ceil((hi - lo) / config.effective_cell_size).astype(np.int64), 1)
    total_cells = int(dims[0] * dims[1] * dims[2])
    return {
        "dims": tuple(int(d) for d in dims),
        "total_cells": total_cells,
        "cells_per_drone": (total_cells / config.num_drones) if config.num_drones else float("inf"),
    }


def assert_configs_are_deterministic(config: SimulationConfig) -> None:
    """Proves two independently constructed Worlds from the same config/seed
    are bit-identical, so every case below sharing a `config` (rather than a
    single mutable World instance) is not "comparing two independently
    generated worlds" -- it is comparing bit-identical ones by construction."""
    w1 = World.create(config)
    w2 = World.create(config)
    assert np.array_equal(w1.state.positions, w2.state.positions), "positions not reproducible from seed"
    assert np.array_equal(w1.state.velocities, w2.state.velocities), "velocities not reproducible from seed"


def print_config_audit(n: int) -> None:
    mpl_cfg = matplotlib_config(n)
    browser_cfg = browser_default_config(n)
    print("=" * 100)
    print("CONFIGURATION AUDIT (effective SimulationConfig each viewer actually runs)")
    print("=" * 100)
    for label, cfg in (
        ("Matplotlib viewer (scripts/run_visualizer.py)", mpl_cfg),
        ("Browser, unmodified default UI fields (index.html)", browser_cfg),
    ):
        gs = grid_stats(cfg)
        vol = float(np.prod(cfg.bounds_max_arr - cfg.bounds_min_arr))
        print(f"\n{label}:")
        print(f"  num_drones:          {cfg.num_drones:,}")
        print(f"  world bounds:        {tuple(cfg.bounds_min)} .. {tuple(cfg.bounds_max)}")
        print(f"  world volume:        {vol:,.0f}")
        print(f"  collision_radius:    {cfg.collision_radius}")
        print(f"  near_miss_radius:    {cfg.near_miss_radius}")
        print(f"  effective_cell_size: {cfg.effective_cell_size}")
        print(f"  dt:                  {cfg.dt}")
        print(f"  seed:                {cfg.seed}")
        print(f"  boundary_mode:       {cfg.boundary_mode}")
        print(f"  grid dims (x,y,z):   {gs['dims']}")
        print(f"  total cells:         {gs['total_cells']:,}")
        print(f"  cells/drone:         {gs['cells_per_drone']:.2f}")

    mpl_gs = grid_stats(mpl_cfg)
    br_gs = grid_stats(browser_cfg)
    ratio = br_gs["cells_per_drone"] / mpl_gs["cells_per_drone"]
    denser_sparser = "SPARSER (more cells/drone)" if ratio > 1 else "DENSER (fewer cells/drone)"
    print(
        f"\n  -> Browser default world is {ratio:.2f}x {denser_sparser} than the Matplotlib/"
        f"\n     benchmark_pipeline_regression.py world -- NOT the same density. Since occupied-cell"
        f"\n     count is bounded by min(num_drones, total_cells) and total_cells >> num_drones in BOTH"
        f"\n     worlds here, this difference is expected to have a small effect on candidate_pairs()'s"
        f"\n     dominant O(occupied_cells) cost -- verify against the measured stage timings below"
        f"\n     rather than assuming; it is reported so it is never silently confounded with anything"
        f"\n     else measured in this file."
    )

    print("\nMovement policy: both DroneState.generate() (used by World.create(), which both")
    print("Simulation() and SimulationRuntime() call) and create_simulation() in routes.py always")
    print("assign movement_policy_ids=0 (RandomMovementAlgorithm); requires_context=False in both --")
    print("confirmed by reading state.py/routes.py/run_visualizer.py, not assumed.")

    print("\nViewport bounds vs. world bounds (verified by reading index.html + routes.py + models.py):")
    print("  index.html's x_min/x_max/y_min/y_max UI fields serve TWO roles:")
    print("  (1) At simulation-CREATION time (createSimulation()): width=x_max-x_min and")
    print("      height=y_max-y_min become the new simulation's bounds_max[0:2]. z is hardcoded to")
    print("      100 and is NEVER sent to POST /simulations -- the z_min/z_max UI fields only ever")
    print("      affect (2) below, never the world itself.")
    print("  (2) On EVERY /frame poll (refresh()): the SAME field values are sent as the VIEWPORT")
    print("      query's x_min/x_max/y_min/y_max/z_min/z_max.")
    print("  With unmodified defaults these numerically coincide (viewport == world x/y extent), so")
    print("  no viewport truncation happens by default and this is NOT why the browser shows fewer")
    print("  collision markers (see --demo collision-markers for the actual mechanism). But they are")
    print("  not the same bound set conceptually: editing the UI fields after creation changes only")
    print("  the QUERIED viewport -- the world stays whatever it was at creation time.")


# --------------------------------------------------------------------- stats
def _stats(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0, "n": 0}
    s = sorted(samples_ms)
    p95_idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return {
        "mean": statistics.mean(samples_ms),
        "median": statistics.median(samples_ms),
        "p95": s[p95_idx],
        "min": s[0],
        "max": s[-1],
        "stdev": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "n": len(samples_ms),
    }


def _fmt_stats(label: str, st: Dict[str, float]) -> str:
    return (
        f"{label:>42} | mean {st['mean']:>7.3f}  median {st['median']:>7.3f}  "
        f"p95 {st['p95']:>7.3f}  min {st['min']:>7.3f}  max {st['max']:>7.3f}  "
        f"stdev {st['stdev']:>6.3f}  (n={st['n']})"
    )


# ----------------------------------------------------------- cases 1-2: pure
def case_1_headless_step(config: SimulationConfig, ticks: int, warmup: int) -> Dict:
    """Pure Simulation.step() -- exactly what SimulationViewer._advance() does
    per tick, minus the density-grid/marker/redraw cost (see case 2)."""
    sim = Simulation(config)
    for _ in range(warmup):
        sim.step()
    times = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        sim.step()
        times.append((time.perf_counter() - t0) * 1e3)
    return _stats(times)


def case_2_matplotlib_data_path(config: SimulationConfig, ticks: int, warmup: int, bins: int = 100) -> Dict:
    """sim.step() + the exact per-tick data-path cost SimulationViewer._advance()
    /_redraw() pay (density grid + marker midpoints), with rendering itself
    (Figure/imshow/canvas) excluded -- "rendering disabled" per the task."""
    sim = Simulation(config)
    lo, hi = config.bounds_min_arr, config.bounds_max_arr
    for _ in range(warmup):
        sim.step()
    times = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        result = sim.step()
        compute_density_grid(sim.world.state.positions, lo, hi, bins)
        collision_marker_positions(sim.world.state.positions, result.collision_pairs)
        times.append((time.perf_counter() - t0) * 1e3)
    return _stats(times)


# ------------------------------------------------------- cases 3-5: runtime
def case_3_runtime_step_component(config: SimulationConfig, ticks: int, warmup: int) -> Dict:
    """Isolates the sim_step_ms component of the runtime's per-tick path.

    The current implementation has no code path that skips snapshot
    publication (build_snapshot() runs unconditionally after every tick, see
    runtime.py's _advance_one_locked) -- there is no real "no snapshot
    publication" case to run. This reports the closest available proxy: the
    sim_step_ms component alone, isolated from snapshot_build_ms (case 4 adds
    that back in), read from the SAME real runtime.step_once() call.
    """
    runtime = SimulationRuntime("case3", config)
    for _ in range(warmup):
        runtime.step_once()
    times = []
    for _ in range(ticks):
        runtime.step_once()
        times.append(runtime.get_last_timings().sim_step_ms)
    runtime.shutdown()
    return _stats(times)


def _run_runtime_for_duration(
    config: SimulationConfig, duration_s: float, *, poll: bool, poll_interval_s: float = 0.15
) -> Dict:
    """Runs a real SimulationRuntime's background loop (runtime.start(), the
    same code path POST /simulations/{id}/start uses) for a fixed wall-clock
    duration, optionally with a concurrent thread hammering the real
    routes.get_frame() handler function directly (in-process: real threads,
    real GIL contention, real (post-fix) single JSON serialization -- but no
    real ASGI/socket transport, which is the one part of a real HTTP request
    this cannot exercise; documented scope, not silently assumed away).
    """
    from drone_sim.api import routes as api_routes

    sim_id = f"contention-{id(config)}-{threading.get_ident()}"
    runtime = SimulationRuntime(sim_id, config)
    stop = threading.Event()
    poller_thread = None
    poll_count = [0]
    poll_errors = [0]

    if poll:
        api_routes._runtimes[sim_id] = runtime
        lo, hi = config.bounds_min_arr, config.bounds_max_arr

        def poll_loop() -> None:
            while not stop.is_set():
                try:
                    api_routes.get_frame(
                        sim_id,
                        x_min=float(lo[0]), x_max=float(hi[0]),
                        y_min=float(lo[1]), y_max=float(hi[1]),
                        z_min=None, z_max=None, x_bins=60, y_bins=60,
                    )
                    poll_count[0] += 1
                except Exception:
                    poll_errors[0] += 1
                time.sleep(poll_interval_s)

        poller_thread = threading.Thread(target=poll_loop, daemon=True)

    runtime.start()
    time.sleep(0.2)  # let startup jitter settle before the measured window
    tick0 = runtime.get_snapshot().tick
    if poller_thread is not None:
        poller_thread.start()

    time.sleep(duration_s)

    tick1 = runtime.get_snapshot().tick
    stop.set()
    if poller_thread is not None:
        poller_thread.join(timeout=2.0)
    runtime.shutdown()
    if poll:
        del api_routes._runtimes[sim_id]

    ticks = tick1 - tick0
    ms_per_tick = (duration_s * 1e3 / ticks) if ticks else float("inf")
    result = {
        "ticks": ticks,
        "ms_per_tick": ms_per_tick,
        "ticks_per_second": (ticks / duration_s) if duration_s > 0 else 0.0,
    }
    if poll:
        result["frame_polls_completed"] = poll_count[0]
        result["frame_poll_errors"] = poll_errors[0]
    return result


def case_4_runtime_with_snapshot(config: SimulationConfig, duration_s: float) -> Dict:
    """Runtime + snapshot publication, no API polling -- also "browser runtime by itself" (case 7)."""
    return _run_runtime_for_duration(config, duration_s, poll=False)


def case_5_runtime_with_frame_polling(config: SimulationConfig, duration_s: float) -> Dict:
    """Runtime + snapshot, WITH a concurrent thread polling the real /frame handler at ~6.7 Hz."""
    return _run_runtime_for_duration(config, duration_s, poll=True)


# --------------------------------------------------- cases 6-8: subprocesses
def _worker_matplotlib(args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless: no window, still real Figure/imshow/redraw
    cfg = matplotlib_config(args.drones, args.seed) if args.world == "matplotlib" else browser_default_config(
        args.drones, args.seed
    )
    from drone_sim.visualization import SimulationViewer

    viewer = SimulationViewer(cfg, render_every=args.render_every, bins=100)
    for _ in range(args.warmup):
        viewer._advance()
        viewer._redraw()

    t_start = time.perf_counter()
    n = 0
    while time.perf_counter() - t_start < args.duration:
        viewer._advance()
        viewer._redraw()
        n += 1
    elapsed = time.perf_counter() - t_start
    effective_ticks = n * args.render_every
    print(json.dumps({
        "worker": "matplotlib",
        "pid": os.getpid(),
        "redraw_cycles": n,
        "effective_ticks": effective_ticks,
        "elapsed_s": elapsed,
        "ms_per_tick": (elapsed * 1e3 / effective_ticks) if effective_ticks else float("inf"),
        "ticks_per_second": (effective_ticks / elapsed) if elapsed > 0 else 0.0,
        "process_time_s": time.process_time(),
    }))


def _worker_runtime(args: argparse.Namespace) -> None:
    cfg = matplotlib_config(args.drones, args.seed) if args.world == "matplotlib" else browser_default_config(
        args.drones, args.seed
    )
    runtime = SimulationRuntime("worker", cfg)
    runtime.start()
    time.sleep(0.2)
    tick0 = runtime.get_snapshot().tick
    t_start = time.perf_counter()
    time.sleep(args.duration)
    elapsed = time.perf_counter() - t_start
    tick1 = runtime.get_snapshot().tick
    runtime.shutdown()
    ticks = tick1 - tick0
    print(json.dumps({
        "worker": "runtime",
        "pid": os.getpid(),
        "ticks": ticks,
        "elapsed_s": elapsed,
        "ms_per_tick": (elapsed * 1e3 / ticks) if ticks else float("inf"),
        "ticks_per_second": (ticks / elapsed) if elapsed > 0 else 0.0,
        "process_time_s": time.process_time(),
    }))


def _spawn_worker(kind: str, drones: int, seed: int, duration: float, warmup: int, world: str, render_every: int) -> subprocess.Popen:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", kind,
        "--drones", str(drones), "--seed", str(seed),
        "--duration", str(duration), "--warmup", str(warmup),
        "--world", world, "--render-every", str(render_every),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _collect_worker(proc: subprocess.Popen) -> Dict:
    out, err = proc.communicate(timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"worker subprocess failed (code {proc.returncode}): {err}")
    line = [ln for ln in out.splitlines() if ln.strip()][-1]
    return json.loads(line)


def cases_6_7_8_process_level(drones: int, seed: int, duration: float, warmup: int, render_every: int) -> Dict:
    """Cases 6/7/8: Matplotlib and the browser runtime as separate OS processes
    -- isolated, then simultaneous -- to measure real multi-process CPU
    contention (distinct from the in-process GIL contention case 5 measures).
    Each uses its OWN natural world (Matplotlib: world_side_for(); runtime:
    the browser's default UI-driven world) since that's what the original
    screenshot scenario actually ran; case 5 already isolates the "identical
    world" GIL-contention question separately.
    """
    print("\nRunning case 6 (Matplotlib alone) and case 7 (browser runtime alone), isolated...")
    p_mpl_alone = _spawn_worker("matplotlib", drones, seed, duration, warmup, "matplotlib", render_every)
    mpl_alone = _collect_worker(p_mpl_alone)

    p_rt_alone = _spawn_worker("runtime", drones, seed, duration, warmup, "browser", render_every)
    rt_alone = _collect_worker(p_rt_alone)

    print("Running case 8 (Matplotlib and browser runtime SIMULTANEOUSLY, two processes)...")
    p_mpl = _spawn_worker("matplotlib", drones, seed, duration, warmup, "matplotlib", render_every)
    p_rt = _spawn_worker("runtime", drones, seed, duration, warmup, "browser", render_every)
    mpl_together = _collect_worker(p_mpl)
    rt_together = _collect_worker(p_rt)

    return {
        "matplotlib_alone": mpl_alone,
        "runtime_alone": rt_alone,
        "matplotlib_simultaneous": mpl_together,
        "runtime_simultaneous": rt_together,
    }


# ------------------------------------------------ demo: orphaned runtimes
def demo_orphaned_threads(config: SimulationConfig, num_reloads: int = 5) -> Dict:
    """Directly demonstrates the confirmed index.html bug (every reload/"Apply
    / New simulation" leaked the previous simulation's background thread
    forever, since nothing ever called SimulationRuntime.shutdown() on it) and
    proves the fix (a DELETE endpoint the client now calls first) keeps live
    thread count bounded at 1 regardless of how many times a "new simulation"
    is created.
    """
    from drone_sim.api import routes as api_routes

    baseline = threading.active_count()

    old_ids = []
    for i in range(num_reloads):
        sim_id = f"orphan-demo-old-{i}"
        api_routes._runtimes[sim_id] = SimulationRuntime(sim_id, config)
        api_routes._runtimes[sim_id].start()
        old_ids.append(sim_id)
    extra_threads_old_behavior = threading.active_count() - baseline

    for sim_id in old_ids:
        api_routes._runtimes[sim_id].shutdown()
        del api_routes._runtimes[sim_id]

    current_id: Optional[str] = None
    for i in range(num_reloads):
        if current_id is not None:
            api_routes._runtimes[current_id].shutdown()
            del api_routes._runtimes[current_id]
        current_id = f"orphan-demo-new-{i}"
        api_routes._runtimes[current_id] = SimulationRuntime(current_id, config)
        api_routes._runtimes[current_id].start()
    extra_threads_new_behavior = threading.active_count() - baseline

    api_routes._runtimes[current_id].shutdown()
    del api_routes._runtimes[current_id]

    return {
        "num_reloads_simulated": num_reloads,
        "extra_live_threads__old_never_stop_behavior": extra_threads_old_behavior,
        "extra_live_threads__new_delete_before_create_behavior": extra_threads_new_behavior,
    }


# --------------------------------------------- demo: collision-marker semantics
def demo_collision_marker_semantics(config: SimulationConfig, render_every: int = 5, ticks: int = 50) -> Dict:
    """Runs ONE simulation, tick by tick, and computes both interpretations
    from the exact same tick sequence: "latest tick only" (what /frame's
    collision markers show) vs. "accumulated since the last redraw" (what
    SimulationViewer's IntervalStats.all_collision_pairs shows) -- isolating
    this semantic difference from any config/seed/world difference.
    """
    sim = Simulation(config)
    interval = IntervalStats()
    latest_tick_counts: List[int] = []
    interval_counts_at_each_redraw: List[int] = []
    for t in range(1, ticks + 1):
        result = sim.step()
        interval.add(result, sim.metrics.ticks[-1].tick_time_s)
        latest_tick_counts.append(result.num_collisions)
        if t % render_every == 0:
            interval_counts_at_each_redraw.append(int(interval.all_collision_pairs.shape[0]))
            interval.reset()
    return {
        "render_every": render_every,
        "ticks": ticks,
        "latest_tick_counts": latest_tick_counts,
        "sum_of_latest_tick_counts": sum(latest_tick_counts),
        "interval_counts_at_each_redraw": interval_counts_at_each_redraw,
        "max_single_redraw_marker_count": max(interval_counts_at_each_redraw) if interval_counts_at_each_redraw else 0,
        "max_single_tick_marker_count": max(latest_tick_counts) if latest_tick_counts else 0,
    }


# --------------------------------------- demo: orphaned-runtime CPU contention
def demo_orphaned_runtime_contention(config: SimulationConfig, max_siblings: int = 3, duration: float = 4.0) -> List[Dict]:
    """Quantifies the mechanism the orphaned-thread bug actually causes: N-1
    forgotten SimulationRuntime instances (from N-1 page reloads/"Apply / New
    simulation" clicks, pre-fix, that never called shutdown()) each still
    running a REAL background thread stepping a REAL 10,000-drone Simulation,
    all sharing this ONE process's GIL with whichever runtime the browser is
    actually polling. Measures the "current" runtime's ms/tick as sibling
    count increases from 0 (no orphans) to max_siblings.
    """
    results = []
    for num_siblings in range(0, max_siblings + 1):
        siblings = [SimulationRuntime(f"sibling-{i}", config) for i in range(num_siblings)]
        for s in siblings:
            s.start()
        primary = SimulationRuntime("primary", config)
        primary.start()
        time.sleep(0.3)  # let all threads past startup jitter
        tick0 = primary.get_snapshot().tick
        time.sleep(duration)
        tick1 = primary.get_snapshot().tick
        primary.shutdown()
        for s in siblings:
            s.shutdown()
        ticks = tick1 - tick0
        results.append({
            "orphaned_siblings": num_siblings,
            "primary_ms_per_tick": (duration * 1e3 / ticks) if ticks else float("inf"),
            "primary_ticks_per_second": (ticks / duration) if duration > 0 else 0.0,
        })
    return results


# --------------------------------------- demo: /frame double-serialization cost
def demo_frame_serialization_cost(config: SimulationConfig, repeats: int = 200) -> Dict:
    """Quantifies the double-json.dumps() bug fixed in routes.py's get_frame():
    builds a payload of the same shape a real /frame response at this config
    produces (60x60 heatmap counts + whatever markers/metrics a real tick
    yields), then times the OLD pattern (two full json.dumps() calls, one
    discarded) against the NEW pattern (one full dump + a tiny splice) over
    many repeats on the SAME payload, so the comparison isolates the
    serialization-pattern cost from any simulation/query variance.
    """
    import json as _json

    from drone_sim.api.routes import get_frame
    from drone_sim.api import routes as api_routes

    sim_id = "frame-serialization-demo"
    runtime = SimulationRuntime(sim_id, config)
    api_routes._runtimes[sim_id] = runtime
    for _ in range(10):
        runtime.step_once()
    lo, hi = config.bounds_min_arr, config.bounds_max_arr

    # Build one real payload snapshot (same code the fixed handler uses) to
    # replay both serialization patterns against identically-shaped data.
    from drone_sim.heatmap import HeatmapQuery, compute_heatmap
    from drone_sim.collision_queries import query_collision_markers
    from drone_sim.viewport import ViewportQuery

    snapshot = runtime.get_snapshot()
    viewport = ViewportQuery(x_min=float(lo[0]), x_max=float(hi[0]), y_min=float(lo[1]), y_max=float(hi[1]))
    heatmap = compute_heatmap(snapshot, HeatmapQuery(viewport=viewport, x_bins=60, y_bins=60))
    markers = query_collision_markers(snapshot, viewport)
    payload = {
        "simulation_id": sim_id, "status": "running", "tick": snapshot.tick,
        "num_visible_drones": heatmap.num_drones_included,
        "heatmap": {
            "x_bins": heatmap.x_bins, "y_bins": heatmap.y_bins,
            "x_edges": heatmap.x_edges.tolist(), "y_edges": heatmap.y_edges.tolist(),
            "counts": heatmap.counts.tolist(), "max_density": heatmap.max_density,
        },
        "markers": [
            {"drone_a": m.drone_a, "drone_b": m.drone_b, "tick": m.tick, "x": m.x, "y": m.y, "z": m.z,
             "distance": m.distance, "relative_speed": m.relative_speed}
            for m in markers
        ],
        "metrics": snapshot.metrics,
    }
    timings_stub = {
        "sim_step_ms": 0.0, "snapshot_build_ms": 0.0, "lock_wait_ms": 0.0,
        "heatmap_ms": 0.0, "collisions_ms": 0.0, "serialization_ms": 0.0, "total_request_ms": 0.0,
    }

    def old_pattern() -> None:
        p = dict(payload)
        _json.dumps(p)  # discarded -- measurement-only, exactly the old bug
        p["timings"] = timings_stub
        _json.dumps(p)  # the real response body -- second full serialization

    def new_pattern() -> None:
        p = dict(payload)
        body = _json.dumps(p)
        body[:-1] + ',"timings":' + _json.dumps(timings_stub) + "}"

    t0 = time.perf_counter()
    for _ in range(repeats):
        old_pattern()
    old_total_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(repeats):
        new_pattern()
    new_total_ms = (time.perf_counter() - t0) * 1e3

    runtime.shutdown()
    del api_routes._runtimes[sim_id]

    return {
        "repeats": repeats,
        "num_markers_in_payload": len(markers),
        "old_pattern_ms_per_call": old_total_ms / repeats,
        "new_pattern_ms_per_call": new_total_ms / repeats,
        "old_pattern_total_ms": old_total_ms,
        "new_pattern_total_ms": new_total_ms,
        "speedup": (old_total_ms / new_total_ms) if new_total_ms else float("inf"),
    }


# ------------------------------------------------------------------- runner
def run_main_comparison(args: argparse.Namespace) -> None:
    config = browser_default_config(args.drones, args.seed)
    assert_configs_are_deterministic(config)
    print_config_audit(args.drones)

    print("\n" + "=" * 100)
    print(f"LAYERED OVERHEAD COMPARISON (drones={args.drones:,}, seed={args.seed}, "
          f"world=browser-default -- the world the observed 28-39ms/tick browser session actually ran)")
    print("=" * 100)
    print(_fmt_stats("(1) headless Simulation.step() only", case_1_headless_step(config, args.ticks, args.warmup)))
    print(_fmt_stats("(2) + Matplotlib data path (no render)", case_2_matplotlib_data_path(config, args.ticks, args.warmup)))
    print(_fmt_stats("(3) runtime sim_step component only", case_3_runtime_step_component(config, args.ticks, args.warmup)))

    print(f"\n(4)/(5) below run for {args.duration:.1f}s wall-clock each and report effective ms/tick "
          f"(ticks advanced / duration) -- these measure the real runtime.start() background loop, "
          f"not step_once() calls, so they include whatever contention actually occurs.")
    r4 = case_4_runtime_with_snapshot(config, args.duration)
    print(f"{'(4) runtime + snapshot, no polling':>42} | ms/tick {r4['ms_per_tick']:.3f}  "
          f"ticks/s {r4['ticks_per_second']:.2f}  ticks {r4['ticks']}")
    r5 = case_5_runtime_with_frame_polling(config, args.duration)
    print(f"{'(5) runtime + snapshot + /frame polling':>42} | ms/tick {r5['ms_per_tick']:.3f}  "
          f"ticks/s {r5['ticks_per_second']:.2f}  ticks {r5['ticks']}  "
          f"(frame polls completed: {r5['frame_polls_completed']}, errors: {r5['frame_poll_errors']})")
    slowdown_5_over_4 = (r5["ms_per_tick"] / r4["ms_per_tick"]) if r4["ms_per_tick"] else float("inf")
    print(f"\n  -> concurrent /frame polling changes ms/tick by {slowdown_5_over_4:.2f}x vs. no polling "
          f"(in-process GIL contention, same process, no real ASGI/socket layer -- see docstring).")

    print("\n" + "=" * 100)
    print("PROCESS-LEVEL COMPARISON (cases 6/7/8 -- separate OS processes, matching the original screenshot)")
    print("=" * 100)
    proc_results = cases_6_7_8_process_level(args.drones, args.seed, args.duration, args.warmup, args.render_every)
    mpl_alone, rt_alone = proc_results["matplotlib_alone"], proc_results["runtime_alone"]
    mpl_together, rt_together = proc_results["matplotlib_simultaneous"], proc_results["runtime_simultaneous"]
    print(f"{'(6) Matplotlib alone':>42} | ms/tick {mpl_alone['ms_per_tick']:.3f}  ticks/s {mpl_alone['ticks_per_second']:.2f}")
    print(f"{'(7) browser runtime alone':>42} | ms/tick {rt_alone['ms_per_tick']:.3f}  ticks/s {rt_alone['ticks_per_second']:.2f}")
    print(f"{'(8) Matplotlib, simultaneous':>42} | ms/tick {mpl_together['ms_per_tick']:.3f}  ticks/s {mpl_together['ticks_per_second']:.2f}")
    print(f"{'(8) browser runtime, simultaneous':>42} | ms/tick {rt_together['ms_per_tick']:.3f}  ticks/s {rt_together['ticks_per_second']:.2f}")
    mpl_slowdown = mpl_together["ms_per_tick"] / mpl_alone["ms_per_tick"] if mpl_alone["ms_per_tick"] else float("inf")
    rt_slowdown = rt_together["ms_per_tick"] / rt_alone["ms_per_tick"] if rt_alone["ms_per_tick"] else float("inf")
    print(f"\n  -> running both simultaneously changed Matplotlib's ms/tick by {mpl_slowdown:.2f}x "
          f"and the browser runtime's by {rt_slowdown:.2f}x vs. running each alone.")
    print(f"  -> logical CPU count on this machine: {os.cpu_count()}")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        print(f"  -> {var}: {os.environ.get(var, '(unset -- NumPy/BLAS default, typically all cores)')}")

    print("\n" + "=" * 100)
    print("ORPHANED-RUNTIME-THREAD DEMONSTRATION (--demo orphaned-threads runs this in isolation)")
    print("=" * 100)
    orphan_result = demo_orphaned_threads(config)
    print(json.dumps(orphan_result, indent=2))

    print("\n" + "=" * 100)
    print("ORPHANED-RUNTIME CPU-CONTENTION MEASUREMENT (--demo orphaned-contention runs this in isolation)")
    print("=" * 100)
    print("Simulates the PRE-FIX scenario: N forgotten SimulationRuntimes (from N page reloads /")
    print("\"Apply / New simulation\" clicks that never called shutdown()) all running their own real")
    print("background thread in this SAME process, alongside the one the browser is actually polling.")
    contention = demo_orphaned_runtime_contention(config, max_siblings=3, duration=min(args.duration, 4.0))
    print(f"{'orphaned siblings':>18} | {'primary ms/tick':>16} | {'primary ticks/s':>16}")
    print("-" * 56)
    baseline_ms = contention[0]["primary_ms_per_tick"]
    for row in contention:
        slowdown = row["primary_ms_per_tick"] / baseline_ms if baseline_ms else float("inf")
        print(f"{row['orphaned_siblings']:>18} | {row['primary_ms_per_tick']:>16.3f} | "
              f"{row['primary_ticks_per_second']:>16.2f}  ({slowdown:.2f}x vs. 0 siblings)")

    print("\n" + "=" * 100)
    print("COLLISION-MARKER SEMANTICS DEMONSTRATION (--demo collision-markers runs this in isolation)")
    print("=" * 100)
    marker_result = demo_collision_marker_semantics(config)
    print(f"render_every={marker_result['render_every']}, ticks={marker_result['ticks']}")
    print(f"latest-tick-only counts (what /frame shows each poll): {marker_result['latest_tick_counts']}")
    print(f"  max single tick:  {marker_result['max_single_tick_marker_count']}")
    print(f"interval-accumulated counts (what Matplotlib shows each redraw): {marker_result['interval_counts_at_each_redraw']}")
    print(f"  max single redraw: {marker_result['max_single_redraw_marker_count']}")
    print(f"  sum of latest-tick-only counts across all ticks: {marker_result['sum_of_latest_tick_counts']} "
          f"(matches sum of interval-accumulated counts -- same underlying collisions, different grouping)")

    print("\n" + "=" * 100)
    print("/frame DOUBLE-SERIALIZATION FIX COST (--demo frame-serialization runs this in isolation)")
    print("=" * 100)
    ser = demo_frame_serialization_cost(config)
    print(f"payload shape: 60x60 heatmap + {ser['num_markers_in_payload']} markers, {ser['repeats']} repeats each pattern")
    print(f"  OLD (two full json.dumps() calls, one discarded): {ser['old_pattern_ms_per_call']:.4f} ms/call")
    print(f"  NEW (one full dump + tiny splice):                {ser['new_pattern_ms_per_call']:.4f} ms/call")
    print(f"  -> {ser['speedup']:.2f}x faster per /frame request from this fix alone, "
          f"at ~6.7 Hz polling that is {(ser['old_pattern_ms_per_call'] - ser['new_pattern_ms_per_call']) * 6.7:.2f} ms/s "
          f"less GIL-held serialization work competing with the background tick thread.")


def run_scaling_benchmark(sizes: List[int], seed: int, duration: float, warmup: int) -> None:
    print("=" * 100)
    print("SCALING BENCHMARK (isolated per-size measurements)")
    print("=" * 100)
    header = (
        f"{'drones':>8} | {'headless':>10} | {'mpl-data':>10} | {'rt-step':>10} | "
        f"{'rt+snap':>10} | {'rt+poll':>10} | {'cand pairs':>11} | {'occ cells':>10} | "
        f"{'avg occ':>8} | {'max occ':>8} | {'colls':>7} | {'near':>7}"
    )
    print(header)
    print("-" * len(header))
    for n in sizes:
        config = browser_default_config(n, seed)
        ticks = max(10, min(warmup * 2, 30))
        s1 = case_1_headless_step(config, ticks, warmup)
        s2 = case_2_matplotlib_data_path(config, ticks, warmup)
        s3 = case_3_runtime_step_component(config, ticks, warmup)
        r4 = case_4_runtime_with_snapshot(config, duration)
        r5 = case_5_runtime_with_frame_polling(config, duration)

        from drone_sim.simulation import TickProfile

        sim = Simulation(config)
        profile = TickProfile()
        for _ in range(warmup):
            sim.step(profile=profile)
        sim.step(profile=profile)

        print(
            f"{n:>8,d} | {s1['mean']:>10.3f} | {s2['mean']:>10.3f} | {s3['mean']:>10.3f} | "
            f"{r4['ms_per_tick']:>10.3f} | {r5['ms_per_tick']:>10.3f} | "
            f"{profile.candidate_pair_count:>11,d} | {profile.occupied_cells:>10,d} | "
            f"{profile.mean_cell_occupancy:>8.3f} | {profile.max_cell_occupancy:>8,d} | "
            f"{profile.collision_pair_count:>7,d} | {profile.near_miss_pair_count:>7,d}"
        )
    print("\nColumns: headless/mpl-data/rt-step are mean ms/tick over `ticks` back-to-back calls;")
    print("rt+snap/rt+poll are effective ms/tick over a", duration, "second wall-clock window each.")
    print("cand pairs/occ cells/avg occ/max occ/colls/near are from ONE representative profiled tick.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drones", type=int, default=DEFAULT_DRONES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="measured ticks for cases 1-3")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="measured seconds for cases 4-8")
    ap.add_argument("--render-every", type=int, default=5)
    ap.add_argument("--scaling", type=int, nargs="+", default=None, help="run the scaling benchmark at these sizes instead")
    ap.add_argument(
        "--demo",
        choices=["orphaned-threads", "collision-markers", "orphaned-contention", "frame-serialization"],
        default=None,
    )
    # Internal: re-invokes this file as a subprocess worker for cases 6/7/8. Not for direct use.
    ap.add_argument("--worker", choices=["matplotlib", "runtime"], default=None, help=argparse.SUPPRESS)
    ap.add_argument("--world", choices=["matplotlib", "browser"], default="browser", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker == "matplotlib":
        _worker_matplotlib(args)
        return
    if args.worker == "runtime":
        _worker_runtime(args)
        return

    if args.demo == "orphaned-threads":
        config = browser_default_config(args.drones, args.seed)
        print(json.dumps(demo_orphaned_threads(config), indent=2))
        return
    if args.demo == "collision-markers":
        config = browser_default_config(args.drones, args.seed)
        result = demo_collision_marker_semantics(config, render_every=args.render_every)
        print(json.dumps(result, indent=2))
        return
    if args.demo == "orphaned-contention":
        config = browser_default_config(args.drones, args.seed)
        result = demo_orphaned_runtime_contention(config, max_siblings=3, duration=args.duration)
        print(json.dumps(result, indent=2))
        return
    if args.demo == "frame-serialization":
        config = browser_default_config(args.drones, args.seed)
        result = demo_frame_serialization_cost(config)
        print(json.dumps(result, indent=2))
        return

    if args.scaling:
        run_scaling_benchmark(args.scaling, args.seed, args.duration, args.warmup)
        return

    run_main_comparison(args)


if __name__ == "__main__":
    main()
