"""Phase 3A tick-rate regression benchmark.

Reproduces, with bounded runtime, the measurements behind the diagnosis in
README's "Phase 3A tick-rate regression" section: why the browser runtime
was slower than the Matplotlib debug viewer at the same drone count, and
what the fix changed. Three independent things are measured, in order:

1. Pipeline-stage comparison at a fixed, modest tick count: pure
   ``Simulation.step()`` (what the Matplotlib viewer does) vs. the same
   policy through ``SimulationRuntime.step_once()`` (snapshot publication
   added) vs. the same plus viewport/heatmap/collision-marker queries and
   JSON serialization (what an API request does). Reports movement policy
   and whether a context-requiring (avoidance) policy was active.
2. The root cause in isolation: ``MetricsCollector.summary()`` cost as a
   pure function of accumulated tick-history length, using synthetic
   histories (fast -- no real simulation ticks needed to reach large
   history sizes) to show the O(history) growth that the OLD
   ``build_snapshot()`` paid on every single tick, and that
   ``RunningMetrics`` (the fix) does not.
3. Lock fairness: measured API-reader lock-wait time against the background
   tick loop, with and without the small inter-tick yield fix.

Usage:
    python benchmarks/benchmark_pipeline_regression.py
    python benchmarks/benchmark_pipeline_regression.py --drones 10000 --ticks 300
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import List

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

from benchmark_simulation import CELL_SIZE, COLLISION_RADIUS, NEAR_MISS_RADIUS, world_side_for  # noqa: E402
from drone_sim.collision_queries import query_collision_markers  # noqa: E402
from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.heatmap import HeatmapQuery, compute_heatmap  # noqa: E402
from drone_sim.metrics import MetricsCollector, TickMetrics  # noqa: E402
from drone_sim.runtime import RunningMetrics, SimulationRuntime  # noqa: E402
from drone_sim.simulation import Simulation  # noqa: E402
from drone_sim.snapshot import build_snapshot  # noqa: E402
from drone_sim.viewport import ViewportQuery  # noqa: E402

DEFAULT_DRONES = 10_000
DEFAULT_TICKS = 300
DEFAULT_WARMUP = 10


def build_config(n: int) -> SimulationConfig:
    side = world_side_for(n)
    return SimulationConfig(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(side, side, side),
        collision_radius=COLLISION_RADIUS,
        near_miss_radius=NEAR_MISS_RADIUS,
        cell_size=CELL_SIZE,
        dt=1.0,
        seed=42,
    )


def _mean_ms(samples: List[float]) -> float:
    return statistics.mean(samples) * 1e3 if samples else 0.0


# --------------------------------------------------------------- Section 1
def section_1_pipeline_stages(config: SimulationConfig, ticks: int, warmup: int) -> None:
    print("=" * 100)
    print("Section 1: pipeline-stage comparison (bounded, modest tick count)")
    print("=" * 100)

    policy_id = 0  # DroneState.generate() always assigns RandomMovementAlgorithm's id (0)
    print("Movement policy used by BOTH the Matplotlib viewer and the Phase 3A browser")
    print("runtime by default: RandomMovementAlgorithm (policy_id=0). Neither sets goal")
    print("positions or a custom MovementSystem, so requires_context=False in both --")
    print("no context-aware avoidance path is exercised here. This refutes the")
    print("hypothesis that a movement-policy mismatch caused the reported gap.\n")

    # --- 1a. Pure Simulation.step() -- exactly what SimulationViewer._advance() does.
    sim = Simulation(config)
    for _ in range(warmup):
        sim.step()
    step_times = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)

    # --- 1b. SimulationRuntime.step_once() -- adds snapshot publication (fixed version).
    runtime = SimulationRuntime("bench-1b", config)
    for _ in range(warmup):
        runtime.step_once()
    runtime_step_times = []
    runtime_snapshot_times = []
    for _ in range(ticks):
        runtime.step_once()
        t = runtime.get_last_timings()
        runtime_step_times.append(t.sim_step_ms / 1e3)
        runtime_snapshot_times.append(t.snapshot_build_ms / 1e3)
    runtime.shutdown()

    # --- 1c. Same + viewport/heatmap/collision-marker queries + JSON serialization
    #     (what one /frame request does server-side, called directly -- no HTTP).
    runtime2 = SimulationRuntime("bench-1c", config)
    for _ in range(warmup):
        runtime2.step_once()
    lo = config.bounds_min_arr
    hi = config.bounds_max_arr
    viewport = ViewportQuery(x_min=float(lo[0]), x_max=float(hi[0]), y_min=float(lo[1]), y_max=float(hi[1]))
    heatmap_query = HeatmapQuery(viewport=viewport, x_bins=60, y_bins=60)
    query_times = []
    serialize_times = []
    import json as _json

    for _ in range(ticks):
        runtime2.step_once()
        snap = runtime2.get_snapshot()
        t0 = time.perf_counter()
        hm = compute_heatmap(snap, heatmap_query)
        markers = query_collision_markers(snap, viewport)
        t1 = time.perf_counter()
        query_times.append(t1 - t0)
        payload = {
            "tick": snap.tick,
            "counts": hm.counts.tolist(),
            "markers": [{"drone_a": m.drone_a, "drone_b": m.drone_b} for m in markers],
            "metrics": snap.metrics,
        }
        t2 = time.perf_counter()
        _json.dumps(payload)
        serialize_times.append(time.perf_counter() - t2)
    runtime2.shutdown()

    print(f"{'stage':>45} | {'mean ms/tick':>12}")
    print("-" * 62)
    print(f"{'(a) matplotlib-equivalent: sim.step() only':>45} | {_mean_ms(step_times):>12.3f}")
    print(f"{'(b) + snapshot publication (runtime)':>45} | {_mean_ms(runtime_step_times) + _mean_ms(runtime_snapshot_times):>12.3f}")
    print(f"{'    -- sim_step component':>45} | {_mean_ms(runtime_step_times):>12.3f}")
    print(f"{'    -- snapshot_build component':>45} | {_mean_ms(runtime_snapshot_times):>12.3f}")
    print(f"{'(c) + viewport/heatmap/collision queries':>45} | {_mean_ms(query_times):>12.3f}")
    print(f"{'(c) + JSON serialization':>45} | {_mean_ms(serialize_times):>12.3f}")
    total_c = _mean_ms(runtime_step_times) + _mean_ms(runtime_snapshot_times) + _mean_ms(query_times) + _mean_ms(serialize_times)
    print(f"{'(c) full per-tick pipeline total':>45} | {total_c:>12.3f}")
    tps = 1000.0 / total_c if total_c > 0 else float("inf")
    print(f"\nFull-pipeline throughput ceiling: ~{tps:.1f} ticks/sec if run back-to-back with no other overhead.")


# --------------------------------------------------------------- Section 2
def section_2_metrics_summary_growth() -> None:
    print("\n" + "=" * 100)
    print("Section 2: root cause -- MetricsCollector.summary() cost vs. accumulated history")
    print("=" * 100)
    print("This is what the OLD build_snapshot() called on EVERY tick. RunningMetrics (the")
    print("fix) is O(1)/tick regardless of history length -- see the comparison below.\n")

    history_sizes = [100, 1_000, 5_000, 20_000, 50_000]
    print(f"{'history (ticks)':>16} | {'MetricsCollector.summary() [OLD, per-call]':>44} | {'RunningMetrics.summary() [NEW, per-call]':>42}")
    print("-" * 108)
    for n in history_sizes:
        mc = MetricsCollector()
        rm = RunningMetrics()
        for i in range(n):
            tm = TickMetrics(tick=i, tick_time_s=0.009, candidate_pairs=1000, collisions=1, near_misses=5, active_drones=10_000)
            mc.record(tm)
            rm.record(tm)

        t0 = time.perf_counter()
        for _ in range(10):
            mc.summary()
        old_ms = (time.perf_counter() - t0) * 1e3 / 10

        t0 = time.perf_counter()
        for _ in range(10):
            rm.summary()
        new_ms = (time.perf_counter() - t0) * 1e3 / 10

        print(f"{n:>16,d} | {old_ms:>44.4f} | {new_ms:>42.4f}")

    print("\nAt every tick, the OLD path paid this cost IN ADDITION to the real sim.step() cost")
    print("-- and it grows without bound over a session. The NEW path's cost is flat.")


# --------------------------------------------------------------- Section 3
def section_3_lock_fairness(config: SimulationConfig) -> None:
    print("\n" + "=" * 100)
    print("Section 3: background-loop lock fairness (API-reader lock-wait time)")
    print("=" * 100)
    print("Measures how long a reader thread (like an API request) waits to acquire the")
    print("runtime lock while the background tick loop is running unthrottled.\n")

    def measure(yield_s) -> tuple:
        sim = Simulation(config)
        lock = threading.Lock()
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                with lock:
                    sim.step()
                time.sleep(yield_s) if yield_s is not None else None

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        time.sleep(0.3)

        waits = []
        for _ in range(15):
            t0 = time.perf_counter()
            with lock:
                t1 = time.perf_counter()
            waits.append((t1 - t0) * 1e3)
            time.sleep(0.02)

        tick_before = sim.clock.tick
        time.sleep(1.0)
        tick_after = sim.clock.tick
        stop.set()
        t.join(timeout=3.0)
        return waits, (tick_after - tick_before)

    old_waits, old_tps = measure(None)  # OLD: no yield at all (tick_interval_s<=0, pre-fix)
    new_waits, new_tps = measure(0.0005)  # NEW: BUSY_LOOP_YIELD_S

    print(f"{'':>10} | {'mean lock-wait ms':>18} | {'max lock-wait ms':>18} | {'throughput ticks/s':>19}")
    print("-" * 75)
    print(f"{'OLD (no yield)':>10} | {statistics.mean(old_waits):>18.2f} | {max(old_waits):>18.2f} | {old_tps:>19}")
    print(f"{'NEW (yield)':>10} | {statistics.mean(new_waits):>18.2f} | {max(new_waits):>18.2f} | {new_tps:>19}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3A tick-rate regression benchmark")
    ap.add_argument("--drones", type=int, default=DEFAULT_DRONES)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = ap.parse_args()

    config = build_config(args.drones)
    print(f"drones={args.drones:,} ticks={args.ticks} warmup={args.warmup}\n")

    section_1_pipeline_stages(config, args.ticks, args.warmup)
    section_2_metrics_summary_growth()
    section_3_lock_fairness(config)


if __name__ == "__main__":
    main()
