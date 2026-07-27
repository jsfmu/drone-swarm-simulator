"""Phase 3B benchmark: SSE dashboard-stream cost, measured in-process.

Reports four separate, never-combined numbers per drone count:

1. Simulation tick throughput with NO active stream client (baseline).
2. Simulation tick throughput WITH one active stream client -- an in-process
   poller thread calling the exact same frame-building function the real
   ``GET .../stream`` endpoint uses (``routes._build_and_serialize_stream_frame``),
   at the same configurable publication rate, against the same live
   ``SimulationRuntime``. This mirrors ``benchmark_viewer_comparison.py``'s
   existing concurrent-poller technique (real threads, real GIL contention)
   rather than spinning up a real uvicorn server + HTTP client, since the
   thing worth measuring is the in-process cost of building/serializing
   dashboard frames alongside the tick loop -- not network/ASGI overhead,
   which ``benchmark_visualization.py`` already covers for the underlying
   query cost and this file does not re-measure.
3. Publication-rate bookkeeping: the configured Hz, mean serialized payload
   size, mean serialization/generation time, and how many ticks were
   superseded (occurred between two publishes and were never individually
   sent) -- proof frames are skipped, not queued.
4. Dashboard-frame generation time (same measurement as timings["generation_ms"]
   in the real endpoint).

These four categories are kept visually and numerically separate in the
output -- simulation throughput, publication-rate bookkeeping, and payload/
timing stats are never combined into one composite number.

Usage:
    python benchmarks/benchmark_streaming.py
    python benchmarks/benchmark_streaming.py --sizes 1000 10000 --duration 2 --hz 8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

from benchmark_simulation import build_config  # noqa: E402
from drone_sim.api.routes import _build_and_serialize_stream_frame  # noqa: E402
from drone_sim.heatmap import HeatmapQuery  # noqa: E402
from drone_sim.runtime import SimulationRuntime  # noqa: E402
from drone_sim.viewport import ViewportQuery  # noqa: E402

DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_DURATION_S = 2.0
DEFAULT_HZ = 8.0
DEFAULT_SEED = 0


def _measure_tick_throughput(runtime: SimulationRuntime, duration_s: float) -> float:
    start_tick = runtime.get_snapshot().tick
    t0 = time.perf_counter()
    time.sleep(duration_s)
    elapsed = time.perf_counter() - t0
    end_tick = runtime.get_snapshot().tick
    return (end_tick - start_tick) / elapsed if elapsed > 0 else 0.0


class _Poller:
    """Mirrors GET .../stream's own loop: build+serialize at a fixed Hz,
    no queue, always reading whatever tick is currently published."""

    def __init__(self, runtime: SimulationRuntime, viewport: ViewportQuery, heatmap_query: HeatmapQuery, hz: float):
        self._runtime = runtime
        self._viewport = viewport
        self._heatmap_query = heatmap_query
        self._interval_s = 1.0 / hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.payload_sizes: List[int] = []
        self.serialization_ms: List[float] = []
        self.generation_ms: List[float] = []
        self.lock_wait_ms: List[float] = []
        self.superseded_ticks = 0
        self.frames_sent = 0
        self._last_tick: int | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            seq += 1
            body, tick = _build_and_serialize_stream_frame(self._runtime, self._viewport, self._heatmap_query, seq)
            self.frames_sent += 1
            self.payload_sizes.append(len(body.encode("utf-8")))
            timings = json.loads(body)["timings"]
            self.serialization_ms.append(timings["serialization_ms"])
            self.generation_ms.append(timings["generation_ms"])
            self.lock_wait_ms.append(timings["lock_wait_ms"])
            if self._last_tick is not None and tick > self._last_tick + 1:
                self.superseded_ticks += tick - self._last_tick - 1
            self._last_tick = tick
            time.sleep(self._interval_s)


def run_one(n: int, duration_s: float, hz: float, seed: int = DEFAULT_SEED) -> Dict:
    config = build_config(n, seed=seed)
    viewport = ViewportQuery(x_min=0.0, x_max=config.bounds_max[0], y_min=0.0, y_max=config.bounds_max[1])
    heatmap_query = HeatmapQuery(viewport=viewport, x_bins=60, y_bins=60)

    # 1. Baseline: no stream client.
    baseline_runtime = SimulationRuntime("bench-baseline", config)
    baseline_runtime.start()
    try:
        baseline_ticks_per_s = _measure_tick_throughput(baseline_runtime, duration_s)
    finally:
        baseline_runtime.shutdown()

    # 2. With one active stream client (same config/seed -> same tick cost).
    streamed_runtime = SimulationRuntime("bench-streamed", config)
    streamed_runtime.start()
    poller = _Poller(streamed_runtime, viewport, heatmap_query, hz)
    poller.start()
    try:
        streamed_ticks_per_s = _measure_tick_throughput(streamed_runtime, duration_s)
    finally:
        poller.stop()
        streamed_runtime.shutdown()

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    return {
        "n": n,
        "baseline_ticks_per_s": baseline_ticks_per_s,
        "streamed_ticks_per_s": streamed_ticks_per_s,
        "slowdown": (baseline_ticks_per_s / streamed_ticks_per_s) if streamed_ticks_per_s > 0 else float("inf"),
        "configured_hz": hz,
        "actual_frames_per_s": poller.frames_sent / duration_s,
        "mean_payload_bytes": mean(poller.payload_sizes),
        "mean_serialization_ms": mean(poller.serialization_ms),
        "mean_generation_ms": mean(poller.generation_ms),
        "mean_lock_wait_ms": mean(poller.lock_wait_ms),
        "superseded_ticks": poller.superseded_ticks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="seconds measured per phase")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="stream publication rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("=" * 100)
    print("PHASE 3B STREAMING BENCHMARK")
    print("=" * 100)
    print(
        f"duration/phase: {args.duration}s   configured publish rate: {args.hz} Hz   seed: {args.seed}\n"
    )

    print("-- 1. Simulation tick throughput: no stream client vs. one stream client --")
    print(f"{'drones':>8} {'baseline ticks/s':>18} {'with-stream ticks/s':>20} {'slowdown':>10}")
    results = []
    for n in args.sizes:
        r = run_one(n, args.duration, args.hz, seed=args.seed)
        results.append(r)
        print(f"{r['n']:>8,} {r['baseline_ticks_per_s']:>18.2f} {r['streamed_ticks_per_s']:>20.2f} {r['slowdown']:>9.2f}x")

    print("\n-- 2. Publication-rate bookkeeping (independent of simulation throughput above) --")
    print(f"{'drones':>8} {'configured Hz':>14} {'actual frames/s':>16} {'superseded ticks':>17}")
    for r in results:
        print(f"{r['n']:>8,} {r['configured_hz']:>14.1f} {r['actual_frames_per_s']:>16.2f} {r['superseded_ticks']:>17,}")

    print("\n-- 3. Per-frame payload / timing stats (mean over the measured window) --")
    print(f"{'drones':>8} {'payload bytes':>14} {'lock wait ms':>13} {'serialization ms':>18} {'generation ms':>15}")
    for r in results:
        print(
            f"{r['n']:>8,} {r['mean_payload_bytes']:>14,.0f} {r['mean_lock_wait_ms']:>13.3f} "
            f"{r['mean_serialization_ms']:>18.3f} {r['mean_generation_ms']:>15.3f}"
        )

    print(
        "\nNote: 'superseded ticks' counts ticks that occurred between two publishes and were\n"
        "never individually sent -- this is expected and by design (see README's streaming-design\n"
        "section), not a defect. It is NOT combined with throughput or payload-size numbers above.\n"
        "'generation ms' is total per-frame cost (lock wait + heatmap/collision query + serialize);\n"
        "'lock wait ms' is broken out separately since at large drone counts, where a single\n"
        "Simulation.step() tick itself already takes tens to hundreds of ms (see\n"
        "benchmark_simulation.py), a stream poll landing mid-tick can wait nearly a full tick for\n"
        "the runtime lock -- a pre-existing Phase 1 tick-cost characteristic, not something this\n"
        "streaming design adds, and not query/serialization cost being slow."
    )


if __name__ == "__main__":
    main()
