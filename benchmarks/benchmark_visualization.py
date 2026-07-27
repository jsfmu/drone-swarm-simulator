"""Phase 3A visualization-query benchmark.

Measures snapshot creation, viewport filtering, heatmap generation,
collision-marker queries, and JSON-ready response conversion -- separately
from each other and separately from the simulation tick path. These are a
different workload than `benchmark_simulation.py` / `benchmark_avoidance.py`
(which measure `Simulation.step()`) and are never combined with those
results: a tick is timed with visualization work fully outside the loop,
and here visualization work is timed on top of an already-advanced world.

Usage:
    python benchmarks/benchmark_visualization.py
    python benchmarks/benchmark_visualization.py --sizes 1000 10000 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[0] / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_BENCH_DIR))

import numpy as np  # noqa: E402

from benchmark_simulation import CELL_SIZE, COLLISION_RADIUS, NEAR_MISS_RADIUS, world_side_for  # noqa: E402
from drone_sim.collision_queries import query_collision_markers  # noqa: E402
from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.heatmap import HeatmapQuery, compute_heatmap  # noqa: E402
from drone_sim.simulation import Simulation  # noqa: E402
from drone_sim.snapshot import build_snapshot  # noqa: E402
from drone_sim.viewport import ViewportQuery, find_visible_drones  # noqa: E402

DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_REPEATS = 5
DEFAULT_WARMUP_TICKS = 3


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


@dataclass
class StageTimings:
    snapshot_ms: List[float]
    viewport_ms: List[float]
    heatmap_ms: List[float]
    collisions_ms: List[float]
    json_conversion_ms: List[float]

    def mean(self, field: str) -> float:
        values = getattr(self, field)
        return float(np.mean(values)) if values else 0.0

    def std(self, field: str) -> float:
        values = getattr(self, field)
        return float(np.std(values)) if values else 0.0


def _viewport_for(config: SimulationConfig) -> ViewportQuery:
    lo = config.bounds_min_arr
    hi = config.bounds_max_arr
    # A central sub-region (half the world on each axis) rather than the
    # whole world, so the viewport filter is measuring an actual filter.
    span = (hi - lo) * 0.25
    center = (hi + lo) / 2
    return ViewportQuery(
        x_min=float(center[0] - span[0]), x_max=float(center[0] + span[0]),
        y_min=float(center[1] - span[1]), y_max=float(center[1] + span[1]),
    )


def run_for_size(n: int, repeats: int, warmup_ticks: int) -> StageTimings:
    config = build_config(n)
    sim = Simulation(config)
    for _ in range(warmup_ticks):
        sim.step()
    last_result = sim.step()

    viewport = _viewport_for(config)
    timings = StageTimings([], [], [], [], [])

    for _ in range(repeats):
        t0 = time.perf_counter()
        # {} for metrics -- this benchmark isolates snapshot/query cost, not
        # RunningMetrics.summary() cost (see runtime.py / benchmark_pipeline.py
        # for that, and README's "Phase 3A tick-rate regression" section).
        snapshot = build_snapshot(f"bench-{n}", sim, last_result, {})
        t1 = time.perf_counter()
        timings.snapshot_ms.append((t1 - t0) * 1e3)

        t0 = time.perf_counter()
        visible = find_visible_drones(snapshot, viewport, limit=5_000)
        t1 = time.perf_counter()
        timings.viewport_ms.append((t1 - t0) * 1e3)

        t0 = time.perf_counter()
        heatmap = compute_heatmap(snapshot, HeatmapQuery(viewport=viewport, x_bins=100, y_bins=100))
        t1 = time.perf_counter()
        timings.heatmap_ms.append((t1 - t0) * 1e3)

        t0 = time.perf_counter()
        markers = query_collision_markers(snapshot, viewport)
        t1 = time.perf_counter()
        timings.collisions_ms.append((t1 - t0) * 1e3)

        t0 = time.perf_counter()
        payload = {
            "tick": snapshot.tick,
            "drones": [
                {"drone_id": int(d), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for d, p in zip(visible.drone_ids, visible.positions)
            ],
            "heatmap": {
                "counts": heatmap.counts.tolist(),
                "x_edges": heatmap.x_edges.tolist(),
                "y_edges": heatmap.y_edges.tolist(),
            },
            "markers": [
                {"drone_a": m.drone_a, "drone_b": m.drone_b, "x": m.x, "y": m.y, "z": m.z}
                for m in markers
            ],
        }
        json.dumps(payload)
        t1 = time.perf_counter()
        timings.json_conversion_ms.append((t1 - t0) * 1e3)

    return timings


def format_row(n: int, timings: StageTimings) -> str:
    fields = ["snapshot_ms", "viewport_ms", "heatmap_ms", "collisions_ms", "json_conversion_ms"]
    cells = " | ".join(f"{timings.mean(f):>9.3f}+/-{timings.std(f):<6.3f}" for f in fields)
    return f"{n:>8,d} | {cells}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3A visualization-query benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--warmup-ticks", type=int, default=DEFAULT_WARMUP_TICKS)
    args = ap.parse_args()

    print(f"numpy {np.__version__}")
    print(f"Benchmarking sizes={args.sizes} repeats={args.repeats} warmup_ticks={args.warmup_ticks}\n")
    print(
        "This measures snapshot/viewport/heatmap/collision-query/JSON-conversion "
        "cost ONLY -- Simulation.step() ticks are not included in any timed region "
        "here (see benchmark_simulation.py / benchmark_avoidance.py for tick throughput).\n"
    )

    header = (
        f"{'drones':>8} | {'snapshot ms':>16} | {'viewport ms':>16} | "
        f"{'heatmap ms':>16} | {'collisions ms':>16} | {'json ms':>16}"
    )
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        try:
            timings = run_for_size(n, args.repeats, args.warmup_ticks)
        except Exception as exc:  # noqa: BLE001 - report honestly, don't hide it
            print(f"{n:>8,d} | FAILED: {type(exc).__name__}: {exc}")
            continue
        print(format_row(n, timings))


if __name__ == "__main__":
    main()
