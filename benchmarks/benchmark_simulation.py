"""Phase 1 benchmark harness.

Runs the local kernel at 1,000 / 10,000 / 100,000 drones and reports tick
latency, throughput, candidate pairs, collisions, and near misses.

The world size scales with drone count so volumetric density (and therefore
the collision rate) stays roughly constant and low, matching the design goal
of collisions being uncommon during normal operation.

Usage:
    python benchmarks/benchmark_simulation.py
    python benchmarks/benchmark_simulation.py --ticks 20 --sizes 1000 10000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.simulation import Simulation  # noqa: E402

# Spacing target: many cells per drone so the collision sphere is a tiny
# fraction of the world volume and collisions stay uncommon during normal
# operation (a small nonzero count still exercises the resolution pipeline).
CELLS_PER_DRONE = 64.0
COLLISION_RADIUS = 1.0
NEAR_MISS_RADIUS = 2.0
CELL_SIZE = NEAR_MISS_RADIUS  # minimum legal cell size


def world_side_for(n: int) -> float:
    """World edge length giving ~CELLS_PER_DRONE cells per drone."""
    target_cells = max(CELLS_PER_DRONE * n, 1.0)
    cells_per_axis = target_cells ** (1.0 / 3.0)
    return cells_per_axis * CELL_SIZE


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
        seed=seed,
    )


def run_one(n: int, ticks: int, warmup: int = 2) -> dict:
    cfg = build_config(n)
    t0 = time.perf_counter()
    sim = Simulation(cfg)
    init_s = time.perf_counter() - t0

    # Warm up (first ticks pay allocation / cache costs) then measure.
    for _ in range(warmup):
        sim.step()
    sim.metrics.reset()

    for _ in range(ticks):
        sim.step()

    summary = sim.metrics.summary()
    summary["num_drones"] = n
    summary["world_side"] = world_side_for(n)
    summary["init_s"] = init_s
    return summary


def format_row(s: dict) -> str:
    return (
        f"{s['num_drones']:>8,d} | "
        f"{s['mean_tick_ms']:>9.2f} | "
        f"{s['median_tick_ms']:>9.2f} | "
        f"{s['p95_tick_ms']:>9.2f} | "
        f"{s['ticks_per_second']:>8.2f} | "
        f"{s['mean_candidate_pairs']:>13,.0f} | "
        f"{s['total_collisions']:>10,d} | "
        f"{s['total_near_misses']:>10,d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Drone simulator Phase 1 benchmark")
    ap.add_argument("--ticks", type=int, default=10, help="measured ticks per size")
    ap.add_argument("--warmup", type=int, default=2, help="warmup ticks per size")
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000],
        help="drone counts to benchmark",
    )
    args = ap.parse_args()

    print(f"numpy {np.__version__}")
    print(
        f"Benchmarking sizes={args.sizes} ticks={args.ticks} warmup={args.warmup}\n"
    )
    header = (
        f"{'drones':>8} | {'mean ms':>9} | {'med ms':>9} | {'p95 ms':>9} | "
        f"{'ticks/s':>8} | {'cand pairs/t':>13} | {'collis':>10} | {'near miss':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for n in args.sizes:
        s = run_one(n, args.ticks, args.warmup)
        results.append(s)
        print(format_row(s))

    print("\nNotes:")
    print(f"  world scales as ~{CELLS_PER_DRONE:.0f} cells/drone; collisions stay rare by design.")
    print("  candidate pairs << N^2 (all-pairs would be ~5e9 at 100k drones).")


if __name__ == "__main__":
    main()
