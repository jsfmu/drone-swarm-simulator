"""Launch the local Matplotlib debug viewer for the drone simulation.

This is a debugging/prototype tool, not the production web dashboard planned
for Phase 3 (no React, FastAPI, WebSocket/SSE, or Redis). It reuses the
existing simulation kernel unchanged.

Usage:
    python scripts/run_visualizer.py
    python scripts/run_visualizer.py --drones 10000 --render-every 5
    python scripts/run_visualizer.py --drones 1000 --seed 42 --bins 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.visualization import SimulationViewer  # noqa: E402

DEFAULT_DRONES = 10_000
CELLS_PER_DRONE = 64.0
COLLISION_RADIUS = 1.0
NEAR_MISS_RADIUS = 2.0
CELL_SIZE = NEAR_MISS_RADIUS  # minimum legal cell size


def world_side_for(n: int) -> float:
    """World edge length giving ~CELLS_PER_DRONE cells per drone (matches the benchmark)."""
    target_cells = max(CELLS_PER_DRONE * n, 1.0)
    return (target_cells ** (1.0 / 3.0)) * CELL_SIZE


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Local Matplotlib debug viewer for the drone simulation (prototype, not the production dashboard)."
    )
    ap.add_argument("--drones", type=int, default=DEFAULT_DRONES, help="number of drones to simulate")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument(
        "--render-every", type=int, default=5,
        help="simulation ticks to run between each redraw",
    )
    ap.add_argument("--bins", type=int, default=100, help="density grid resolution per axis")
    args = ap.parse_args()

    side = world_side_for(args.drones)
    config = SimulationConfig(
        num_drones=args.drones,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(side, side, side),
        collision_radius=COLLISION_RADIUS,
        near_miss_radius=NEAR_MISS_RADIUS,
        cell_size=CELL_SIZE,
        dt=1.0,
        max_speed=5.0,
        seed=args.seed,
    )

    viewer = SimulationViewer(config, render_every=args.render_every, bins=args.bins)
    viewer.show()


if __name__ == "__main__":
    main()
