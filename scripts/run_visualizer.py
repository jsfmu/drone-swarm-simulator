"""Launch the local Matplotlib debug viewer for the drone simulation.

This is a debugging/prototype tool, not the production web dashboard planned
for Phase 3 (no React, WebSocket/SSE, or Redis). It reuses the existing
simulation kernel unchanged.

Two modes:

Local (default) -- this process owns the ``Simulation`` and steps it itself:
    python scripts/run_visualizer.py
    python scripts/run_visualizer.py --drones 10000 --render-every 5
    python scripts/run_visualizer.py --drones 1000 --seed 42 --bins 60

Remote (--remote) -- polls a running ``uvicorn drone_sim.api.app:app``
server instead, via ``drone_sim.api_client`` (stdlib HTTP, no FastAPI import
here). Lets this window and a browser tab opened to the printed join URL
display the exact same live simulation -- one ``Simulation`` advancing on the
server's background thread, not two independent ones:
    uvicorn drone_sim.api.app:app --reload             # in one terminal
    python scripts/run_visualizer.py --remote          # in another
    python scripts/run_visualizer.py --remote --drones 5000 --seed 7
    # attach to a simulation already running (e.g. one the browser created):
    python scripts/run_visualizer.py --remote --simulation-id abc123 --x-max 500 --y-max 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_sim.config import SimulationConfig  # noqa: E402
from drone_sim.visualization import RemoteSimulationViewer, SimulationViewer  # noqa: E402

DEFAULT_DRONES = 10_000
CELLS_PER_DRONE = 64.0
COLLISION_RADIUS = 1.0
NEAR_MISS_RADIUS = 2.0
CELL_SIZE = NEAR_MISS_RADIUS  # minimum legal cell size
DEFAULT_API_URL = "http://127.0.0.1:8000"


def world_side_for(n: int) -> float:
    """World edge length giving ~CELLS_PER_DRONE cells per drone (matches the benchmark)."""
    target_cells = max(CELLS_PER_DRONE * n, 1.0)
    return (target_cells ** (1.0 / 3.0)) * CELL_SIZE


def _run_local(args: argparse.Namespace) -> None:
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


def _run_remote(args: argparse.Namespace) -> None:
    if args.simulation_id:
        if args.x_max is None or args.y_max is None:
            raise SystemExit("--x-max and --y-max are required when attaching with --simulation-id")
        viewer = RemoteSimulationViewer(
            args.api_url,
            simulation_id=args.simulation_id,
            viewport=(0.0, args.x_max, 0.0, args.y_max),
            x_bins=args.x_bins,
            y_bins=args.y_bins,
            poll_interval_ms=args.poll_interval_ms,
        )
    else:
        side = world_side_for(args.drones)
        viewer = RemoteSimulationViewer(
            args.api_url,
            create_kwargs=dict(
                num_drones=args.drones,
                bounds_max=(side, side, side),
                seed=args.seed,
                dt=1.0,
                max_speed=5.0,
                collision_radius=COLLISION_RADIUS,
                near_miss_radius=NEAR_MISS_RADIUS,
            ),
            x_bins=args.x_bins,
            y_bins=args.y_bins,
            poll_interval_ms=args.poll_interval_ms,
        )
        print(f"Created remote simulation {viewer.simulation_id} on {args.api_url}")
        print(f"Open in a browser to view the same simulation: {viewer.join_url()}")

    viewer.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Local Matplotlib debug viewer for the drone simulation (prototype, not the production dashboard)."
    )
    ap.add_argument("--drones", type=int, default=DEFAULT_DRONES, help="number of drones to simulate")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument(
        "--render-every", type=int, default=5,
        help="simulation ticks to run between each redraw (local mode only)",
    )
    ap.add_argument("--bins", type=int, default=100, help="density grid resolution per axis (local mode only)")

    ap.add_argument(
        "--remote", action="store_true",
        help="poll a running `uvicorn drone_sim.api.app:app` server instead of simulating "
             "locally, so this viewer and a browser tab can show the same live simulation",
    )
    ap.add_argument("--api-url", default=DEFAULT_API_URL, help="base URL of the running API server (--remote only)")
    ap.add_argument(
        "--simulation-id", default=None,
        help="attach to an existing simulation instead of creating one (--remote only)",
    )
    ap.add_argument("--x-max", type=float, default=None, help="viewport x_max; required with --simulation-id (--remote only)")
    ap.add_argument("--y-max", type=float, default=None, help="viewport y_max; required with --simulation-id (--remote only)")
    ap.add_argument("--x-bins", type=int, default=60, help="heatmap x bins requested from the server (--remote only)")
    ap.add_argument("--y-bins", type=int, default=60, help="heatmap y bins requested from the server (--remote only)")
    ap.add_argument(
        "--poll-interval-ms", type=int, default=150,
        help="polling cadence in ms -- matches static/index.html's default (--remote only)",
    )
    args = ap.parse_args()

    if args.remote:
        _run_remote(args)
    else:
        _run_local(args)


if __name__ == "__main__":
    main()
