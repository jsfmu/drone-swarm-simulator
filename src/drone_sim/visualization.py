"""Local Matplotlib debug viewer for the Phase 1 simulation.

This is a prototype/debugging tool, not the Phase 3 production dashboard
(no React, FastAPI, REST, WebSocket/SSE, Redis, or GPU code here). It renders
the existing :class:`~drone_sim.simulation.Simulation` top-down (x/y only)
by reading its real ``World``/``DroneState``/``DetectionResult`` output —
movement, boundaries, spatial hashing, and collision logic are untouched.

Matplotlib is only imported inside :class:`SimulationViewer`, so the pure
grid/marker calculations below can be unit tested without a display backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .collisions import DetectionResult
from .config import SimulationConfig
from .simulation import Simulation

CONTROLS_TEXT = "Space: pause/resume   R: reset (same seed)   Esc / close window: quit"


def compute_density_grid(
    positions: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin drone x/y positions into a top-down 2D density grid.

    Vectorized via ``numpy.histogram2d`` over the world's actual x/y bounds
    (no per-drone Python loop). Returns ``(grid, xedges, yedges)`` where
    ``grid`` is already transposed for direct use with
    ``imshow(..., origin="lower")``.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    x_range = (float(bounds_min[0]), float(bounds_max[0]))
    y_range = (float(bounds_min[1]), float(bounds_max[1]))
    grid, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
    return grid.T, xedges, yedges


def collision_marker_positions(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Return the (x, y) midpoint of each collision pair for marker plotting.

    ``pairs`` is an ``(K, 2)`` int array of drone indices, e.g.
    ``DetectionResult.collision_pairs``. Empty input yields an empty
    ``(0, 2)`` array. Does not alter collision detection in any way.
    """
    if pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    p = positions.astype(np.float64)
    mid = (p[pairs[:, 0]] + p[pairs[:, 1]]) * 0.5
    return mid[:, :2]


@dataclass
class IntervalStats:
    """Accumulates collision/near-miss/timing data between redraws."""

    collisions: int = 0
    near_misses: int = 0
    tick_times_s: List[float] = field(default_factory=list)
    collision_pairs: List[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.collisions = 0
        self.near_misses = 0
        self.tick_times_s = []
        self.collision_pairs = []

    def add(self, result: DetectionResult, tick_time_s: float) -> None:
        self.collisions += result.num_collisions
        self.near_misses += result.num_near_misses
        self.tick_times_s.append(tick_time_s)
        if result.num_collisions:
            self.collision_pairs.append(result.collision_pairs)

    @property
    def ticks_per_second(self) -> float:
        total = sum(self.tick_times_s)
        if total <= 0.0:
            return 0.0
        return len(self.tick_times_s) / total

    @property
    def all_collision_pairs(self) -> np.ndarray:
        if not self.collision_pairs:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(self.collision_pairs, axis=0)


class SimulationViewer:
    """Matplotlib top-down debug viewer over a real ``Simulation`` instance.

    Consumes the existing simulation kernel unchanged; this class only reads
    ``sim.world.state`` / ``sim.step()`` results and renders them. Not a
    replacement for the headless benchmark, which stays independent of this
    module.
    """

    def __init__(self, config: SimulationConfig, render_every: int = 5, bins: int = 100) -> None:
        import matplotlib.pyplot as plt  # local import keeps this class GUI-optional at module load

        self.config = config
        self.render_every = max(1, int(render_every))
        self.bins = int(bins)

        self.sim = Simulation(config)
        self.interval = IntervalStats()
        self.cumulative_collisions = 0
        self.paused = False
        self.closed = False

        self.fig, self.ax = plt.subplots(figsize=(7.5, 7.5))
        self.im = self.ax.imshow(
            np.zeros((self.bins, self.bins)),
            origin="lower",
            extent=[
                config.bounds_min[0], config.bounds_max[0],
                config.bounds_min[1], config.bounds_max[1],
            ],
            cmap="inferno",
            aspect="equal",
        )
        self.fig.colorbar(self.im, ax=self.ax, label="drones / cell (top-down density)")
        self.collision_scatter = self.ax.scatter(
            [], [], c="red", marker="x", s=70, linewidths=2.2, zorder=5, label="collision"
        )
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Drone Collision Simulator — top-down debug viewer (prototype)")
        self.ax.legend(loc="upper right")

        self.metrics_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left",
            color="white", fontsize=9, family="monospace",
            bbox=dict(facecolor="black", alpha=0.55, boxstyle="round"),
        )
        self.fig.text(0.5, 0.01, CONTROLS_TEXT, ha="center", fontsize=9)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._redraw()  # initial frame before any ticks run

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ("r", "R"):
            self.reset()
        elif event.key == "escape":
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _on_close(self, _event) -> None:
        self.closed = True

    def reset(self) -> None:
        """Recreate the simulation from the same config (and therefore seed)."""
        self.sim = Simulation(self.config)
        self.interval.reset()
        self.cumulative_collisions = 0

    def _advance(self) -> None:
        self.interval.reset()
        for _ in range(self.render_every):
            result = self.sim.step()
            tick_time_s = self.sim.metrics.ticks[-1].tick_time_s
            self.interval.add(result, tick_time_s)
            self.cumulative_collisions += result.num_collisions

    def _redraw(self) -> None:
        state = self.sim.world.state
        grid, _, _ = compute_density_grid(
            state.positions, self.config.bounds_min_arr, self.config.bounds_max_arr, self.bins
        )
        self.im.set_data(grid)
        vmax = float(grid.max())
        self.im.set_clim(0, vmax if vmax > 0 else 1.0)

        markers = collision_marker_positions(state.positions, self.interval.all_collision_pairs)
        if markers.shape[0]:
            self.collision_scatter.set_offsets(markers)
        else:
            self.collision_scatter.set_offsets(np.empty((0, 2)))

        last_tick_ms = self.interval.tick_times_s[-1] * 1e3 if self.interval.tick_times_s else 0.0
        self.metrics_text.set_text(
            "\n".join(
                [
                    f"drones (configured): {self.config.num_drones:,}",
                    f"tick: {self.sim.clock.tick}",
                    f"last tick time: {last_tick_ms:.2f} ms",
                    f"ticks/s (approx): {self.interval.ticks_per_second:,.1f}",
                    "-- current interval --",
                    f"collisions: {self.interval.collisions}",
                    f"near misses: {self.interval.near_misses}",
                    "-- cumulative --",
                    f"collisions: {self.cumulative_collisions}",
                ]
            )
        )

    def show(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def _update(_frame):
            if self.closed:
                return [self.im, self.collision_scatter, self.metrics_text]
            if not self.paused:
                self._advance()
                self._redraw()
            return [self.im, self.collision_scatter, self.metrics_text]

        self._anim = FuncAnimation(self.fig, _update, interval=1, cache_frame_data=False)
        plt.show()
