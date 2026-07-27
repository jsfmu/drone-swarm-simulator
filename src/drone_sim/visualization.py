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
REMOTE_CONTROLS_TEXT = (
    "Space: pause/resume (shared)   R: reset (shared)   Esc / close window: quit"
)


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


def _build_figure(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_shape: Tuple[int, int],
    title: str,
    controls_text: str,
):
    """Shared Matplotlib scaffold for both :class:`SimulationViewer` and
    :class:`RemoteSimulationViewer` -- density heatmap image, collision-marker
    scatter, and a text overlay for live metrics. The two classes differ only
    in *how* a fresh grid/markers/metrics triple is obtained each frame (local
    ``Simulation.step()`` vs. polling a remote API), not in how it's drawn.
    """
    import matplotlib.pyplot as plt  # local import keeps callers GUI-optional at module load

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(
        np.zeros(grid_shape),
        origin="lower",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap="inferno",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="drones / cell (top-down density)")
    collision_scatter = ax.scatter(
        [], [], c="red", marker="x", s=70, linewidths=2.2, zorder=5, label="collision"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right")

    metrics_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=9, family="monospace",
        bbox=dict(facecolor="black", alpha=0.55, boxstyle="round"),
    )
    fig.text(0.5, 0.01, controls_text, ha="center", fontsize=9)
    return fig, ax, im, collision_scatter, metrics_text


class SimulationViewer:
    """Matplotlib top-down debug viewer over a real ``Simulation`` instance.

    Consumes the existing simulation kernel unchanged; this class only reads
    ``sim.world.state`` / ``sim.step()`` results and renders them. Not a
    replacement for the headless benchmark, which stays independent of this
    module.
    """

    def __init__(self, config: SimulationConfig, render_every: int = 5, bins: int = 100) -> None:
        self.config = config
        self.render_every = max(1, int(render_every))
        self.bins = int(bins)

        self.sim = Simulation(config)
        self.interval = IntervalStats()
        self.cumulative_collisions = 0
        self.paused = False
        self.closed = False

        self.fig, self.ax, self.im, self.collision_scatter, self.metrics_text = _build_figure(
            (config.bounds_min[0], config.bounds_max[0]),
            (config.bounds_min[1], config.bounds_max[1]),
            (self.bins, self.bins),
            "Drone Collision Simulator — top-down debug viewer (prototype)",
            CONTROLS_TEXT,
        )

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


class RemoteSimulationViewer:
    """Matplotlib top-down viewer that polls a running ``drone_sim`` API
    server instead of owning a local ``Simulation``.

    This is the same role ``drone_sim/api/static/index.html`` plays in the
    browser: both are read-mostly clients of one server-side
    :class:`~drone_sim.runtime.SimulationRuntime`, polling
    ``GET /simulations/{id}/frame`` (see ``api_client.get_frame``). Pointing
    this viewer and a browser tab at the same ``simulation_id`` makes them
    display the exact same live tick -- there is exactly one
    ``Simulation`` advancing, on the server's background thread, not two
    independent ones. Space/R act on the *shared* simulation (pause/resume/
    reset run on the server), not just this window's polling.

    The grid drawn here is the server's already-binned ``heatmap.counts``
    (see ``heatmap.py``) -- this class never recomputes a density grid or
    touches raw drone positions, so it stays a thin renderer exactly like
    :class:`SimulationViewer`, just fed from HTTP instead of from
    ``sim.world.state``.
    """

    def __init__(
        self,
        base_url: str,
        simulation_id: str | None = None,
        *,
        create_kwargs: dict | None = None,
        viewport: Tuple[float, float, float, float] | None = None,
        x_bins: int = 60,
        y_bins: int = 60,
        poll_interval_ms: int = 150,
    ) -> None:
        from . import api_client

        self._api = api_client
        self.base_url = base_url.rstrip("/")
        self.x_bins = int(x_bins)
        self.y_bins = int(y_bins)
        self.poll_interval_ms = int(poll_interval_ms)
        self.owns_simulation = simulation_id is None
        self.closed = False
        self.last_error: str | None = None
        self.server_status = "unknown"

        if simulation_id is None:
            create_kwargs = dict(create_kwargs or {})
            simulation_id = self._api.create_simulation(self.base_url, **create_kwargs)
            self._api.start_simulation(self.base_url, simulation_id)
            if viewport is None:
                bounds_max = create_kwargs.get("bounds_max", (1000.0, 1000.0, 1000.0))
                viewport = (0.0, float(bounds_max[0]), 0.0, float(bounds_max[1]))
        elif viewport is None:
            raise ValueError(
                "viewport=(x_min, x_max, y_min, y_max) is required when attaching "
                "to an existing simulation_id (there is no server endpoint to "
                "recover a simulation's world bounds from its id alone)"
            )

        self.simulation_id = simulation_id
        self.viewport = viewport

        x_min, x_max, y_min, y_max = viewport
        self.fig, self.ax, self.im, self.collision_scatter, self.metrics_text = _build_figure(
            (x_min, x_max),
            (y_min, y_max),
            (self.y_bins, self.x_bins),
            f"Drone Collision Simulator — remote viewer ({self.base_url}, sim {simulation_id})",
            REMOTE_CONTROLS_TEXT,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def join_url(self) -> str:
        """Browser URL that views this exact simulation (see index.html's
        ``?simulation_id=`` join support).

        Carries this viewer's own viewport (x_min/x_max/y_min/y_max) as query
        params so index.html can set its input boxes to match instead of
        keeping its hardcoded 0-500 defaults -- otherwise the two clients can
        query different windows of the same simulation and count different
        collisions, even though both are polling the one shared ``/frame``.
        """
        import urllib.parse

        x_min, x_max, y_min, y_max = self.viewport
        qs = urllib.parse.urlencode({
            "simulation_id": self.simulation_id,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        })
        return f"{self.base_url}/?{qs}"

    def _on_key(self, event) -> None:
        if event.key == " ":
            self._toggle_pause_resume()
        elif event.key in ("r", "R"):
            try:
                self._api.reset_simulation(self.base_url, self.simulation_id)
                self.last_error = None
            except self._api.RemoteSimulationError as exc:
                self.last_error = str(exc)
        elif event.key == "escape":
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _on_close(self, _event) -> None:
        self.closed = True
        if self.owns_simulation:
            # Best-effort, mirrors static/index.html's stopSimulationIfAny --
            # a viewer that created its own simulation must not leave its
            # background thread running forever after the window closes.
            self._api.delete_simulation(self.base_url, self.simulation_id)

    def _toggle_pause_resume(self) -> None:
        """Pause if running, resume if paused -- guessed from ``self.server_status``,
        which is only as fresh as the last poll.

        ``self.server_status`` can be stale relative to the server's actual
        state (another client paused/resumed it since our last poll, or the
        user pressed Space twice faster than one poll interval). Guessing
        wrong sends a request the server rejects with 409 (pause/resume each
        require one specific prior status) -- if that happens, try the other
        action instead of silently swallowing the keypress, since a 409 here
        is proof positive of which state the server was actually in.
        """
        paused = self.server_status == "paused"
        try:
            if paused:
                self._api.resume_simulation(self.base_url, self.simulation_id)
            else:
                self._api.pause_simulation(self.base_url, self.simulation_id)
            self.server_status = "running" if paused else "paused"
            self.last_error = None
            return
        except self._api.RemoteSimulationError:
            pass  # guessed wrong (stale status) -- fall through and try the other action

        try:
            if paused:
                self._api.pause_simulation(self.base_url, self.simulation_id)
            else:
                self._api.resume_simulation(self.base_url, self.simulation_id)
            self.server_status = "paused" if paused else "running"
            self.last_error = None
        except self._api.RemoteSimulationError as exc:
            self.last_error = str(exc)

    def _poll_and_redraw(self) -> None:
        x_min, x_max, y_min, y_max = self.viewport
        try:
            frame = self._api.get_frame(
                self.base_url, self.simulation_id,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                x_bins=self.x_bins, y_bins=self.y_bins,
            )
        except self._api.RemoteSimulationError as exc:
            self.last_error = str(exc)
            return

        self.last_error = None
        self.server_status = frame.get("status", "unknown")

        counts = np.asarray(frame["heatmap"]["counts"], dtype=np.float64)
        self.im.set_data(counts)
        vmax = float(counts.max()) if counts.size else 0.0
        self.im.set_clim(0, vmax if vmax > 0 else 1.0)

        markers = frame.get("markers") or []
        if markers:
            offsets = np.array([[m["x"], m["y"]] for m in markers], dtype=np.float64)
            self.collision_scatter.set_offsets(offsets)
        else:
            self.collision_scatter.set_offsets(np.empty((0, 2)))

        m = frame.get("metrics") or {}
        self.metrics_text.set_text(
            "\n".join(
                [
                    f"simulation: {self.simulation_id} ({self.base_url})",
                    f"status: {self.server_status}",
                    f"tick: {frame.get('tick', 0)}",
                    f"drones in viewport: {frame.get('num_visible_drones', 0):,}",
                    # Single-tick count, directly comparable to index.html's
                    # "collision markers: N" line -- unlike "collisions" below.
                    f"collision markers: {len(markers)}",
                    f"ticks/s (server): {m.get('ticks_per_second', 0.0):,.1f}",
                    "-- cumulative (server, all clients) --",
                    f"collisions: {m.get('total_collisions', 0)}",
                    f"near misses: {m.get('total_near_misses', 0)}",
                ]
            )
        )

    def show(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def _update(_frame):
            if self.closed:
                return [self.im, self.collision_scatter, self.metrics_text]
            self._poll_and_redraw()  # always poll: the shared sim may advance
            if self.last_error:  # or be paused/resumed by another client
                self.metrics_text.set_text(f"ERROR: {self.last_error}")
            return [self.im, self.collision_scatter, self.metrics_text]

        self._anim = FuncAnimation(
            self.fig, _update, interval=self.poll_interval_ms, cache_frame_data=False
        )
        plt.show()
