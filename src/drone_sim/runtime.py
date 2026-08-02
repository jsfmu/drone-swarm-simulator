"""Independent simulation runtime: advances a ``Simulation`` on a background thread.

FastAPI request handlers must never drive the simulation's main loop inline
-- :class:`SimulationRuntime` owns a single background thread that repeatedly
calls ``Simulation.step()`` and publishes a fresh :class:`SimulationSnapshot`
after every completed tick. Request handlers only ever call ``get_snapshot()``.

Thread safety: ``_lock`` guards mutation of ``_sim``/``_status``/``_snapshot``
and is held only for the cheap parts of a tick. It is never held during JSON
serialization or heatmap/viewport calculation -- those read the already
-published, immutable snapshot after the lock has been released.

Metrics: ``RunningMetrics`` deliberately does NOT call
``MetricsCollector.summary()`` on every tick. ``summary()`` rebuilds NumPy
arrays from *every* tick ever recorded and sorts them for percentiles -- an
O(number of ticks so far) cost that grows without bound over a long-running
session. ``RunningMetrics`` instead updates O(1) running totals from the
single just-recorded ``TickMetrics`` (``sim.metrics.ticks[-1]``, itself an
O(1) list index) and keeps only a small bounded window of recent tick times
for the display-only median/p95 figures. This was a real, measured bug (see
README's "Phase 3A tick-rate regression" section) responsible for the
Matplotlib debug viewer (which never calls ``summary()`` -- it reads
``sim.metrics.ticks[-1]`` directly) staying fast while the browser runtime
degraded over a session.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Deque, Dict, Union

import numpy as np

from . import checkpoint as _checkpoint
from .collisions import DetectionResult
from .config import SimulationConfig
from .movement import MovementSystem
from .simulation import Simulation
from .snapshot import SimulationSnapshot, build_snapshot
from .state import World

PathLike = Union[str, Path]

#: How many of the most recent tick times to keep for median/p95 display.
#: Bounded so RunningMetrics.summary() stays O(1) regardless of how long the
#: simulation has been running (see module docstring).
RECENT_WINDOW = 200

#: Yield inserted between ticks when the background loop is unthrottled
#: (``tick_interval_s <= 0``, the default). A tight loop that releases and
#: immediately re-acquires the same ``threading.Lock`` every iteration can
#: starve another thread waiting on it -- measured at 10,000 drones: an API
#: reader's lock-wait spiked past 1000ms with no yield, dropping to under
#: 15ms with this one inserted (see README's "Phase 3A tick-rate regression"
#: section for the full measurement). ``time.sleep(0)`` alone was tried
#: first and was not reliably enough (Windows' scheduler quantum is coarser
#: than a bare yield); this small real sleep is. Its throughput cost is a
#: fixed ~0.5ms/tick, negligible at the 1k-100k drone scale this project
#: targets (multi-millisecond ticks) but proportionally larger for very
#: small/fast simulations -- see known limitations in README.
BUSY_LOOP_YIELD_S = 0.0005


class RuntimeStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RuntimeState:
    simulation_id: str
    status: RuntimeStatus
    tick: int
    num_drones: int


@dataclass(frozen=True)
class TickTimings:
    """Per-tick timing for the most recently completed tick, in milliseconds.

    ``sim_step_ms`` is the pure ``Simulation.step()`` cost (movement,
    boundaries, spatial hash, detection, resolution, metrics recording) --
    nothing from the visualization layer. ``snapshot_build_ms`` is the cost
    of ``build_snapshot()`` (array copies + O(1) metrics lookup) that runs
    immediately after. Neither includes viewport/heatmap/collision queries,
    JSON serialization, or any API request handling -- those are measured
    separately, at query time, by the API layer (see ``routes.py``'s
    ``/frame`` endpoint).
    """

    sim_step_ms: float = 0.0
    snapshot_build_ms: float = 0.0


@dataclass
class RunningMetrics:
    """O(1)-per-tick metrics accumulator (see module docstring for why).

    Mirrors ``MetricsCollector.summary()``'s key names so API/UI consumers
    don't need to change, but ``median_tick_ms``/``p95_tick_ms`` are computed
    from only the last ``RECENT_WINDOW`` ticks rather than the full history --
    an intentional, documented, bounded approximation. ``mean_tick_ms``,
    ``ticks_per_second``, and all totals remain exact (simple running sums).
    """

    num_ticks: int = 0
    total_time_s: float = 0.0
    total_candidate_pairs: int = 0
    total_collisions: int = 0
    total_near_misses: int = 0
    min_tick_ms: float = float("inf")
    max_tick_ms: float = 0.0
    recent_tick_times_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))

    def record(self, tm) -> None:
        ms = tm.tick_time_s * 1e3
        self.num_ticks += 1
        self.total_time_s += tm.tick_time_s
        self.total_candidate_pairs += tm.candidate_pairs
        self.total_collisions += tm.collisions
        self.total_near_misses += tm.near_misses
        self.min_tick_ms = min(self.min_tick_ms, ms)
        self.max_tick_ms = max(self.max_tick_ms, ms)
        self.recent_tick_times_ms.append(ms)

    def summary(self) -> Dict[str, float]:
        if self.num_ticks == 0:
            return {}
        recent = np.asarray(self.recent_tick_times_ms, dtype=np.float64)
        return {
            "num_ticks": self.num_ticks,
            "total_time_s": self.total_time_s,
            "mean_tick_ms": (self.total_time_s / self.num_ticks) * 1e3,
            "median_tick_ms": float(np.median(recent)),
            "p95_tick_ms": float(np.percentile(recent, 95)),
            "min_tick_ms": self.min_tick_ms,
            "max_tick_ms": self.max_tick_ms,
            "ticks_per_second": (self.num_ticks / self.total_time_s) if self.total_time_s > 0 else float("inf"),
            "mean_candidate_pairs": self.total_candidate_pairs / self.num_ticks,
            "total_candidate_pairs": self.total_candidate_pairs,
            "total_collisions": self.total_collisions,
            "total_near_misses": self.total_near_misses,
        }


class SimulationRuntime:
    """Owns one ``Simulation`` and advances it independently of any request.

    ``movement``/``world_factory`` are optional, additive hooks for Phase 3B's
    policy/scenario selection (see ``api/routes.py``'s ``_build_movement_system()``/
    ``_build_world_factory()``) -- ``SimulationRuntime`` itself knows nothing
    about policies or scenarios, it only passes these straight through to
    ``Simulation`` exactly as ``Simulation`` already accepts them. Leaving both
    ``None`` (every pre-Phase-3B call site) reproduces the exact previous
    behavior: ``Simulation(config)`` with its own default ``MovementSystem()``
    and ``World.create(config)``. ``world_factory`` must be a pure function of
    ``config`` (no closures over mutable state) so ``reset()`` reproducing the
    identical initial world by calling it again stays deterministic.
    """

    def __init__(
        self,
        simulation_id: str,
        config: SimulationConfig,
        movement: MovementSystem | None = None,
        world_factory: Callable[[SimulationConfig], World] | None = None,
    ) -> None:
        self.simulation_id = simulation_id
        self._config = config
        self._movement = movement
        self._world_factory = world_factory
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set == paused

        self._sim = Simulation(config, movement=movement, world=self._make_world(config))
        self._status = RuntimeStatus.CREATED
        self._running_metrics = RunningMetrics()
        self._last_timings = TickTimings()
        self._snapshot: SimulationSnapshot = build_snapshot(simulation_id, self._sim, None, {})

    def _make_world(self, config: SimulationConfig) -> World | None:
        return None if self._world_factory is None else self._world_factory(config)

    # ------------------------------------------------------------ lifecycle
    def start(self, tick_interval_s: float = 0.0) -> None:
        """Start the background tick loop. Raises if already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(f"simulation {self.simulation_id!r} is already running")
            self._stop_event.clear()
            self._pause_event.clear()
            self._status = RuntimeStatus.RUNNING
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(tick_interval_s,),
                daemon=True,
                name=f"drone-sim-{self.simulation_id}",
            )
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self._status != RuntimeStatus.RUNNING:
                raise RuntimeError(f"simulation {self.simulation_id!r} is not running")
            self._pause_event.set()
            self._status = RuntimeStatus.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._status != RuntimeStatus.PAUSED:
                raise RuntimeError(f"simulation {self.simulation_id!r} is not paused")
            self._pause_event.clear()
            self._status = RuntimeStatus.RUNNING

    def step_once(self) -> SimulationSnapshot:
        """Advance exactly one tick. Requires the runtime not be actively running."""
        with self._lock:
            if self._status == RuntimeStatus.RUNNING:
                raise RuntimeError(
                    f"simulation {self.simulation_id!r} is running -- pause it before stepping"
                )
            self._advance_one_locked()
            return self._snapshot

    def reset(self) -> SimulationSnapshot:
        """Recreate the simulation from the original config (same seed -> identical run)."""
        with self._lock:
            if self._status == RuntimeStatus.RUNNING:
                raise RuntimeError(
                    f"simulation {self.simulation_id!r} is running -- pause it before resetting"
                )
            self._sim = Simulation(self._config, movement=self._movement, world=self._make_world(self._config))
            self._status = RuntimeStatus.CREATED
            self._running_metrics = RunningMetrics()
            self._last_timings = TickTimings()
            self._snapshot = build_snapshot(self.simulation_id, self._sim, None, {})
            return self._snapshot

    def shutdown(self) -> None:
        """Stop the background loop (if any) and block until it exits."""
        self._stop_event.set()
        self._pause_event.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._status = RuntimeStatus.STOPPED

    # ---------------------------------------------------------------- reads
    def get_snapshot(self) -> SimulationSnapshot:
        """Return the latest published snapshot. Cheap: no simulation work happens here."""
        with self._lock:
            return self._snapshot

    def get_snapshot_with_lock_wait(self) -> tuple[SimulationSnapshot, float]:
        """Like ``get_snapshot`` but also reports time spent waiting to acquire
        the lock, in milliseconds -- for diagnostics/benchmarking only."""
        t0 = time.perf_counter()
        with self._lock:
            t1 = time.perf_counter()
            snapshot = self._snapshot
        return snapshot, (t1 - t0) * 1e3

    def get_snapshot_and_status_with_lock_wait(self) -> tuple[SimulationSnapshot, RuntimeState, float]:
        """Like ``get_snapshot_with_lock_wait`` but also returns ``RuntimeState``
        from the *same* lock acquisition, instead of a caller doing a second,
        separate ``get_status()`` call (a second, independent lock wait).

        A caller needing both a snapshot and status (``routes.py``'s ``/frame``
        handler) previously acquired ``self._lock`` twice per request -- once
        via ``get_snapshot_with_lock_wait()`` (measured as ``lock_wait_ms``)
        and once via ``get_status()`` (never measured at all, and itself able
        to block for as long as the background loop is mid-tick). That second,
        uninstrumented wait was real: measured at 10,000 drones under active
        ``/frame`` polling, it explained a large share of the gap between
        ``lock_wait_ms + heatmap_ms + collisions_ms + serialization_ms`` and
        the reported ``total_request_ms`` (see README's "Full /frame request
        timing" section). Reading ``status``/``tick`` from the same lock hold
        that fetches the snapshot also guarantees they describe the same
        instant -- the old two-call sequence could, in principle, return a
        status/tick pair from a *later* tick than the snapshot if the
        background loop advanced between the two acquisitions.
        """
        t0 = time.perf_counter()
        with self._lock:
            t1 = time.perf_counter()
            snapshot = self._snapshot
            state = RuntimeState(
                simulation_id=self.simulation_id,
                status=self._status,
                tick=snapshot.tick,
                num_drones=self._sim.world.state.num_drones,
            )
        return snapshot, state, (t1 - t0) * 1e3

    def get_status(self) -> RuntimeState:
        with self._lock:
            return RuntimeState(
                simulation_id=self.simulation_id,
                status=self._status,
                tick=self._snapshot.tick,
                num_drones=self._sim.world.state.num_drones,
            )

    def get_last_timings(self) -> TickTimings:
        """Timing for the most recently completed tick (see ``TickTimings``)."""
        with self._lock:
            return self._last_timings

    # ------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path: PathLike) -> Dict[str, int]:
        """Write a Phase 5 checkpoint (see ``checkpoint.save_checkpoint``) of
        the simulation's current state.

        Lock-protected like every other read here: waits for an in-progress
        tick to finish rather than racing it, so the written checkpoint always
        reflects one fully-completed tick, never a half-mutated one. Safe to
        call while the background loop is running -- unlike ``load_checkpoint``,
        this never mutates ``self._sim``.
        """
        with self._lock:
            _checkpoint.save_checkpoint(self._sim, path)
            return {"tick": self._sim.clock.tick, "num_drones": self._sim.world.state.num_drones}

    def load_checkpoint(self, path: PathLike) -> SimulationSnapshot:
        """Replace the current simulation with one restored from ``path``.

        Requires the runtime not be actively running (same guard as
        ``reset()``/``step_once()``) -- replacing ``self._sim`` out from under
        a live background tick would be a real race, not merely a semantic
        surprise. On success the restored simulation becomes ``self._sim``
        (its ``config`` -- which may describe a different drone count than
        the simulation originally had -- replaces ``self._config`` too, so a
        later ``reset()`` reproduces the *loaded* state, not the pre-load
        one), status moves to ``PAUSED`` (loaded and ready, not "never
        started"), and per-session running metrics/timings are cleared --
        exactly ``reset()``'s contract, only sourcing the world/clock/RNG
        from the checkpoint instead of a fresh ``World.create(config)``.
        """
        with self._lock:
            if self._status == RuntimeStatus.RUNNING:
                raise RuntimeError(
                    f"simulation {self.simulation_id!r} is running -- pause it before loading a checkpoint"
                )
            loaded = _checkpoint.load_checkpoint(path, movement=self._movement)
            self._sim = loaded
            self._config = loaded.config
            self._status = RuntimeStatus.PAUSED
            self._running_metrics = RunningMetrics()
            self._last_timings = TickTimings()
            self._snapshot = build_snapshot(self.simulation_id, self._sim, None, {})
            return self._snapshot

    # ------------------------------------------------------------- internal
    def _advance_one_locked(self) -> None:
        """Advance the simulation by one tick. Caller must hold ``self._lock``."""
        t0 = time.perf_counter()
        result: DetectionResult = self._sim.step()
        t1 = time.perf_counter()

        # O(1): the TickMetrics just recorded by step(), not a history rescan.
        self._running_metrics.record(self._sim.metrics.ticks[-1])
        metrics = self._running_metrics.summary()
        self._snapshot = build_snapshot(self.simulation_id, self._sim, result, metrics)
        t2 = time.perf_counter()

        self._last_timings = TickTimings(
            sim_step_ms=(t1 - t0) * 1e3,
            snapshot_build_ms=(t2 - t1) * 1e3,
        )

    def _run_loop(self, tick_interval_s: float) -> None:
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.02)
                continue
            with self._lock:
                if self._stop_event.is_set():
                    break
                self._advance_one_locked()
            if tick_interval_s > 0:
                time.sleep(tick_interval_s)
            else:
                time.sleep(BUSY_LOOP_YIELD_S)
        with self._lock:
            if self._status == RuntimeStatus.RUNNING:
                self._status = RuntimeStatus.PAUSED
