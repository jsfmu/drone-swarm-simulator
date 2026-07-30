"""Independent distributed simulation runtime: advances a ``DistributedCoordinator``
on a background thread.

This deliberately mirrors ``runtime.py``'s ``SimulationRuntime`` almost line
for line -- same lock/thread/pause-event skeleton, same snapshot-publishing
contract, same public method surface -- but drives a
:class:`~drone_sim.coordinator.DistributedCoordinator` instead of a plain
:class:`~drone_sim.simulation.Simulation`. It is NOT extracted into a shared
base class with ``SimulationRuntime``: that file has ~20+ dedicated tests and
is exercised transitively by nearly every API/stream test, so threading a
distributed-mode branch through its tested internals would touch the
highest-blast-radius file in the project for what is an opt-in feature.
Duplicating the ~80 lines of lifecycle scaffolding here is the deliberate
trade for zero risk to ``runtime.py``'s existing, proven behavior -- the same
philosophy that put ``DistributedCoordinator`` in its own module in Phase 4
rather than branching it into ``Simulation``.

``build_snapshot()`` (``snapshot.py``) is type-hinted ``sim: Simulation`` but
only ever reads ``.world``, ``.clock.tick``, ``.clock.time_s`` -- all of which
``DistributedCoordinator`` exposes identically (same ``World``/
``SimulationClock`` classes, confirmed by reading ``coordinator.py``), so it
works here unmodified via duck typing. Likewise ``RunningMetrics.record()``
only reads ``coord.metrics.ticks[-1]`` (same ``MetricsCollector``/
``TickMetrics`` classes ``Simulation`` uses).
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .collisions import DetectionResult
from .config import SimulationConfig
from .coordinator import DistributedConfig, DistributedCoordinator
from .movement import MovementSystem
from .runtime import BUSY_LOOP_YIELD_S, RuntimeState, RuntimeStatus, RunningMetrics, TickTimings
from .snapshot import SimulationSnapshot, build_snapshot
from .state import World


class DistributedSimulationRuntime:
    """Owns one ``DistributedCoordinator`` and advances it independently of
    any request. Public interface is a drop-in match for ``SimulationRuntime``
    (see module docstring) plus one addition, :meth:`get_distributed_metrics`.

    ``movement``/``world_factory`` behave exactly as they do on
    ``SimulationRuntime`` -- ``world_factory`` is called once (here, not
    inside ``DistributedCoordinator``, which only accepts a pre-built
    ``World``) via :meth:`_make_world`, reused verbatim by :meth:`reset`.
    """

    def __init__(
        self,
        simulation_id: str,
        config: SimulationConfig,
        dist_config: DistributedConfig,
        movement: MovementSystem | None = None,
        world_factory: Callable[[SimulationConfig], World] | None = None,
    ) -> None:
        self.simulation_id = simulation_id
        self._config = config
        self._dist_config = dist_config
        self._movement = movement
        self._world_factory = world_factory
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set == paused

        # Raises NotImplementedError here (a requires_context policy, e.g.
        # LocalAvoidanceMovementAlgorithm) or ValueError (bad DistributedConfig)
        # -- always BEFORE any WorkerPool/process is created (see
        # DistributedCoordinator.__init__: the policy check runs before
        # self.pool = WorkerPool(...)), so a rejected construction here is
        # inherently leak-safe. Let it propagate; routes.py turns it into a 400.
        self._coord = DistributedCoordinator(
            config, dist_config, movement=movement, world=self._make_world(config)
        )
        self._status = RuntimeStatus.CREATED
        self._running_metrics = RunningMetrics()
        self._last_timings = TickTimings()
        self._snapshot: SimulationSnapshot = build_snapshot(simulation_id, self._coord, None, {})

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
                name=f"drone-sim-dist-{self.simulation_id}",
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
        """Recreate the coordinator from the original config (same seed ->
        identical run). Shuts down the OLD coordinator's worker pool (a no-op
        unless ``use_processes=True``) before building the new one, so a
        process pool is never replaced without releasing the one it replaces.

        Safe to do while holding ``self._lock``: this method already requires
        ``status != RUNNING`` (checked below, same guarantee
        ``SimulationRuntime.reset()`` relies on), so the background thread is
        either not running, already joined, or parked in ``_run_loop``'s
        ``PAUSED`` branch -- which never touches ``self._coord`` -- meaning
        this can never race a concurrent tick.
        """
        with self._lock:
            if self._status == RuntimeStatus.RUNNING:
                raise RuntimeError(
                    f"simulation {self.simulation_id!r} is running -- pause it before resetting"
                )
            self._coord.shutdown()
            self._coord = DistributedCoordinator(
                self._config, self._dist_config, movement=self._movement,
                world=self._make_world(self._config),
            )
            self._status = RuntimeStatus.CREATED
            self._running_metrics = RunningMetrics()
            self._last_timings = TickTimings()
            self._snapshot = build_snapshot(self.simulation_id, self._coord, None, {})
            return self._snapshot

    def shutdown(self) -> None:
        """Stop the background loop (if any), block until it exits, then
        release the coordinator's worker pool (no-op unless
        ``use_processes=True`` -- see ``DistributedCoordinator.shutdown()``)."""
        self._stop_event.set()
        self._pause_event.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._status = RuntimeStatus.STOPPED
            self._coord.shutdown()

    # ---------------------------------------------------------------- reads
    def get_snapshot(self) -> SimulationSnapshot:
        """Return the latest published snapshot. Cheap: no simulation work happens here."""
        with self._lock:
            return self._snapshot

    def get_snapshot_with_lock_wait(self) -> tuple[SimulationSnapshot, float]:
        t0 = time.perf_counter()
        with self._lock:
            t1 = time.perf_counter()
            snapshot = self._snapshot
        return snapshot, (t1 - t0) * 1e3

    def get_snapshot_and_status_with_lock_wait(self) -> tuple[SimulationSnapshot, RuntimeState, float]:
        """Same one-lock-acquisition contract as
        ``SimulationRuntime.get_snapshot_and_status_with_lock_wait()`` -- see
        that method's docstring for why this matters for ``/frame``."""
        t0 = time.perf_counter()
        with self._lock:
            t1 = time.perf_counter()
            snapshot = self._snapshot
            state = RuntimeState(
                simulation_id=self.simulation_id,
                status=self._status,
                tick=snapshot.tick,
                num_drones=self._coord.world.state.num_drones,
            )
        return snapshot, state, (t1 - t0) * 1e3

    def get_status(self) -> RuntimeState:
        with self._lock:
            return RuntimeState(
                simulation_id=self.simulation_id,
                status=self._status,
                tick=self._snapshot.tick,
                num_drones=self._coord.world.state.num_drones,
            )

    def get_last_timings(self) -> TickTimings:
        with self._lock:
            return self._last_timings

    def get_distributed_metrics(self) -> dict:
        """Phase 5 monitoring: worker/partition health, rebalances, load --
        see ``DistributedCoordinator.metrics_summary()``. Lock-protected: the
        coordinator has no lock of its own (only this wrapper does), and
        ``metrics_summary()`` reads mutable state (``clock.tick``,
        ``partition_worker``, ``last_load_stats``) that ``_advance_one_locked``
        mutates without any internal synchronization."""
        with self._lock:
            return self._coord.metrics_summary()

    # ------------------------------------------------------------- internal
    def _advance_one_locked(self) -> None:
        """Advance the coordinator by one tick. Caller must hold ``self._lock``."""
        t0 = time.perf_counter()
        result: DetectionResult = self._coord.step()
        t1 = time.perf_counter()

        # O(1): the TickMetrics just recorded by step(), not a history rescan.
        self._running_metrics.record(self._coord.metrics.ticks[-1])
        metrics = self._running_metrics.summary()
        self._snapshot = build_snapshot(self.simulation_id, self._coord, result, metrics)
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
