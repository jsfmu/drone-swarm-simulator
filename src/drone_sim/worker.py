"""Worker abstraction and worker pool for distributed execution (Phase 4).

A :class:`Worker` owns zero or more partitions *transiently*, for one tick at
a time. It carries no persistent simulation state between calls -- everything
needed for a tick (drone arrays, config, movement policies, an RNG seed) is
passed explicitly via :class:`WorkerMovementInput` / :class:`WorkerDetectionInput`,
never read from mutable global state. This is what makes a worker freely
reassignable to a different partition (rebalancing) or replaceable after a
failure without carrying stale state, and is what keeps the execution backend
(sequential, threaded, and -- unimplemented here -- process/remote in the
future) swappable without touching :mod:`drone_sim.coordinator`.

A worker never advances (integrates positions for) a drone it does not own.
Boundary/ghost drones are read-only inputs to the detection phase only; they
are never assigned an owner, never appear in a movement phase, and are never
written back anywhere.

The tick is split into two phases with a synchronisation point between them
(orchestrated by the coordinator, not this module):

1. **Movement phase** -- each partition's owned drones are moved and
   boundary-constrained, using the existing, unmodified
   :class:`~drone_sim.movement.MovementSystem` / :class:`~drone_sim.boundaries.BoundaryManager`.
2. **Detection phase** -- each partition builds a local
   :class:`~drone_sim.spatial_hash.SpatialHashGrid` over its own
   post-movement owned drones plus read-only ghost snapshots of neighbouring
   partitions' post-movement boundary drones, then runs the existing,
   unmodified :class:`~drone_sim.collisions.CollisionDetectionEngine` over it.
   Results are translated back to global drone ids so the coordinator can
   merge them directly.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .boundaries import BoundaryManager
from .collisions import CollisionDetectionEngine
from .config import SimulationConfig
from .movement import MovementSystem
from .spatial_hash import SpatialHashGrid
from .state import DroneState, World


class WorkerLifecycleState(enum.Enum):
    """Explicit worker lifecycle, tracked by :class:`WorkerPool`."""

    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    RECOVERED = "recovered"


class WorkerFailure(Exception):
    """Raised when a worker task fails (real exception or injected fault).

    Carries enough context for the coordinator to preserve authoritative
    state, mark the worker unhealthy, and reassign its partition(s) without
    needing to inspect the pool's internals.
    """

    def __init__(self, worker_id: int, partition_id: int, tick: int, phase: str, cause: BaseException | str) -> None:
        self.worker_id = worker_id
        self.partition_id = partition_id
        self.tick = tick
        self.phase = phase
        self.cause = cause
        super().__init__(
            f"worker {worker_id} failed on partition {partition_id}, tick {tick}, phase={phase!r}: {cause}"
        )


@dataclass(frozen=True)
class WorkerMovementInput:
    """Everything a worker needs to advance one partition's owned drones by
    one tick. Deliberately explicit and self-contained -- no mutable global
    state is referenced."""

    partition_id: int
    tick: int
    config: SimulationConfig
    movement: MovementSystem
    positions: np.ndarray            # (k, 3) owned drones, pre-movement
    velocities: np.ndarray           # (k, 3)
    active_mask: np.ndarray          # (k,)
    movement_policy_ids: np.ndarray  # (k,)
    goal_positions: np.ndarray | None
    global_ids: np.ndarray           # (k,) local row -> global drone id
    rng_seed: object                 # np.random.SeedSequence-compatible seed


@dataclass(frozen=True)
class WorkerMovementResult:
    partition_id: int
    positions: np.ndarray    # (k, 3) post movement + boundary
    velocities: np.ndarray   # (k, 3)
    global_ids: np.ndarray   # (k,) same ordering as input
    movement_ns: int
    boundary_ns: int


@dataclass(frozen=True)
class WorkerDetectionInput:
    """Everything a worker needs to detect collisions/near-misses for one
    partition: its own post-movement owned drones plus read-only ghost
    snapshots from neighbouring partitions."""

    partition_id: int
    tick: int
    config: SimulationConfig
    owned_global_ids: np.ndarray     # (o,)
    owned_positions: np.ndarray      # (o, 3) post-movement
    ghost_global_ids: np.ndarray     # (g,) read-only, never written back
    ghost_positions: np.ndarray      # (g, 3) read-only, post-movement snapshot


@dataclass(frozen=True)
class WorkerDetectionResult:
    partition_id: int
    collision_pairs: np.ndarray       # (K, 2) GLOBAL ids, i<j
    collision_distances: np.ndarray   # (K,)
    near_miss_pairs: np.ndarray       # (M, 2) GLOBAL ids, i<j
    near_miss_distances: np.ndarray   # (M,)
    candidate_pairs: np.ndarray       # (C, 2) GLOBAL ids, i<j -- local (possibly
    #: cross-partition-duplicated) candidate set; see coordinator dedup rule.
    candidate_pair_count: int         # == candidate_pairs.shape[0]; local/raw,
    #: a per-partition LOAD metric (may double-count boundary pairs also seen
    #: by a neighbour) -- NOT the authoritative global count.
    owned_drone_count: int
    ghost_drone_count: int
    grid_ns: int
    pairs_ns: int
    detect_ns: int


def _to_global_pairs(local_pairs: np.ndarray, combined_global_ids: np.ndarray) -> np.ndarray:
    if local_pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    g = combined_global_ids[local_pairs]
    gi = np.minimum(g[:, 0], g[:, 1])
    gj = np.maximum(g[:, 0], g[:, 1])
    return np.stack([gi, gj], axis=1)


class Worker:
    """Owns zero or more partitions transiently, for one tick at a time."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self.state = WorkerLifecycleState.IDLE

    def run_movement_phase(self, inp: WorkerMovementInput) -> WorkerMovementResult:
        """Advance exactly the owned drones in ``inp`` by one tick (movement
        + boundary). Never touches any drone this worker does not own --
        there is no ghost/boundary concept at this phase at all, by
        construction (``inp`` only ever contains owned-drone arrays)."""
        self.state = WorkerLifecycleState.RUNNING
        local_state = DroneState(
            positions=inp.positions.copy(),
            velocities=inp.velocities.copy(),
            active_mask=inp.active_mask.copy(),
            movement_policy_ids=inp.movement_policy_ids.copy(),
            goal_positions=None if inp.goal_positions is None else inp.goal_positions.copy(),
        )
        local_world = World(config=inp.config, state=local_state)
        rng = np.random.default_rng(inp.rng_seed)

        t0 = time.perf_counter_ns()
        inp.movement.step(local_state, rng, inp.config, inp.tick, context=None)
        t1 = time.perf_counter_ns()
        BoundaryManager().apply(local_world)
        t2 = time.perf_counter_ns()

        self.state = WorkerLifecycleState.IDLE
        return WorkerMovementResult(
            partition_id=inp.partition_id,
            positions=local_state.positions,
            velocities=local_state.velocities,
            global_ids=inp.global_ids,
            movement_ns=t1 - t0,
            boundary_ns=t2 - t1,
        )

    def run_detection_phase(self, inp: WorkerDetectionInput) -> WorkerDetectionResult:
        """Detect collisions/near-misses for this partition's owned drones,
        using neighbours' ghost snapshots as read-only context. Ghosts are
        never mutated, never returned, and never treated as owned."""
        self.state = WorkerLifecycleState.RUNNING

        if inp.ghost_positions.shape[0]:
            combined_positions = np.concatenate([inp.owned_positions, inp.ghost_positions], axis=0)
            combined_global_ids = np.concatenate([inp.owned_global_ids, inp.ghost_global_ids])
        else:
            combined_positions = inp.owned_positions
            combined_global_ids = inp.owned_global_ids
        m = combined_positions.shape[0]

        grid = SpatialHashGrid(inp.config)
        t0 = time.perf_counter_ns()
        grid.build(combined_positions, np.arange(m, dtype=np.int64))
        t1 = time.perf_counter_ns()
        local_pairs = grid.candidate_pairs()
        t2 = time.perf_counter_ns()

        # A throwaway DroneState purely to reuse CollisionDetectionEngine.detect()
        # unchanged -- only .positions is read by classification; velocities/
        # policy ids are irrelevant placeholders never inspected by detect().
        dummy_state = DroneState(
            positions=combined_positions,
            velocities=np.zeros_like(combined_positions),
            active_mask=np.ones(m, dtype=bool),
            movement_policy_ids=np.zeros(m, dtype=np.int32),
        )
        engine = CollisionDetectionEngine(inp.config)
        result = engine.detect(dummy_state, grid, pairs=local_pairs)
        t3 = time.perf_counter_ns()

        self.state = WorkerLifecycleState.IDLE
        return WorkerDetectionResult(
            partition_id=inp.partition_id,
            collision_pairs=_to_global_pairs(result.collision_pairs, combined_global_ids),
            collision_distances=result.collision_distances,
            near_miss_pairs=_to_global_pairs(result.near_miss_pairs, combined_global_ids),
            near_miss_distances=result.near_miss_distances,
            candidate_pairs=_to_global_pairs(local_pairs, combined_global_ids),
            candidate_pair_count=int(local_pairs.shape[0]),
            owned_drone_count=int(inp.owned_positions.shape[0]),
            ghost_drone_count=int(inp.ghost_positions.shape[0]),
            grid_ns=t1 - t0,
            pairs_ns=t2 - t1,
            detect_ns=t3 - t2,
        )


#: (worker_id, tick, phase) -> True if this call should simulate a failure.
FaultInjector = Callable[[int, int, str], bool]


def _process_run_movement(worker_id: int, inp: WorkerMovementInput) -> WorkerMovementResult:
    """Module-level (picklable) process-pool target -- see
    ``WorkerPool``'s ``executor="process"`` docstring for why this can't be a
    bound method. Builds its own throwaway ``Worker`` in the child process
    (never the parent's ``self.workers[worker_id]``, which never crosses the
    process boundary) and wraps any failure in :class:`WorkerFailure` exactly
    like :meth:`WorkerPool.run_movement` does for the sequential/threaded
    paths, so a crashing worker is reported identically regardless of which
    executor is configured.
    """
    try:
        return Worker(worker_id).run_movement_phase(inp)
    except Exception as exc:  # noqa: BLE001 - deliberately wrap any worker crash
        raise WorkerFailure(worker_id, inp.partition_id, inp.tick, "movement", exc) from exc


def _process_run_detection(worker_id: int, inp: WorkerDetectionInput) -> WorkerDetectionResult:
    """Process-pool counterpart to :func:`_process_run_movement`, detection phase."""
    try:
        return Worker(worker_id).run_detection_phase(inp)
    except Exception as exc:  # noqa: BLE001
        raise WorkerFailure(worker_id, inp.partition_id, inp.tick, "detection", exc) from exc


class WorkerPool:
    """Runs a fixed number of logical :class:`Worker` instances locally.

    Sequential by default (fully deterministic wall-clock ordering, simplest
    to reason about). ``use_threads=True`` runs jobs on a
    ``concurrent.futures.ThreadPoolExecutor`` instead -- numerical results are
    identical either way (workers never share mutable state), only wall-clock
    behaviour differs.

    ``use_processes=True`` runs jobs on a persistent
    ``concurrent.futures.ProcessPoolExecutor`` instead (mutually exclusive
    with ``use_threads``) -- created lazily on first use and kept alive across
    ticks (creating one per tick would pay process-spawn cost every tick,
    defeating the point); call :meth:`shutdown` when done with this pool to
    avoid leaking worker processes. This was added only after measuring that
    threading gives no real benefit here (Python's GIL is not released long
    enough by these NumPy calls to outweigh thread-scheduling overhead --
    see benchmarks/phase5_results/), while a real process pool measured a
    genuine, if modest and worker-count/scale-dependent, 1.05x-1.66x speedup
    at 50,000-100,000 drones. Numerical results are identical to the
    sequential path either way (workers never share mutable state; a worker
    process only ever receives explicit, already-copied DTOs and returns new
    arrays, exactly like the threaded path). Fault injection is checked in
    THIS process before a job is ever submitted (see :meth:`_maybe_fail`), not
    inside the worker process, so :meth:`set_fault_injector` works identically
    across all three execution modes without requiring the injector callable
    itself to be picklable.

    Windows note: ``ProcessPoolExecutor`` uses spawn on Windows, which
    re-imports the launching script in each child process -- any script that
    constructs a ``WorkerPool``/``DistributedCoordinator`` with
    ``use_processes=True`` must guard its entry point with
    ``if __name__ == "__main__":`` (already true of every benchmark script
    and of pytest's own process model), or child processes can recursively
    re-run the parent script instead of just importing it.

    This split is what keeps the execution backend replaceable: a later
    remote pool only needs to implement the same
    ``run_movement_batch``/``run_detection_batch`` interface.
    """

    def __init__(self, num_workers: int, use_threads: bool = False, use_processes: bool = False) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if use_threads and use_processes:
            raise ValueError("use_threads and use_processes are mutually exclusive")
        self.num_workers = num_workers
        self.use_threads = use_threads
        self.use_processes = use_processes
        self.workers: dict[int, Worker] = {i: Worker(i) for i in range(num_workers)}
        self._health: dict[int, WorkerLifecycleState] = {
            i: WorkerLifecycleState.IDLE for i in range(num_workers)
        }
        self._fault_injector: FaultInjector | None = None
        self._process_executor = None  # created lazily -- see run_movement_batch/run_detection_batch

    # ------------------------------------------------------------- health
    def set_fault_injector(self, fn: FaultInjector | None) -> None:
        self._fault_injector = fn

    def healthy_worker_ids(self) -> list[int]:
        return sorted(w for w, s in self._health.items() if s != WorkerLifecycleState.FAILED)

    def mark_failed(self, worker_id: int) -> None:
        self._health[worker_id] = WorkerLifecycleState.FAILED

    def mark_recovered(self, worker_id: int) -> None:
        self._health[worker_id] = WorkerLifecycleState.RECOVERED

    def worker_state(self, worker_id: int) -> WorkerLifecycleState:
        return self._health[worker_id]

    # -------------------------------------------------------------- runs
    def _maybe_fail(self, worker_id: int, partition_id: int, tick: int, phase: str) -> None:
        if self._fault_injector is not None and self._fault_injector(worker_id, tick, phase):
            raise WorkerFailure(worker_id, partition_id, tick, phase, "injected fault")

    def run_movement(self, worker_id: int, inp: WorkerMovementInput) -> WorkerMovementResult:
        self._maybe_fail(worker_id, inp.partition_id, inp.tick, "movement")
        try:
            return self.workers[worker_id].run_movement_phase(inp)
        except WorkerFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately wrap any worker crash
            raise WorkerFailure(worker_id, inp.partition_id, inp.tick, "movement", exc) from exc

    def run_detection(self, worker_id: int, inp: WorkerDetectionInput) -> WorkerDetectionResult:
        self._maybe_fail(worker_id, inp.partition_id, inp.tick, "detection")
        try:
            return self.workers[worker_id].run_detection_phase(inp)
        except WorkerFailure:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkerFailure(worker_id, inp.partition_id, inp.tick, "detection", exc) from exc

    def run_movement_batch(
        self, jobs: list[tuple[int, WorkerMovementInput]]
    ) -> dict[int, WorkerMovementResult]:
        """``jobs``: list of ``(worker_id, input)``. Returns
        ``partition_id -> result``. Raises :class:`WorkerFailure` on the
        first failing job -- the caller must treat the whole batch as not
        having happened (no partial results are usable)."""
        if self.use_processes:
            return self._run_batch_process(jobs, "movement", _process_run_movement)
        if self.use_threads:
            return self._run_batch_threaded(jobs, self.run_movement)
        out: dict[int, WorkerMovementResult] = {}
        for worker_id, inp in jobs:
            out[inp.partition_id] = self.run_movement(worker_id, inp)
        return out

    def run_detection_batch(
        self, jobs: list[tuple[int, WorkerDetectionInput]]
    ) -> dict[int, WorkerDetectionResult]:
        if self.use_processes:
            return self._run_batch_process(jobs, "detection", _process_run_detection)
        if self.use_threads:
            return self._run_batch_threaded(jobs, self.run_detection)
        out: dict[int, WorkerDetectionResult] = {}
        for worker_id, inp in jobs:
            out[inp.partition_id] = self.run_detection(worker_id, inp)
        return out

    def _run_batch_threaded(self, jobs, run_fn):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {ex.submit(run_fn, worker_id, inp): inp.partition_id for worker_id, inp in jobs}
            for fut in as_completed(futures):
                partition_id = futures[fut]
                out[partition_id] = fut.result()  # re-raises WorkerFailure here
        return out

    def _ensure_process_executor(self):
        if self._process_executor is None:
            from concurrent.futures import ProcessPoolExecutor

            self._process_executor = ProcessPoolExecutor(max_workers=self.num_workers)
        return self._process_executor

    def _run_batch_process(self, jobs, phase: str, target_fn):
        from concurrent.futures import as_completed

        # Fault injection is checked HERE, in this process, before any job is
        # submitted across the process boundary -- see class docstring for
        # why (the injector callable never needs to be picklable this way,
        # and a triggered fault fails fast without submitting any work).
        for worker_id, inp in jobs:
            self._maybe_fail(worker_id, inp.partition_id, inp.tick, phase)

        ex = self._ensure_process_executor()
        out = {}
        futures = {ex.submit(target_fn, worker_id, inp): inp.partition_id for worker_id, inp in jobs}
        for fut in as_completed(futures):
            partition_id = futures[fut]
            out[partition_id] = fut.result()  # re-raises WorkerFailure here
        return out

    def shutdown(self) -> None:
        """Release the persistent process pool, if one was ever created (a
        no-op otherwise, including for the sequential/threaded paths, which
        create no long-lived resources to begin with). Always call this when
        done with a ``WorkerPool`` constructed with ``use_processes=True`` --
        an unclosed process pool leaks worker processes for the parent's
        entire remaining lifetime, exactly the kind of leak
        ``SimulationRuntime.shutdown()`` already exists to prevent for
        background threads (see README's "orphaned runtime threads")."""
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None
