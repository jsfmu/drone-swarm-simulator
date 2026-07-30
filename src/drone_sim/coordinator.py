"""Distributed-execution coordinator (Phase 4).

:class:`DistributedCoordinator` is a drop-in alternative to
:class:`~drone_sim.simulation.Simulation` that advances the same authoritative
:class:`~drone_sim.state.World` through :class:`~drone_sim.worker.WorkerPool`-
managed workers instead of a single in-process loop, while producing the same
kind of per-tick :class:`~drone_sim.collisions.DetectionResult` and
:class:`~drone_sim.metrics.MetricsCollector` history.

It does not replace or modify :class:`~drone_sim.simulation.Simulation` --
that class, and the plain single-process path it drives, are unchanged and
remain the default/simplest way to run a simulation. This module is
additive: a new orchestration layer built on top of the existing,
unmodified movement/boundary/spatial-hash/collision kernel classes.

Ownership model
----------------
There is exactly one authoritative :class:`~drone_sim.state.World` (owned by
the coordinator). A drone's owning partition is *derived*, each tick, from
its current position via :class:`~drone_sim.partition.PartitionGrid` -- it is
never stored or transmitted as separate state, so there is nothing to
desynchronize. "Ownership transfer" when a drone crosses a partition boundary
is therefore automatic: next tick, :meth:`PartitionGrid.assign` simply
returns a different partition id for it.

Partitions are assigned to workers via ``self.partition_worker`` (a plain
``dict``), which is what rebalancing and failure recovery actually mutate --
never drone-to-partition assignment, which stays purely spatial.

Cross-partition collision deduplication
----------------------------------------
Two neighbouring partitions' local detection passes both see any pair
straddling their shared boundary (each owns one drone and receives the other
as a read-only ghost), so summing every partition's local results would
double-count every cross-partition pair. The rule used to keep exactly one
copy: a pair ``(i, j)`` is kept from partition ``p``'s results only if
``p == min(owner(i), owner(j))`` -- i.e. **the lower-numbered partition
always wins** the tie. This is arbitrary but fixed and deterministic, applied
uniformly to collision pairs, near-miss pairs, and candidate pairs, so the
merged result is byte-for-byte the same regardless of which partition
"noticed" the pair first.

Tick-level transactional behaviour
------------------------------------
A tick's movement and detection results are always computed into freshly
allocated staging arrays (copies of the current authoritative state), never
written into ``self.world.state`` in place. The real state is mutated only
once, at the very end of a *successful* attempt (see :meth:`step`'s commit
step). If any worker task raises :class:`~drone_sim.worker.WorkerFailure`
partway through an attempt, the exception propagates out of the (still
in-progress) attempt before any commit happens -- the authoritative state is
therefore always exactly what it was before ``step()`` was called, for every
failed attempt. Retries recompute the whole tick from that same unchanged
state; because per-partition movement RNG is derived purely from
``(config.seed, tick, partition_id)`` (never from worker identity or attempt
count), a retried tick reproduces the same result regardless of how many
attempts it took or which healthy worker ended up running which partition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .collisions import CollisionResolutionEngine, DetectionResult
from .config import SimulationConfig
from .metrics import MetricsCollector, TickMetrics
from .movement import MovementSystem
from .partition import PartitionGrid
from .simulation import SimulationClock
from .state import DroneState, World
from .worker import (
    WorkerDetectionInput,
    WorkerDetectionResult,
    WorkerFailure,
    WorkerMovementInput,
    WorkerMovementResult,
    WorkerPool,
)


@dataclass(frozen=True)
class DistributedConfig:
    """Tunables for :class:`DistributedCoordinator`.

    Defaults (``num_workers=1``, ``num_partitions=None`` -> 1) preserve
    single-worker, single-partition behaviour: with one partition there are
    no neighbours, no ghost exchange, and no cross-partition dedup to apply,
    so a :class:`DistributedCoordinator` run is then a faithful (RNG-stream
    caveat aside -- see README.md) reproduction of the plain
    :class:`~drone_sim.simulation.Simulation` path.
    """

    num_workers: int = 1
    num_partitions: int | None = None  # defaults to num_workers
    halo_distance: float | None = None  # defaults to config.interaction_radius
    rebalance_interval_ticks: int = 20
    rebalance_imbalance_threshold: float = 1.5  # max/mean worker load ratio that triggers a move
    worker_retry_limit: int = 2
    use_threads: bool = False
    #: Phase 5: run workers on a persistent concurrent.futures.ProcessPoolExecutor
    #: instead (mutually exclusive with use_threads). Off by default -- see
    #: WorkerPool's docstring for the measured evidence behind this (threading
    #: gave no benefit here; a real process pool measured a modest, scale-
    #: dependent 1.05x-1.66x speedup at 50k-100k drones). Call
    #: DistributedCoordinator.shutdown() when done with a coordinator
    #: constructed this way, or its worker processes will leak.
    use_processes: bool = False

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.num_partitions is not None and self.num_partitions < 1:
            raise ValueError("num_partitions must be >= 1")
        if self.rebalance_interval_ticks < 1:
            raise ValueError("rebalance_interval_ticks must be >= 1")
        if self.rebalance_imbalance_threshold <= 1.0:
            raise ValueError("rebalance_imbalance_threshold must be > 1.0")
        if self.worker_retry_limit < 1:
            raise ValueError("worker_retry_limit must be >= 1")
        if self.use_threads and self.use_processes:
            raise ValueError("use_threads and use_processes are mutually exclusive")


@dataclass
class PartitionLoadStats:
    """Per-partition load measurement for one tick, used by the rebalancer
    and exposed to callers/benchmarks."""

    partition_id: int
    owned_drone_count: int
    ghost_drone_count: int
    candidate_pair_count: int  # raw/local -- see WorkerDetectionResult docstring
    tick_duration_s: float


class TickCommitError(RuntimeError):
    """Raised when a tick could not be completed within ``worker_retry_limit``
    attempts. The authoritative state is guaranteed unchanged (see module
    docstring) -- this is a clean, all-or-nothing failure, never a partial
    commit."""

    def __init__(self, tick: int, attempts: int, last_error: str) -> None:
        self.tick = tick
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"tick {tick} failed after {attempts} attempt(s): {last_error}")


class DistributedCoordinator:
    """Creates partitions, assigns them to workers, and drives one tick at a
    time across a :class:`~drone_sim.worker.WorkerPool`.

    ``movement``/``world`` mirror :class:`~drone_sim.simulation.Simulation`'s
    constructor for the same reason: reuse, not duplication.
    """

    def __init__(
        self,
        config: SimulationConfig,
        dist_config: DistributedConfig,
        movement: MovementSystem | None = None,
        world: World | None = None,
    ) -> None:
        self.movement = movement if movement is not None else MovementSystem()
        for policy in self.movement.policies.values():
            if getattr(policy, "requires_context", False):
                raise NotImplementedError(
                    f"DistributedCoordinator does not support requires_context "
                    f"movement policies ({policy.__class__.__name__}): correct "
                    f"cross-partition MovementContext exchange (a second, "
                    f"pre-movement ghost round-trip for trajectory prediction) "
                    f"is a documented, out-of-scope follow-up for this phase. "
                    f"Use drone_sim.simulation.Simulation directly for "
                    f"LocalAvoidanceMovementAlgorithm."
                )

        self.config = config
        self.dist_config = dist_config
        self.world = world if world is not None else World.create(config)
        self.clock = SimulationClock(dt=config.dt)
        self.metrics = MetricsCollector()
        self.resolver = CollisionResolutionEngine(config)

        num_partitions = dist_config.num_partitions or dist_config.num_workers
        self.partition_grid = PartitionGrid(config, num_partitions)

        self.halo_distance = (
            dist_config.halo_distance if dist_config.halo_distance is not None else config.interaction_radius
        )
        if self.halo_distance < config.interaction_radius:
            raise ValueError(
                f"halo_distance ({self.halo_distance}) must be >= config.interaction_radius "
                f"({config.interaction_radius}) so no cross-partition interacting pair is missed "
                f"-- the same guarantee cell_size >= near_miss_radius gives SpatialHashGrid."
            )

        self.pool = WorkerPool(
            dist_config.num_workers,
            use_threads=dist_config.use_threads,
            use_processes=dist_config.use_processes,
        )
        # Deterministic initial assignment: partition i -> worker (i % num_workers).
        self.partition_worker: dict[int, int] = {
            p.partition_id: p.partition_id % dist_config.num_workers for p in self.partition_grid.partitions
        }

        self.last_load_stats: list[PartitionLoadStats] = []
        self.last_tick_attempts: int = 0
        self.reassignment_log: list[tuple[int, int, int]] = []  # (tick, partition_id, new_worker_id)

    def set_fault_injector(self, fn) -> None:
        self.pool.set_fault_injector(fn)

    def shutdown(self) -> None:
        """Release this coordinator's worker pool (only meaningful, and only
        needed, when ``dist_config.use_processes=True`` -- see
        ``WorkerPool.shutdown()``). A no-op for the default sequential/
        threaded pools, so it is always safe to call."""
        self.pool.shutdown()

    def metrics_summary(self) -> dict:
        """Phase 5 monitoring: distributed-execution metrics for this
        coordinator, as a plain, JSON-serializable dict.

        Not currently exposed as a live HTTP endpoint -- the FastAPI layer
        (``drone_sim.api``) only ever drives a plain ``Simulation`` via
        ``SimulationRuntime``, never a ``DistributedCoordinator`` (see
        README.md's Phase 4/5 sections), so there is no live coordinator
        instance for an API process to report on today. This method exists so
        any caller that *does* run a coordinator directly (tests, benchmarks,
        a future integration) has one place to read the metrics the Phase 5
        spec asks for, rather than reaching into private attributes.
        """
        healthy = set(self.pool.healthy_worker_ids())
        partitions_per_worker: dict[int, int] = {}
        for worker_id in self.partition_worker.values():
            partitions_per_worker[worker_id] = partitions_per_worker.get(worker_id, 0) + 1

        return {
            "tick": self.clock.tick,
            "num_workers": self.dist_config.num_workers,
            "num_partitions": len(self.partition_grid.partitions),
            "healthy_worker_count": len(healthy),
            "unhealthy_worker_count": self.dist_config.num_workers - len(healthy),
            "partitions_per_worker": partitions_per_worker,
            "last_tick_attempts": self.last_tick_attempts,
            "total_reassignments": len(self.reassignment_log),
            "reassignments_this_tick": sum(1 for t, _, _ in self.reassignment_log if t == self.clock.tick),
            "ghost_drone_count_last_tick": sum(s.ghost_drone_count for s in self.last_load_stats),
            "owned_drone_count_last_tick": sum(s.owned_drone_count for s in self.last_load_stats),
            "candidate_pair_count_last_tick": sum(s.candidate_pair_count for s in self.last_load_stats),
            "per_partition_load": [
                {
                    "partition_id": s.partition_id,
                    "owned_drone_count": s.owned_drone_count,
                    "ghost_drone_count": s.ghost_drone_count,
                    "candidate_pair_count": s.candidate_pair_count,
                    "tick_duration_s": s.tick_duration_s,
                }
                for s in self.last_load_stats
            ],
        }

    # ------------------------------------------------------------- helpers
    def _current_owners(self) -> np.ndarray:
        """Owning partition id for every drone (active or not), derived
        purely from current position -- see module docstring."""
        return self.partition_grid.assign(self.world.state.positions)

    def _derive_seed(self, tick: int, partition_id: int) -> np.random.SeedSequence:
        """Movement RNG seed as a pure function of (config.seed, tick,
        partition_id) -- independent of worker identity and attempt count, so
        retries and rebalancing never change numerical results."""
        return np.random.SeedSequence([self.config.seed, tick, partition_id])

    # ------------------------------------------------------------- ticking
    def step(self) -> DetectionResult:
        t_start = time.perf_counter()
        owners = self._current_owners()

        attempt = 0
        last_error = ""
        staged_state = merged_result = load_stats = None
        while attempt < self.dist_config.worker_retry_limit:
            attempt += 1
            try:
                staged_state, merged_result, load_stats = self._attempt_tick(owners)
                break
            except WorkerFailure as exc:
                last_error = str(exc)
                self._handle_failure(exc)

        if staged_state is None:
            raise TickCommitError(tick=self.clock.tick, attempts=attempt, last_error=last_error)

        # Commit: the only point at which authoritative state changes.
        self.world.state.positions[:] = staged_state.positions
        self.world.state.velocities[:] = staged_state.velocities

        self.last_tick_attempts = attempt
        self.last_load_stats = load_stats

        tick_time = time.perf_counter() - t_start
        self.metrics.record(
            TickMetrics(
                tick=self.clock.tick,
                tick_time_s=tick_time,
                candidate_pairs=merged_result.num_candidate_pairs,
                collisions=merged_result.num_collisions,
                near_misses=merged_result.num_near_misses,
                active_drones=int(self.world.state.active_mask.sum()),
            )
        )
        self.clock.advance()

        if self.clock.tick % self.dist_config.rebalance_interval_ticks == 0:
            self._maybe_rebalance()

        return merged_result

    def run(self, num_ticks: int) -> MetricsCollector:
        for _ in range(num_ticks):
            self.step()
        return self.metrics

    # --------------------------------------------------------- one attempt
    def _attempt_tick(self, owners: np.ndarray):
        state = self.world.state
        n = state.num_drones
        all_ids = np.arange(n, dtype=np.int64)
        active_owned = {
            part.partition_id: all_ids[(owners == part.partition_id) & state.active_mask]
            for part in self.partition_grid.partitions
        }

        # ---- phase 1: movement + boundary, owned drones only ----
        movement_jobs: list[tuple[int, WorkerMovementInput]] = []
        for part in self.partition_grid.partitions:
            pid = part.partition_id
            owned_idx = active_owned[pid]
            worker_id = self.partition_worker[pid]
            movement_jobs.append(
                (
                    worker_id,
                    WorkerMovementInput(
                        partition_id=pid,
                        tick=self.clock.tick,
                        config=self.config,
                        movement=self.movement,
                        positions=state.positions[owned_idx],
                        velocities=state.velocities[owned_idx],
                        active_mask=state.active_mask[owned_idx],
                        movement_policy_ids=state.movement_policy_ids[owned_idx],
                        goal_positions=None if state.goal_positions is None else state.goal_positions[owned_idx],
                        global_ids=owned_idx,
                        rng_seed=self._derive_seed(self.clock.tick, pid),
                    ),
                )
            )
        movement_results: dict[int, WorkerMovementResult] = self.pool.run_movement_batch(movement_jobs)

        staged_positions = state.positions.copy()
        staged_velocities = state.velocities.copy()
        for res in movement_results.values():
            staged_positions[res.global_ids] = res.positions
            staged_velocities[res.global_ids] = res.velocities

        # ---- ghost exchange (post-movement) + phase 2: detection ----
        detection_jobs: list[tuple[int, WorkerDetectionInput]] = []
        for part in self.partition_grid.partitions:
            pid = part.partition_id
            owned_idx = active_owned[pid]
            ghost_id_parts: list[np.ndarray] = []
            for neighbor_id in self.partition_grid.neighbors(pid):
                neighbor_owned = active_owned[neighbor_id]
                exports = self.partition_grid.ghost_export_indices(
                    staged_positions, neighbor_owned, neighbor_id, self.halo_distance
                )
                idxs = exports.get(pid)
                if idxs is not None and idxs.size:
                    ghost_id_parts.append(idxs)
            ghost_ids = np.concatenate(ghost_id_parts) if ghost_id_parts else np.empty(0, dtype=np.int64)

            worker_id = self.partition_worker[pid]
            detection_jobs.append(
                (
                    worker_id,
                    WorkerDetectionInput(
                        partition_id=pid,
                        tick=self.clock.tick,
                        config=self.config,
                        owned_global_ids=owned_idx,
                        owned_positions=staged_positions[owned_idx],
                        ghost_global_ids=ghost_ids,
                        ghost_positions=staged_positions[ghost_ids],
                    ),
                )
            )
        detection_results: dict[int, WorkerDetectionResult] = self.pool.run_detection_batch(detection_jobs)

        merged_result, load_stats = self._merge_detection_results(detection_results, owners)

        staged_state = DroneState(
            positions=staged_positions,
            velocities=staged_velocities,
            active_mask=state.active_mask.copy(),
            movement_policy_ids=state.movement_policy_ids.copy(),
            goal_positions=None if state.goal_positions is None else state.goal_positions.copy(),
        )
        # Resolve exactly once, on the merged/deduplicated global pair set,
        # directly against the (not-yet-committed) staged state -- this is
        # what lets cross-partition collisions update both drones correctly
        # without any worker ever writing to a drone it doesn't own: neither
        # worker resolves anything, only the coordinator does, once, here.
        self.resolver.resolve(staged_state, merged_result)

        return staged_state, merged_result, load_stats

    def _merge_detection_results(
        self, detection_results: dict[int, WorkerDetectionResult], owners: np.ndarray
    ) -> tuple[DetectionResult, list[PartitionLoadStats]]:
        def _kept_by(pairs: np.ndarray, pid: int) -> np.ndarray:
            if pairs.shape[0] == 0:
                return np.zeros(0, dtype=bool)
            owner_i = owners[pairs[:, 0]]
            owner_j = owners[pairs[:, 1]]
            return np.minimum(owner_i, owner_j) == pid

        coll_pairs, coll_dists, near_pairs, near_dists, cand_pairs = [], [], [], [], []
        load_stats: list[PartitionLoadStats] = []

        for pid in sorted(detection_results):
            res = detection_results[pid]
            load_stats.append(
                PartitionLoadStats(
                    partition_id=pid,
                    owned_drone_count=res.owned_drone_count,
                    ghost_drone_count=res.ghost_drone_count,
                    candidate_pair_count=res.candidate_pair_count,
                    tick_duration_s=(res.grid_ns + res.pairs_ns + res.detect_ns) / 1e9,
                )
            )

            keep_c = _kept_by(res.collision_pairs, pid)
            keep_n = _kept_by(res.near_miss_pairs, pid)
            keep_cand = _kept_by(res.candidate_pairs, pid)
            coll_pairs.append(res.collision_pairs[keep_c])
            coll_dists.append(res.collision_distances[keep_c])
            near_pairs.append(res.near_miss_pairs[keep_n])
            near_dists.append(res.near_miss_distances[keep_n])
            cand_pairs.append(res.candidate_pairs[keep_cand])

        def _cat(parts, width=None):
            if not parts:
                return np.empty(0) if width is None else np.empty((0, width), dtype=np.int64)
            return np.concatenate(parts, axis=0)

        collision_pairs = _cat(coll_pairs, width=2)
        collision_distances = _cat(coll_dists)
        near_miss_pairs = _cat(near_pairs, width=2)
        near_miss_distances = _cat(near_dists)
        candidate_pairs_total = _cat(cand_pairs, width=2)

        merged = DetectionResult(
            collision_pairs=collision_pairs.astype(np.int64) if collision_pairs.size else np.empty((0, 2), dtype=np.int64),
            collision_distances=collision_distances.astype(np.float64),
            near_miss_pairs=near_miss_pairs.astype(np.int64) if near_miss_pairs.size else np.empty((0, 2), dtype=np.int64),
            near_miss_distances=near_miss_distances.astype(np.float64),
            num_candidate_pairs=int(candidate_pairs_total.shape[0]),
        )
        return merged, load_stats

    # ------------------------------------------------------------ failure
    def _handle_failure(self, exc: WorkerFailure) -> None:
        """Mark the failed worker unhealthy and reassign every partition it
        owned to a remaining healthy worker (round robin, deterministic by
        partition id). The caller retries the whole tick afterward -- nothing
        about the authoritative state was ever touched by the failed
        attempt (see module docstring)."""
        self.pool.mark_failed(exc.worker_id)
        healthy = self.pool.healthy_worker_ids()
        if not healthy:
            raise RuntimeError(
                f"all {self.dist_config.num_workers} worker(s) are marked FAILED; cannot recover"
            ) from exc

        affected = sorted(pid for pid, wid in self.partition_worker.items() if wid == exc.worker_id)
        for k, pid in enumerate(affected):
            new_worker = healthy[k % len(healthy)]
            self.partition_worker[pid] = new_worker
            self.reassignment_log.append((self.clock.tick, pid, new_worker))

    def recover_worker(self, worker_id: int) -> None:
        """Explicitly mark a previously failed worker healthy again, making
        it eligible for future partition assignment/rebalancing."""
        self.pool.mark_recovered(worker_id)

    # --------------------------------------------------------- rebalancing
    def _maybe_rebalance(self) -> None:
        """Deterministic policy: if the busiest worker's load exceeds the
        mean by more than ``rebalance_imbalance_threshold``, move ONE whole
        partition (the busiest owned by the busiest worker) to the idlest
        worker. Never migrates individual drones -- drone ownership stays
        purely spatial (:meth:`PartitionGrid.assign`); only which *worker*
        runs a given partition changes here."""
        if not self.last_load_stats:
            return

        healthy = set(self.pool.healthy_worker_ids())
        if len(healthy) < 2:
            return

        worker_load: dict[int, float] = {w: 0.0 for w in healthy}
        partitions_by_worker: dict[int, list[tuple[int, float]]] = {w: [] for w in healthy}
        for stat in self.last_load_stats:
            wid = self.partition_worker.get(stat.partition_id)
            if wid not in healthy:
                continue
            load = float(stat.owned_drone_count + stat.candidate_pair_count)
            worker_load[wid] += load
            partitions_by_worker[wid].append((stat.partition_id, load))

        if not worker_load:
            return
        mean_load = sum(worker_load.values()) / len(worker_load)
        if mean_load <= 0:
            return
        busiest_worker = max(worker_load, key=lambda w: (worker_load[w], w))
        idlest_worker = min(worker_load, key=lambda w: (worker_load[w], w))
        if busiest_worker == idlest_worker:
            return
        if worker_load[busiest_worker] / mean_load < self.dist_config.rebalance_imbalance_threshold:
            return

        candidates = partitions_by_worker.get(busiest_worker, [])
        if not candidates:
            return
        heaviest_partition_id, _ = max(candidates, key=lambda t: (t[1], t[0]))
        self.partition_worker[heaviest_partition_id] = idlest_worker
        self.reassignment_log.append((self.clock.tick, heaviest_partition_id, idlest_worker))
