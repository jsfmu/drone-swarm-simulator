import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.movement import MovementSystem, ScriptedMovementAlgorithm
from drone_sim.worker import (
    Worker,
    WorkerDetectionInput,
    WorkerFailure,
    WorkerLifecycleState,
    WorkerMovementInput,
    WorkerPool,
)


def cfg(**kw):
    base = dict(
        num_drones=10,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(100.0, 100.0, 100.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        cell_size=2.0,
        seed=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _movement_input(config, positions, velocities, global_ids, tick=0, partition_id=0):
    n = positions.shape[0]
    return WorkerMovementInput(
        partition_id=partition_id,
        tick=tick,
        config=config,
        movement=MovementSystem(),  # default registry: Random(0)/Scripted(1)
        positions=positions,
        velocities=velocities,
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=np.full(n, ScriptedMovementAlgorithm.policy_id, dtype=np.int32),
        goal_positions=None,
        global_ids=global_ids,
        rng_seed=np.random.SeedSequence([config.seed, tick, partition_id]),
    )


# -------------------------------------------------------------- movement phase
def test_movement_phase_integrates_only_the_given_owned_drones():
    c = cfg()
    positions = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float32)
    velocities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    global_ids = np.array([3, 7], dtype=np.int64)

    worker = Worker(worker_id=0)
    result = worker.run_movement_phase(_movement_input(c, positions, velocities, global_ids))

    # ScriptedMovementAlgorithm: constant velocity, pure integration.
    expected = positions + velocities * c.dt
    np.testing.assert_allclose(result.positions, expected)
    np.testing.assert_array_equal(result.global_ids, global_ids)
    assert result.movement_ns >= 0
    assert result.boundary_ns >= 0


def test_movement_phase_returns_worker_to_idle():
    c = cfg()
    positions = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    global_ids = np.array([0], dtype=np.int64)
    worker = Worker(worker_id=0)
    assert worker.state == WorkerLifecycleState.IDLE
    worker.run_movement_phase(_movement_input(c, positions, velocities, global_ids))
    assert worker.state == WorkerLifecycleState.IDLE


def test_movement_phase_empty_input_does_not_crash():
    c = cfg()
    positions = np.empty((0, 3), dtype=np.float32)
    velocities = np.empty((0, 3), dtype=np.float32)
    global_ids = np.empty(0, dtype=np.int64)
    worker = Worker(worker_id=0)
    result = worker.run_movement_phase(_movement_input(c, positions, velocities, global_ids))
    assert result.positions.shape == (0, 3)


# ------------------------------------------------------------- detection phase
def test_cross_partition_collision_detected_via_ghost():
    """One owned drone and one read-only ghost, close enough to collide:
    detection must find the pair, reported with GLOBAL ids."""
    c = cfg(collision_radius=1.0, near_miss_radius=2.0, cell_size=2.0)
    owned_ids = np.array([5], dtype=np.int64)
    owned_pos = np.array([[50.0, 50.0, 50.0]], dtype=np.float32)
    ghost_ids = np.array([9], dtype=np.int64)
    ghost_pos = np.array([[50.5, 50.0, 50.0]], dtype=np.float32)  # distance 0.5 < collision_radius

    worker = Worker(worker_id=0)
    result = worker.run_detection_phase(
        WorkerDetectionInput(
            partition_id=0,
            tick=0,
            config=c,
            owned_global_ids=owned_ids,
            owned_positions=owned_pos,
            ghost_global_ids=ghost_ids,
            ghost_positions=ghost_pos,
        )
    )
    assert result.collision_pairs.shape == (1, 2)
    np.testing.assert_array_equal(result.collision_pairs[0], [5, 9])
    assert result.owned_drone_count == 1
    assert result.ghost_drone_count == 1


def test_no_double_advancement_of_ghost_drones():
    """WorkerDetectionResult carries no position/velocity fields at all --
    structurally impossible for a ghost to be "advanced" by the detection
    phase, since nothing about drone motion is computed or returned there."""
    c = cfg()
    owned_ids = np.array([0], dtype=np.int64)
    owned_pos = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    ghost_ids = np.array([1], dtype=np.int64)
    ghost_pos = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)

    worker = Worker(worker_id=0)
    result = worker.run_detection_phase(
        WorkerDetectionInput(
            partition_id=0, tick=0, config=c,
            owned_global_ids=owned_ids, owned_positions=owned_pos,
            ghost_global_ids=ghost_ids, ghost_positions=ghost_pos,
        )
    )
    result_fields = {f for f in result.__dataclass_fields__}
    assert not ({"positions", "velocities"} & result_fields)


def test_detection_phase_with_no_ghosts_matches_owned_only():
    c = cfg(collision_radius=1.0, near_miss_radius=2.0, cell_size=2.0)
    owned_ids = np.array([0, 1], dtype=np.int64)
    owned_pos = np.array([[10.0, 10.0, 10.0], [10.5, 10.0, 10.0]], dtype=np.float32)
    worker = Worker(worker_id=0)
    result = worker.run_detection_phase(
        WorkerDetectionInput(
            partition_id=0, tick=0, config=c,
            owned_global_ids=owned_ids, owned_positions=owned_pos,
            ghost_global_ids=np.empty(0, dtype=np.int64),
            ghost_positions=np.empty((0, 3), dtype=np.float32),
        )
    )
    np.testing.assert_array_equal(result.collision_pairs[0], [0, 1])


# ------------------------------------------------------------------ WorkerPool
def test_worker_pool_healthy_ids_and_mark_failed():
    pool = WorkerPool(num_workers=3)
    assert pool.healthy_worker_ids() == [0, 1, 2]
    pool.mark_failed(1)
    assert pool.healthy_worker_ids() == [0, 2]
    assert pool.worker_state(1) == WorkerLifecycleState.FAILED
    pool.mark_recovered(1)
    assert pool.worker_state(1) == WorkerLifecycleState.RECOVERED


def test_fault_injector_raises_worker_failure():
    pool = WorkerPool(num_workers=2)
    pool.set_fault_injector(lambda worker_id, tick, phase: worker_id == 0 and phase == "movement")

    c = cfg()
    positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    global_ids = np.array([0], dtype=np.int64)
    inp = _movement_input(c, positions, velocities, global_ids)

    with pytest.raises(WorkerFailure):
        pool.run_movement(0, inp)

    # A different worker (not targeted by the injector) succeeds fine.
    result = pool.run_movement(1, inp)
    assert result.partition_id == 0


def test_run_movement_batch_sequential_matches_threaded():
    c = cfg()
    jobs = []
    for pid in range(3):
        positions = np.array([[float(pid) * 10.0, 0.0, 0.0]], dtype=np.float32)
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        global_ids = np.array([pid], dtype=np.int64)
        jobs.append((pid, _movement_input(c, positions, velocities, global_ids, partition_id=pid)))

    pool_seq = WorkerPool(num_workers=3, use_threads=False)
    pool_thr = WorkerPool(num_workers=3, use_threads=True)
    out_seq = pool_seq.run_movement_batch(jobs)
    out_thr = pool_thr.run_movement_batch(jobs)

    assert set(out_seq) == set(out_thr) == {0, 1, 2}
    for pid in out_seq:
        np.testing.assert_allclose(out_seq[pid].positions, out_thr[pid].positions)


# --------------------------------------------------- Phase 5: process executor
def test_use_threads_and_use_processes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        WorkerPool(num_workers=2, use_threads=True, use_processes=True)


def test_run_movement_batch_process_matches_sequential():
    c = cfg()
    jobs = []
    for pid in range(2):
        positions = np.array([[float(pid) * 10.0, 0.0, 0.0]], dtype=np.float32)
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        global_ids = np.array([pid], dtype=np.int64)
        jobs.append((pid, _movement_input(c, positions, velocities, global_ids, partition_id=pid)))

    pool_seq = WorkerPool(num_workers=2, use_threads=False)
    pool_proc = WorkerPool(num_workers=2, use_processes=True)
    try:
        out_seq = pool_seq.run_movement_batch(jobs)
        out_proc = pool_proc.run_movement_batch(jobs)
    finally:
        pool_proc.shutdown()

    assert set(out_seq) == set(out_proc) == {0, 1}
    for pid in out_seq:
        np.testing.assert_allclose(out_seq[pid].positions, out_proc[pid].positions)
        np.testing.assert_array_equal(out_seq[pid].global_ids, out_proc[pid].global_ids)


def test_run_detection_batch_process_matches_sequential():
    c = cfg(collision_radius=1.0, near_miss_radius=2.0, cell_size=2.0)
    jobs = [
        (
            0,
            WorkerDetectionInput(
                partition_id=0, tick=0, config=c,
                owned_global_ids=np.array([0, 1], dtype=np.int64),
                owned_positions=np.array([[10.0, 10.0, 10.0], [10.5, 10.0, 10.0]], dtype=np.float32),
                ghost_global_ids=np.empty(0, dtype=np.int64),
                ghost_positions=np.empty((0, 3), dtype=np.float32),
            ),
        )
    ]
    pool_seq = WorkerPool(num_workers=1, use_threads=False)
    pool_proc = WorkerPool(num_workers=1, use_processes=True)
    try:
        out_seq = pool_seq.run_detection_batch(jobs)
        out_proc = pool_proc.run_detection_batch(jobs)
    finally:
        pool_proc.shutdown()

    np.testing.assert_array_equal(out_seq[0].collision_pairs, out_proc[0].collision_pairs)


def test_process_executor_fault_injection_raises_without_starting_pool():
    """Fault injection is checked in the parent process before any job is
    submitted (see WorkerPool docstring) -- confirm it still raises
    WorkerFailure for the process executor, and that no process pool is
    created when the very first job already fails."""
    pool = WorkerPool(num_workers=2, use_processes=True)
    pool.set_fault_injector(lambda worker_id, tick, phase: True)

    c = cfg()
    positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    global_ids = np.array([0], dtype=np.int64)
    jobs = [(0, _movement_input(c, positions, velocities, global_ids))]

    with pytest.raises(WorkerFailure):
        pool.run_movement_batch(jobs)
    assert pool._process_executor is None  # never created -- fault check happened first
    pool.shutdown()  # no-op, must not raise


def test_process_executor_shutdown_is_idempotent():
    pool = WorkerPool(num_workers=2, use_processes=True)
    c = cfg()
    positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    global_ids = np.array([0], dtype=np.int64)
    jobs = [(0, _movement_input(c, positions, velocities, global_ids))]

    pool.run_movement_batch(jobs)
    assert pool._process_executor is not None
    pool.shutdown()
    assert pool._process_executor is None
    pool.shutdown()  # calling twice must not raise
