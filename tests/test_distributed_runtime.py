import threading
import time

import pytest

from drone_sim.config import SimulationConfig
from drone_sim.coordinator import DistributedConfig
from drone_sim.distributed_runtime import DistributedSimulationRuntime
from drone_sim.movement import LocalAvoidanceMovementAlgorithm, MovementSystem
from drone_sim.runtime import RuntimeStatus


def cfg(n=100, world=50.0, seed=0):
    return SimulationConfig(num_drones=n, bounds_min=(0, 0, 0), bounds_max=(world, world, world), seed=seed)


def dist_cfg(**kw):
    base = dict(num_workers=1)
    base.update(kw)
    return DistributedConfig(**base)


def wait_until(predicate, timeout=2.0, interval=0.01):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ------------------------------------------------------------ lifecycle (mirrors test_runtime.py)
def test_simulation_advances_without_external_driving():
    runtime = DistributedSimulationRuntime("d1", cfg(), dist_cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 3)
    finally:
        runtime.shutdown()


def test_start_does_not_create_duplicate_loops():
    runtime = DistributedSimulationRuntime("d2", cfg(), dist_cfg())
    try:
        runtime.start()
        with pytest.raises(RuntimeError):
            runtime.start()
    finally:
        runtime.shutdown()


def test_pause_stops_advancement():
    runtime = DistributedSimulationRuntime("d3", cfg(), dist_cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 2)
        runtime.pause()
        time.sleep(0.1)
        stable_tick = runtime.get_snapshot().tick
        time.sleep(0.2)
        assert runtime.get_snapshot().tick == stable_tick
        assert runtime.get_status().status == RuntimeStatus.PAUSED
    finally:
        runtime.shutdown()


def test_resume_restarts_advancement():
    runtime = DistributedSimulationRuntime("d4", cfg(), dist_cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
        runtime.pause()
        paused_tick = runtime.get_snapshot().tick
        runtime.resume()
        assert wait_until(lambda: runtime.get_snapshot().tick > paused_tick)
    finally:
        runtime.shutdown()


def test_single_step_advances_exactly_one_tick_while_paused():
    runtime = DistributedSimulationRuntime("d5", cfg(), dist_cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
        runtime.pause()
        time.sleep(0.1)
        before = runtime.get_snapshot().tick
        snap = runtime.step_once()
        assert snap.tick == before + 1
        time.sleep(0.1)
        assert runtime.get_snapshot().tick == before + 1
    finally:
        runtime.shutdown()


def test_step_once_works_when_never_started():
    runtime = DistributedSimulationRuntime("d6", cfg(), dist_cfg())
    try:
        snap = runtime.step_once()
        assert snap.tick == 1
    finally:
        runtime.shutdown()


def test_step_while_running_is_rejected():
    runtime = DistributedSimulationRuntime("d7", cfg(), dist_cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
        with pytest.raises(RuntimeError):
            runtime.step_once()
    finally:
        runtime.shutdown()


def test_reset_restores_deterministic_initial_state():
    runtime = DistributedSimulationRuntime("d8", cfg(seed=7), dist_cfg())
    try:
        initial_positions = runtime.get_snapshot().positions.copy()
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 3)
        runtime.pause()
        snap = runtime.reset()
        assert snap.tick == 0
        assert (snap.positions == initial_positions).all()
    finally:
        runtime.shutdown()


def test_shutdown_stops_the_loop():
    runtime = DistributedSimulationRuntime("d9", cfg(), dist_cfg())
    runtime.start()
    assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
    runtime.shutdown()
    tick_after_shutdown = runtime.get_snapshot().tick
    time.sleep(0.2)
    assert runtime.get_snapshot().tick == tick_after_shutdown
    assert runtime.get_status().status == RuntimeStatus.STOPPED


def test_get_snapshot_and_status_with_lock_wait_matches_separate_calls():
    runtime = DistributedSimulationRuntime("d11", cfg(n=50), dist_cfg())
    try:
        runtime.step_once()
        runtime.step_once()
        snapshot, state, lock_wait_ms = runtime.get_snapshot_and_status_with_lock_wait()
        assert snapshot.tick == runtime.get_snapshot().tick
        assert state.status == runtime.get_status().status
        assert state.tick == snapshot.tick
        assert state.num_drones == 50
        assert lock_wait_ms >= 0.0
    finally:
        runtime.shutdown()


def test_get_snapshot_and_status_with_lock_wait_reflects_current_status():
    runtime = DistributedSimulationRuntime("d12", cfg(n=50), dist_cfg())
    try:
        _, state, _ = runtime.get_snapshot_and_status_with_lock_wait()
        assert state.status == RuntimeStatus.CREATED

        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
        _, state, _ = runtime.get_snapshot_and_status_with_lock_wait()
        assert state.status == RuntimeStatus.RUNNING

        runtime.pause()
        _, state, _ = runtime.get_snapshot_and_status_with_lock_wait()
        assert state.status == RuntimeStatus.PAUSED
    finally:
        runtime.shutdown()


def test_concurrent_readers_never_see_a_torn_snapshot():
    runtime = DistributedSimulationRuntime("d10", cfg(n=200), dist_cfg())
    errors = []

    def reader():
        for _ in range(50):
            snap = runtime.get_snapshot()
            if not (snap.positions.shape[0] == snap.velocities.shape[0] == snap.drone_ids.shape[0]):
                errors.append(snap.tick)

    try:
        runtime.start()
        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()
        for t in readers:
            t.join()
        assert errors == []
    finally:
        runtime.shutdown()


# --------------------------------------------------------- distributed-specific
def test_requires_context_policy_raises_not_implemented_error_at_construction():
    algo = LocalAvoidanceMovementAlgorithm()
    ms = MovementSystem(policies={algo.policy_id: algo})
    with pytest.raises(NotImplementedError):
        DistributedSimulationRuntime("d13", cfg(), dist_cfg(), movement=ms)


def test_reset_shuts_down_old_process_pool_before_creating_new_one():
    runtime = DistributedSimulationRuntime("d14", cfg(n=20), dist_cfg(num_workers=2, use_processes=True))
    try:
        runtime.step_once()  # forces lazy ProcessPoolExecutor creation
        old_coord = runtime._coord
        assert old_coord.pool._process_executor is not None

        runtime.reset()

        assert old_coord.pool._process_executor is None  # old pool released
        assert runtime._coord.pool._process_executor is None  # new coord: nothing created yet
    finally:
        runtime.shutdown()


def test_shutdown_releases_process_pool():
    runtime = DistributedSimulationRuntime("d15", cfg(n=20), dist_cfg(num_workers=2, use_processes=True))
    runtime.step_once()
    assert runtime._coord.pool._process_executor is not None
    runtime.shutdown()
    assert runtime._coord.pool._process_executor is None


def test_get_distributed_metrics_reflects_tick_and_worker_health():
    runtime = DistributedSimulationRuntime("d16", cfg(n=30), dist_cfg(num_workers=2))
    try:
        runtime.step_once()
        runtime.step_once()
        metrics = runtime.get_distributed_metrics()
        assert metrics["tick"] == 2
        assert metrics["num_workers"] == 2
        assert metrics["healthy_worker_count"] == 2
        assert metrics["unhealthy_worker_count"] == 0
    finally:
        runtime.shutdown()
