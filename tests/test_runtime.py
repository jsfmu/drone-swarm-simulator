import time

import pytest

from drone_sim.config import SimulationConfig
from drone_sim.runtime import RuntimeStatus, SimulationRuntime


def cfg(n=100, world=50.0, seed=0):
    return SimulationConfig(num_drones=n, bounds_min=(0, 0, 0), bounds_max=(world, world, world), seed=seed)


def wait_until(predicate, timeout=2.0, interval=0.01):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_simulation_advances_without_external_driving():
    runtime = SimulationRuntime("r1", cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 3)
    finally:
        runtime.shutdown()


def test_start_does_not_create_duplicate_loops():
    runtime = SimulationRuntime("r2", cfg())
    try:
        runtime.start()
        with pytest.raises(RuntimeError):
            runtime.start()
    finally:
        runtime.shutdown()


def test_pause_stops_advancement():
    runtime = SimulationRuntime("r3", cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 2)
        runtime.pause()
        # pause() racing the in-flight tick can let at most one more tick land
        # right after it returns; give that a moment to settle before treating
        # the tick as frozen.
        time.sleep(0.1)
        stable_tick = runtime.get_snapshot().tick
        time.sleep(0.2)
        assert runtime.get_snapshot().tick == stable_tick
        assert runtime.get_status().status == RuntimeStatus.PAUSED
    finally:
        runtime.shutdown()


def test_resume_restarts_advancement():
    runtime = SimulationRuntime("r4", cfg())
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
    runtime = SimulationRuntime("r5", cfg())
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
    runtime = SimulationRuntime("r6", cfg())
    try:
        snap = runtime.step_once()
        assert snap.tick == 1
    finally:
        runtime.shutdown()


def test_step_while_running_is_rejected():
    runtime = SimulationRuntime("r7", cfg())
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
        with pytest.raises(RuntimeError):
            runtime.step_once()
    finally:
        runtime.shutdown()


def test_reset_restores_deterministic_initial_state():
    runtime = SimulationRuntime("r8", cfg(seed=7))
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
    runtime = SimulationRuntime("r9", cfg())
    runtime.start()
    assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
    runtime.shutdown()
    tick_after_shutdown = runtime.get_snapshot().tick
    time.sleep(0.2)
    assert runtime.get_snapshot().tick == tick_after_shutdown
    assert runtime.get_status().status == RuntimeStatus.STOPPED


def test_get_snapshot_and_status_with_lock_wait_matches_separate_calls():
    """The combined accessor must report the same snapshot/status a caller
    would get from get_snapshot() + get_status() separately, while only
    acquiring the lock once (see routes.py's get_frame(), which used to call
    both separately -- a second, unmeasured lock acquisition)."""
    runtime = SimulationRuntime("r11", cfg(n=50))
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
    runtime = SimulationRuntime("r12", cfg(n=50))
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
    """Many threads calling get_snapshot() while the background loop steps
    must always see a fully-formed snapshot (never partially-updated arrays)."""
    runtime = SimulationRuntime("r10", cfg(n=200))
    errors = []

    def reader():
        for _ in range(200):
            snap = runtime.get_snapshot()
            if not (snap.positions.shape[0] == snap.velocities.shape[0] == snap.drone_ids.shape[0]):
                errors.append(snap.tick)

    try:
        runtime.start()
        import threading

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()
        for t in readers:
            t.join()
        assert errors == []
    finally:
        runtime.shutdown()
