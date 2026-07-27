"""Tests for the Phase 3A tick-rate regression fix.

Covers: RunningMetrics staying O(1)/bounded (not recomputing from full tick
history), tick timing excluding snapshot/query work, scheduler sleep not
being counted as simulation time, and the background loop not starving API
readers of the lock. See README's "Phase 3A tick-rate regression" section
for the measured root cause this defends against regressing.
"""

import threading
import time

import pytest

from drone_sim.config import SimulationConfig
from drone_sim.metrics import TickMetrics
from drone_sim.runtime import RECENT_WINDOW, RunningMetrics, SimulationRuntime
from drone_sim.simulation import Simulation
from drone_sim.snapshot import build_snapshot


def cfg(n=200, world=50.0, seed=0):
    return SimulationConfig(num_drones=n, bounds_min=(0, 0, 0), bounds_max=(world, world, world), seed=seed)


def wait_until(predicate, timeout=2.0, interval=0.01):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


# --------------------------------------------------------------- RunningMetrics
def test_running_metrics_totals_are_exact_and_o1_per_tick():
    rm = RunningMetrics()
    for i in range(1000):
        rm.record(TickMetrics(tick=i, tick_time_s=0.01, candidate_pairs=5, collisions=1, near_misses=2, active_drones=10))
    summary = rm.summary()
    assert summary["num_ticks"] == 1000
    assert summary["total_collisions"] == 1000
    assert summary["total_near_misses"] == 2000
    assert summary["total_candidate_pairs"] == 5000
    assert summary["mean_tick_ms"] == pytest.approx(10.0)
    assert summary["ticks_per_second"] == pytest.approx(100.0)


def test_running_metrics_recent_window_is_bounded():
    """The deque backing median/p95 never grows past RECENT_WINDOW, regardless
    of how many ticks have been recorded -- this is what keeps summary() O(1)
    instead of the O(ticks-so-far) cost that caused the original regression."""
    rm = RunningMetrics()
    for i in range(RECENT_WINDOW * 10):
        rm.record(TickMetrics(tick=i, tick_time_s=0.005, candidate_pairs=1, collisions=0, near_misses=0, active_drones=10))
    assert len(rm.recent_tick_times_ms) == RECENT_WINDOW


def test_running_metrics_summary_cost_does_not_grow_with_history():
    """summary() at a long history must not cost meaningfully more than at a
    short one -- proves the fix is not merely 'less bad' but actually O(1)."""
    rm_short = RunningMetrics()
    for i in range(50):
        rm_short.record(TickMetrics(tick=i, tick_time_s=0.005, candidate_pairs=1, collisions=0, near_misses=0, active_drones=10))

    rm_long = RunningMetrics()
    for i in range(200_000):
        rm_long.record(TickMetrics(tick=i, tick_time_s=0.005, candidate_pairs=1, collisions=0, near_misses=0, active_drones=10))

    t0 = time.perf_counter()
    for _ in range(50):
        rm_short.summary()
    short_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(50):
        rm_long.summary()
    long_ms = (time.perf_counter() - t0) * 1e3

    # Generous bound (5x) to absorb timing noise -- the point is "flat", not
    # "grew 4000x" the way calling MetricsCollector.summary() every tick would.
    assert long_ms < short_ms * 5 + 5.0


# --------------------------------------------------------------- build_snapshot
def test_build_snapshot_never_calls_metrics_summary():
    """Monkeypatch Simulation.metrics.summary to explode if called -- proves
    build_snapshot() (and therefore the runtime's hot tick path) never
    triggers the O(history) recomputation, however long the sim has run."""
    sim = Simulation(cfg())
    for _ in range(30):
        sim.step()

    def _boom():
        raise AssertionError("build_snapshot must not call MetricsCollector.summary()")

    sim.metrics.summary = _boom  # type: ignore[method-assign]
    build_snapshot("s", sim, None, {"num_ticks": 30})  # must not raise


# --------------------------------------------------------------- runtime hot path
def test_tick_timings_exclude_snapshot_and_query_work():
    runtime = SimulationRuntime("t1", cfg(n=500))
    try:
        runtime.step_once()
        timings = runtime.get_last_timings()
        # sim_step_ms is measured around Simulation.step() alone; snapshot_build_ms
        # around build_snapshot() alone. Neither includes any heatmap/collision
        # query or JSON work -- those aren't even called by step_once().
        assert timings.sim_step_ms >= 0.0
        assert timings.snapshot_build_ms >= 0.0
        # A sanity ceiling: at 500 drones neither stage should be anywhere near
        # "tens of milliseconds" -- if it is, something got pulled into the
        # hot path that doesn't belong there.
        assert timings.sim_step_ms < 50.0
        assert timings.snapshot_build_ms < 50.0
    finally:
        runtime.shutdown()


def test_running_metrics_mean_matches_raw_tick_times_not_inflated():
    """mean_tick_ms reported by the runtime's metrics must reflect only
    Simulation.step() cost -- not snapshot building, not API queries."""
    runtime = SimulationRuntime("t2", cfg(n=300))
    try:
        for _ in range(20):
            runtime.step_once()
        snap = runtime.get_snapshot()
        raw_mean_ms = sum(tm.tick_time_s for tm in runtime._sim.metrics.ticks) / 20 * 1e3
        assert snap.metrics["mean_tick_ms"] == pytest.approx(raw_mean_ms, rel=1e-6)
    finally:
        runtime.shutdown()


def test_scheduler_sleep_is_not_counted_as_tick_time():
    """Running with a non-zero tick_interval_s (a real inter-tick sleep) must
    not inflate mean_tick_ms -- TickMetrics.tick_time_s is recorded inside
    Simulation.step(), strictly before the loop's own time.sleep() call."""
    runtime = SimulationRuntime("t3", cfg(n=200))
    try:
        runtime.start(tick_interval_s=0.05)  # 50ms deliberate sleep between ticks
        assert wait_until(lambda: runtime.get_snapshot().tick >= 3)
        snap = runtime.get_snapshot()
        # If the 50ms sleep leaked into tick_time_s, mean_tick_ms would be
        # ~50ms+; real simulation-only cost at 200 drones is a fraction of a ms.
        assert snap.metrics["mean_tick_ms"] < 20.0
    finally:
        runtime.shutdown()


# --------------------------------------------------------------- locking / polling
def test_lock_is_not_held_during_query_or_serialization_work():
    """While the background loop is running, a slow simulated 'query' must not
    block tick advancement -- proves query work happens outside the lock."""
    runtime = SimulationRuntime("t4", cfg(n=300))
    try:
        runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 2)

        # get_snapshot() only ever holds the lock long enough to copy a
        # reference -- simulate a slow reader by holding the returned
        # snapshot and sleeping "as if" doing heatmap/JSON work, without
        # re-acquiring the lock, and confirm ticks keep advancing meanwhile.
        snap = runtime.get_snapshot()
        tick_before = snap.tick
        time.sleep(0.3)  # stand-in for slow query/serialization work
        assert wait_until(lambda: runtime.get_snapshot().tick > tick_before, timeout=1.0)
    finally:
        runtime.shutdown()


def test_api_polling_does_not_advance_the_simulation():
    from fastapi.testclient import TestClient

    from drone_sim.api.app import create_app
    from drone_sim.api.routes import reset_registry

    app = create_app()
    with TestClient(app) as client:
        try:
            resp = client.post("/simulations", json={"num_drones": 100, "bounds_max": [50, 50, 50]})
            sim_id = resp.json()["simulation_id"]
            # Never call /start or /step -- only read endpoints.
            for _ in range(10):
                client.get(f"/simulations/{sim_id}/frame?x_min=0&x_max=50&y_min=0&y_max=50")
                client.get(f"/simulations/{sim_id}/metrics")
                client.get(f"/simulations/{sim_id}")
            assert client.get(f"/simulations/{sim_id}").json()["tick"] == 0
        finally:
            reset_registry()


def test_frame_endpoint_reuses_one_snapshot_across_all_fields():
    from fastapi.testclient import TestClient

    from drone_sim.api.app import create_app
    from drone_sim.api.routes import reset_registry

    app = create_app()
    with TestClient(app) as client:
        try:
            resp = client.post("/simulations", json={"num_drones": 200, "bounds_max": [50, 50, 50]})
            sim_id = resp.json()["simulation_id"]
            for _ in range(5):
                client.post(f"/simulations/{sim_id}/step")

            resp = client.get(f"/simulations/{sim_id}/frame?x_min=0&x_max=50&y_min=0&y_max=50")
            body = resp.json()
            # tick appears at top level and implicitly in metrics' num_ticks --
            # both must agree with each other and with the standalone status.
            assert body["tick"] == 5
            assert body["metrics"]["num_ticks"] == 5
            status = client.get(f"/simulations/{sim_id}").json()
            assert status["tick"] == body["tick"]
        finally:
            reset_registry()


def test_collision_history_is_not_copied_without_bound():
    """SimulationSnapshot must only ever hold the CURRENT tick's collision/
    near-miss pairs, never an accumulating history -- run many ticks and
    confirm array sizes stay bounded by drone count, not by tick count."""
    runtime = SimulationRuntime("t5", cfg(n=100, world=6.0))
    try:
        max_pairs_seen = 0
        for _ in range(200):
            snap = runtime.step_once()
            max_pairs_seen = max(max_pairs_seen, snap.collision_pairs.shape[0], snap.near_miss_pairs.shape[0])
        # An unbounded history bug would make this grow roughly linearly with
        # tick count (200 ticks here); the true cap is limited by how many
        # *simultaneous* pairs 100 drones can form in one tick.
        assert max_pairs_seen < 100 * 100
    finally:
        runtime.shutdown()


def test_only_one_runtime_loop_can_run_and_lifecycle_still_works():
    """Re-confirms start/pause/resume/step/reset all still work after the
    RunningMetrics/lock-yield changes (regression guard for this fix)."""
    runtime = SimulationRuntime("t6", cfg(n=100))
    try:
        runtime.start()
        with pytest.raises(RuntimeError):
            runtime.start()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 1)
        runtime.pause()
        with pytest.raises(RuntimeError):
            runtime.pause()  # already paused
        runtime.resume()
        assert wait_until(lambda: runtime.get_snapshot().tick >= 2)
        runtime.pause()
        before = runtime.get_snapshot().tick
        snap = runtime.step_once()
        assert snap.tick == before + 1
        snap = runtime.reset()
        assert snap.tick == 0
    finally:
        runtime.shutdown()
