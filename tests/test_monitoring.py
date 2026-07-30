import pytest
from fastapi.testclient import TestClient

from drone_sim.api.app import create_app
from drone_sim.api.monitoring import app_ready_state, request_stats
from drone_sim.api.routes import reset_registry


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_registry()


def create_sim(client, num_drones=50, bounds_max=(50.0, 50.0, 50.0), **kwargs):
    resp = client.post("/simulations", json={"num_drones": num_drones, "bounds_max": list(bounds_max), **kwargs})
    assert resp.status_code == 200
    return resp.json()["simulation_id"]


# ---------------------------------------------------------------------- health
def test_health_always_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["uptime_s"] >= 0


def test_health_does_not_require_any_simulation_to_exist(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# --------------------------------------------------------------------- ready
def test_ready_is_true_once_lifespan_startup_has_run(client):
    """TestClient's `with` block runs the lifespan startup handler before
    yielding -- so by the time a request can be made, readiness must be True."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_returns_503_before_startup_has_run():
    """Direct check of app_ready_state's semantics, independent of TestClient's
    own lifespan timing: an app that has never completed startup must not
    report ready."""
    app_ready_state["ready"] = False
    try:
        app = create_app()
        # Deliberately NOT using `with TestClient(app)` -- skips the lifespan
        # startup handler, so app_ready_state stays exactly as we just set it.
        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"
    finally:
        app_ready_state["ready"] = True  # restore for any other test in this session


# -------------------------------------------------------------------- metrics
def test_metrics_reports_zero_simulations_when_none_exist(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_simulations"] == 0
    assert body["simulations"] == {}
    assert body["streaming"]["total_active_stream_consumers"] == 0
    assert body["streaming"]["queue_depth"] == 0


def test_metrics_reports_per_simulation_fields_after_stepping(client):
    sim_id = create_sim(client, num_drones=30)
    client.post(f"/simulations/{sim_id}/step")
    client.post(f"/simulations/{sim_id}/step")

    resp = client.get("/metrics")
    body = resp.json()
    assert body["total_simulations"] == 1
    sim_metrics = body["simulations"][sim_id]
    assert sim_metrics["tick"] == 2
    assert sim_metrics["status"] == "created"  # step_once() never sets RUNNING
    assert sim_metrics["active_drone_count"] == 30
    assert "mean_tick_ms" in sim_metrics
    assert "ticks_per_second" in sim_metrics
    assert sim_metrics["current_collision_count"] >= 0
    assert sim_metrics["current_near_miss_count"] >= 0
    assert sim_metrics["active_stream_consumers"] == 0


def test_metrics_includes_process_and_api_sections(client):
    client.get("/health")
    client.get("/health")
    resp = client.get("/metrics")
    body = resp.json()
    assert "process" in body and "uptime_s" in body["process"]
    assert body["api"]["request_count"] >= 2  # at least the two /health calls above
    assert body["api"]["mean_request_latency_ms"] >= 0


def test_request_stats_middleware_counts_requests(client):
    before = request_stats["count"]
    client.get("/health")
    client.get("/health")
    client.get("/health")
    assert request_stats["count"] >= before + 3
