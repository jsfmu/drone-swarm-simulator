import time

import pytest
from fastapi.testclient import TestClient

from drone_sim.api.app import create_app
from drone_sim.api.routes import MAX_VISIBLE_DRONES, _runtimes, reset_registry


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_registry()


def wait_until(predicate, timeout=2.0, interval=0.02):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def create_sim(client, num_drones=100, bounds_max=(50.0, 50.0, 50.0), **kwargs):
    resp = client.post("/simulations", json={"num_drones": num_drones, "bounds_max": list(bounds_max), **kwargs})
    assert resp.status_code == 200
    return resp.json()["simulation_id"]


def test_simulation_creation_succeeds(client):
    resp = client.post("/simulations", json={"num_drones": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"
    assert body["tick"] == 0
    assert body["num_drones"] == 50


def test_unknown_simulation_id_returns_404(client):
    assert client.get("/simulations/does-not-exist").status_code == 404
    assert client.get("/simulations/does-not-exist/viewport?x_min=0&x_max=1&y_min=0&y_max=1").status_code == 404


def test_invalid_viewport_parameters_return_400_or_422(client):
    sim_id = create_sim(client)
    # Reversed bounds -> 400 (rejected by ViewportQuery, caught and converted).
    resp = client.get(f"/simulations/{sim_id}/viewport?x_min=10&x_max=0&y_min=0&y_max=10")
    assert resp.status_code == 400
    # Missing required query param -> 422 (FastAPI/Pydantic validation).
    resp = client.get(f"/simulations/{sim_id}/viewport?x_min=0&y_min=0&y_max=10")
    assert resp.status_code == 422


def test_heatmap_response_includes_tick_and_counts(client):
    sim_id = create_sim(client, num_drones=200, bounds_max=(50.0, 50.0, 50.0))
    resp = client.get(f"/simulations/{sim_id}/heatmap?x_min=0&x_max=50&y_min=0&y_max=50&x_bins=5&y_bins=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tick"] == 0
    total = sum(sum(row) for row in body["counts"])
    assert total == body["num_drones_included"] == 200


def test_collision_response_contains_required_marker_fields(client):
    # Small, dense world (many drones, tiny bounds) so a collision is essentially
    # certain to appear within a few ticks -- deterministic given the seed.
    sim_id = create_sim(
        client, num_drones=100, bounds_max=(5.0, 5.0, 5.0),
        seed=1, collision_radius=1.0, near_miss_radius=2.0,
    )
    for _ in range(10):
        resp = client.post(f"/simulations/{sim_id}/step")
        assert resp.status_code == 200

    resp = client.get(f"/simulations/{sim_id}/collisions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tick"] == 10
    if body["markers"]:
        m = body["markers"][0]
        for field in ("drone_a", "drone_b", "tick", "x", "y", "z", "distance", "relative_speed"):
            assert field in m


def test_responses_do_not_leak_numpy_types(client):
    sim_id = create_sim(client, num_drones=50)
    client.post(f"/simulations/{sim_id}/step")
    resp = client.get(f"/simulations/{sim_id}/viewport?x_min=0&x_max=50&y_min=0&y_max=50")
    body = resp.json()  # json.loads succeeding at all rules out numpy scalar leakage
    assert isinstance(body["tick"], int)
    if body["drones"]:
        d = body["drones"][0]
        assert isinstance(d["drone_id"], int)
        assert isinstance(d["x"], float)


def test_raw_drone_response_limit_is_enforced(client):
    sim_id = create_sim(client, num_drones=200, bounds_max=(20.0, 20.0, 20.0))
    resp = client.get(
        f"/simulations/{sim_id}/viewport?x_min=0&x_max=20&y_min=0&y_max=20&limit=10"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["returned"] <= 10
    if body["total_visible"] > 10:
        assert body["truncated"] is True

    # Requesting more than the hard cap is rejected outright (422).
    resp = client.get(
        f"/simulations/{sim_id}/viewport?x_min=0&x_max=20&y_min=0&y_max=20&limit={MAX_VISIBLE_DRONES + 1}"
    )
    assert resp.status_code == 422


def test_api_endpoints_remain_responsive_while_simulation_runs(client):
    sim_id = create_sim(client, num_drones=500, bounds_max=(100.0, 100.0, 100.0))
    resp = client.post(f"/simulations/{sim_id}/start")
    assert resp.status_code == 200
    assert wait_until(
        lambda: client.get(f"/simulations/{sim_id}").json()["tick"] >= 1
    )
    resp = client.get(f"/simulations/{sim_id}/metrics")
    assert resp.status_code == 200
    resp = client.post(f"/simulations/{sim_id}/pause")
    assert resp.status_code == 200


def test_delete_simulation_stops_and_removes_it(client):
    from drone_sim.api.routes import _runtimes

    sim_id = create_sim(client, num_drones=100, bounds_max=(50.0, 50.0, 50.0))
    client.post(f"/simulations/{sim_id}/start")
    assert wait_until(lambda: client.get(f"/simulations/{sim_id}").json()["tick"] >= 1)

    resp = client.delete(f"/simulations/{sim_id}")
    assert resp.status_code == 204
    assert sim_id not in _runtimes
    assert client.get(f"/simulations/{sim_id}").status_code == 404


def test_delete_unknown_simulation_returns_404(client):
    assert client.delete("/simulations/does-not-exist").status_code == 404


def test_delete_stops_the_background_thread(client):
    """DELETE must actually stop the background thread, not just remove the
    registry entry -- otherwise it keeps consuming CPU forever even though
    it's no longer reachable via the API. This is the server-side capability
    that fixes the orphaned-runtime-thread bug: before this endpoint existed,
    nothing (other than the test-only reset_registry(), which wipes every
    simulation) could ever stop a runtime started via POST .../start."""
    from drone_sim.api.routes import _runtimes

    sim_id = create_sim(client, num_drones=100, bounds_max=(50.0, 50.0, 50.0))
    runtime = _runtimes[sim_id]
    client.post(f"/simulations/{sim_id}/start")
    assert wait_until(lambda: runtime.get_snapshot().tick >= 1)

    resp = client.delete(f"/simulations/{sim_id}")
    assert resp.status_code == 204

    tick_after_delete = runtime.get_snapshot().tick
    time.sleep(0.2)
    assert runtime.get_snapshot().tick == tick_after_delete
    assert runtime.get_status().status.value == "stopped"


def test_frame_does_not_serialize_full_payload_twice(client, monkeypatch):
    """Regression guard for the double-json.dumps() bug in get_frame(): the
    full heatmap/markers/metrics payload must be serialized exactly once per
    request. The old implementation dumped it twice -- once into a variable
    used only to measure serialization_ms and then discarded, once more for
    the actual response body."""
    import drone_sim.api.routes as routes_module

    sim_id = create_sim(client, num_drones=50)
    full_payload_dump_count = {"n": 0}
    real_dumps = routes_module.json.dumps

    def counting_dumps(obj, *a, **kw):
        if isinstance(obj, dict) and "heatmap" in obj:
            full_payload_dump_count["n"] += 1
        return real_dumps(obj, *a, **kw)

    routes_module.json.dumps = counting_dumps
    try:
        resp = client.get(f"/simulations/{sim_id}/frame?x_min=0&x_max=50&y_min=0&y_max=50")
    finally:
        routes_module.json.dumps = real_dumps
    assert resp.status_code == 200
    assert full_payload_dump_count["n"] == 1


def test_frame_does_not_call_get_status_separately(client, monkeypatch):
    """Regression guard: get_frame() must read status from the SAME lock
    acquisition that fetches the snapshot (get_snapshot_and_status_with_lock_wait()),
    not a second, separate runtime.get_status() call -- the old implementation's
    second call was a second, entirely unmeasured lock-wait per request."""
    from drone_sim.api.routes import _runtimes

    sim_id = create_sim(client, num_drones=50)
    runtime = _runtimes[sim_id]

    def _boom():
        raise AssertionError("get_frame() must not call runtime.get_status() separately")

    monkeypatch.setattr(runtime, "get_status", _boom)
    resp = client.get(f"/simulations/{sim_id}/frame?x_min=0&x_max=50&y_min=0&y_max=50")
    assert resp.status_code == 200


def test_frame_timings_sum_does_not_exceed_total_request(client):
    """Before the fix, lock_wait_ms + heatmap_ms + collisions_ms +
    serialization_ms could fall far short of total_request_ms -- a second,
    unmeasured lock acquisition (get_status()) and a second, unmeasured full
    JSON serialization pass both happened after total_request_ms was already
    captured. Every one of these stages is now a strict sub-interval of the
    single measured span, so their sum must never exceed the reported total
    (beyond negligible float/bookkeeping slack)."""
    sim_id = create_sim(client, num_drones=300, bounds_max=(60.0, 60.0, 60.0))
    client.post(f"/simulations/{sim_id}/start")
    try:
        resp = client.get(f"/simulations/{sim_id}/frame?x_min=0&x_max=60&y_min=0&y_max=60")
        assert resp.status_code == 200
        t = resp.json()["timings"]
        accounted = t["lock_wait_ms"] + t["heatmap_ms"] + t["collisions_ms"] + t["serialization_ms"]
        assert accounted <= t["total_request_ms"] + 0.5
    finally:
        client.post(f"/simulations/{sim_id}/pause")


def test_static_browser_page_loads_successfully(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Drone Collision Simulator" in resp.text


def test_cors_allows_the_vite_dev_server_origin(client):
    """Regression guard: the Phase 3B React dashboard runs on its own Vite
    dev server (a different origin from this API), so a request carrying an
    Origin header must get back a matching Access-Control-Allow-Origin --
    otherwise the browser blocks it before any handler runs, surfacing in
    the frontend as a generic 'TypeError: Failed to fetch'."""
    origin = "http://localhost:5173"
    resp = client.get("/simulations/does-not-exist", headers={"Origin": origin})
    assert resp.headers.get("access-control-allow-origin") == origin


def test_cors_preflight_succeeds_for_the_vite_dev_server_origin(client):
    origin = "http://localhost:5173"
    resp = client.options(
        "/simulations",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin


def test_cors_rejects_an_unrelated_origin(client):
    resp = client.get("/simulations/does-not-exist", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


# ------------------------------------------------------ Phase 5: distributed mode
def test_create_simulation_distributed_true_returns_distributed_execution_mode(client):
    resp = client.post(
        "/simulations",
        json={"num_drones": 60, "bounds_max": [50.0, 50.0, 50.0], "distributed": True, "num_workers": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["execution_mode"] == "distributed"
    assert body["num_workers"] == 2


def test_create_simulation_default_is_single_process(client):
    """Backward compatibility: distributed=False (the default) must reproduce
    the exact pre-Phase-5 response shape/behavior."""
    resp = client.post("/simulations", json={"num_drones": 60, "bounds_max": [50.0, 50.0, 50.0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["execution_mode"] == "single_process"
    assert body["num_workers"] is None


def test_create_simulation_distributed_with_local_avoidance_returns_400(client):
    before = set(_runtimes)
    resp = client.post(
        "/simulations",
        json={
            "num_drones": 30, "bounds_max": [50.0, 50.0, 50.0],
            "distributed": True, "policy": "local_avoidance",
        },
    )
    assert resp.status_code == 400
    # A rejected creation must never leave a partial/orphaned registry entry.
    assert set(_runtimes) == before


def test_create_simulation_distributed_process_executor_end_to_end(client):
    resp = client.post(
        "/simulations",
        json={
            "num_drones": 40, "bounds_max": [50.0, 50.0, 50.0],
            "distributed": True, "num_workers": 2, "executor": "processes",
        },
    )
    assert resp.status_code == 200
    sim_id = resp.json()["simulation_id"]
    runtime = _runtimes[sim_id]

    resp = client.post(f"/simulations/{sim_id}/step")
    assert resp.status_code == 200

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "distributed" in resp.json()["simulations"][sim_id]

    resp = client.get(f"/simulations/{sim_id}/metrics")
    assert resp.status_code == 200
    assert resp.json()["distributed_metrics"] is not None

    resp = client.delete(f"/simulations/{sim_id}")
    assert resp.status_code == 204
    # DELETE calls runtime.shutdown() -- the process pool must be released,
    # not leaked (mirrors tests/test_worker.py's leak-check style).
    assert runtime._coord.pool._process_executor is None


def test_metrics_endpoint_includes_distributed_key_only_for_distributed_sims(client):
    dist_resp = client.post(
        "/simulations",
        json={"num_drones": 20, "bounds_max": [50.0, 50.0, 50.0], "distributed": True, "num_workers": 1},
    )
    plain_resp = client.post("/simulations", json={"num_drones": 20, "bounds_max": [50.0, 50.0, 50.0]})
    dist_id = dist_resp.json()["simulation_id"]
    plain_id = plain_resp.json()["simulation_id"]

    body = client.get("/metrics").json()
    assert "distributed" in body["simulations"][dist_id]
    assert "distributed" not in body["simulations"][plain_id]
