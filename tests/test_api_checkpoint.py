"""Tests for the Phase 5 checkpoint HTTP endpoints (routes.py: POST
.../checkpoint, POST .../checkpoint/load, GET /checkpoints).

These previously existed only as plain Python functions in checkpoint.py
(see tests/test_checkpoint.py for the deep round-trip/determinism coverage of
that layer) -- this file only exercises the new HTTP contract wrapping them:
status codes, response shapes, the single-process-only rejection, and the
filesystem-backed listing. It deliberately does not re-verify checkpoint.py's
own array/RNG-state correctness.
"""

import pytest
from fastapi.testclient import TestClient

from drone_sim.api import routes as routes_module
from drone_sim.api.app import create_app
from drone_sim.api.routes import reset_registry


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_module, "CHECKPOINT_DIR", tmp_path)
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_registry()


def create_sim(client, num_drones=50, bounds_max=(50.0, 50.0, 50.0), **kwargs):
    resp = client.post("/simulations", json={"num_drones": num_drones, "bounds_max": list(bounds_max), **kwargs})
    assert resp.status_code == 200
    return resp.json()["simulation_id"]


def test_save_checkpoint_succeeds(client):
    sim_id = create_sim(client)
    for _ in range(3):
        assert client.post(f"/simulations/{sim_id}/step").status_code == 200

    resp = client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["simulation_id"] == sim_id
    assert body["name"] == "demo"
    assert body["tick"] == 3
    assert body["num_drones"] == 50
    assert body["size_bytes"] > 0
    assert body["saved_at"]


def test_save_then_list_shows_metadata(client):
    sim_id = create_sim(client, num_drones=30)
    client.post(f"/simulations/{sim_id}/step")

    assert client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "alpha"}).status_code == 200

    resp = client.get("/checkpoints")
    assert resp.status_code == 200
    checkpoints = resp.json()["checkpoints"]
    assert len(checkpoints) == 1
    assert checkpoints[0]["name"] == "alpha"
    assert checkpoints[0]["tick"] == 1
    assert checkpoints[0]["num_drones"] == 30
    assert checkpoints[0]["size_bytes"] > 0
    assert checkpoints[0]["modified_at"]


def test_list_checkpoints_empty_when_none_saved(client):
    resp = client.get("/checkpoints")
    assert resp.status_code == 200
    assert resp.json() == {"checkpoints": []}


def test_load_checkpoint_restores_tick_and_pauses(client):
    sim_id = create_sim(client)
    for _ in range(5):
        client.post(f"/simulations/{sim_id}/step")
    assert client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "snap"}).status_code == 200

    for _ in range(4):
        client.post(f"/simulations/{sim_id}/step")
    status = client.get(f"/simulations/{sim_id}").json()
    assert status["tick"] == 9

    resp = client.post(f"/simulations/{sim_id}/checkpoint/load", json={"name": "snap"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tick"] == 5
    assert body["status"] == "paused"

    status = client.get(f"/simulations/{sim_id}").json()
    assert status["tick"] == 5


def test_load_unknown_checkpoint_returns_404(client):
    sim_id = create_sim(client)
    resp = client.post(f"/simulations/{sim_id}/checkpoint/load", json={"name": "does-not-exist"})
    assert resp.status_code == 404


def test_checkpoint_actions_on_unknown_simulation_return_404(client):
    resp = client.post("/simulations/does-not-exist/checkpoint", json={"name": "x"})
    assert resp.status_code == 404
    resp = client.post("/simulations/does-not-exist/checkpoint/load", json={"name": "x"})
    assert resp.status_code == 404


def test_invalid_checkpoint_name_rejected(client):
    sim_id = create_sim(client)
    resp = client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "../../etc/passwd"})
    assert resp.status_code == 422  # Pydantic Field(pattern=...) rejects before the handler runs
    resp = client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "has spaces"})
    assert resp.status_code == 422


def test_save_and_load_rejected_for_distributed_simulation(client):
    sim_id = create_sim(client, num_drones=20, distributed=True, num_workers=1)
    resp = client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "x"})
    assert resp.status_code == 400
    assert "single-process" in resp.json()["detail"]

    resp = client.post(f"/simulations/{sim_id}/checkpoint/load", json={"name": "x"})
    assert resp.status_code == 400
    assert "single-process" in resp.json()["detail"]


def test_load_checkpoint_while_running_returns_409(client):
    sim_id = create_sim(client)
    assert client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "snap"}).status_code == 200
    assert client.post(f"/simulations/{sim_id}/start").status_code == 200
    try:
        resp = client.post(f"/simulations/{sim_id}/checkpoint/load", json={"name": "snap"})
        assert resp.status_code == 409
    finally:
        client.post(f"/simulations/{sim_id}/pause")


def test_save_checkpoint_while_running_is_allowed(client):
    """Unlike load, save never mutates the runtime -- it only needs the same
    lock every other read acquires, so it must be safe to call at any time,
    including while the background loop is actively running."""
    sim_id = create_sim(client)
    assert client.post(f"/simulations/{sim_id}/start").status_code == 200
    try:
        resp = client.post(f"/simulations/{sim_id}/checkpoint", json={"name": "while-running"})
        assert resp.status_code == 200
    finally:
        client.post(f"/simulations/{sim_id}/pause")
