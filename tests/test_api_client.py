"""Integration tests for api_client.py against a real, live uvicorn server.

These exercise actual HTTP (urllib) over a real socket -- not FastAPI's
TestClient, which never opens a port -- because api_client is specifically
the thing that talks real HTTP to a running `uvicorn drone_sim.api.app:app`
process (see scripts/run_visualizer.py --remote).
"""

import socket
import threading
import time

import pytest
import uvicorn

from drone_sim import api_client
from drone_sim.api.app import create_app
from drone_sim.api.routes import reset_registry


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server():
    port = _free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    assert server.started, "uvicorn server did not start in time"

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5.0)
    reset_registry()


def wait_until(predicate, timeout=2.0, interval=0.02):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_create_start_and_get_frame(live_server):
    sim_id = api_client.create_simulation(
        live_server, num_drones=50, bounds_max=(50.0, 50.0, 50.0), seed=1,
    )
    assert sim_id

    status = api_client.get_status(live_server, sim_id)
    assert status["status"] == "created"
    assert status["num_drones"] == 50

    api_client.start_simulation(live_server, sim_id)

    frame = api_client.get_frame(
        live_server, sim_id, x_min=0, x_max=50, y_min=0, y_max=50, x_bins=5, y_bins=5,
    )
    assert frame["simulation_id"] == sim_id
    assert frame["status"] in ("running", "paused")
    assert len(frame["heatmap"]["counts"]) == 5
    assert len(frame["heatmap"]["counts"][0]) == 5
    assert "markers" in frame
    assert "metrics" in frame

    api_client.delete_simulation(live_server, sim_id)
    with pytest.raises(api_client.RemoteSimulationError):
        api_client.get_status(live_server, sim_id)


def test_pause_resume_reset(live_server):
    sim_id = api_client.create_simulation(live_server, num_drones=20, bounds_max=(20.0, 20.0, 20.0))
    api_client.start_simulation(live_server, sim_id)

    assert wait_until(lambda: api_client.get_status(live_server, sim_id)["tick"] > 0)

    api_client.pause_simulation(live_server, sim_id)
    assert api_client.get_status(live_server, sim_id)["status"] == "paused"

    api_client.resume_simulation(live_server, sim_id)
    assert api_client.get_status(live_server, sim_id)["status"] == "running"

    api_client.pause_simulation(live_server, sim_id)  # reset requires not-running
    api_client.reset_simulation(live_server, sim_id)
    assert api_client.get_status(live_server, sim_id)["tick"] == 0

    api_client.delete_simulation(live_server, sim_id)


def test_unknown_simulation_raises_remote_error(live_server):
    with pytest.raises(api_client.RemoteSimulationError):
        api_client.get_status(live_server, "does-not-exist")


def test_delete_simulation_is_best_effort_on_unknown_id(live_server):
    # Must not raise even though the id doesn't exist (404) -- mirrors
    # static/index.html's stopSimulationIfAny "best effort" contract.
    api_client.delete_simulation(live_server, "does-not-exist")
