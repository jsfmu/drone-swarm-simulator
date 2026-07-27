"""Tests for GET /simulations/{id}/stream (Phase 3B SSE dashboard streaming).

These run against a REAL live uvicorn server (see ``live_server``, copied from
``test_api_client.py``'s established pattern) rather than FastAPI's
``TestClient``. This is not a style choice: httpx's in-memory ``ASGITransport``
(what ``TestClient`` uses) buffers an entire ASGI response before returning
anything to the caller, and its mock ``receive()`` channel only ever reports
``http.disconnect`` *after* the response is already complete -- there is no
way for a deliberately-until-disconnected generator to ever finish under that
transport, so any attempt to read a stream through ``TestClient`` deadlocks.
A real socket has real TCP-level disconnect semantics, which is what
``Request.is_disconnected()`` is actually designed to observe.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from drone_sim.api.app import create_app
from drone_sim.api.routes import _runtimes, _stream_connection_counts, reset_registry
from drone_sim.movement import GoalDirectedMovementAlgorithm, LocalAvoidanceMovementAlgorithm


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


def wait_until(predicate, timeout=3.0, interval=0.02):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _post(base_url, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else b""
    req = urllib.request.Request(f"{base_url}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read())


def _create_sim(base_url, num_drones=50, bounds_max=(50.0, 50.0, 50.0), **kwargs):
    body = {"num_drones": num_drones, "bounds_max": list(bounds_max), **kwargs}
    return _post(base_url, "/simulations", body)["simulation_id"]


def _stream_url(base_url, sim_id, hz=20, bounds=(0, 50, 0, 50)):
    x_min, x_max, y_min, y_max = bounds
    return (
        f"{base_url}/simulations/{sim_id}/stream"
        f"?x_min={x_min}&x_max={x_max}&y_min={y_min}&y_max={y_max}&hz={hz}"
    )


class _SseReader:
    """Minimal incremental SSE parser over a real, open urllib response."""

    def __init__(self, resp):
        self._resp = resp

    def read_events(self, max_events: int, timeout_s: float = 10.0):
        """Read up to ``max_events`` (event_type, data_dict) pairs."""
        events = []
        event_type = "message"
        data_lines: list[str] = []
        deadline = time.time() + timeout_s
        while len(events) < max_events and time.time() < deadline:
            raw = self._resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\n").rstrip("\r")
            if line == "":
                if data_lines:
                    events.append((event_type, json.loads("\n".join(data_lines))))
                event_type = "message"
                data_lines = []
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            # "id:" lines are ignored here; seq is also inside the JSON payload.
        return events

    def close(self):
        self._resp.close()


def _open_stream(url, timeout_s=10.0) -> _SseReader:
    resp = urllib.request.urlopen(url, timeout=timeout_s)
    return _SseReader(resp)


# --------------------------------------------------------------------- tests
def test_stream_returns_initial_valid_frame(live_server):
    sim_id = _create_sim(live_server)
    _post(live_server, f"/simulations/{sim_id}/step")

    reader = _open_stream(_stream_url(live_server, sim_id))
    try:
        events = reader.read_events(1)
    finally:
        reader.close()

    assert len(events) == 1
    event_type, frame = events[0]
    assert event_type == "message"
    assert frame["tick"] == 1
    assert frame["simulation_id"] == sim_id
    for key in ("status", "num_visible_drones", "heatmap", "markers", "metrics", "seq", "server_time", "timings"):
        assert key in frame
    assert frame["heatmap"]["counts"]
    assert frame["seq"] == 1


def test_stream_advances_tick_while_running(live_server):
    sim_id = _create_sim(live_server, num_drones=200)
    _post(live_server, f"/simulations/{sim_id}/start")

    reader = _open_stream(_stream_url(live_server, sim_id, hz=15))
    try:
        events = reader.read_events(5)
    finally:
        reader.close()

    ticks = [frame["tick"] for _, frame in events]
    seqs = [frame["seq"] for _, frame in events]
    assert seqs == sorted(seqs)
    assert ticks == sorted(ticks)
    assert ticks[-1] > ticks[0], "tick must advance across frames while the simulation runs"


def test_stream_frame_fields_describe_one_snapshot(live_server):
    sim_id = _create_sim(live_server, num_drones=50)
    for _ in range(3):
        _post(live_server, f"/simulations/{sim_id}/step")

    reader = _open_stream(_stream_url(live_server, sim_id))
    try:
        _, frame = reader.read_events(1)[0]
    finally:
        reader.close()

    assert frame["tick"] == 3
    total = sum(sum(row) for row in frame["heatmap"]["counts"])
    assert total == frame["num_visible_drones"]


def test_stream_behavior_while_paused(live_server):
    sim_id = _create_sim(live_server, num_drones=50)
    _post(live_server, f"/simulations/{sim_id}/step")
    # Freshly created + step_once()'d simulation is not running -- stream must
    # still deliver frames describing the frozen tick without erroring.

    reader = _open_stream(_stream_url(live_server, sim_id))
    try:
        events = reader.read_events(3)
    finally:
        reader.close()

    assert len(events) == 3
    assert all(frame["tick"] == 1 for _, frame in events)
    assert all(frame["status"] == "created" for _, frame in events)


def test_stream_client_disconnect_cleanup(live_server):
    sim_id = _create_sim(live_server, num_drones=50)
    _post(live_server, f"/simulations/{sim_id}/start")

    reader = _open_stream(_stream_url(live_server, sim_id, hz=20))
    reader.read_events(1)
    assert _stream_connection_counts.get(sim_id, 0) >= 1
    reader.close()

    assert wait_until(lambda: _stream_connection_counts.get(sim_id, 0) == 0)
    _post(live_server, f"/simulations/{sim_id}/pause")


def test_stream_invalid_simulation_id_returns_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_stream_url(live_server, "does-not-exist"), timeout=5.0)
    assert excinfo.value.code == 404


def test_stream_deleted_simulation_closes(live_server):
    sim_id = _create_sim(live_server, num_drones=50)
    _post(live_server, f"/simulations/{sim_id}/start")

    reader = _open_stream(_stream_url(live_server, sim_id, hz=20))
    try:
        reader.read_events(1)  # at least one normal frame first

        req = urllib.request.Request(f"{live_server}/simulations/{sim_id}", method="DELETE")
        urllib.request.urlopen(req, timeout=5.0)

        events = reader.read_events(1, timeout_s=5.0)
    finally:
        reader.close()

    assert len(events) == 1
    event_type, data = events[0]
    assert event_type == "closed"
    assert data["reason"] == "simulation_deleted"


def test_stream_bounded_publication_rate(live_server):
    # A tiny, fast-ticking simulation: many more ticks/sec than the stream's
    # publication rate, so this proves the stream is NOT publishing every tick.
    sim_id = _create_sim(live_server, num_drones=10, bounds_max=(1000.0, 1000.0, 1000.0))
    _post(live_server, f"/simulations/{sim_id}/start")
    time.sleep(0.5)  # let many ticks accumulate

    hz = 4
    duration_s = 1.0
    reader = _open_stream(_stream_url(live_server, sim_id, hz=hz, bounds=(0, 1000, 0, 1000)))
    try:
        t0 = time.time()
        events = []
        while time.time() - t0 < duration_s:
            got = reader.read_events(1, timeout_s=1.0)
            if not got:
                break
            events.append(got[0])
    finally:
        reader.close()

    # Bounded near hz*duration, not anywhere near the (much higher) tick count.
    assert 0 < len(events) <= hz * duration_s + 3


def test_stream_latest_frame_skips_intermediate_ticks(live_server):
    sim_id = _create_sim(live_server, num_drones=10, bounds_max=(1000.0, 1000.0, 1000.0))
    _post(live_server, f"/simulations/{sim_id}/start")
    time.sleep(0.5)

    reader = _open_stream(_stream_url(live_server, sim_id, hz=3, bounds=(0, 1000, 0, 1000)))
    try:
        events = reader.read_events(3, timeout_s=5.0)
    finally:
        reader.close()

    ticks = [frame["tick"] for _, frame in events]
    jumps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert any(j > 1 for j in jumps), (
        f"expected at least one skipped-tick jump (no per-frame queue) between polls, got ticks={ticks}"
    )


def test_stream_policy_selection_selects_intended_policy(live_server):
    sim_id = _create_sim(live_server, num_drones=30, policy="goal_directed")
    runtime = _runtimes[sim_id]
    ids = set(runtime._sim.world.state.movement_policy_ids.tolist())
    assert ids == {GoalDirectedMovementAlgorithm.policy_id}
    assert GoalDirectedMovementAlgorithm.policy_id in runtime._sim.engine.movement.policies
    assert runtime._sim.world.state.goal_positions is not None


def test_stream_local_avoidance_policy_selection(live_server):
    sim_id = _create_sim(live_server, num_drones=30, policy="local_avoidance")
    runtime = _runtimes[sim_id]
    ids = set(runtime._sim.world.state.movement_policy_ids.tolist())
    assert ids == {LocalAvoidanceMovementAlgorithm.policy_id}
    assert runtime._sim.engine.needs_context is True


def test_stream_same_seed_scenario_policy_reproducible(live_server):
    sim_a = _create_sim(live_server, num_drones=40, seed=7, policy="goal_directed")
    sim_b = _create_sim(live_server, num_drones=40, seed=7, policy="goal_directed")
    for _ in range(5):
        _post(live_server, f"/simulations/{sim_a}/step")
        _post(live_server, f"/simulations/{sim_b}/step")

    pos_a = _runtimes[sim_a]._sim.world.state.positions
    pos_b = _runtimes[sim_b]._sim.world.state.positions
    assert (pos_a == pos_b).all()


def test_stream_scenario_selection_reproducible(live_server):
    sim_a = _create_sim(live_server, num_drones=2, scenario="head_on_collision")
    sim_b = _create_sim(live_server, num_drones=2, scenario="head_on_collision")
    for _ in range(6):
        _post(live_server, f"/simulations/{sim_a}/step")
        _post(live_server, f"/simulations/{sim_b}/step")

    snap_a = _runtimes[sim_a].get_snapshot()
    snap_b = _runtimes[sim_b].get_snapshot()
    assert (snap_a.positions == snap_b.positions).all()
    assert snap_a.collision_pairs.shape == snap_b.collision_pairs.shape


def test_frame_endpoint_and_rest_controls_still_work(live_server):
    """Regression guard: Phase 3B's routes.py refactor must not change
    /frame's or the REST controls' observable behavior."""
    sim_id = _create_sim(live_server, num_drones=80)
    _post(live_server, f"/simulations/{sim_id}/start")
    assert wait_until(lambda: _post(live_server, f"/simulations/{sim_id}/pause") or True)

    req = urllib.request.Request(
        f"{live_server}/simulations/{sim_id}/frame?x_min=0&x_max=50&y_min=0&y_max=50", method="GET"
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        frame = json.loads(resp.read())
    assert frame["simulation_id"] == sim_id
    assert "timings" in frame and "total_request_ms" in frame["timings"]
