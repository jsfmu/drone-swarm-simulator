"""Minimal stdlib-only HTTP client for the Phase 3A drone_sim API.

Lets a non-FastAPI process (the Matplotlib debug viewer, in particular) act
as a second poller of a running ``drone_sim.api`` server -- the same role
``static/index.html`` already plays in the browser -- so a CLI viewer and a
browser tab can display the exact same live ``simulation_id`` at once instead
of each owning an independent ``Simulation``.

Uses ``urllib.request`` (stdlib) rather than ``httpx``/``requests`` so the
kernel/viz side of the package never gains a hard dependency on whatever HTTP
library the API side happens to use -- this module talks HTTP as a plain
client, it does not import anything from ``drone_sim.api``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

DEFAULT_TIMEOUT_S = 5.0


class RemoteSimulationError(RuntimeError):
    """A request to a ``drone_sim`` API server failed."""


def _request(method: str, url: str, body: Mapping[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RemoteSimulationError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RemoteSimulationError(f"{method} {url} -> unreachable: {exc.reason}") from exc


def create_simulation(
    base_url: str,
    *,
    num_drones: int,
    bounds_max: Sequence[float] = (1000.0, 1000.0, 1000.0),
    seed: int = 0,
    dt: float = 1.0,
    max_speed: float = 5.0,
    collision_radius: float = 1.0,
    near_miss_radius: float = 2.0,
) -> str:
    """Create a simulation on the server and return its ``simulation_id``."""
    body = {
        "num_drones": num_drones,
        "bounds_max": list(bounds_max),
        "seed": seed,
        "dt": dt,
        "max_speed": max_speed,
        "collision_radius": collision_radius,
        "near_miss_radius": near_miss_radius,
    }
    resp = _request("POST", f"{base_url}/simulations", body)
    return resp["simulation_id"]


def start_simulation(base_url: str, simulation_id: str) -> None:
    _request("POST", f"{base_url}/simulations/{simulation_id}/start")


def pause_simulation(base_url: str, simulation_id: str) -> None:
    _request("POST", f"{base_url}/simulations/{simulation_id}/pause")


def resume_simulation(base_url: str, simulation_id: str) -> None:
    _request("POST", f"{base_url}/simulations/{simulation_id}/resume")


def reset_simulation(base_url: str, simulation_id: str) -> None:
    _request("POST", f"{base_url}/simulations/{simulation_id}/reset")


def delete_simulation(base_url: str, simulation_id: str) -> None:
    """Stop and remove ``simulation_id``. Best-effort: never raises.

    Mirrors ``static/index.html``'s ``stopSimulationIfAny`` -- a viewer that
    created its own simulation should not leave its background thread running
    forever after the window closes.
    """
    try:
        _request("DELETE", f"{base_url}/simulations/{simulation_id}")
    except RemoteSimulationError:
        pass


def get_status(base_url: str, simulation_id: str) -> dict:
    return _request("GET", f"{base_url}/simulations/{simulation_id}")


def get_frame(
    base_url: str,
    simulation_id: str,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_bins: int = 60,
    y_bins: int = 60,
) -> dict:
    """Fetch one combined heatmap + collision-marker + metrics frame.

    Same endpoint ``static/index.html`` polls, so a CLI viewer and a browser
    tab pointed at the same ``simulation_id`` are guaranteed to describe the
    same tick on any given call (see ``routes.py``'s ``/frame`` docstring).
    """
    params = {
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "x_bins": x_bins, "y_bins": y_bins,
    }
    qs = urllib.parse.urlencode(params)
    return _request("GET", f"{base_url}/simulations/{simulation_id}/frame?{qs}")
