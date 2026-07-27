"""Phase 3A FastAPI endpoints.

Every handler is a thin adapter: it resolves a :class:`SimulationRuntime`
from the in-process registry, reads a single immutable
:class:`~drone_sim.snapshot.SimulationSnapshot` via ``get_snapshot()``, and
runs the vectorized viewport/heatmap/collision queries against that snapshot
-- never against the live, possibly-mutating simulation arrays.

A single local process registry (``_runtimes``) is intentional for Phase 3A;
see CLAUDE.md's scope restrictions (no distributed workers, no multi-tenant
orchestration).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..collision_queries import query_collision_markers
from ..config import SimulationConfig
from ..heatmap import HeatmapQuery, compute_heatmap
from ..runtime import SimulationRuntime
from ..viewport import ViewportQuery, find_visible_drones
from .models import (
    CollisionMarkerResponse,
    CollisionsResponse,
    CreateSimulationRequest,
    DronePosition,
    HeatmapResponse,
    MetricsResponse,
    SimulationStatusResponse,
    ViewportResponse,
)

router = APIRouter()

_runtimes: dict[str, SimulationRuntime] = {}

#: Hard cap on raw drone positions returned by /viewport in one response.
#: A viewport with more visible drones than this still gets full heatmap
#: data; raw positions are truncated to a deterministic prefix and
#: ``truncated=True`` is reported (see ViewportResponse).
MAX_VISIBLE_DRONES = 5_000


def reset_registry() -> None:
    """Test-only: clear all runtimes (and stop their threads) between tests."""
    for runtime in _runtimes.values():
        runtime.shutdown()
    _runtimes.clear()


def _get_runtime(simulation_id: str) -> SimulationRuntime:
    runtime = _runtimes.get(simulation_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"unknown simulation_id {simulation_id!r}")
    return runtime


def _status_response(runtime: SimulationRuntime) -> SimulationStatusResponse:
    state = runtime.get_status()
    return SimulationStatusResponse(
        simulation_id=state.simulation_id,
        status=state.status.value,
        tick=state.tick,
        num_drones=state.num_drones,
    )


def _build_viewport(
    x_min: float, x_max: float, y_min: float, y_max: float,
    z_min: Optional[float], z_max: Optional[float],
) -> ViewportQuery:
    try:
        return ViewportQuery(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/simulations", response_model=SimulationStatusResponse)
def create_simulation(req: CreateSimulationRequest) -> SimulationStatusResponse:
    simulation_id = uuid.uuid4().hex[:12]
    try:
        config = SimulationConfig(
            num_drones=req.num_drones,
            bounds_max=req.bounds_max,
            seed=req.seed,
            dt=req.dt,
            max_speed=req.max_speed,
            collision_radius=req.collision_radius,
            near_miss_radius=req.near_miss_radius,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _runtimes[simulation_id] = SimulationRuntime(simulation_id, config)
    return _status_response(_runtimes[simulation_id])


@router.get("/simulations/{simulation_id}", response_model=SimulationStatusResponse)
def get_simulation(simulation_id: str) -> SimulationStatusResponse:
    return _status_response(_get_runtime(simulation_id))


@router.delete("/simulations/{simulation_id}", status_code=204)
def delete_simulation(simulation_id: str) -> Response:
    """Stop ``simulation_id``'s background thread and remove it from the registry.

    Until this endpoint existed, there was no way to stop a simulation short
    of the test-only ``reset_registry()`` (which wipes every simulation at
    once) -- every ``POST /simulations`` call left the previous one's
    background thread running forever, since ``SimulationRuntime.start()``'s
    loop only exits on ``shutdown()``. The static browser page now calls this
    before creating a replacement simulation (see ``static/index.html``'s
    ``createSimulation()``), which is what actually stops the leak; this
    endpoint is the server-side capability that made that possible at all.
    """
    runtime = _get_runtime(simulation_id)
    runtime.shutdown()
    del _runtimes[simulation_id]
    return Response(status_code=204)


@router.post("/simulations/{simulation_id}/start", response_model=SimulationStatusResponse)
def start_simulation(simulation_id: str) -> SimulationStatusResponse:
    runtime = _get_runtime(simulation_id)
    try:
        runtime.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_response(runtime)


@router.post("/simulations/{simulation_id}/pause", response_model=SimulationStatusResponse)
def pause_simulation(simulation_id: str) -> SimulationStatusResponse:
    runtime = _get_runtime(simulation_id)
    try:
        runtime.pause()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_response(runtime)


@router.post("/simulations/{simulation_id}/resume", response_model=SimulationStatusResponse)
def resume_simulation(simulation_id: str) -> SimulationStatusResponse:
    runtime = _get_runtime(simulation_id)
    try:
        runtime.resume()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_response(runtime)


@router.post("/simulations/{simulation_id}/step", response_model=SimulationStatusResponse)
def step_simulation(simulation_id: str) -> SimulationStatusResponse:
    runtime = _get_runtime(simulation_id)
    try:
        runtime.step_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_response(runtime)


@router.post("/simulations/{simulation_id}/reset", response_model=SimulationStatusResponse)
def reset_simulation(simulation_id: str) -> SimulationStatusResponse:
    runtime = _get_runtime(simulation_id)
    try:
        runtime.reset()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_response(runtime)


@router.get("/simulations/{simulation_id}/viewport", response_model=ViewportResponse)
def get_viewport(
    simulation_id: str,
    x_min: float = Query(...),
    x_max: float = Query(...),
    y_min: float = Query(...),
    y_max: float = Query(...),
    z_min: Optional[float] = Query(None),
    z_max: Optional[float] = Query(None),
    limit: int = Query(MAX_VISIBLE_DRONES, gt=0, le=MAX_VISIBLE_DRONES),
) -> ViewportResponse:
    runtime = _get_runtime(simulation_id)
    viewport = _build_viewport(x_min, x_max, y_min, y_max, z_min, z_max)
    snapshot = runtime.get_snapshot()
    visible = find_visible_drones(snapshot, viewport, limit=limit)

    drones = [
        DronePosition(
            drone_id=int(did),
            x=float(p[0]), y=float(p[1]), z=float(p[2]),
            vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
        )
        for did, p, v in zip(visible.drone_ids, visible.positions, visible.velocities)
    ]
    return ViewportResponse(
        simulation_id=simulation_id,
        tick=visible.tick,
        total_visible=visible.total_visible,
        returned=len(drones),
        truncated=visible.truncated,
        drones=drones,
    )


@router.get("/simulations/{simulation_id}/heatmap", response_model=HeatmapResponse)
def get_heatmap(
    simulation_id: str,
    x_min: float = Query(...),
    x_max: float = Query(...),
    y_min: float = Query(...),
    y_max: float = Query(...),
    z_min: Optional[float] = Query(None),
    z_max: Optional[float] = Query(None),
    x_bins: int = Query(50, gt=0),
    y_bins: int = Query(50, gt=0),
) -> HeatmapResponse:
    runtime = _get_runtime(simulation_id)
    viewport = _build_viewport(x_min, x_max, y_min, y_max, z_min, z_max)
    try:
        query = HeatmapQuery(viewport=viewport, x_bins=x_bins, y_bins=y_bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot = runtime.get_snapshot()
    result = compute_heatmap(snapshot, query)
    return HeatmapResponse(
        simulation_id=simulation_id,
        tick=result.tick,
        x_bins=result.x_bins,
        y_bins=result.y_bins,
        x_edges=result.x_edges.tolist(),
        y_edges=result.y_edges.tolist(),
        counts=result.counts.tolist(),
        max_density=result.max_density,
        num_drones_included=result.num_drones_included,
    )


@router.get("/simulations/{simulation_id}/collisions", response_model=CollisionsResponse)
def get_collisions(
    simulation_id: str,
    x_min: Optional[float] = Query(None),
    x_max: Optional[float] = Query(None),
    y_min: Optional[float] = Query(None),
    y_max: Optional[float] = Query(None),
    z_min: Optional[float] = Query(None),
    z_max: Optional[float] = Query(None),
) -> CollisionsResponse:
    runtime = _get_runtime(simulation_id)

    viewport = None
    provided = (x_min, x_max, y_min, y_max)
    if any(v is not None for v in provided):
        if any(v is None for v in provided):
            raise HTTPException(
                status_code=400, detail="x_min/x_max/y_min/y_max must all be provided together"
            )
        viewport = _build_viewport(x_min, x_max, y_min, y_max, z_min, z_max)

    snapshot = runtime.get_snapshot()
    markers = query_collision_markers(snapshot, viewport)
    return CollisionsResponse(
        simulation_id=simulation_id,
        tick=snapshot.tick,
        markers=[CollisionMarkerResponse(**vars(m)) for m in markers],
    )


@router.get("/simulations/{simulation_id}/metrics", response_model=MetricsResponse)
def get_metrics(simulation_id: str) -> MetricsResponse:
    runtime = _get_runtime(simulation_id)
    snapshot = runtime.get_snapshot()
    return MetricsResponse(simulation_id=simulation_id, tick=snapshot.tick, metrics=snapshot.metrics)


@router.get("/simulations/{simulation_id}/frame")
def get_frame(
    simulation_id: str,
    x_min: float = Query(...),
    x_max: float = Query(...),
    y_min: float = Query(...),
    y_max: float = Query(...),
    z_min: Optional[float] = Query(None),
    z_max: Optional[float] = Query(None),
    x_bins: int = Query(60, gt=0),
    y_bins: int = Query(60, gt=0),
) -> Response:
    """Combined viewport-driven frame: heatmap + collision markers + metrics
    + timing measurements, computed from ONE ``get_snapshot()`` call.

    This is what the browser page polls -- reusing one snapshot across all
    four kinds of query (instead of the four separate round trips of
    ``/viewport`` + ``/heatmap`` + ``/collisions`` + ``/metrics``) means every
    field in the response is guaranteed to describe the same tick, and avoids
    reacquiring the runtime lock four times per refresh. Never returns raw
    drone positions (use ``/viewport`` explicitly for those).
    """
    t_start = time.perf_counter()
    runtime = _get_runtime(simulation_id)
    viewport = _build_viewport(x_min, x_max, y_min, y_max, z_min, z_max)
    try:
        heatmap_query = HeatmapQuery(viewport=viewport, x_bins=x_bins, y_bins=y_bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Single lock acquisition for BOTH snapshot and status -- see
    # get_snapshot_and_status_with_lock_wait()'s docstring for why calling
    # get_snapshot_with_lock_wait() + get_status() separately (the previous
    # implementation) was a real, unmeasured second lock-wait, not just a
    # style choice.
    snapshot, status, lock_wait_ms = runtime.get_snapshot_and_status_with_lock_wait()

    t0 = time.perf_counter()
    heatmap = compute_heatmap(snapshot, heatmap_query)
    t1 = time.perf_counter()
    markers = query_collision_markers(snapshot, viewport)
    t2 = time.perf_counter()

    tick_timings = runtime.get_last_timings()
    payload = {
        "simulation_id": simulation_id,
        "status": status.status.value,
        "tick": snapshot.tick,
        "num_visible_drones": heatmap.num_drones_included,
        "heatmap": {
            "x_bins": heatmap.x_bins,
            "y_bins": heatmap.y_bins,
            "x_edges": heatmap.x_edges.tolist(),
            "y_edges": heatmap.y_edges.tolist(),
            "counts": heatmap.counts.tolist(),
            "max_density": heatmap.max_density,
        },
        "markers": [
            {
                "drone_a": m.drone_a, "drone_b": m.drone_b, "tick": m.tick,
                "x": m.x, "y": m.y, "z": m.z,
                "distance": m.distance, "relative_speed": m.relative_speed,
            }
            for m in markers
        ],
        "metrics": snapshot.metrics,
        # "timings" deliberately omitted here: serialization_ms/total_request_ms
        # describe the cost of producing this very payload, so they can't be
        # computed until AFTER it's serialized (see below). Adding the whole
        # "timings" block to the response is a second, separate, tiny dump +
        # string concat rather than re-dumping this (potentially large)
        # heatmap/markers payload a second time just to add 7 floats.
    }

    # The previous implementation dumped the equivalent of this same payload
    # TWICE per request: once into a variable used only to measure
    # serialization_ms and then discarded, once more (with a "timings" key
    # added) for the actual response body. That wasted a full serialization
    # pass on every request AND meant total_request_ms was captured before the
    # real (second) dumps() call ever ran, so it never accounted for it --
    # exactly the gap between the reported stage timings and total_request_ms.
    # This is now the only json.dumps() of the heatmap/markers/metrics payload.
    t3 = time.perf_counter()
    body_without_timings = json.dumps(payload)
    t4 = time.perf_counter()
    timings = {
        "sim_step_ms": tick_timings.sim_step_ms,
        "snapshot_build_ms": tick_timings.snapshot_build_ms,
        "lock_wait_ms": lock_wait_ms,
        "heatmap_ms": (t1 - t0) * 1e3,
        "collisions_ms": (t2 - t1) * 1e3,
        "serialization_ms": (t4 - t3) * 1e3,
        "total_request_ms": (t4 - t_start) * 1e3,
    }
    # payload is a non-empty dict, so json.dumps always renders it as
    # '{...}' with no trailing whitespace -- stripping the final '}' and
    # appending the "timings" field (a second, ~7-float dump, negligible next
    # to the payload above) is a safe, simple way to add one more top-level
    # key without re-serializing everything already in body_without_timings.
    body = body_without_timings[:-1] + ',"timings":' + json.dumps(timings) + "}"
    return Response(content=body, media_type="application/json")
