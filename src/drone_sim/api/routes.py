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

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from .. import checkpoint as _checkpoint
from ..checkpoint import CheckpointError
from ..collision_queries import query_collision_markers
from ..config import SimulationConfig
from ..coordinator import DistributedConfig
from ..distributed_runtime import DistributedSimulationRuntime
from ..heatmap import HeatmapQuery, compute_heatmap
from ..movement import (
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementSystem,
)
from ..runtime import SimulationRuntime
from ..scenarios import SCENARIOS
from ..state import World
from ..viewport import ViewportQuery, find_visible_drones
from .models import (
    CheckpointInfo,
    CheckpointListResponse,
    CheckpointLoadRequest,
    CheckpointLoadResponse,
    CheckpointSaveRequest,
    CheckpointSaveResponse,
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

#: Phase 5 checkpoint save/load: on-disk directory checkpoints are written to
#: and listed from. Overridable via env var so the Docker image (WORKDIR
#: /app, non-root appuser -- see Dockerfile) can point this at a directory it
#: actually owns; the bare "checkpoints" default resolves relative to
#: whatever cwd the process is launched from (repo root in every documented
#: dev workflow -- see README's "How to run").
CHECKPOINT_DIR = Path(os.environ.get("DRONE_SIM_CHECKPOINT_DIR", "checkpoints"))

#: Checkpoint names are bare identifiers, never paths (see
#: CheckpointSaveRequest/CheckpointLoadRequest's charset-restricted Field) --
#: this second check is defense in depth, not the only guard, so a future
#: caller that skips Pydantic validation still can't escape CHECKPOINT_DIR.
_SAFE_CHECKPOINT_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


def _resolve_checkpoint_path(name: str) -> Path:
    if not name or not set(name) <= _SAFE_CHECKPOINT_NAME_CHARS:
        raise HTTPException(status_code=400, detail=f"invalid checkpoint name {name!r}")
    return CHECKPOINT_DIR / f"{name}.npz"

#: Phase 5: a simulation's runtime is either a single-process SimulationRuntime
#: (the default) or a DistributedSimulationRuntime (opt-in via
#: CreateSimulationRequest.distributed=True) -- both implement the identical
#: method surface every handler below calls, so nothing else in this file
#: needs to branch on which kind a given simulation_id actually has.
_runtimes: dict[str, "SimulationRuntime | DistributedSimulationRuntime"] = {}

#: Hard cap on raw drone positions returned by /viewport in one response.
#: A viewport with more visible drones than this still gets full heatmap
#: data; raw positions are truncated to a deterministic prefix and
#: ``truncated=True`` is reported (see ViewportResponse).
MAX_VISIBLE_DRONES = 5_000

#: Phase 3B dashboard-stream publication rate bounds/default (Hz). The
#: simulation tick rate is unaffected by this -- see GET .../stream. 8Hz sits
#: in the 5-10Hz range CLAUDE.md's Phase 3B scope specifies.
DEFAULT_STREAM_HZ = 8.0
MIN_STREAM_HZ = 1.0
MAX_STREAM_HZ = 20.0

#: Close a stream after this many consecutive frame-build/serialize failures
#: rather than retrying forever -- a transient error (e.g. a viewport query
#: racing a reset()) must not spin hot, but must also never propagate up and
#: kill the SimulationRuntime, which owns no reference to any stream.
MAX_CONSECUTIVE_STREAM_ERRORS = 5

#: Bound on how long a single ``Request.is_disconnected()`` check may take.
#: Starlette implements it as a best-effort non-blocking receive() wrapped in
#: a cancel scope; under some ASGI transports (observed with the test client,
#: not just a defensive guess) that receive() can fail to yield a checkpoint
#: promptly and stall. Never let one disconnect check hang the whole
#: publication loop -- on timeout, assume "still connected" and let the next
#: loop iteration check again.
DISCONNECT_CHECK_TIMEOUT_S = 0.05


async def _client_disconnected(request: Request) -> bool:
    try:
        return await asyncio.wait_for(request.is_disconnected(), timeout=DISCONNECT_CHECK_TIMEOUT_S)
    except asyncio.TimeoutError:
        return False

#: Active SSE connections per simulation_id, for disconnect-cleanup
#: verification (see tests/test_stream.py) and duplicate-connection
#: diagnostics. Incremented/decremented only in the stream generator's
#: try/finally, so it is accurate even when a client disconnects mid-frame.
_stream_connection_counts: dict[str, int] = {}

#: Phase 5 monitoring (see api/monitoring.py's /metrics): process-wide,
#: best-effort counters across every stream connection this process has ever
#: served. Plain module-level ints rather than a lock-guarded accumulator --
#: these are operational counters for a metrics display, never used for any
#: correctness decision, so CPython's GIL-serialized += is sufficient.
_stream_frames_published_total = 0
_stream_frames_superseded_total = 0


def _default_goal_positions(config: SimulationConfig, positions: np.ndarray) -> np.ndarray:
    """Reflect each starting position through the world center.

    Same idea already used by ``scenarios.rare_collision_background`` for its
    background drones -- a distant, reproducible destination with no
    dependency on the chosen policy. Used here only when a caller asks for
    ``policy=`` on a world that has no scenario-provided goals of its own
    (plain ``num_drones``-style creation, or a scenario factory that doesn't
    set ``goal_positions``), so ``GoalDirectedMovementAlgorithm``/
    ``LocalAvoidanceMovementAlgorithm`` always have somewhere to steer toward.
    This lives here (not in scenarios.py) since it's Phase 3B API-layer
    orchestration, not a new scenario.
    """
    center = (config.bounds_min_arr + config.bounds_max_arr) / 2.0
    return (2.0 * center[None, :] - positions.astype(np.float64)).astype(np.float32)


def _build_movement_system(policy: Optional[str]) -> Optional[MovementSystem]:
    """Build the ``MovementSystem`` for a requested policy, or ``None`` for
    the Phase 3A default (Random/Scripted, unmodified)."""
    if policy is None:
        return None
    if policy == "goal_directed":
        algo = GoalDirectedMovementAlgorithm()
    elif policy == "local_avoidance":
        algo = LocalAvoidanceMovementAlgorithm()
    else:  # pragma: no cover - Literal type already rejects anything else
        raise ValueError(f"unknown policy {policy!r}")
    return MovementSystem(policies={algo.policy_id: algo})


def _build_world_factory(req: CreateSimulationRequest):
    """Build the ``world_factory`` for a requested scenario/policy combination.

    Returns ``None`` for the Phase 3A default path (no scenario, no policy),
    so ``SimulationRuntime`` falls back to its own ``World.create(config)``
    exactly as before. A pure function of ``config`` (no captured mutable
    state) so ``SimulationRuntime.reset()`` reproduces the identical initial
    world on every call -- required for the "same seed stays reproducible"
    acceptance criterion.
    """
    if req.scenario is None and req.policy is None:
        return None

    scenario_name = req.scenario
    policy = req.policy
    policy_id = None
    if policy == "goal_directed":
        policy_id = GoalDirectedMovementAlgorithm.policy_id
    elif policy == "local_avoidance":
        policy_id = LocalAvoidanceMovementAlgorithm.policy_id

    def factory(config: SimulationConfig) -> World:
        if scenario_name is not None:
            world = SCENARIOS[scenario_name](config).world
        else:
            world = World.create(config)

        if policy_id is not None:
            state = world.state
            state.movement_policy_ids = np.full(state.num_drones, policy_id, dtype=np.int32)
            if state.goal_positions is None:
                state.goal_positions = _default_goal_positions(config, state.positions)
        return world

    return factory


def reset_registry() -> None:
    """Test-only: clear all runtimes (and stop their threads) between tests."""
    for runtime in _runtimes.values():
        runtime.shutdown()
    _runtimes.clear()


def _get_runtime(simulation_id: str) -> "SimulationRuntime | DistributedSimulationRuntime":
    runtime = _runtimes.get(simulation_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"unknown simulation_id {simulation_id!r}")
    return runtime


def _status_response(runtime: "SimulationRuntime | DistributedSimulationRuntime") -> SimulationStatusResponse:
    state = runtime.get_status()
    is_distributed = isinstance(runtime, DistributedSimulationRuntime)
    return SimulationStatusResponse(
        simulation_id=state.simulation_id,
        status=state.status.value,
        tick=state.tick,
        num_drones=state.num_drones,
        execution_mode="distributed" if is_distributed else "single_process",
        num_workers=runtime._dist_config.num_workers if is_distributed else None,
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

    movement = _build_movement_system(req.policy)
    world_factory = _build_world_factory(req)

    if req.distributed:
        dist_config = DistributedConfig(
            num_workers=req.num_workers,
            num_partitions=req.num_partitions,
            use_threads=(req.executor == "threads"),
            use_processes=(req.executor == "processes"),
        )
        try:
            runtime: "SimulationRuntime | DistributedSimulationRuntime" = DistributedSimulationRuntime(
                simulation_id, config, dist_config, movement=movement, world_factory=world_factory
            )
        except (ValueError, NotImplementedError) as exc:
            # NotImplementedError: a requires_context policy (e.g.
            # local_avoidance) was requested with distributed=True --
            # DistributedCoordinator rejects this before any worker pool is
            # created (see distributed_runtime.py), so nothing to clean up.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        runtime = SimulationRuntime(simulation_id, config, movement=movement, world_factory=world_factory)

    _runtimes[simulation_id] = runtime
    return _status_response(runtime)


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
    distributed_metrics = (
        runtime.get_distributed_metrics() if isinstance(runtime, DistributedSimulationRuntime) else None
    )
    return MetricsResponse(
        simulation_id=simulation_id, tick=snapshot.tick, metrics=snapshot.metrics,
        distributed_metrics=distributed_metrics,
    )


def _require_single_process(runtime: "SimulationRuntime | DistributedSimulationRuntime") -> SimulationRuntime:
    """Checkpointing (see ``checkpoint.py``) is only ever defined for a plain
    ``Simulation`` -- it reads ``sim.engine.get_rng_state()``, which
    ``DistributedCoordinator`` has no equivalent of (it advances via a
    ``WorkerPool`` of per-partition workers, not one ``SimulationEngine``).
    Reject distributed simulations here, before touching the filesystem,
    exactly like the existing ``distributed=True`` + ``local_avoidance`` 400
    rejects an unsupported combination before any worker pool is created."""
    if isinstance(runtime, DistributedSimulationRuntime):
        raise HTTPException(
            status_code=400,
            detail="checkpointing is only supported for single-process (non-distributed) simulations",
        )
    return runtime


@router.post("/simulations/{simulation_id}/checkpoint", response_model=CheckpointSaveResponse)
def save_checkpoint(simulation_id: str, req: CheckpointSaveRequest) -> CheckpointSaveResponse:
    runtime = _require_single_process(_get_runtime(simulation_id))
    path = _resolve_checkpoint_path(req.name)
    try:
        info = runtime.save_checkpoint(path)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CheckpointSaveResponse(
        simulation_id=simulation_id,
        name=req.name,
        tick=info["tick"],
        num_drones=info["num_drones"],
        size_bytes=path.stat().st_size,
        saved_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/simulations/{simulation_id}/checkpoint/load", response_model=CheckpointLoadResponse)
def load_checkpoint(simulation_id: str, req: CheckpointLoadRequest) -> CheckpointLoadResponse:
    runtime = _require_single_process(_get_runtime(simulation_id))
    path = _resolve_checkpoint_path(req.name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"checkpoint {req.name!r} not found")
    try:
        snapshot = runtime.load_checkpoint(path)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CheckpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    status = runtime.get_status()
    return CheckpointLoadResponse(
        simulation_id=simulation_id,
        name=req.name,
        tick=snapshot.tick,
        num_drones=status.num_drones,
        status=status.status.value,
    )


@router.get("/checkpoints", response_model=CheckpointListResponse)
def list_checkpoints() -> CheckpointListResponse:
    """Best-effort directory listing -- not tied to any simulation_id, since
    a checkpoint file itself carries no simulation_id (see ``checkpoint.py``:
    only config/tick/rng-state/drone arrays are persisted). Skips any file
    that fails validation (corrupt, foreign, or mid-write) rather than
    failing the whole listing -- one bad file must never hide every other
    valid checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for candidate in sorted(CHECKPOINT_DIR.glob("*.npz")):
        try:
            meta = _checkpoint.validate_checkpoint(candidate)
        except CheckpointError:
            continue
        stat = candidate.stat()
        items.append(
            CheckpointInfo(
                name=candidate.stem,
                tick=meta["tick"],
                num_drones=meta["config"]["num_drones"],
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            )
        )
    return CheckpointListResponse(checkpoints=items)


def _build_frame_components(
    runtime: SimulationRuntime, viewport: ViewportQuery, heatmap_query: HeatmapQuery
) -> tuple[dict, dict, int]:
    """One snapshot read -> (payload dict without timings, partial timings, tick).

    Shared by ``GET /frame`` and ``GET /stream`` so both build a dashboard
    frame from the exact same query logic -- one ``get_snapshot_and_status_with_lock_wait()``
    call, then heatmap/collision queries against that single already-published
    snapshot, outside the lock. Neither endpoint recomputes or reclassifies
    anything: this only reads what ``Simulation.step()`` already produced.
    Never returns raw drone positions (use ``/viewport`` explicitly for those).
    """
    snapshot, status, lock_wait_ms = runtime.get_snapshot_and_status_with_lock_wait()

    t0 = time.perf_counter()
    heatmap = compute_heatmap(snapshot, heatmap_query)
    t1 = time.perf_counter()
    markers = query_collision_markers(snapshot, viewport)
    t2 = time.perf_counter()

    tick_timings = runtime.get_last_timings()
    payload = {
        "simulation_id": runtime.simulation_id,
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
    }
    partial_timings = {
        "sim_step_ms": tick_timings.sim_step_ms,
        "snapshot_build_ms": tick_timings.snapshot_build_ms,
        "lock_wait_ms": lock_wait_ms,
        "heatmap_ms": (t1 - t0) * 1e3,
        "collisions_ms": (t2 - t1) * 1e3,
    }
    return payload, partial_timings, snapshot.tick


def _splice_timings(body_without_timings: str, timings: dict) -> str:
    """Append a ``"timings"`` key to an already-serialized payload without
    re-dumping it -- see ``get_frame()``'s docstring for why this exists."""
    return body_without_timings[:-1] + ',"timings":' + json.dumps(timings) + "}"


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
    drone positions (use ``/viewport`` explicitly for those). See
    ``GET .../stream`` for the pushed (SSE) equivalent of this same frame.
    """
    t_start = time.perf_counter()
    runtime = _get_runtime(simulation_id)
    viewport = _build_viewport(x_min, x_max, y_min, y_max, z_min, z_max)
    try:
        heatmap_query = HeatmapQuery(viewport=viewport, x_bins=x_bins, y_bins=y_bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # _build_frame_components() makes the single lock acquisition for BOTH
    # snapshot and status -- see get_snapshot_and_status_with_lock_wait()'s
    # docstring for why calling get_snapshot_with_lock_wait() + get_status()
    # separately (the previous implementation) was a real, unmeasured second
    # lock-wait, not just a style choice.
    payload, partial_timings, _tick = _build_frame_components(runtime, viewport, heatmap_query)
    # "timings" deliberately left out of payload here: serialization_ms/
    # total_request_ms describe the cost of producing this very payload, so
    # they can't be computed until AFTER it's serialized (see _splice_timings).

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
        **partial_timings,
        "serialization_ms": (t4 - t3) * 1e3,
        "total_request_ms": (t4 - t_start) * 1e3,
    }
    body = _splice_timings(body_without_timings, timings)
    return Response(content=body, media_type="application/json")


def _build_and_serialize_stream_frame(
    runtime: SimulationRuntime, viewport: ViewportQuery, heatmap_query: HeatmapQuery, seq: int
) -> tuple[str, int]:
    """Build one dashboard frame and serialize it, returning ``(body, tick)``.

    Runs entirely off the shared ``_build_frame_components()`` (same query
    logic as ``/frame``) so streaming never becomes a second, competing
    visualization pipeline. Called via ``asyncio.to_thread`` by the stream
    generator below so the numpy/JSON work never blocks the asyncio event
    loop other requests (including other open streams) are served from.
    """
    t_start = time.perf_counter()
    payload, partial_timings, tick = _build_frame_components(runtime, viewport, heatmap_query)
    payload["seq"] = seq
    payload["server_time"] = time.time()

    t0 = time.perf_counter()
    body_without_timings = json.dumps(payload)
    t1 = time.perf_counter()
    timings = {
        **partial_timings,
        "serialization_ms": (t1 - t0) * 1e3,
        "generation_ms": (t1 - t_start) * 1e3,
    }
    return _splice_timings(body_without_timings, timings), tick


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/simulations/{simulation_id}/stream")
async def stream_simulation(
    request: Request,
    simulation_id: str,
    x_min: float = Query(...),
    x_max: float = Query(...),
    y_min: float = Query(...),
    y_max: float = Query(...),
    z_min: Optional[float] = Query(None),
    z_max: Optional[float] = Query(None),
    x_bins: int = Query(60, gt=0),
    y_bins: int = Query(60, gt=0),
    hz: float = Query(DEFAULT_STREAM_HZ, ge=MIN_STREAM_HZ, le=MAX_STREAM_HZ),
) -> StreamingResponse:
    """Server-Sent Events stream of the same dashboard frame ``/frame`` returns,
    pushed at a bounded, configurable rate (``hz``, default 8) independent of
    the simulation's own tick rate.

    Each frame is built fresh from whatever the ``SimulationRuntime`` has most
    recently published -- there is no queue of pending frames anywhere in this
    endpoint. A slow client therefore never grows a backlog: the next time its
    connection is ready for more data, this generator fetches and sends
    whatever tick is *then* current, silently superseding any ticks that
    happened in between (the "skip old frames, always send the latest" policy
    the acceptance criteria ask for). The numpy/JSON work for each frame runs
    via ``asyncio.to_thread`` so it can never block this process's asyncio
    event loop -- and therefore never delays any other concurrent request,
    including a different simulation's own stream.
    """
    runtime = _get_runtime(simulation_id)  # 404 raised here, before the stream ever opens
    viewport = _build_viewport(x_min, x_max, y_min, y_max, z_min, z_max)
    try:
        heatmap_query = HeatmapQuery(viewport=viewport, x_bins=x_bins, y_bins=y_bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    interval_s = 1.0 / hz

    async def event_generator():
        global _stream_frames_published_total, _stream_frames_superseded_total
        _stream_connection_counts[simulation_id] = _stream_connection_counts.get(simulation_id, 0) + 1
        seq = 0
        consecutive_errors = 0
        last_tick: Optional[int] = None
        try:
            while True:
                if await _client_disconnected(request):
                    break
                if simulation_id not in _runtimes:
                    # Deleted mid-stream (DELETE /simulations/{id}) -- our
                    # `runtime` reference is still a valid, harmless object
                    # (its background thread was already stopped by
                    # shutdown() before removal), but nothing should keep
                    # polling it. Tell the client why, then stop cleanly.
                    yield _sse_event("closed", {"reason": "simulation_deleted"})
                    break

                seq += 1
                try:
                    body, _tick = await asyncio.to_thread(
                        _build_and_serialize_stream_frame, runtime, viewport, heatmap_query, seq
                    )
                except Exception:
                    # A bad frame (e.g. a viewport query racing a reset())
                    # must never kill the SimulationRuntime or the whole
                    # stream outright -- retry a bounded number of times.
                    consecutive_errors += 1
                    seq -= 1
                    if consecutive_errors >= MAX_CONSECUTIVE_STREAM_ERRORS:
                        yield _sse_event("error", {"detail": "repeated frame errors, closing stream"})
                        break
                    await asyncio.sleep(interval_s)
                    continue
                consecutive_errors = 0

                _stream_frames_published_total += 1
                if last_tick is not None and _tick - last_tick > 1:
                    _stream_frames_superseded_total += _tick - last_tick - 1
                last_tick = _tick

                yield f"id: {seq}\ndata: {body}\n\n"
                await asyncio.sleep(interval_s)
        finally:
            remaining = _stream_connection_counts.get(simulation_id, 1) - 1
            if remaining <= 0:
                _stream_connection_counts.pop(simulation_id, None)
            else:
                _stream_connection_counts[simulation_id] = remaining

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
