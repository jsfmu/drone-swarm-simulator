"""Phase 5 operational endpoints: health, readiness, and metrics.

Kept in its own router/module rather than folded into ``routes.py`` (the
simulation-domain CRUD/query endpoints) or onto ``SimulationConfig``/
``SimulationRuntime`` themselves -- ops-facing concerns stay separate from
simulation-domain code, matching the same "don't turn an already-highly
-connected class into a dumping ground" guidance CLAUDE.md gives for this
phase. Nothing here computes anything the simulation kernel doesn't already
expose: ``/metrics`` only reads already-published ``SimulationSnapshot``
fields and ``RunningMetrics.summary()`` output (via the snapshot), never
recomputes detection or steps a simulation.

Distributed-execution metrics (worker/partition health, rebalances, etc.) ARE
exposed here, nested under a per-simulation ``"distributed"`` key, for any
simulation created with ``distributed=True`` (see
``distributed_runtime.DistributedSimulationRuntime`` and
``coordinator.DistributedCoordinator.metrics_summary()``). Plain,
single-process simulations never have this key -- absence, not a null/empty
value, is how a consumer tells the two kinds apart.
"""

from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter, Response

from ..distributed_runtime import DistributedSimulationRuntime
from ..process_metrics import resident_set_size_bytes
from . import routes as _routes
from .routes import _runtimes, _stream_connection_counts

# _stream_frames_published_total/_stream_frames_superseded_total are read via
# _routes.<name> below, NOT `from .routes import <name>` -- those two counters
# are reassigned (not mutated in place) inside routes.py's event_generator via
# `global`, so a name-imported copy here would freeze at its value at import
# time instead of tracking routes.py's live counter. _runtimes/
# _stream_connection_counts are dicts mutated in place, so importing them by
# name is safe (same underlying object either way).

router = APIRouter()

_START_TIME = time.time()

#: Set True by app.py's startup handler once the app has finished whatever
#: (currently trivial, but explicit) initialization it needs -- see
#: app_ready_state's docstring in app.py for why this is a real, if simple,
#: distinction from "the HTTP server started".
app_ready_state = {"ready": False}

#: Phase 5 monitoring: total request count and cumulative latency, updated by
#: app.py's timing middleware. A plain dict of running sums (not a class) --
#: intentionally the simplest thing that gives mean latency without an
#: unbounded per-request history.
request_stats = {"count": 0, "total_time_s": 0.0}


@router.get("/health")
def health() -> dict:
    """Liveness: is this process able to respond at all.

    Always ``{"status": "ok"}`` once the ASGI app is serving requests --
    never depends on simulation state or readiness. A process that can answer
    this at all is, by definition, alive.
    """
    return {"status": "ok", "uptime_s": round(time.time() - _START_TIME, 3)}


@router.get("/ready")
def ready(response: Response) -> dict:
    """Readiness: can the API accept work right now.

    Distinct from ``/health`` on purpose (see the Phase 5 spec's "do not
    report readiness merely because the HTTP server started"): this reads
    ``app_ready_state``, set only after ``app.py``'s startup handler runs to
    completion, not merely because a request handler executed at all.
    """
    if not app_ready_state["ready"]:
        response.status_code = 503
        return {"status": "not_ready"}
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> dict:
    """Machine-readable operational metrics: every simulation this process
    currently tracks, plus process-wide API/streaming counters.

    Each simulation's entry is built entirely from its already-published
    ``SimulationSnapshot`` (one ``get_snapshot()`` call, no new lock
    acquisition beyond that, no detection/movement work triggered) --
    identical in spirit to how ``/frame`` reads exactly one snapshot per
    request.
    """
    sims: Dict[str, dict] = {}
    for sim_id, runtime in _runtimes.items():
        status = runtime.get_status()
        snapshot = runtime.get_snapshot()
        sims[sim_id] = {
            "status": status.status.value,
            "tick": snapshot.tick,
            "active_drone_count": snapshot.num_active_drones,
            "current_collision_count": int(snapshot.collision_pairs.shape[0]),
            "current_near_miss_count": int(snapshot.near_miss_pairs.shape[0]),
            "active_stream_consumers": _stream_connection_counts.get(sim_id, 0),
            **{
                k: v
                for k, v in snapshot.metrics.items()
                if k in (
                    "mean_tick_ms", "median_tick_ms", "p95_tick_ms", "ticks_per_second",
                    "total_collisions", "total_near_misses",
                    "mean_candidate_pairs", "total_candidate_pairs",
                )
            },
        }
        if isinstance(runtime, DistributedSimulationRuntime):
            sims[sim_id]["distributed"] = runtime.get_distributed_metrics()

    rss = resident_set_size_bytes()
    return {
        "simulations": sims,
        "total_simulations": len(_runtimes),
        "process": {
            "uptime_s": round(time.time() - _START_TIME, 3),
            "resident_set_size_bytes": rss,
        },
        "api": {
            "request_count": request_stats["count"],
            "mean_request_latency_ms": (
                (request_stats["total_time_s"] / request_stats["count"]) * 1e3
                if request_stats["count"] else 0.0
            ),
        },
        "streaming": {
            "total_active_stream_consumers": sum(_stream_connection_counts.values()),
            "frames_published_total": _routes._stream_frames_published_total,
            "frames_superseded_total": _routes._stream_frames_superseded_total,
            "queue_depth": 0,  # no queue by design -- latest-state semantics, see GET .../stream
        },
    }
