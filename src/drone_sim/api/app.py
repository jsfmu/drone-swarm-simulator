"""FastAPI application factory for the Phase 3A local visualization API.

Import boundary: this module (and everything under ``drone_sim.api``) is the
only place in the package that imports FastAPI. ``drone_sim``'s simulation
kernel modules never import from ``drone_sim.api``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import monitoring
from .monitoring import app_ready_state, request_stats
from .routes import router

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Flips GET /ready from 503 to ready on startup, resets it on shutdown.

    This app currently has no slow/async startup dependency to wait for, so
    this fires almost immediately after the ASGI server starts serving -- but
    the distinction from GET /health (always ok) is real and deliberate: a
    future startup step (e.g. warming a cache, checking a required directory)
    has exactly one place to gate readiness on, without touching every route
    handler.
    """
    app_ready_state["ready"] = True
    yield
    app_ready_state["ready"] = False


def create_app() -> FastAPI:
    app = FastAPI(
        title="Drone Collision Simulator API",
        description="Phase 3A local visualization query backend.",
        version="0.1.0",
        lifespan=_lifespan,
    )
    # Phase 3A's static/index.html is served BY this app (same-origin, no CORS
    # needed). Phase 3B's React dashboard runs on its own Vite dev server
    # (default http://localhost:5173) for local dev, or on Render for production
    # -- a genuinely different origin -- so REST calls and the EventSource
    # stream both need CORS allowed here, or the browser blocks them before any
    # handler runs (surfaces in the frontend as a generic "TypeError: Failed to
    # fetch").
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"(http://(localhost|127\.0\.0\.1)(:\d+)?|https://drone-swarm-simulator-8sj3\.onrender\.com)",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _request_stats_middleware(request: Request, call_next):
        """Phase 5 monitoring: total request count + cumulative latency, read
        by GET /metrics. Deliberately just two running sums (see
        monitoring.request_stats's docstring) -- no per-request history kept,
        so this stays O(1) regardless of how long the process has been up."""
        t0 = time.perf_counter()
        response = await call_next(request)
        request_stats["count"] += 1
        request_stats["total_time_s"] += time.perf_counter() - t0
        return response

    app.include_router(router)
    app.include_router(monitoring.router)
    # Mounted last (and at "/") so API routes registered above always take
    # precedence; this only serves the minimal static browser page/assets.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    return app


app = create_app()
