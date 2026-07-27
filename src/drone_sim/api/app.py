"""FastAPI application factory for the Phase 3A local visualization API.

Import boundary: this module (and everything under ``drone_sim.api``) is the
only place in the package that imports FastAPI. ``drone_sim``'s simulation
kernel modules never import from ``drone_sim.api``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Drone Collision Simulator API",
        description="Phase 3A local visualization query backend.",
        version="0.1.0",
    )
    # Phase 3A's static/index.html is served BY this app (same-origin, no CORS
    # needed). Phase 3B's React dashboard runs on its own Vite dev server
    # (default http://localhost:5173) -- a genuinely different origin -- so
    # REST calls and the EventSource stream both need CORS allowed here, or
    # the browser blocks them before any handler runs (surfaces in the
    # frontend as a generic "TypeError: Failed to fetch"). Scoped to
    # localhost/127.0.0.1 on any port rather than "*", since this is a local
    # dev tool with no auth and no cookies -- not a public deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    # Mounted last (and at "/") so API routes registered above always take
    # precedence; this only serves the minimal static browser page/assets.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    return app


app = create_app()
