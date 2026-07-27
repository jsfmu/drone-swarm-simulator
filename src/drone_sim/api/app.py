"""FastAPI application factory for the Phase 3A local visualization API.

Import boundary: this module (and everything under ``drone_sim.api``) is the
only place in the package that imports FastAPI. ``drone_sim``'s simulation
kernel modules never import from ``drone_sim.api``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Drone Collision Simulator API",
        description="Phase 3A local visualization query backend.",
        version="0.1.0",
    )
    app.include_router(router)
    # Mounted last (and at "/") so API routes registered above always take
    # precedence; this only serves the minimal static browser page/assets.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    return app


app = create_app()
