"""Phase 3A local API package (FastAPI). Not imported by the simulation kernel."""

from .app import create_app

__all__ = ["create_app"]
