"""Boundary handling.

The :class:`BoundaryManager` constrains drones to the configured XYZ world
box after movement. It is fully vectorised across all active drones and all
three axes at once.
"""

from __future__ import annotations

import numpy as np

from .config import BoundaryMode
from .state import World


class BoundaryManager:
    """Keeps drones inside the world box each tick."""

    def apply(self, world: World) -> None:
        state = world.state
        active = state.active_mask
        if not active.any():
            return

        pos = state.positions
        vel = state.velocities
        lo = world.bounds_min.astype(np.float32)
        hi = world.bounds_max.astype(np.float32)
        mode = world.config.boundary_mode

        # Only touch active rows, but boolean masks broadcast cleanly over the
        # full arrays; inactive rows are excluded via ``active[:, None]``.
        below = (pos < lo) & active[:, None]
        above = (pos > hi) & active[:, None]

        # Clamp positions back onto the wall.
        if below.any():
            pos[below] = np.broadcast_to(lo, pos.shape)[below]
        if above.any():
            pos[above] = np.broadcast_to(hi, pos.shape)[above]

        hit = below | above
        if not hit.any():
            return

        if mode is BoundaryMode.REFLECT:
            # Negate the velocity component on axes that hit a wall.
            vel[hit] = -vel[hit]
        elif mode is BoundaryMode.CLAMP:
            vel[hit] = 0.0
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown boundary mode: {mode}")
