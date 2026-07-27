"""Vectorized bounding-box (viewport) queries over a :class:`SimulationSnapshot`.

Filtering is a single NumPy boolean-mask pass over ``snapshot.positions`` --
no per-drone Python loop, so this scales the same way ``SpatialHashGrid``
and ``NeighborFeatureBuilder`` do. Bounds are inclusive on every axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .snapshot import SimulationSnapshot


@dataclass(frozen=True)
class ViewportQuery:
    """Inclusive axis-aligned bounding box. ``z_min``/``z_max`` are optional --
    omitting both means "no altitude filtering"."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float | None = None
    z_max: float | None = None

    def __post_init__(self) -> None:
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be <= x_max ({self.x_max})")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be <= y_max ({self.y_max})")
        if self.z_min is not None and self.z_max is not None and self.z_min > self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be <= z_max ({self.z_max})")


@dataclass(frozen=True)
class VisibleDroneData:
    """Drones inside a :class:`ViewportQuery`, aligned across the three arrays."""

    tick: int
    drone_ids: np.ndarray  # (V,) int64
    positions: np.ndarray  # (V, 3) float32
    velocities: np.ndarray  # (V, 3) float32
    total_visible: int  # count matching the viewport, before any truncation
    truncated: bool = False


def find_visible_drones(
    snapshot: SimulationSnapshot,
    query: ViewportQuery,
    *,
    limit: int | None = None,
) -> VisibleDroneData:
    """Return active drones inside ``query``'s bounding box.

    ``limit`` caps the number of *raw positions* returned (a deterministic
    prefix by ascending drone id) while ``total_visible`` always reports the
    true count so callers can tell truncation happened.
    """
    pos = snapshot.positions
    if pos.shape[0] == 0:
        empty3 = np.empty((0, 3), dtype=np.float32)
        return VisibleDroneData(
            tick=snapshot.tick,
            drone_ids=np.empty(0, dtype=np.int64),
            positions=empty3,
            velocities=empty3,
            total_visible=0,
            truncated=False,
        )

    mask = (
        (pos[:, 0] >= query.x_min)
        & (pos[:, 0] <= query.x_max)
        & (pos[:, 1] >= query.y_min)
        & (pos[:, 1] <= query.y_max)
    )
    if query.z_min is not None:
        mask &= pos[:, 2] >= query.z_min
    if query.z_max is not None:
        mask &= pos[:, 2] <= query.z_max

    idx = np.nonzero(mask)[0]
    total = int(idx.size)
    truncated = False
    if limit is not None and total > limit:
        idx = idx[:limit]
        truncated = True

    return VisibleDroneData(
        tick=snapshot.tick,
        drone_ids=snapshot.drone_ids[idx],
        positions=snapshot.positions[idx],
        velocities=snapshot.velocities[idx],
        total_visible=total,
        truncated=truncated,
    )
