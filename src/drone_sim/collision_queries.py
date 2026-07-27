"""Collision-marker queries over a :class:`SimulationSnapshot`.

Reads the canonical ``collision_pairs``/``collision_distances`` already
computed by ``CollisionDetectionEngine`` (captured into the snapshot as-is)
-- this module never recomputes or reclassifies a collision, it only looks
up positions/velocities for markers already known to be real collisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .snapshot import SimulationSnapshot
from .viewport import ViewportQuery


@dataclass(frozen=True)
class CollisionMarker:
    drone_a: int
    drone_b: int
    tick: int
    x: float
    y: float
    z: float
    distance: float
    relative_speed: float


def query_collision_markers(
    snapshot: SimulationSnapshot,
    viewport: ViewportQuery | None = None,
) -> list[CollisionMarker]:
    """Return one :class:`CollisionMarker` per collision pair in ``snapshot``.

    Marker position is the midpoint between the two drones at the captured
    tick. When ``viewport`` is given, a marker is included only if its
    midpoint falls inside the (inclusive) bounds. ``DetectionResult.collision_pairs``
    is already unique and canonically ``i < j`` (``SpatialHashGrid`` never
    produces a pair twice), so no reversed-duplicate filtering is needed here.
    """
    pairs = snapshot.collision_pairs
    if pairs.shape[0] == 0:
        return []

    rows_a = snapshot.id_to_row[pairs[:, 0]]
    rows_b = snapshot.id_to_row[pairs[:, 1]]
    valid = (rows_a >= 0) & (rows_b >= 0)
    pairs = pairs[valid]
    rows_a = rows_a[valid]
    rows_b = rows_b[valid]
    distances = snapshot.collision_distances[valid]

    pos_a = snapshot.positions[rows_a].astype(np.float64)
    pos_b = snapshot.positions[rows_b].astype(np.float64)
    midpoints = (pos_a + pos_b) * 0.5

    vel_a = snapshot.velocities[rows_a].astype(np.float64)
    vel_b = snapshot.velocities[rows_b].astype(np.float64)
    relative_speeds = np.linalg.norm(vel_a - vel_b, axis=1)

    mask = np.ones(pairs.shape[0], dtype=bool)
    if viewport is not None:
        mask &= (midpoints[:, 0] >= viewport.x_min) & (midpoints[:, 0] <= viewport.x_max)
        mask &= (midpoints[:, 1] >= viewport.y_min) & (midpoints[:, 1] <= viewport.y_max)
        if viewport.z_min is not None:
            mask &= midpoints[:, 2] >= viewport.z_min
        if viewport.z_max is not None:
            mask &= midpoints[:, 2] <= viewport.z_max

    markers: list[CollisionMarker] = []
    for k in np.nonzero(mask)[0]:
        a, b = int(pairs[k, 0]), int(pairs[k, 1])
        markers.append(
            CollisionMarker(
                drone_a=min(a, b),
                drone_b=max(a, b),
                tick=snapshot.tick,
                x=float(midpoints[k, 0]),
                y=float(midpoints[k, 1]),
                z=float(midpoints[k, 2]),
                distance=float(distances[k]),
                relative_speed=float(relative_speeds[k]),
            )
        )
    return markers
