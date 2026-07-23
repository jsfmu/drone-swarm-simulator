"""Collision detection and resolution.

The pipeline separates *detection* from *resolution*:

* :class:`CollisionDetectionEngine` turns candidate pairs into collision and
  near-miss results. It offers a spatial-hash detector and a brute-force
  reference detector; the two must agree on small cases.
* :class:`CollisionResolutionEngine` consumes collisions and updates the
  affected drone state (equal-mass elastic response along the line of
  centres). It does not change detection rules.

Results are returned as arrays (no per-drone objects). Lightweight
:class:`CollisionEvent` / :class:`NearMissEvent` dataclasses can be
materialised on demand for logging or the (later) event pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import SimulationConfig
from .spatial_hash import SpatialHashGrid
from .state import DroneState


@dataclass(frozen=True)
class CollisionEvent:
    tick: int
    drone_i: int
    drone_j: int
    position: tuple  # (x, y, z) midpoint
    distance: float
    relative_speed: float


@dataclass(frozen=True)
class NearMissEvent:
    tick: int
    drone_i: int
    drone_j: int
    min_distance: float


@dataclass
class DetectionResult:
    """Arrays describing the collisions and near misses found on a tick."""

    collision_pairs: np.ndarray      # (K, 2) int64, i<j
    collision_distances: np.ndarray  # (K,) float64
    near_miss_pairs: np.ndarray      # (M, 2) int64, i<j
    near_miss_distances: np.ndarray  # (M,) float64
    num_candidate_pairs: int

    @property
    def num_collisions(self) -> int:
        return int(self.collision_pairs.shape[0])

    @property
    def num_near_misses(self) -> int:
        return int(self.near_miss_pairs.shape[0])

    def collision_events(self, state: DroneState, tick: int) -> List[CollisionEvent]:
        events: List[CollisionEvent] = []
        for (i, j), d in zip(self.collision_pairs, self.collision_distances):
            mid = (state.positions[i] + state.positions[j]) * 0.5
            rel = state.velocities[i] - state.velocities[j]
            events.append(
                CollisionEvent(
                    tick=tick,
                    drone_i=int(i),
                    drone_j=int(j),
                    position=(float(mid[0]), float(mid[1]), float(mid[2])),
                    distance=float(d),
                    relative_speed=float(np.linalg.norm(rel)),
                )
            )
        return events

    def near_miss_events(self, tick: int) -> List[NearMissEvent]:
        return [
            NearMissEvent(tick=tick, drone_i=int(i), drone_j=int(j), min_distance=float(d))
            for (i, j), d in zip(self.near_miss_pairs, self.near_miss_distances)
        ]


def _classify(
    pairs: np.ndarray,
    positions: np.ndarray,
    collision_radius: float,
    near_miss_radius: float,
    num_candidates: int,
) -> DetectionResult:
    """Split candidate ``pairs`` into collisions and near misses by distance.

    Distances are computed in float64 for stability. Both detectors funnel
    through this function, so given identical pair sets they produce identical
    classifications.
    """
    if pairs.shape[0] == 0:
        empty_p = np.empty((0, 2), dtype=np.int64)
        empty_d = np.empty(0, dtype=np.float64)
        return DetectionResult(empty_p, empty_d, empty_p, empty_d, num_candidates)

    p = positions.astype(np.float64)
    delta = p[pairs[:, 0]] - p[pairs[:, 1]]
    dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))

    coll = dist <= collision_radius
    near = (dist > collision_radius) & (dist <= near_miss_radius)

    return DetectionResult(
        collision_pairs=pairs[coll],
        collision_distances=dist[coll],
        near_miss_pairs=pairs[near],
        near_miss_distances=dist[near],
        num_candidate_pairs=num_candidates,
    )


class CollisionDetectionEngine:
    """Finds unique collisions and near misses for a tick."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def detect(
        self, state: DroneState, grid: SpatialHashGrid, pairs: np.ndarray | None = None
    ) -> DetectionResult:
        """Spatial-hash detection. ``grid`` must already be built for this tick.

        ``pairs`` lets a caller that already computed ``grid.candidate_pairs()``
        (e.g. stage-by-stage profiling in :class:`~drone_sim.simulation.SimulationEngine`)
        pass them in directly instead of having this method recompute them.
        Omitting it — every existing call site — behaves exactly as before.
        """
        if pairs is None:
            pairs = grid.candidate_pairs()
        return _classify(
            pairs,
            state.positions,
            self.config.collision_radius,
            self.config.near_miss_radius,
            num_candidates=int(pairs.shape[0]),
        )

    def detect_brute_force(self, state: DroneState) -> DetectionResult:
        """O(N^2) reference detector over all active pairs. For verification."""
        active = state.active_indices()
        n = active.size
        if n < 2:
            empty_p = np.empty((0, 2), dtype=np.int64)
            empty_d = np.empty(0, dtype=np.float64)
            return DetectionResult(empty_p, empty_d, empty_p, empty_d, 0)

        ii, jj = np.triu_indices(n, k=1)
        pairs = np.stack([active[ii], active[jj]], axis=1).astype(np.int64)
        return _classify(
            pairs,
            state.positions,
            self.config.collision_radius,
            self.config.near_miss_radius,
            num_candidates=int(pairs.shape[0]),
        )


class CollisionResolutionEngine:
    """Consumes collisions and updates affected drone state.

    Model: equal-mass elastic collision. The velocity components along the
    line of centres are exchanged, and overlapping drones are separated to the
    contact distance so they don't immediately re-collide. Pairs are processed
    sequentially; a drone appearing in several collisions is updated for each.
    Resolution never alters detection rules.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def resolve(self, state: DroneState, result: DetectionResult) -> int:
        pairs = result.collision_pairs
        if pairs.shape[0] == 0:
            return 0

        pos = state.positions
        vel = state.velocities
        r = self.config.collision_radius

        for i, j in pairs:
            pi = pos[i].astype(np.float64)
            pj = pos[j].astype(np.float64)
            n = pi - pj
            dist = float(np.linalg.norm(n))
            if dist == 0.0:
                # Degenerate exact overlap: pick a deterministic axis.
                n = np.array([1.0, 0.0, 0.0])
                dist = 0.0
            else:
                n = n / dist

            vi = vel[i].astype(np.float64)
            vj = vel[j].astype(np.float64)
            # Exchange the velocity component along the normal (equal masses).
            vi_n = float(np.dot(vi, n))
            vj_n = float(np.dot(vj, n))
            transfer = vi_n - vj_n
            vi = vi - transfer * n
            vj = vj + transfer * n
            vel[i] = vi.astype(np.float32)
            vel[j] = vj.astype(np.float32)

            # Separate so centre distance is exactly the contact distance.
            overlap = r - dist
            if overlap > 0:
                shift = 0.5 * overlap * n
                pos[i] = (pi + shift).astype(np.float32)
                pos[j] = (pj - shift).astype(np.float32)

        return int(pairs.shape[0])
