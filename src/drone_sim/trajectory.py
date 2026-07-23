"""Vectorised trajectory prediction.

:class:`TrajectoryPredictionService` estimates, for every candidate pair, how
close two drones will get if they keep their current velocity, and classifies
the resulting *risk* — it never decides whether a collision actually
happened. That remains the sole authority of
:class:`~drone_sim.collisions.CollisionDetectionEngine`, which classifies
*actual* post-movement distances. This module and that one intentionally
never call each other.

Does not inherit from :class:`~drone_sim.movement.MovementAlgorithm` — it has
no velocities to update, only risk to estimate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .state import DroneState


class PredictedRisk(enum.IntEnum):
    """Priority-ordered outcome of a single candidate pair's prediction.

    Checked in this order: a pair whose predicted closest approach is inside
    ``collision_radius`` is a predicted collision even if the pair happens to
    also be flagged "not closing" (e.g. already overlapping while separating);
    only pairs that clear both distance thresholds fall through to the
    not-closing / currently-safe distinction.
    """

    PREDICTED_COLLISION = 0
    PREDICTED_NEAR_MISS = 1
    CURRENTLY_SAFE = 2
    NOT_CLOSING_OR_OUTSIDE_HORIZON = 3


@dataclass
class PredictionResult:
    """Arrays describing the predicted risk for every candidate pair.

    No per-pair Python objects — everything is a flat NumPy array, indexed in
    parallel with ``pairs``.
    """

    pairs: np.ndarray                     # (K, 2) int64, i<j
    time_to_closest_approach: np.ndarray   # (K,) float64, clipped to [0, prediction_horizon]
    predicted_separation: np.ndarray       # (K,) float64
    risk: np.ndarray                       # (K,) int8, see PredictedRisk

    @property
    def num_pairs(self) -> int:
        return int(self.pairs.shape[0])

    def count(self, risk: PredictedRisk) -> int:
        return int(np.count_nonzero(self.risk == risk))


class TrajectoryPredictionService:
    """Estimates time-to-closest-approach and predicted separation for
    candidate pairs, assuming each drone keeps its current velocity.

    Input is ``DroneState`` plus a ``(K, 2)`` candidate-pair array (typically
    from a *pre-movement* :class:`~drone_sim.spatial_hash.SpatialHashGrid`
    build — see the Phase 2 tick flow in README.md).
    """

    def __init__(self, config: SimulationConfig, prediction_horizon: float | None = None) -> None:
        self.config = config
        self.prediction_horizon = (
            config.prediction_horizon if prediction_horizon is None else prediction_horizon
        )

    def predict(self, state: DroneState, candidate_pairs: np.ndarray) -> PredictionResult:
        if candidate_pairs.shape[0] == 0:
            return PredictionResult(
                pairs=np.empty((0, 2), dtype=np.int64),
                time_to_closest_approach=np.empty(0, dtype=np.float64),
                predicted_separation=np.empty(0, dtype=np.float64),
                risk=np.empty(0, dtype=np.int8),
            )

        i = candidate_pairs[:, 0]
        j = candidate_pairs[:, 1]
        pos = state.positions.astype(np.float64)
        vel = state.velocities.astype(np.float64)

        relative_position = pos[j] - pos[i]
        relative_velocity = vel[j] - vel[i]

        denom = np.einsum("ij,ij->i", relative_velocity, relative_velocity)
        numer = -np.einsum("ij,ij->i", relative_position, relative_velocity)

        eps = 1e-12
        safe_denom = np.where(denom > eps, denom, 1.0)
        raw_ttca = np.where(denom > eps, numer / safe_denom, 0.0)

        horizon = self.prediction_horizon
        time_to_closest_approach = np.clip(raw_ttca, 0.0, horizon)

        predicted_delta = relative_position + relative_velocity * time_to_closest_approach[:, None]
        predicted_separation = np.sqrt(np.einsum("ij,ij->i", predicted_delta, predicted_delta))

        not_closing_or_outside = (denom <= eps) | (raw_ttca <= 0.0) | (raw_ttca > horizon)

        collision_r = self.config.collision_radius
        near_r = self.config.near_miss_radius

        risk = np.full(i.shape, PredictedRisk.CURRENTLY_SAFE, dtype=np.int8)
        risk[not_closing_or_outside] = PredictedRisk.NOT_CLOSING_OR_OUTSIDE_HORIZON
        # Distance thresholds take priority over the not-closing flag: a pair
        # already inside a risk band is at risk regardless of instantaneous
        # closing direction.
        risk[predicted_separation <= near_r] = PredictedRisk.PREDICTED_NEAR_MISS
        risk[predicted_separation <= collision_r] = PredictedRisk.PREDICTED_COLLISION

        return PredictionResult(
            pairs=candidate_pairs,
            time_to_closest_approach=time_to_closest_approach,
            predicted_separation=predicted_separation,
            risk=risk,
        )
