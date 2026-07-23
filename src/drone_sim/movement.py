"""Movement policies and the movement system.

Movement algorithms are interchangeable *batched* policies. Each policy
operates on a subset of drones (selected by ``movement_policy_ids``) using
vectorised NumPy, never a per-drone Python loop.

Phase 1 shipped two baseline policies:

* :class:`RandomMovementAlgorithm` — reproducible random walk (baseline).
* :class:`ScriptedMovementAlgorithm` — constant velocity (deterministic).

Phase 2 adds goal-seeking and local avoidance:

* :class:`GoalDirectedMovementAlgorithm` — steers toward ``DroneState.goal_positions``
  with acceleration/speed limits and no avoidance. Serves as the no-avoidance
  comparison baseline for :class:`LocalAvoidanceMovementAlgorithm`.
* :class:`LocalAvoidanceMovementAlgorithm` — goal-directed movement plus a
  bounded correction away from the most urgent predicted threat, read from a
  :class:`MovementContext` built by :class:`NeighborFeatureBuilder`.

``NeuralAvoidanceMovementAlgorithm`` is planned future work (see the
``Remaining prerequisites`` section of the Phase 2 handoff in README.md) and
is intentionally not implemented here.

``DroneState`` never invokes a ``MovementAlgorithm`` and never holds policy
objects — only integer ``movement_policy_ids``. ``MovementSystem`` reads those
ids, groups drones into batches, and dispatches each batch to its policy.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import SimulationConfig
from .state import DroneState
from .trajectory import PredictionResult


@dataclass(frozen=True)
class MovementContext:
    """Read-only per-tick local awareness for context-aware movement policies.

    Built once per tick (by :class:`NeighborFeatureBuilder`) from the
    pre-movement spatial hash and trajectory predictions, then handed to every
    policy dispatched that tick. Arrays are indexed by the *full* drone id
    ``0 .. N-1``; a policy indexes into them with its own ``idx`` batch.
    """

    neighbor_features: np.ndarray       # (N, F) float32 — see NeighborFeatureBuilder.FEATURE_NAMES
    neighbor_valid_mask: np.ndarray     # (N,) bool — False where a drone has no candidate pair
    goal_vectors: np.ndarray | None = None  # (N, 3) float32, goal_position - position; None if no goals


class MovementAlgorithm(abc.ABC):
    """Interface for a batched movement policy.

    A policy updates velocities in place for the given subset of drones. The
    :class:`MovementSystem` is responsible for integrating positions, so a
    policy only decides *how velocities change*.
    """

    #: Stable integer id used in ``DroneState.movement_policy_ids``.
    policy_id: int

    #: Set True on policies that cannot function without a MovementContext
    #: (see LocalAvoidanceMovementAlgorithm). MovementSystem/SimulationEngine
    #: use this to decide whether the pre-movement prediction pipeline is
    #: worth building at all.
    requires_context: bool = False

    @abc.abstractmethod
    def update_velocities(
        self,
        state: DroneState,
        idx: np.ndarray,
        rng: np.random.Generator,
        config: SimulationConfig,
        tick: int,
        context: MovementContext | None = None,
    ) -> None:
        """Update ``state.velocities[idx]`` in place (vectorised)."""


class RandomMovementAlgorithm(MovementAlgorithm):
    """Reproducible random walk: apply a small random acceleration each tick.

    Velocities are clipped to ``+/- max_speed`` per axis so drones stay well
    behaved. Reproducibility comes from the shared seeded ``rng``. Ignores
    ``context`` — it has no notion of neighbours or goals.
    """

    policy_id = 0

    def __init__(self, accel_scale: float = 0.5) -> None:
        self.accel_scale = accel_scale

    def update_velocities(self, state, idx, rng, config, tick, context=None) -> None:
        if idx.size == 0:
            return
        accel = rng.uniform(
            -self.accel_scale, self.accel_scale, size=(idx.size, 3)
        ).astype(np.float32)
        v = state.velocities[idx] + accel
        np.clip(v, -config.max_speed, config.max_speed, out=v)
        state.velocities[idx] = v


class ScriptedMovementAlgorithm(MovementAlgorithm):
    """Deterministic constant-velocity movement (no velocity change).

    Useful for deterministic scenarios and tests: positions are a pure linear
    function of the (fixed) initial velocity and the tick count. Ignores
    ``context``.
    """

    policy_id = 1

    def update_velocities(self, state, idx, rng, config, tick, context=None) -> None:
        # Constant velocity: nothing to update.
        return


class GoalDirectedMovementAlgorithm(MovementAlgorithm):
    """Steers each selected drone toward ``state.goal_positions`` with no
    collision avoidance.

    Serves as the "no-avoidance" comparison baseline against
    :class:`LocalAvoidanceMovementAlgorithm` under identical conditions.
    Reads destinations directly from ``state.goal_positions`` (set during
    scenario generation) rather than from ``context.goal_vectors`` — it needs
    no neighbour awareness, so it works standalone without a MovementContext.

    Acceleration is capped at ``config.max_accel`` and resulting speed at
    ``config.max_speed``. A drone within ``config.goal_tolerance`` of its goal
    is considered arrived and its velocity is zeroed (see the
    "stationary-drone percentage" validation metric).
    """

    policy_id = 2

    def update_velocities(self, state, idx, rng, config, tick, context=None) -> None:
        if idx.size == 0:
            return
        if state.goal_positions is None:
            raise ValueError(
                "GoalDirectedMovementAlgorithm requires DroneState.goal_positions "
                "to be set (during scenario generation) — it does not generate "
                "goals itself."
            )

        pos = state.positions[idx].astype(np.float64)
        vel = state.velocities[idx].astype(np.float64)
        goals = state.goal_positions[idx].astype(np.float64)

        to_goal = goals - pos
        dist = np.linalg.norm(to_goal, axis=1)
        arrived = dist <= config.goal_tolerance

        direction = np.zeros_like(to_goal)
        moving = ~arrived & (dist > 1e-9)
        direction[moving] = to_goal[moving] / dist[moving, None]

        desired_speed = np.minimum(config.max_speed, dist / max(config.dt, 1e-9))
        desired_vel = direction * desired_speed[:, None]
        desired_vel[arrived] = 0.0

        new_vel = _apply_accel_and_speed_limits(
            vel, desired_vel, max_accel=config.max_accel, max_speed=config.max_speed
        )
        new_vel[arrived] = 0.0

        state.velocities[idx] = new_vel.astype(np.float32)


class LocalAvoidanceMovementAlgorithm(MovementAlgorithm):
    """Goal-directed movement plus a bounded correction away from the single
    most urgent predicted threat.

    Requires a :class:`MovementContext` (built by :class:`NeighborFeatureBuilder`
    from trajectory predictions) and fails clearly without one. Deterministic:
    uses no randomness, so identical state + config always produce identical
    output.
    """

    policy_id = 3
    requires_context = True

    def __init__(self, goal_policy: GoalDirectedMovementAlgorithm | None = None) -> None:
        self._goal_policy = goal_policy or GoalDirectedMovementAlgorithm()

    def update_velocities(self, state, idx, rng, config, tick, context=None) -> None:
        if idx.size == 0:
            return
        if context is None:
            raise ValueError(
                "LocalAvoidanceMovementAlgorithm requires a MovementContext — "
                "build one with NeighborFeatureBuilder and pass it through "
                "MovementSystem.step(..., context=...)."
            )

        # 1. Start from the same goal-directed desired velocity (no avoidance).
        self._goal_policy.update_velocities(state, idx, rng, config, tick, context)
        desired_vel = state.velocities[idx].astype(np.float64).copy()

        valid = context.neighbor_valid_mask[idx]
        if not valid.any():
            return  # nothing nearby to avoid; goal-directed velocity stands.

        feats = context.neighbor_features[idx].astype(np.float64)
        rel_pos = feats[:, 0:3]   # from self toward the most urgent threat
        separation = feats[:, 6]
        ttca = feats[:, 8]
        predicted_separation = feats[:, 9]

        away = np.zeros_like(rel_pos)
        nonzero = separation > 1e-9
        away[nonzero] = -rel_pos[nonzero] / separation[nonzero, None]

        horizon = max(config.prediction_horizon, 1e-9)
        near_r = max(config.near_miss_radius, 1e-9)
        # dist_urgency is the gate: TrajectoryPredictionService guarantees
        # predicted_separation > near_miss_radius whenever a pair is NOT
        # currently at risk (CURRENTLY_SAFE / NOT_CLOSING_OR_OUTSIDE_HORIZON),
        # so dist_urgency is provably 0 for those — including the "not
        # closing" fallback where ttca is clamped to 0 and would otherwise
        # look maximally imminent. time_urgency only modulates the strength
        # of an already-real threat; it can never manufacture one.
        dist_urgency = np.clip(1.0 - predicted_separation / near_r, 0.0, 1.0)
        time_urgency = np.clip(1.0 - ttca / horizon, 0.0, 1.0)
        urgency = dist_urgency * (0.5 + 0.5 * time_urgency)
        urgency[~valid] = 0.0

        max_avoid_accel = config.effective_avoidance_max_accel
        correction = away * (urgency * config.avoidance_strength * max_avoid_accel)[:, None]
        correction_mag = np.linalg.norm(correction, axis=1)
        over = correction_mag > max_avoid_accel
        if over.any():
            correction[over] *= (max_avoid_accel / np.maximum(correction_mag[over], 1e-9))[:, None]

        new_vel = desired_vel + correction
        speed = np.linalg.norm(new_vel, axis=1)
        over_speed = speed > config.max_speed
        if over_speed.any():
            new_vel[over_speed] *= (config.max_speed / np.maximum(speed[over_speed], 1e-9))[:, None]

        state.velocities[idx] = new_vel.astype(np.float32)


def _apply_accel_and_speed_limits(
    current_vel: np.ndarray, desired_vel: np.ndarray, *, max_accel: float, max_speed: float
) -> np.ndarray:
    """Move ``current_vel`` toward ``desired_vel`` under an acceleration cap,
    then clip the result to ``max_speed``. Fully vectorised over the batch.
    """
    accel = desired_vel - current_vel
    accel_mag = np.linalg.norm(accel, axis=1)
    over_accel = accel_mag > max_accel
    if over_accel.any():
        accel[over_accel] *= (max_accel / np.maximum(accel_mag[over_accel], 1e-9))[:, None]

    new_vel = current_vel + accel
    speed = np.linalg.norm(new_vel, axis=1)
    over_speed = speed > max_speed
    if over_speed.any():
        new_vel[over_speed] *= (max_speed / np.maximum(speed[over_speed], 1e-9))[:, None]
    return new_vel


class NeighborFeatureBuilder:
    """Vectorised construction of :class:`MovementContext` from a
    :class:`~drone_sim.trajectory.PredictionResult`.

    For every drone, picks the single most urgent candidate pair it
    participates in (smallest predicted separation) using the same
    stable-sort "first occurrence per group" trick already used by
    :class:`~drone_sim.spatial_hash.SpatialHashGrid` for cell grouping — no
    per-drone Python loop.
    """

    #: rel_pos(3) + rel_vel(3) + separation(1) + closing_speed(1)
    #: + time_to_closest_approach(1) + predicted_separation(1) + neighbor_count(1)
    NUM_FEATURES = 11
    FEATURE_NAMES = (
        "rel_pos_x", "rel_pos_y", "rel_pos_z",
        "rel_vel_x", "rel_vel_y", "rel_vel_z",
        "separation", "closing_speed",
        "time_to_closest_approach", "predicted_separation",
        "neighbor_count",
    )

    def build(
        self,
        state: DroneState,
        prediction: PredictionResult,
        goal_positions: np.ndarray | None = None,
    ) -> MovementContext:
        n = state.num_drones
        features = np.zeros((n, self.NUM_FEATURES), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        pairs = prediction.pairs
        if pairs.shape[0]:
            i = pairs[:, 0]
            j = pairs[:, 1]
            pos = state.positions.astype(np.float64)
            vel = state.velocities.astype(np.float64)

            rel_pos_ij = pos[j] - pos[i]
            rel_vel_ij = vel[j] - vel[i]
            current_sep = np.linalg.norm(rel_pos_ij, axis=1)
            safe_sep = np.maximum(current_sep, 1e-9)
            closing_speed = -np.einsum("ij,ij->i", rel_pos_ij, rel_vel_ij) / safe_sep

            # Doubled, directed rows: one from i's point of view, one from j's
            # (mirrored). This lets every drone see the pair "from its side".
            self_idx = np.concatenate([i, j])
            rel_pos_self = np.concatenate([rel_pos_ij, -rel_pos_ij])
            rel_vel_self = np.concatenate([rel_vel_ij, -rel_vel_ij])
            sep_self = np.concatenate([current_sep, current_sep])
            closing_self = np.concatenate([closing_speed, closing_speed])
            ttca_self = np.concatenate([prediction.time_to_closest_approach, prediction.time_to_closest_approach])
            predsep_self = np.concatenate([prediction.predicted_separation, prediction.predicted_separation])

            # Most urgent = smallest predicted separation. lexsort's primary
            # key is the LAST argument, so this groups by self_idx (ascending)
            # and, within each group, orders by ascending urgency.
            order = np.lexsort((predsep_self, self_idx))
            self_sorted = self_idx[order]
            group_mask = np.empty(self_sorted.shape[0], dtype=bool)
            group_mask[0] = True
            group_mask[1:] = self_sorted[1:] != self_sorted[:-1]
            group_start = np.nonzero(group_mask)[0]
            best_rows = order[group_start]
            ids = self_sorted[group_start]

            features[ids, 0:3] = rel_pos_self[best_rows].astype(np.float32)
            features[ids, 3:6] = rel_vel_self[best_rows].astype(np.float32)
            features[ids, 6] = sep_self[best_rows].astype(np.float32)
            features[ids, 7] = closing_self[best_rows].astype(np.float32)
            features[ids, 8] = ttca_self[best_rows].astype(np.float32)
            features[ids, 9] = predsep_self[best_rows].astype(np.float32)
            valid[ids] = True

            counts = np.bincount(self_idx, minlength=n)
            features[:, 10] = counts.astype(np.float32)

        goal_vectors = None
        if goal_positions is not None:
            goal_vectors = (goal_positions.astype(np.float32) - state.positions.astype(np.float32))

        return MovementContext(
            neighbor_features=features,
            neighbor_valid_mask=valid,
            goal_vectors=goal_vectors,
        )


class MovementSystem:
    """Runs the active movement policies then integrates positions.

    The system groups drones by ``movement_policy_ids`` and dispatches each
    group to its policy as a single batch. A loop over the (small) policy
    registry is fine; a loop over individual drones is not — and never
    happens here.
    """

    def __init__(self, policies: Dict[int, MovementAlgorithm] | None = None) -> None:
        if policies is None:
            policies = {
                RandomMovementAlgorithm.policy_id: RandomMovementAlgorithm(),
                ScriptedMovementAlgorithm.policy_id: ScriptedMovementAlgorithm(),
            }
        self.policies = policies

    def step(
        self,
        state: DroneState,
        rng: np.random.Generator,
        config: SimulationConfig,
        tick: int,
        context: MovementContext | None = None,
    ) -> None:
        active = state.active_mask
        policy_ids = state.movement_policy_ids

        active_ids = np.unique(policy_ids[active]) if active.any() else np.empty(0, dtype=policy_ids.dtype)
        unknown = set(int(p) for p in active_ids) - set(self.policies)
        if unknown:
            raise ValueError(f"unknown movement_policy_ids for active drones: {sorted(unknown)}")

        # 1. Update velocities, one batched call per policy present.
        for pid, policy in self.policies.items():
            idx = np.nonzero(active & (policy_ids == pid))[0]
            if idx.size:
                policy.update_velocities(state, idx, rng, config, tick, context)

        # 2. Integrate positions for all active drones (fixed time step).
        act = np.nonzero(active)[0]
        if act.size:
            state.positions[act] += state.velocities[act] * config.dt
