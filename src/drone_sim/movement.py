"""Movement policies and the movement system.

Movement algorithms are interchangeable *batched* policies. Each policy
operates on a subset of drones (selected by ``movement_policy_ids``) using
vectorised NumPy, never a per-drone Python loop.

Phase 1 ships two baseline policies:

* :class:`RandomMovementAlgorithm` — reproducible random walk (baseline).
* :class:`ScriptedMovementAlgorithm` — constant velocity (deterministic).

AI / avoidance policies are Phase 2 and intentionally omitted.
"""

from __future__ import annotations

import abc
from typing import Dict

import numpy as np

from .config import SimulationConfig
from .state import DroneState


class MovementAlgorithm(abc.ABC):
    """Interface for a batched movement policy.

    A policy updates velocities in place for the given subset of drones. The
    :class:`MovementSystem` is responsible for integrating positions, so a
    policy only decides *how velocities change*.
    """

    #: Stable integer id used in ``DroneState.movement_policy_ids``.
    policy_id: int

    @abc.abstractmethod
    def update_velocities(
        self,
        state: DroneState,
        idx: np.ndarray,
        rng: np.random.Generator,
        config: SimulationConfig,
        tick: int,
    ) -> None:
        """Update ``state.velocities[idx]`` in place (vectorised)."""


class RandomMovementAlgorithm(MovementAlgorithm):
    """Reproducible random walk: apply a small random acceleration each tick.

    Velocities are clipped to ``+/- max_speed`` per axis so drones stay well
    behaved. Reproducibility comes from the shared seeded ``rng``.
    """

    policy_id = 0

    def __init__(self, accel_scale: float = 0.5) -> None:
        self.accel_scale = accel_scale

    def update_velocities(self, state, idx, rng, config, tick) -> None:
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
    function of the (fixed) initial velocity and the tick count.
    """

    policy_id = 1

    def update_velocities(self, state, idx, rng, config, tick) -> None:
        # Constant velocity: nothing to update.
        return


class MovementSystem:
    """Runs the active movement policies then integrates positions.

    The system groups drones by ``movement_policy_ids`` and dispatches each
    group to its policy as a single batch.
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
    ) -> None:
        active = state.active_mask
        policy_ids = state.movement_policy_ids

        # 1. Update velocities, one batched call per policy present.
        for pid, policy in self.policies.items():
            idx = np.nonzero(active & (policy_ids == pid))[0]
            if idx.size:
                policy.update_velocities(state, idx, rng, config, tick)

        # 2. Integrate positions for all active drones (fixed time step).
        act = np.nonzero(active)[0]
        if act.size:
            state.positions[act] += state.velocities[act] * config.dt
