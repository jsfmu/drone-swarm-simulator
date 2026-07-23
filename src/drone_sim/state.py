"""Data-oriented drone state.

``Drone`` is a *logical* domain entity in the architecture. The performance
critical simulation stores drone state structure-of-arrays with NumPy, so we
never allocate 100,000 Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass
class DroneState:
    """Structure-of-arrays state for all drones.

    All arrays are indexed by drone id ``0 .. N-1``.
    """

    positions: np.ndarray            # (N, 3) float32
    velocities: np.ndarray           # (N, 3) float32
    active_mask: np.ndarray          # (N,)   bool
    movement_policy_ids: np.ndarray  # (N,)   int32
    # Optional (N, 3) float32 destinations for goal-directed / avoidance
    # policies. ``None`` for Phase 1 scenarios that never assign goals.
    # Destination initialization belongs in scenario generation, never inside
    # MovementSystem.step().
    goal_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        assert self.positions.shape == (n, 3)
        assert self.velocities.shape == (n, 3)
        assert self.active_mask.shape == (n,)
        assert self.movement_policy_ids.shape == (n,)
        if self.goal_positions is not None:
            assert self.goal_positions.shape == (n, 3)

    @property
    def num_drones(self) -> int:
        return self.positions.shape[0]

    def active_indices(self) -> np.ndarray:
        return np.nonzero(self.active_mask)[0]

    @classmethod
    def generate(cls, config: SimulationConfig) -> "DroneState":
        """Deterministically generate initial state from ``config.seed``.

        Positions are uniform inside the world box; velocities are uniform in
        a cube of half-width ``max_speed``. Using a single seeded Generator
        makes generation fully reproducible.
        """
        n = config.num_drones
        rng = np.random.default_rng(config.seed)

        lo = config.bounds_min_arr
        hi = config.bounds_max_arr

        positions = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
        velocities = rng.uniform(
            -config.max_speed, config.max_speed, size=(n, 3)
        ).astype(np.float32)
        active_mask = np.ones(n, dtype=bool)
        movement_policy_ids = np.zeros(n, dtype=np.int32)

        return cls(
            positions=positions,
            velocities=velocities,
            active_mask=active_mask,
            movement_policy_ids=movement_policy_ids,
        )


@dataclass
class World:
    """The bounded XYZ world plus the drone state it contains.

    ``World`` owns the drone state (SoA) and exposes the collision / near-miss
    radii. In the target architecture ``World`` aggregates ``Drone`` entities;
    here those entities are rows in the SoA arrays.
    """

    config: SimulationConfig
    state: DroneState

    @classmethod
    def create(cls, config: SimulationConfig) -> "World":
        return cls(config=config, state=DroneState.generate(config))

    @property
    def bounds_min(self) -> np.ndarray:
        return self.config.bounds_min_arr

    @property
    def bounds_max(self) -> np.ndarray:
        return self.config.bounds_max_arr

    @property
    def collision_radius(self) -> float:
        return self.config.collision_radius

    @property
    def near_miss_radius(self) -> float:
        return self.config.near_miss_radius
