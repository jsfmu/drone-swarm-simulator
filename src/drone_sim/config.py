"""Simulation configuration.

All tunable Phase 1 parameters live here. The config is deliberately a plain
frozen dataclass so that a simulation is fully described (and therefore
reproducible) by ``SimulationConfig`` + the random seed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


class BoundaryMode(enum.Enum):
    """How drones behave when they reach a world boundary."""

    REFLECT = "reflect"  # clamp position to the wall and negate that velocity axis
    CLAMP = "clamp"      # clamp position to the wall and zero that velocity axis


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable description of a simulation.

    Attributes
    ----------
    num_drones:
        Number of drones to simulate.
    bounds_min / bounds_max:
        Inclusive XYZ world bounds. Drones are kept inside this box.
    collision_radius:
        Two drones collide when their centre distance is <= this value.
    near_miss_radius:
        Two drones are a near miss when collision_radius < distance <= this
        value. Must be >= collision_radius.
    cell_size:
        Edge length of a uniform spatial-hash cell. Must be >=
        near_miss_radius so that no interacting pair spans non-adjacent cells.
        Defaults to ``near_miss_radius`` when left as ``None``.
    dt:
        Fixed simulation time step.
    seed:
        Master random seed. Given the same config + seed the simulation is
        bit-for-bit reproducible.
    max_speed:
        Speed magnitude used when generating / perturbing velocities.
    boundary_mode:
        See :class:`BoundaryMode`.
    max_accel:
        Acceleration limit (units/tick) used by :class:`GoalDirectedMovementAlgorithm`
        and :class:`LocalAvoidanceMovementAlgorithm` when steering toward a goal.
    avoidance_max_accel:
        Acceleration limit for the avoidance correction applied by
        :class:`LocalAvoidanceMovementAlgorithm`. Defaults to ``max_accel`` when
        left as ``None``.
    avoidance_strength:
        Scales the magnitude of the avoidance correction (0 disables it).
    goal_tolerance:
        Distance within which a drone is considered to have reached its goal
        (see :class:`GoalDirectedMovementAlgorithm`).
    prediction_horizon:
        Look-ahead time (in the same units as ``dt``) used by
        :class:`TrajectoryPredictionService` to bound time-to-closest-approach.
    """

    num_drones: int
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    collision_radius: float = 1.0
    near_miss_radius: float = 2.0
    cell_size: float | None = None
    dt: float = 1.0
    seed: int = 0
    max_speed: float = 5.0
    boundary_mode: BoundaryMode = BoundaryMode.REFLECT
    max_accel: float = 3.0
    avoidance_max_accel: float | None = None
    avoidance_strength: float = 1.0
    goal_tolerance: float = 1.0
    prediction_horizon: float = 5.0

    # Derived, cached as plain fields (frozen dataclass -> use object.__setattr__)
    _bounds_min_arr: np.ndarray = field(init=False, repr=False, compare=False)
    _bounds_max_arr: np.ndarray = field(init=False, repr=False, compare=False)
    _effective_cell_size: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.num_drones < 0:
            raise ValueError("num_drones must be non-negative")
        if self.near_miss_radius < self.collision_radius:
            raise ValueError("near_miss_radius must be >= collision_radius")
        if self.collision_radius <= 0:
            raise ValueError("collision_radius must be positive")
        if self.max_accel <= 0:
            raise ValueError("max_accel must be positive")
        if self.avoidance_max_accel is not None and self.avoidance_max_accel <= 0:
            raise ValueError("avoidance_max_accel must be positive when set")
        if self.avoidance_strength < 0:
            raise ValueError("avoidance_strength must be non-negative")
        if self.goal_tolerance < 0:
            raise ValueError("goal_tolerance must be non-negative")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

        bmin = np.asarray(self.bounds_min, dtype=np.float64)
        bmax = np.asarray(self.bounds_max, dtype=np.float64)
        if not np.all(bmax > bmin):
            raise ValueError("bounds_max must be strictly greater than bounds_min on every axis")

        cell = self.near_miss_radius if self.cell_size is None else self.cell_size
        if cell < self.near_miss_radius:
            raise ValueError(
                f"cell_size ({cell}) must be >= near_miss_radius ({self.near_miss_radius}) "
                "so interacting pairs are never split across non-adjacent cells"
            )

        object.__setattr__(self, "_bounds_min_arr", bmin)
        object.__setattr__(self, "_bounds_max_arr", bmax)
        object.__setattr__(self, "_effective_cell_size", float(cell))

    @property
    def bounds_min_arr(self) -> np.ndarray:
        return self._bounds_min_arr

    @property
    def bounds_max_arr(self) -> np.ndarray:
        return self._bounds_max_arr

    @property
    def effective_cell_size(self) -> float:
        return self._effective_cell_size

    @property
    def interaction_radius(self) -> float:
        """The largest radius the spatial hash must not miss."""
        return self.near_miss_radius

    @property
    def effective_avoidance_max_accel(self) -> float:
        """``avoidance_max_accel`` if set, else ``max_accel``."""
        return self.max_accel if self.avoidance_max_accel is None else self.avoidance_max_accel
