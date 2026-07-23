"""Deterministic, seeded scenario factory for Phase 2 validation.

Produces fully-formed :class:`~drone_sim.state.World` instances (positions,
velocities, policy ids, and — where useful for goal-directed/avoidance
comparisons — ``goal_positions``) for controlled collision, near-miss, and
avoidance testing.

Scenario generation is intentionally separate from
:class:`~drone_sim.movement.MovementAlgorithm`: policies only decide how
velocities change each tick; scenarios decide starting conditions and (fixed)
destinations. Ground truth (expected collision/near-miss pairs) is
precomputed analytically here, not discovered by running the simulation —
:class:`~drone_sim.collisions.CollisionDetectionEngine` remains the sole
authority for whether a collision actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import SimulationConfig
from .movement import ScriptedMovementAlgorithm
from .state import DroneState, World


@dataclass(frozen=True)
class ScenarioResult:
    """A deterministic scenario: the world plus ground truth for validation."""

    name: str
    world: World
    description: str = ""
    expected_collision_pairs: Tuple[Tuple[int, int], ...] = ()
    expected_near_miss_pairs: Tuple[Tuple[int, int], ...] = ()


def _make_world(
    config: SimulationConfig,
    positions,
    velocities,
    policy_ids=None,
    goal_positions=None,
) -> World:
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    n = positions.shape[0]
    state = DroneState(
        positions=positions,
        velocities=velocities,
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=(
            np.full(n, ScriptedMovementAlgorithm.policy_id, dtype=np.int32)
            if policy_ids is None
            else np.asarray(policy_ids, dtype=np.int32)
        ),
        goal_positions=None if goal_positions is None else np.asarray(goal_positions, dtype=np.float32),
    )
    return World(config=config, state=state)


# Ticks-to-closest-approach used to lay out timed scenarios. Multiplying by
# config.dt (rather than hardcoding a distance) keeps the geometry exact
# regardless of the configured time step, so the precomputed ground truth
# always lands on a real simulated tick.
_CLOSE_TICKS = 5


def _closing_speed(config: SimulationConfig, desired: float = 1.0) -> float:
    return min(desired, config.max_speed)


def head_on_collision(config: SimulationConfig) -> ScenarioResult:
    """Two drones on a guaranteed head-on collision course.

    Meet exactly ``_CLOSE_TICKS`` ticks after tick 0, at distance 0.
    """
    speed = _closing_speed(config)
    gap = 2.0 * speed * _CLOSE_TICKS * config.dt
    positions = [[0.0, 0.0, 0.0], [gap, 0.0, 0.0]]
    velocities = [[speed, 0.0, 0.0], [-speed, 0.0, 0.0]]
    world = _make_world(config, positions, velocities)
    return ScenarioResult(
        name="head_on_collision",
        world=world,
        description="Two scripted drones closing head-on along x; guaranteed to collide.",
        expected_collision_pairs=((0, 1),),
    )


def crossing_paths(config: SimulationConfig) -> ScenarioResult:
    """Two drones on perpendicular paths that intersect at the same tick."""
    speed = _closing_speed(config)
    d = speed * _CLOSE_TICKS * config.dt
    positions = [[0.0, d, 0.0], [d, 0.0, 0.0]]
    velocities = [[speed, 0.0, 0.0], [0.0, speed, 0.0]]
    world = _make_world(config, positions, velocities)
    return ScenarioResult(
        name="crossing_paths",
        world=world,
        description="Perpendicular scripted paths meeting at the same point and tick.",
        expected_collision_pairs=((0, 1),),
    )


def near_miss(config: SimulationConfig) -> ScenarioResult:
    """Two drones passing with closest approach just outside the collision
    radius (well inside the near-miss band, close to its inner edge)."""
    speed = _closing_speed(config)
    gap = 2.0 * speed * _CLOSE_TICKS * config.dt
    band = config.near_miss_radius - config.collision_radius
    offset = config.collision_radius + 0.15 * band
    positions = [[0.0, 0.0, 0.0], [gap, offset, 0.0]]
    velocities = [[speed, 0.0, 0.0], [-speed, 0.0, 0.0]]
    world = _make_world(config, positions, velocities)
    return ScenarioResult(
        name="near_miss",
        world=world,
        description=(
            "Closing paths offset in y so the closest approach lands just "
            "outside collision_radius, inside the near-miss band."
        ),
        expected_near_miss_pairs=((0, 1),),
    )


def parallel_safe(config: SimulationConfig) -> ScenarioResult:
    """Two drones moving in parallel; separation never changes."""
    speed = _closing_speed(config)
    far = config.near_miss_radius * 5.0
    positions = [[0.0, 0.0, 0.0], [0.0, far, 0.0]]
    velocities = [[speed, 0.0, 0.0], [speed, 0.0, 0.0]]
    world = _make_world(config, positions, velocities)
    return ScenarioResult(
        name="parallel_safe",
        world=world,
        description="Same-velocity parallel drones; separation is constant and always safe.",
    )


def stationary_obstacle(config: SimulationConfig) -> ScenarioResult:
    """One stationary drone; a second flies directly into it."""
    speed = _closing_speed(config)
    gap = speed * _CLOSE_TICKS * config.dt
    positions = [[0.0, 0.0, 0.0], [gap, 0.0, 0.0]]
    velocities = [[0.0, 0.0, 0.0], [-speed, 0.0, 0.0]]
    world = _make_world(config, positions, velocities)
    return ScenarioResult(
        name="stationary_obstacle",
        world=world,
        description="Drone 0 stationary; drone 1 flies directly into it.",
        expected_collision_pairs=((0, 1),),
    )


def converging_group(config: SimulationConfig, num_drones: int = 6) -> ScenarioResult:
    """Multiple drones converging on the world's center from a ring, all
    arriving at (approximately) the same tick."""
    lo = config.bounds_min_arr
    hi = config.bounds_max_arr
    center = (lo + hi) / 2.0
    speed = _closing_speed(config)
    radius = speed * _CLOSE_TICKS * config.dt
    angles = np.linspace(0.0, 2.0 * np.pi, num_drones, endpoint=False)
    positions = np.stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(num_drones, center[2]),
        ],
        axis=1,
    )
    directions = center[None, :] - positions
    norms = np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
    velocities = (directions / norms) * speed
    world = _make_world(config, positions, velocities)
    pairs = tuple((i, j) for i in range(num_drones) for j in range(i + 1, num_drones))
    return ScenarioResult(
        name="converging_group",
        world=world,
        description=f"{num_drones} drones converging on the world center from a ring, all arriving together.",
        expected_collision_pairs=pairs,
    )


def rare_collision_background(
    config: SimulationConfig,
    num_background: int = 200,
    num_injected_collisions: int = 2,
    num_injected_near_misses: int = 2,
    seed: int | None = None,
) -> ScenarioResult:
    """Many safe background drones plus a small, known number of injected
    collision courses and near misses — the rare-collision validation scenario.

    Reproducible from ``seed`` (defaults to ``config.seed``). Background
    drones get simple reflective goals (start position mirrored through the
    world center) so this scenario can also drive the
    :class:`~drone_sim.validation.CollisionRateValidator` policy comparison;
    a ``ScriptedMovementAlgorithm`` run simply ignores ``goal_positions``.
    """
    effective_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(effective_seed)

    lo = config.bounds_min_arr
    hi = config.bounds_max_arr
    center = (lo + hi) / 2.0

    bg_positions = rng.uniform(lo, hi, size=(num_background, 3))
    bg_speed = min(config.max_speed * 0.5, 1.0)
    bg_directions = rng.uniform(-1.0, 1.0, size=(num_background, 3))
    bg_directions /= np.maximum(np.linalg.norm(bg_directions, axis=1, keepdims=True), 1e-9)
    bg_velocities = bg_directions * bg_speed
    # Reflection through the center: a distant, reproducible destination that
    # generally routes each background drone back across the busy middle.
    bg_goals = 2.0 * center[None, :] - bg_positions

    speed = _closing_speed(config)
    gap = 2.0 * speed * _CLOSE_TICKS * config.dt
    band = config.near_miss_radius - config.collision_radius
    offset = config.collision_radius + 0.15 * band

    injected_positions: List[List[float]] = []
    injected_velocities: List[List[float]] = []
    injected_goals: List[List[float]] = []
    expected_collisions: List[Tuple[int, int]] = []
    expected_near_misses: List[Tuple[int, int]] = []

    def _add_pair(base: np.ndarray, lateral: float) -> Tuple[int, int]:
        idx0 = num_background + len(injected_positions)
        injected_positions.append(base.tolist())
        injected_positions.append((base + np.array([gap, lateral, 0.0])).tolist())
        injected_velocities.append([speed, 0.0, 0.0])
        injected_velocities.append([-speed, 0.0, 0.0])
        # "Continue straight through" goals: far beyond the meeting point on
        # each drone's original heading. This is the no-avoidance continuation
        # used to compare GoalDirectedMovementAlgorithm/LocalAvoidanceMovementAlgorithm
        # against the ScriptedMovementAlgorithm ground truth on the same pair.
        far = gap * 4.0
        injected_goals.append((base + np.array([far, 0.0, 0.0])).tolist())
        injected_goals.append((base + np.array([gap - far, lateral, 0.0])).tolist())
        return (idx0, idx0 + 1)

    for k in range(num_injected_collisions):
        base = center + np.array([0.0, 20.0 * (k + 1), 0.0])
        expected_collisions.append(_add_pair(base, lateral=0.0))

    for k in range(num_injected_near_misses):
        base = center + np.array([40.0, 20.0 * (k + 1), 0.0])
        expected_near_misses.append(_add_pair(base, lateral=offset))

    injected_positions_arr = np.asarray(injected_positions).reshape(-1, 3)
    injected_velocities_arr = np.asarray(injected_velocities).reshape(-1, 3)
    injected_goals_arr = np.asarray(injected_goals).reshape(-1, 3)

    positions = np.concatenate([bg_positions, injected_positions_arr], axis=0)
    velocities = np.concatenate([bg_velocities, injected_velocities_arr], axis=0)
    goals = np.concatenate([bg_goals, injected_goals_arr], axis=0)
    n = positions.shape[0]
    policy_ids = np.full(n, ScriptedMovementAlgorithm.policy_id, dtype=np.int32)

    world = _make_world(config, positions, velocities, policy_ids=policy_ids, goal_positions=goals)
    return ScenarioResult(
        name="rare_collision_background",
        world=world,
        description=(
            f"{num_background} safe background drones (with reflective goals) + "
            f"{num_injected_collisions} injected collision courses + "
            f"{num_injected_near_misses} injected near misses (with straight-through goals)."
        ),
        expected_collision_pairs=tuple(expected_collisions),
        expected_near_miss_pairs=tuple(expected_near_misses),
    )


SCENARIOS = {
    "head_on_collision": head_on_collision,
    "crossing_paths": crossing_paths,
    "near_miss": near_miss,
    "parallel_safe": parallel_safe,
    "stationary_obstacle": stationary_obstacle,
    "converging_group": converging_group,
    "rare_collision_background": rare_collision_background,
}
