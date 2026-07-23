"""Simulation orchestration.

Assembles the pipeline and runs it on a fixed time step. For policies that
never need local awareness (Phase 1's Random/Scripted, and any run with no
context-requiring policy registered) the tick flow is unchanged from Phase 1:

    MovementSystem
      -> BoundaryManager
      -> SpatialHashGrid
      -> CollisionDetectionEngine
      -> CollisionResolutionEngine
      -> MetricsCollector

When at least one registered policy sets ``MovementAlgorithm.requires_context``
(currently only :class:`~drone_sim.movement.LocalAvoidanceMovementAlgorithm`),
:class:`SimulationEngine` additionally builds a *pre-movement* spatial hash,
runs :class:`~drone_sim.trajectory.TrajectoryPredictionService`, and builds a
:class:`~drone_sim.movement.MovementContext` before dispatching movement —
see the "Phase 2 tick flow" section of README.md. The post-movement spatial
hash rebuild and actual collision detection are never skipped or replaced by
the pre-movement one: prediction only estimates risk, it is never reused as
the authority for whether a collision actually happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .boundaries import BoundaryManager
from .collisions import (
    CollisionDetectionEngine,
    CollisionResolutionEngine,
    DetectionResult,
)
from .config import SimulationConfig
from .metrics import MetricsCollector, TickMetrics
from .movement import MovementSystem, NeighborFeatureBuilder
from .spatial_hash import SpatialHashGrid
from .state import World
from .trajectory import TrajectoryPredictionService


@dataclass
class SimulationClock:
    """Tracks tick count and simulated time for a fixed time step."""

    dt: float
    tick: int = 0
    time_s: float = 0.0

    def advance(self) -> None:
        self.tick += 1
        self.time_s += self.dt


class SimulationEngine:
    """Runs one ordered pipeline pass per tick over the world state.

    Accepts an optional pre-built ``movement`` system so callers (scenario
    validation, comparisons across policies) can swap in
    goal-directed/avoidance policies without duplicating the pipeline. When
    none of the registered policies require a MovementContext, this behaves
    exactly as Phase 1 did — no pre-movement grid, no prediction, no context
    construction, so Phase 1 timing/behavior is unchanged.
    """

    def __init__(self, config: SimulationConfig, movement: MovementSystem | None = None) -> None:
        self.config = config
        self.movement = movement if movement is not None else MovementSystem()
        self.boundaries = BoundaryManager()
        self.detector = CollisionDetectionEngine(config)
        self.resolver = CollisionResolutionEngine(config)
        self.grid = SpatialHashGrid(config)
        # A dedicated RNG stream for movement, seeded off the master seed so
        # runs stay reproducible while staying independent of state generation.
        self._rng = np.random.default_rng(config.seed + 1)

        self._needs_context = any(
            getattr(policy, "requires_context", False) for policy in self.movement.policies.values()
        )
        if self._needs_context:
            self._pre_grid = SpatialHashGrid(config)
            self._predictor = TrajectoryPredictionService(config)
            self._feature_builder = NeighborFeatureBuilder()
        else:
            self._pre_grid = None
            self._predictor = None
            self._feature_builder = None

    def step(self, world: World, clock: SimulationClock) -> tuple[DetectionResult, float]:
        state = world.state
        t0 = time.perf_counter()

        context = None
        if self._needs_context:
            # Pre-movement grid + prediction + context, used only to inform
            # movement policies this tick — never reused for actual detection.
            self._pre_grid.build(state.positions, state.active_indices())
            pre_pairs = self._pre_grid.candidate_pairs()
            prediction = self._predictor.predict(state, pre_pairs)
            context = self._feature_builder.build(state, prediction, state.goal_positions)

        # 1-2. Movement (batched policies, optionally context-aware) + fixed-step integration.
        self.movement.step(state, self._rng, self.config, clock.tick, context=context)
        # 2. Boundary constraints.
        self.boundaries.apply(world)
        # 3. Spatial hash (re)build for this tick — the actual detection authority.
        self.grid.build(state.positions, state.active_indices())
        # 4. Detection.
        result = self.detector.detect(state, self.grid)
        # 5. Resolution.
        self.resolver.resolve(state, result)

        tick_time = time.perf_counter() - t0
        return result, tick_time


class Simulation:
    """Top-level simulation: owns the world, engine, clock, metrics.

    ``movement`` and ``world`` are optional so scenario/validation code can
    inject a custom policy registry and/or a hand-built ``World`` (fixed
    positions, velocities, policy ids, goals) while reusing this exact
    pipeline — no duplicate simulation classes.
    """

    def __init__(
        self,
        config: SimulationConfig,
        movement: MovementSystem | None = None,
        world: World | None = None,
    ) -> None:
        self.config = config
        self.world = world if world is not None else World.create(config)
        self.clock = SimulationClock(dt=config.dt)
        self.engine = SimulationEngine(config, movement=movement)
        self.metrics = MetricsCollector()

    def step(self) -> DetectionResult:
        result, tick_time = self.engine.step(self.world, self.clock)
        self.metrics.record(
            TickMetrics(
                tick=self.clock.tick,
                tick_time_s=tick_time,
                candidate_pairs=result.num_candidate_pairs,
                collisions=result.num_collisions,
                near_misses=result.num_near_misses,
                active_drones=int(self.world.state.active_mask.sum()),
            )
        )
        self.clock.advance()
        return result

    def run(self, num_ticks: int) -> MetricsCollector:
        for _ in range(num_ticks):
            self.step()
        return self.metrics
