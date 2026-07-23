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


@dataclass
class TickProfile:
    """Opt-in, per-tick stage-timing result object for :meth:`SimulationEngine.step`.

    Disabled by default: passing ``profile=None`` (every existing call site)
    means ``step()`` takes none of the branches below and pays no timing
    overhead beyond a handful of cheap ``is not None`` checks. Pass a fresh
    ``TickProfile()`` instance to have ``step()`` fill it in with
    nanosecond-resolution stage timings (``time.perf_counter_ns()``).

    When no registered policy sets ``requires_context`` (e.g. a
    ``GoalDirectedMovementAlgorithm``-only run), the four context-building
    stages are never executed — they are left at their default ``0`` and
    ``context_stages_skipped`` is set ``True``, so a skipped stage is never
    confused with a stage that failed to record a measurement.

    Measuring ``post_pairs_ns`` and ``detection_ns`` separately costs one
    extra (otherwise redundant) call to ``grid.candidate_pairs()`` compared to
    the non-profiled path, which calls ``CollisionDetectionEngine.detect()``
    without precomputed pairs and lets it compute them internally exactly
    once. This is the only behavioral difference profiling introduces, it is
    pure overhead confined to the profiled run, and it does not change any
    detection result (candidate-pair generation is a pure, deterministic
    function of the already-built grid).
    """

    pre_grid_ns: int = 0
    pre_pairs_ns: int = 0
    prediction_ns: int = 0
    context_ns: int = 0
    movement_ns: int = 0
    boundary_ns: int = 0
    post_grid_ns: int = 0
    post_pairs_ns: int = 0
    detection_ns: int = 0
    resolution_ns: int = 0
    total_ns: int = 0
    context_stages_skipped: bool = False


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

    @property
    def needs_context(self) -> bool:
        """True if any registered policy requires a MovementContext (see
        ``MovementAlgorithm.requires_context``) — i.e. whether this engine
        runs the extended Phase 2 tick flow or the plain Phase 1 one."""
        return self._needs_context

    def step(
        self, world: World, clock: SimulationClock, profile: TickProfile | None = None
    ) -> tuple[DetectionResult, float]:
        state = world.state
        t0 = time.perf_counter()

        context = None
        if self._needs_context:
            if profile is not None:
                s0 = time.perf_counter_ns()
                self._pre_grid.build(state.positions, state.active_indices())
                s1 = time.perf_counter_ns()
                pre_pairs = self._pre_grid.candidate_pairs()
                s2 = time.perf_counter_ns()
                prediction = self._predictor.predict(state, pre_pairs)
                s3 = time.perf_counter_ns()
                context = self._feature_builder.build(state, prediction, state.goal_positions)
                s4 = time.perf_counter_ns()
                profile.pre_grid_ns = s1 - s0
                profile.pre_pairs_ns = s2 - s1
                profile.prediction_ns = s3 - s2
                profile.context_ns = s4 - s3
            else:
                # Pre-movement grid + prediction + context, used only to inform
                # movement policies this tick — never reused for actual detection.
                self._pre_grid.build(state.positions, state.active_indices())
                pre_pairs = self._pre_grid.candidate_pairs()
                prediction = self._predictor.predict(state, pre_pairs)
                context = self._feature_builder.build(state, prediction, state.goal_positions)
        elif profile is not None:
            profile.context_stages_skipped = True

        if profile is None:
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
        else:
            s5 = time.perf_counter_ns()
            self.movement.step(state, self._rng, self.config, clock.tick, context=context)
            s6 = time.perf_counter_ns()
            self.boundaries.apply(world)
            s7 = time.perf_counter_ns()
            self.grid.build(state.positions, state.active_indices())
            s8 = time.perf_counter_ns()
            post_pairs = self.grid.candidate_pairs()
            s9 = time.perf_counter_ns()
            result = self.detector.detect(state, self.grid, pairs=post_pairs)
            s10 = time.perf_counter_ns()
            self.resolver.resolve(state, result)
            s11 = time.perf_counter_ns()
            profile.movement_ns = s6 - s5
            profile.boundary_ns = s7 - s6
            profile.post_grid_ns = s8 - s7
            profile.post_pairs_ns = s9 - s8
            profile.detection_ns = s10 - s9
            profile.resolution_ns = s11 - s10

        tick_time = time.perf_counter() - t0
        if profile is not None:
            profile.total_ns = int(tick_time * 1e9)
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

    def step(self, profile: TickProfile | None = None) -> DetectionResult:
        result, tick_time = self.engine.step(self.world, self.clock, profile=profile)
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
