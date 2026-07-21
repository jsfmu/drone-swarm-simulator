"""Simulation orchestration.

Assembles the Phase 1 pipeline and runs it on a fixed time step:

    MovementSystem
      -> BoundaryManager
      -> SpatialHashGrid
      -> CollisionDetectionEngine
      -> CollisionResolutionEngine
      -> MetricsCollector
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
from .movement import MovementSystem
from .spatial_hash import SpatialHashGrid
from .state import World


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
    """Runs one ordered pipeline pass per tick over the world state."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.movement = MovementSystem()
        self.boundaries = BoundaryManager()
        self.detector = CollisionDetectionEngine(config)
        self.resolver = CollisionResolutionEngine(config)
        self.grid = SpatialHashGrid(config)
        # A dedicated RNG stream for movement, seeded off the master seed so
        # runs stay reproducible while staying independent of state generation.
        self._rng = np.random.default_rng(config.seed + 1)

    def step(self, world: World, clock: SimulationClock) -> tuple[DetectionResult, float]:
        state = world.state
        t0 = time.perf_counter()

        # 1-2. Movement (batched policies) + fixed-step integration.
        self.movement.step(state, self._rng, self.config, clock.tick)
        # 2. Boundary constraints.
        self.boundaries.apply(world)
        # 3. Spatial hash (re)build for this tick.
        self.grid.build(state.positions, state.active_indices())
        # 4. Detection.
        result = self.detector.detect(state, self.grid)
        # 5. Resolution.
        self.resolver.resolve(state, result)

        tick_time = time.perf_counter() - t0
        return result, tick_time


class Simulation:
    """Top-level Phase 1 simulation: owns the world, engine, clock, metrics."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.world = World.create(config)
        self.clock = SimulationClock(dt=config.dt)
        self.engine = SimulationEngine(config)
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
