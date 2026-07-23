"""Collision-rate validation: compares movement policies under identical
starting conditions.

Lives outside :class:`~drone_sim.movement.MovementSystem` and
:class:`~drone_sim.collisions.CollisionDetectionEngine` — it only *consumes*
their output. :class:`CollisionRateValidator` runs the same
:class:`~drone_sim.scenarios.ScenarioResult` world under several policies
(deep-copied so runs never share mutable state) and reports aggregate metrics.

Two distinct collision measurements are reported, both computed by
:class:`CollisionEventAccumulator`:

* ``collision_pair_ticks`` — every (pair, tick) observation, including
  persistent ones. Measures total time spent colliding.
* ``unique_collision_events`` — a pair only counts as a new event on the tick
  it *enters* the collision (or near-miss) state. While it keeps appearing in
  the detector's output on subsequent ticks, it is not re-counted; it can be
  counted again only after it leaves that state and later re-enters it.

``average_collision_duration_ticks = collision_pair_ticks / unique_collision_events``
is how long, on average, each distinct collision incident lasted.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import SimulationConfig
from .movement import (
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementAlgorithm,
    MovementSystem,
    ScriptedMovementAlgorithm,
)
from .scenarios import ScenarioResult, rare_collision_background
from .simulation import Simulation
from .state import DroneState, World


def _pair_set(arr: np.ndarray) -> Set[Tuple[int, int]]:
    return {(int(a), int(b)) for a, b in arr}


@dataclass
class CollisionEventAccumulator:
    """Tracks collision-pair-tick and unique-collision-event bookkeeping
    across a sequence of ticks, given each tick's set of currently-colliding
    (or currently-near-missing) unordered pairs.

    Two distinct measurements, both defined precisely in README.md /
    CLAUDE.md:

    * ``pair_ticks`` — every (pair, tick) observation, including persistent
      ones. Counts total time spent colliding.
    * ``unique_events`` — a pair only counts as a new event on the tick it
      *enters* the state (was not present last tick, is present this tick).
      A continuously-colliding pair is not recounted while it persists; if it
      separates for at least one tick and later re-collides, that is a new
      event.

    Pairs are canonicalized to ``(min(a, b), max(a, b))`` on every call, so
    ``(2, 7)`` and ``(7, 2)`` are always treated as the same pair regardless
    of the order the detector happened to report them in. A fresh instance
    starts with empty ``previous_pairs``, so state never leaks across policy
    runs or seeds — each :meth:`~CollisionRateValidator.run_policy` call
    creates its own accumulator.
    """

    previous_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    pair_ticks: int = 0
    unique_events: int = 0

    def record_tick(self, current_pairs: Iterable[Tuple[int, int]]) -> None:
        canonical = {(min(a, b), max(a, b)) for a, b in current_pairs}
        self.pair_ticks += len(canonical)
        self.unique_events += len(canonical - self.previous_pairs)
        self.previous_pairs = canonical


@dataclass
class PolicyRunMetrics:
    """Aggregate metrics for one policy over one run (or the mean of several
    seeded runs when ``num_runs > 1``).

    ``avoidance_success_rate`` and ``destination_completion_rate``/``avg_travel_time``
    are ``None`` when the underlying scenario provides no ground truth for
    them (no injected collision pairs, or no ``goal_positions``).

    ``average_collision_duration_ticks`` is ``0.0`` (not ``NaN``/``None``)
    when ``unique_collision_events`` is zero.
    """

    policy_name: str
    num_runs: int
    num_ticks: int
    unique_collision_events: float
    collision_pair_ticks: float
    average_collision_pairs_per_tick: float
    average_collision_duration_ticks: float
    collisions_per_10k_drone_seconds: float
    unique_near_miss_events: float
    near_misses_per_10k_drone_seconds: float
    min_observed_separation: float
    avg_drone_speed: float
    stationary_percentage: float
    avoidance_success_rate: Optional[float] = None
    destination_completion_rate: Optional[float] = None
    avg_travel_time: Optional[float] = None


def _copy_world(world: World) -> World:
    """Deep-copy a World's arrays so independent policy runs never alias state."""
    state = world.state
    return World(
        config=world.config,
        state=DroneState(
            positions=state.positions.copy(),
            velocities=state.velocities.copy(),
            active_mask=state.active_mask.copy(),
            movement_policy_ids=state.movement_policy_ids.copy(),
            goal_positions=None if state.goal_positions is None else state.goal_positions.copy(),
        ),
    )


def default_policies() -> Dict[str, Tuple[int, MovementAlgorithm]]:
    """The minimum comparison set required by the Phase 2 validation spec."""
    return {
        "scripted_baseline": (ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm()),
        "goal_directed_no_avoidance": (GoalDirectedMovementAlgorithm.policy_id, GoalDirectedMovementAlgorithm()),
        "local_avoidance": (LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm()),
    }


class CollisionRateValidator:
    """Runs scenarios under one or more policies and reports comparable metrics."""

    def run_policy(
        self,
        world: World,
        config: SimulationConfig,
        policy_id: int,
        policy: MovementAlgorithm,
        num_ticks: int,
        expected_collision_pairs: Sequence[Tuple[int, int]] = (),
    ) -> PolicyRunMetrics:
        """Run ``num_ticks`` of a copy of ``world`` with every active drone
        assigned to ``policy_id``/``policy``, and summarise the run.
        """
        run_world = _copy_world(world)
        run_world.state.movement_policy_ids[:] = policy_id
        num_active = int(run_world.state.active_mask.sum())

        sim = Simulation(config, movement=MovementSystem(policies={policy_id: policy}), world=run_world)

        # Fresh per call: state (previous_pairs, counts) never leaks across
        # policy runs or seeds.
        collision_tracker = CollisionEventAccumulator()
        near_miss_tracker = CollisionEventAccumulator()
        ever_collided: Set[Tuple[int, int]] = set()
        min_separation = float("inf")
        speed_sum = 0.0
        speed_count = 0

        n = run_world.state.num_drones
        has_goals = run_world.state.goal_positions is not None
        arrival_tick = np.full(n, -1, dtype=np.int64) if has_goals else None

        for tick in range(num_ticks):
            result = sim.step()
            state = sim.world.state

            coll_pairs = _pair_set(result.collision_pairs)
            near_pairs = _pair_set(result.near_miss_pairs)

            collision_tracker.record_tick(coll_pairs)
            near_miss_tracker.record_tick(near_pairs)
            ever_collided |= coll_pairs

            if result.collision_distances.size:
                min_separation = min(min_separation, float(result.collision_distances.min()))
            if result.near_miss_distances.size:
                min_separation = min(min_separation, float(result.near_miss_distances.min()))

            active_idx = np.nonzero(state.active_mask)[0]
            speed = np.linalg.norm(state.velocities[active_idx], axis=1)
            speed_sum += float(speed.sum())
            speed_count += speed.size

            if has_goals:
                dist_to_goal = np.linalg.norm(
                    state.goal_positions[active_idx] - state.positions[active_idx], axis=1
                )
                arrived_now = dist_to_goal <= config.goal_tolerance
                not_yet = arrival_tick[active_idx] < 0
                newly = active_idx[arrived_now & not_yet]
                arrival_tick[newly] = tick

        total_seconds = num_ticks * config.dt
        drone_seconds = max(num_active * total_seconds, 1e-9)

        collision_pair_ticks = collision_tracker.pair_ticks
        unique_collision_events = collision_tracker.unique_events
        average_collision_pairs_per_tick = collision_pair_ticks / num_ticks if num_ticks else 0.0
        average_collision_duration_ticks = (
            collision_pair_ticks / unique_collision_events if unique_collision_events else 0.0
        )

        avoidance_success_rate = None
        if expected_collision_pairs:
            avoided = sum(1 for p in expected_collision_pairs if tuple(p) not in ever_collided)
            avoidance_success_rate = avoided / len(expected_collision_pairs)

        destination_completion_rate = None
        avg_travel_time = None
        if has_goals:
            arrived_mask = arrival_tick >= 0
            destination_completion_rate = float(arrived_mask.mean())
            if arrived_mask.any():
                avg_travel_time = float(arrival_tick[arrived_mask].mean() * config.dt)

        final_speed = np.linalg.norm(
            sim.world.state.velocities[sim.world.state.active_mask], axis=1
        )
        stationary_percentage = (
            float((final_speed < 1e-3).mean()) if final_speed.size else 0.0
        )

        return PolicyRunMetrics(
            policy_name=policy.__class__.__name__,
            num_runs=1,
            num_ticks=num_ticks,
            unique_collision_events=float(unique_collision_events),
            collision_pair_ticks=float(collision_pair_ticks),
            average_collision_pairs_per_tick=average_collision_pairs_per_tick,
            average_collision_duration_ticks=average_collision_duration_ticks,
            collisions_per_10k_drone_seconds=unique_collision_events / drone_seconds * 10_000.0,
            unique_near_miss_events=float(near_miss_tracker.unique_events),
            near_misses_per_10k_drone_seconds=near_miss_tracker.unique_events / drone_seconds * 10_000.0,
            min_observed_separation=min_separation,
            avg_drone_speed=(speed_sum / speed_count) if speed_count else 0.0,
            stationary_percentage=stationary_percentage,
            avoidance_success_rate=avoidance_success_rate,
            destination_completion_rate=destination_completion_rate,
            avg_travel_time=avg_travel_time,
        )

    def compare(
        self,
        scenario: ScenarioResult,
        config: SimulationConfig,
        num_ticks: int,
        policies: Optional[Dict[str, Tuple[int, MovementAlgorithm]]] = None,
    ) -> Dict[str, PolicyRunMetrics]:
        """Run every policy in ``policies`` against a copy of ``scenario.world``."""
        policies = policies or default_policies()
        return {
            name: self.run_policy(
                scenario.world, config, policy_id, policy, num_ticks, scenario.expected_collision_pairs
            )
            for name, (policy_id, policy) in policies.items()
        }

    def compare_seed_suite(
        self,
        base_config: SimulationConfig,
        seeds: Sequence[int],
        num_ticks: int,
        policies: Optional[Dict[str, Tuple[int, MovementAlgorithm]]] = None,
        **scenario_kwargs,
    ) -> Dict[str, PolicyRunMetrics]:
        """Run :func:`~drone_sim.scenarios.rare_collision_background` across a
        deterministic seed suite for every policy, returning per-policy
        aggregates (means across seeds — individual seeded runs are not
        required to each improve, only the aggregate).
        """
        policies = policies or default_policies()
        per_policy_runs: Dict[str, List[PolicyRunMetrics]] = {name: [] for name in policies}

        for seed in seeds:
            cfg = dataclasses.replace(base_config, seed=seed)
            scenario = rare_collision_background(cfg, seed=seed, **scenario_kwargs)
            for name, metrics in self.compare(scenario, cfg, num_ticks, policies).items():
                per_policy_runs[name].append(metrics)

        return {name: _aggregate(name, runs) for name, runs in per_policy_runs.items()}


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _aggregate(name: str, runs: List[PolicyRunMetrics]) -> PolicyRunMetrics:
    # Ratio-like fields (rates, per-tick/per-event averages) are aggregated as
    # the mean of each run's own computed ratio, consistent with how
    # collisions_per_10k_drone_seconds was already aggregated — not
    # recomputed from summed numerators/denominators across runs.
    return PolicyRunMetrics(
        policy_name=name,
        num_runs=len(runs),
        num_ticks=runs[0].num_ticks,
        unique_collision_events=float(np.mean([r.unique_collision_events for r in runs])),
        collision_pair_ticks=float(np.mean([r.collision_pair_ticks for r in runs])),
        average_collision_pairs_per_tick=float(np.mean([r.average_collision_pairs_per_tick for r in runs])),
        average_collision_duration_ticks=float(np.mean([r.average_collision_duration_ticks for r in runs])),
        collisions_per_10k_drone_seconds=float(np.mean([r.collisions_per_10k_drone_seconds for r in runs])),
        unique_near_miss_events=float(np.mean([r.unique_near_miss_events for r in runs])),
        near_misses_per_10k_drone_seconds=float(np.mean([r.near_misses_per_10k_drone_seconds for r in runs])),
        min_observed_separation=float(min(r.min_observed_separation for r in runs)),
        avg_drone_speed=float(np.mean([r.avg_drone_speed for r in runs])),
        stationary_percentage=float(np.mean([r.stationary_percentage for r in runs])),
        avoidance_success_rate=_mean_or_none([r.avoidance_success_rate for r in runs]),
        destination_completion_rate=_mean_or_none([r.destination_completion_rate for r in runs]),
        avg_travel_time=_mean_or_none([r.avg_travel_time for r in runs]),
    )
