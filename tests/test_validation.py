import time

import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.movement import LocalAvoidanceMovementAlgorithm, MovementSystem, ScriptedMovementAlgorithm
from drone_sim.scenarios import near_miss, parallel_safe, rare_collision_background
from drone_sim.simulation import Simulation
from drone_sim.state import DroneState, World
from drone_sim.validation import CollisionEventAccumulator, CollisionRateValidator


def cfg(**kw):
    base = dict(
        num_drones=0,
        bounds_min=(-500.0, -500.0, -500.0),
        bounds_max=(500.0, 500.0, 500.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        cell_size=2.0,
        dt=1.0,
        seed=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _overlapping_stationary_world(config):
    """Two drones already overlapping, at rest, never moving — a continuous
    collision every tick with no dynamics to complicate it."""
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    state = DroneState(
        positions=positions,
        velocities=velocities,
        active_mask=np.ones(2, dtype=bool),
        movement_policy_ids=np.full(2, ScriptedMovementAlgorithm.policy_id, dtype=np.int32),
    )
    return World(config=config, state=state)


def test_continuous_overlap_is_not_counted_as_a_new_event_every_tick():
    config = cfg()
    world = _overlapping_stationary_world(config)
    validator = CollisionRateValidator()

    metrics = validator.run_policy(
        world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=6
    )

    # The pair collides on every one of the 6 ticks, but it only *entered*
    # the collision state once.
    assert metrics.unique_collision_events == 1.0


def test_rates_are_normalized_correctly():
    config = cfg(dt=1.0)
    world = _overlapping_stationary_world(config)
    validator = CollisionRateValidator()

    num_ticks = 4
    metrics = validator.run_policy(
        world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=num_ticks
    )

    # 2 active drones * 4 ticks * dt=1.0s = 8 drone-seconds; 1 unique event.
    expected_rate = 1 / 8 * 10_000.0
    assert metrics.collisions_per_10k_drone_seconds == pytest.approx(expected_rate)


# --------------------------------------------- CollisionEventAccumulator unit
def test_collision_event_accumulator_exact_timeline_example():
    """The exact worked example from the task spec:

    Tick 1: A-B safe
    Tick 2: A-B colliding
    Tick 3: A-B still colliding
    Tick 4: A-B still colliding
    Tick 5: A-B safe
    Tick 6: A-B colliding

    Expected: collision_pair_ticks=4, unique_collision_events=2,
    average_collision_pairs_per_tick=4/6, average_collision_duration_ticks=2.0.
    """
    tracker = CollisionEventAccumulator()
    timeline = [set(), {(0, 1)}, {(0, 1)}, {(0, 1)}, set(), {(0, 1)}]
    for pairs in timeline:
        tracker.record_tick(pairs)

    assert tracker.pair_ticks == 4
    assert tracker.unique_events == 2

    num_ticks = len(timeline)
    avg_per_tick = tracker.pair_ticks / num_ticks
    assert avg_per_tick == pytest.approx(4 / 6)
    avg_duration = tracker.pair_ticks / tracker.unique_events
    assert avg_duration == pytest.approx(2.0)


def test_collision_event_accumulator_no_collisions():
    tracker = CollisionEventAccumulator()
    for _ in range(5):
        tracker.record_tick(set())
    assert tracker.pair_ticks == 0
    assert tracker.unique_events == 0


def test_collision_event_accumulator_collision_on_first_tick():
    tracker = CollisionEventAccumulator()
    tracker.record_tick({(0, 1)})
    tracker.record_tick({(0, 1)})
    assert tracker.unique_events == 1  # counts as one event even though it started on tick 1
    assert tracker.pair_ticks == 2


def test_collision_event_accumulator_continues_until_episode_end():
    tracker = CollisionEventAccumulator()
    for _ in range(5):
        tracker.record_tick({(0, 1)})
    assert tracker.unique_events == 1
    assert tracker.pair_ticks == 5


def test_collision_event_accumulator_two_simultaneous_pairs():
    tracker = CollisionEventAccumulator()
    tracker.record_tick({(0, 1), (2, 3)})
    tracker.record_tick({(0, 1), (2, 3)})
    assert tracker.unique_events == 2
    assert tracker.pair_ticks == 4


def test_collision_event_accumulator_reversed_pair_ordering_is_canonical():
    tracker = CollisionEventAccumulator()
    # Same pair, drone ids given in reversed order on the second tick —
    # canonical (min, max) ordering must treat them as identical, not as two
    # different pairs.
    tracker.record_tick({(2, 7)})
    tracker.record_tick({(7, 2)})
    assert tracker.unique_events == 1
    assert tracker.pair_ticks == 2


def test_collision_event_accumulator_separates_then_recollides():
    tracker = CollisionEventAccumulator()
    tracker.record_tick({(0, 1)})  # collide (event 1)
    tracker.record_tick(set())  # separate
    tracker.record_tick({(0, 1)})  # collide again (event 2)
    assert tracker.unique_events == 2
    assert tracker.pair_ticks == 2


def test_collision_event_accumulator_state_resets_across_instances():
    a = CollisionEventAccumulator()
    a.record_tick({(0, 1)})
    a.record_tick({(0, 1)})

    # A fresh instance (as run_policy creates per call) must not inherit
    # anything from `a` — the pair entering here is a brand-new event.
    b = CollisionEventAccumulator()
    b.record_tick({(0, 1)})

    assert a.unique_events == 1
    assert b.unique_events == 1
    assert b.pair_ticks == 1


def test_run_policy_resets_collision_state_between_calls():
    config = cfg()
    world = _overlapping_stationary_world(config)
    validator = CollisionRateValidator()

    first = validator.run_policy(
        world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=3
    )
    second = validator.run_policy(
        world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=3
    )

    # Each independent call sees the pair "enter" the collision state once,
    # not accumulated across calls.
    assert first.unique_collision_events == 1.0
    assert second.unique_collision_events == 1.0
    assert first.collision_pair_ticks == 3.0
    assert second.collision_pair_ticks == 3.0


# ----------------------------------------- PolicyRunMetrics integration checks
def test_policy_run_metrics_collision_pair_tick_fields():
    config = cfg()
    world = _overlapping_stationary_world(config)
    validator = CollisionRateValidator()

    metrics = validator.run_policy(
        world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=6
    )

    assert metrics.collision_pair_ticks == 6.0  # colliding on all 6 ticks
    assert metrics.unique_collision_events == 1.0
    assert metrics.average_collision_pairs_per_tick == pytest.approx(6 / 6)
    assert metrics.average_collision_duration_ticks == pytest.approx(6.0)


def test_average_collision_duration_is_zero_when_no_collisions():
    config = cfg()
    scenario = parallel_safe(config)
    validator = CollisionRateValidator()

    metrics = validator.run_policy(
        scenario.world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=10
    )

    assert metrics.unique_collision_events == 0.0
    assert metrics.collision_pair_ticks == 0.0
    assert metrics.average_collision_duration_ticks == 0.0
    assert metrics.average_collision_pairs_per_tick == 0.0


def test_near_misses_do_not_enter_collision_metrics():
    config = cfg()
    scenario = near_miss(config)
    validator = CollisionRateValidator()

    metrics = validator.run_policy(
        scenario.world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=12
    )

    assert metrics.unique_collision_events == 0.0
    assert metrics.collision_pair_ticks == 0.0
    assert metrics.average_collision_duration_ticks == 0.0
    assert metrics.unique_near_miss_events >= 1.0


def test_run_policy_does_not_mutate_the_original_world():
    config = cfg(seed=2)
    scenario = rare_collision_background(config, num_background=20, num_injected_collisions=1, seed=2)
    positions_before = scenario.world.state.positions.copy()
    velocities_before = scenario.world.state.velocities.copy()

    validator = CollisionRateValidator()
    validator.compare(scenario, config, num_ticks=6)

    assert np.array_equal(scenario.world.state.positions, positions_before)
    assert np.array_equal(scenario.world.state.velocities, velocities_before)


def test_baseline_and_avoidance_runs_share_identical_initial_conditions():
    config = cfg(seed=3)
    scenario = rare_collision_background(config, num_background=15, num_injected_collisions=1, seed=3)
    validator = CollisionRateValidator()

    # Two independent run_policy calls on the same scenario world must each
    # start from the exact same positions/velocities/goals (only the policy
    # differs) — verified indirectly via run_policy's internal deep copy.
    world_snapshot = scenario.world.state.positions.copy()
    validator.run_policy(scenario.world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), 5)
    assert np.array_equal(scenario.world.state.positions, world_snapshot)
    validator.run_policy(
        scenario.world,
        config,
        LocalAvoidanceMovementAlgorithm.policy_id,
        LocalAvoidanceMovementAlgorithm(),
        5,
    )
    assert np.array_equal(scenario.world.state.positions, world_snapshot)


def test_deterministic_seed_suite_produces_stable_aggregate_metrics():
    config = cfg(max_accel=3.0, prediction_horizon=6.0)
    validator = CollisionRateValidator()
    seeds = [1, 2, 3]

    first = validator.compare_seed_suite(config, seeds, num_ticks=8, num_background=15)
    second = validator.compare_seed_suite(config, seeds, num_ticks=8, num_background=15)

    for name in first:
        assert first[name].unique_collision_events == pytest.approx(second[name].unique_collision_events)
        assert first[name].collisions_per_10k_drone_seconds == pytest.approx(
            second[name].collisions_per_10k_drone_seconds
        )
        assert first[name].num_runs == len(seeds) == second[name].num_runs


def test_local_avoidance_lowers_injected_collision_risk():
    # Deliberately generous avoidance parameters (matching max_speed so
    # goal-directed reproduces the scripted ground-truth trajectory exactly
    # when undisturbed) make this a clear, non-flaky demonstration of the
    # mechanism rather than a claim about tuned production defaults.
    config = cfg(
        max_speed=1.0,
        max_accel=5.0,
        avoidance_max_accel=5.0,
        avoidance_strength=1.0,
        prediction_horizon=8.0,
        goal_tolerance=0.5,
    )
    scenario = rare_collision_background(
        config, num_background=0, num_injected_collisions=1, num_injected_near_misses=0, seed=0
    )
    validator = CollisionRateValidator()
    results = validator.compare(scenario, config, num_ticks=20)

    # Ground truth: the injected pair collides under the no-avoidance baselines.
    assert results["scripted_baseline"].avoidance_success_rate == 0.0
    assert results["goal_directed_no_avoidance"].avoidance_success_rate == 0.0
    # Local avoidance must do at least as well, and demonstrably better here.
    assert results["local_avoidance"].avoidance_success_rate == 1.0
    assert (
        results["local_avoidance"].collisions_per_10k_drone_seconds
        <= results["scripted_baseline"].collisions_per_10k_drone_seconds
    )


def test_avoidance_success_rate_is_none_without_expected_pairs():
    config = cfg()
    scenario = rare_collision_background(config, num_background=10, num_injected_collisions=0, num_injected_near_misses=0)
    validator = CollisionRateValidator()
    metrics = validator.run_policy(
        scenario.world, config, ScriptedMovementAlgorithm.policy_id, ScriptedMovementAlgorithm(), num_ticks=3
    )
    assert metrics.avoidance_success_rate is None


# --------------------------------------------------------------- performance
def test_avoidance_enabled_tick_flow_performance_smoke():
    """Sanity-check that the context-building pipeline (pre-movement grid +
    trajectory prediction + neighbor features) stays vectorised at a moderate
    drone count. Generous budget — this is a smoke test, not a benchmark."""
    n = 2000
    config = cfg(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(300.0, 300.0, 300.0),
        cell_size=2.0,
        max_speed=3.0,
    )
    from drone_sim.state import World as _World

    world = _World.create(config)
    world.state.movement_policy_ids[:] = LocalAvoidanceMovementAlgorithm.policy_id
    # Give every drone a distant goal so GoalDirectedMovementAlgorithm (used
    # internally by LocalAvoidance) doesn't raise for missing goal_positions.
    rng = np.random.default_rng(0)
    goals = rng.uniform(config.bounds_min_arr, config.bounds_max_arr, size=(n, 3)).astype(np.float32)
    world.state.goal_positions = goals

    sim = Simulation(
        config,
        movement=MovementSystem(policies={LocalAvoidanceMovementAlgorithm.policy_id: LocalAvoidanceMovementAlgorithm()}),
        world=world,
    )

    t0 = time.perf_counter()
    for _ in range(5):
        sim.step()
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"avoidance tick flow took {elapsed:.2f}s for {n} drones x 5 ticks — investigate"
