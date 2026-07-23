import numpy as np
import pytest

from drone_sim.collisions import CollisionDetectionEngine
from drone_sim.config import SimulationConfig
from drone_sim.movement import (
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementSystem,
)
from drone_sim.simulation import Simulation, TickProfile
from drone_sim.spatial_hash import SpatialHashGrid
from drone_sim.state import DroneState, World


def make_config(**kw):
    base = dict(
        num_drones=20,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(50.0, 50.0, 50.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        cell_size=2.0,
        seed=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _world_with_goals(config, policy_id):
    state = DroneState.generate(config)
    state.movement_policy_ids[:] = policy_id
    rng = np.random.default_rng(1)
    state.goal_positions = rng.uniform(
        config.bounds_min_arr, config.bounds_max_arr, size=(config.num_drones, 3)
    ).astype(np.float32)
    return World(config=config, state=state)


def _simulation_for(config, policy_id, policy, world):
    return Simulation(config, movement=MovementSystem(policies={policy_id: policy}), world=world)


# ------------------------------------------------------------- profiling off
def test_profiling_disabled_preserves_existing_behavior():
    """profile=None (the default, and every existing call site) must behave
    exactly as before this feature existed: no new required arguments, same
    return type, same semantics."""
    config = make_config()
    sim = Simulation(config)
    result = sim.step()  # the pre-existing call site, unchanged
    assert result is not None
    assert sim.clock.tick == 1


# -------------------------------------------------------------- stage names
def test_profiling_enabled_returns_all_expected_stage_names():
    config = make_config()
    world = _world_with_goals(config, LocalAvoidanceMovementAlgorithm.policy_id)
    sim = _simulation_for(config, LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm(), world)

    profile = TickProfile()
    sim.step(profile=profile)

    expected_fields = {
        "pre_grid_ns", "pre_pairs_ns", "prediction_ns", "context_ns",
        "movement_ns", "boundary_ns", "post_grid_ns", "post_pairs_ns",
        "detection_ns", "resolution_ns", "total_ns", "context_stages_skipped",
    }
    assert expected_fields <= set(vars(profile))


def test_stage_timings_are_finite_and_non_negative():
    config = make_config()
    world = _world_with_goals(config, LocalAvoidanceMovementAlgorithm.policy_id)
    sim = _simulation_for(config, LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm(), world)

    profile = TickProfile()
    sim.step(profile=profile)

    for name in (
        "pre_grid_ns", "pre_pairs_ns", "prediction_ns", "context_ns",
        "movement_ns", "boundary_ns", "post_grid_ns", "post_pairs_ns",
        "detection_ns", "resolution_ns", "total_ns",
    ):
        value = getattr(profile, name)
        assert np.isfinite(value)
        assert value >= 0


# --------------------------------------------------------- skip vs. execute
def test_goal_directed_mode_skips_avoidance_only_stages():
    config = make_config()
    world = _world_with_goals(config, GoalDirectedMovementAlgorithm.policy_id)
    sim = _simulation_for(config, GoalDirectedMovementAlgorithm.policy_id, GoalDirectedMovementAlgorithm(), world)
    assert sim.engine.needs_context is False

    profile = TickProfile()
    sim.step(profile=profile)

    assert profile.context_stages_skipped is True
    assert profile.pre_grid_ns == 0
    assert profile.pre_pairs_ns == 0
    assert profile.prediction_ns == 0
    assert profile.context_ns == 0
    # The rest of the pipeline still ran, and was still timed.
    assert profile.movement_ns > 0
    assert profile.detection_ns >= 0


def test_local_avoidance_mode_executes_prediction_and_context_stages():
    config = make_config()
    world = _world_with_goals(config, LocalAvoidanceMovementAlgorithm.policy_id)
    sim = _simulation_for(config, LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm(), world)
    assert sim.engine.needs_context is True

    profile = TickProfile()
    sim.step(profile=profile)

    assert profile.context_stages_skipped is False
    assert profile.pre_grid_ns > 0
    assert profile.prediction_ns >= 0
    assert profile.context_ns >= 0


# --------------------------------------------------- profiling correctness
def test_profiled_and_unprofiled_runs_produce_identical_detection_results():
    """The pairs= passthrough used only while profiling must not change what
    is actually detected — same seed, same state, same collisions."""
    config = make_config(num_drones=80, bounds_max=(20.0, 20.0, 20.0))

    world_a = _world_with_goals(config, LocalAvoidanceMovementAlgorithm.policy_id)
    sim_a = _simulation_for(config, LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm(), world_a)
    result_a = sim_a.step()  # unprofiled

    world_b = _world_with_goals(config, LocalAvoidanceMovementAlgorithm.policy_id)
    sim_b = _simulation_for(config, LocalAvoidanceMovementAlgorithm.policy_id, LocalAvoidanceMovementAlgorithm(), world_b)
    result_b = sim_b.step(profile=TickProfile())  # profiled

    assert np.array_equal(result_a.collision_pairs, result_b.collision_pairs)
    assert np.array_equal(result_a.near_miss_pairs, result_b.near_miss_pairs)
    assert np.array_equal(sim_a.world.state.positions, sim_b.world.state.positions)
    assert np.array_equal(sim_a.world.state.velocities, sim_b.world.state.velocities)


def test_detect_accepts_precomputed_pairs_and_matches_default():
    config = make_config(num_drones=60, bounds_max=(15.0, 15.0, 15.0))
    state = DroneState.generate(config)
    grid = SpatialHashGrid(config)
    grid.build(state.positions, state.active_indices())

    det = CollisionDetectionEngine(config)
    default_result = det.detect(state, grid)
    pairs = grid.candidate_pairs()
    explicit_result = det.detect(state, grid, pairs=pairs)

    assert np.array_equal(default_result.collision_pairs, explicit_result.collision_pairs)
    assert np.array_equal(default_result.near_miss_pairs, explicit_result.near_miss_pairs)
    assert default_result.num_candidate_pairs == explicit_result.num_candidate_pairs
