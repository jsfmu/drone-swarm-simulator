import numpy as np
import pytest

from drone_sim.config import SimulationConfig, BoundaryMode
from drone_sim.state import DroneState, World
from drone_sim.movement import (
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementContext,
    MovementSystem,
    NeighborFeatureBuilder,
    ScriptedMovementAlgorithm,
    RandomMovementAlgorithm,
)
from drone_sim.simulation import Simulation
from drone_sim.trajectory import TrajectoryPredictionService


def make_config(**kw):
    base = dict(
        num_drones=200,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(100.0, 100.0, 100.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        dt=1.0,
        seed=7,
    )
    base.update(kw)
    return SimulationConfig(**base)


def test_generation_is_reproducible():
    cfg = make_config()
    a = DroneState.generate(cfg)
    b = DroneState.generate(cfg)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.velocities, b.velocities)


def test_generation_within_bounds():
    cfg = make_config(num_drones=1000)
    s = DroneState.generate(cfg)
    assert (s.positions >= cfg.bounds_min_arr).all()
    assert (s.positions <= cfg.bounds_max_arr).all()


def test_state_dtypes_and_shapes():
    cfg = make_config(num_drones=50)
    s = DroneState.generate(cfg)
    assert s.positions.shape == (50, 3)
    assert s.positions.dtype == np.float32
    assert s.velocities.dtype == np.float32
    assert s.active_mask.dtype == np.bool_
    assert s.active_mask.all()


def test_scripted_movement_is_constant_velocity():
    # A single scripted drone should move linearly by v*dt each tick.
    cfg = make_config(num_drones=1, bounds_max=(1e9, 1e9, 1e9), dt=0.5)
    world = World.create(cfg)
    world.state.movement_policy_ids[:] = ScriptedMovementAlgorithm.policy_id
    v0 = world.state.velocities[0].copy()
    p0 = world.state.positions[0].copy()

    ms = MovementSystem()
    rng = np.random.default_rng(0)
    for t in range(5):
        ms.step(world.state, rng, cfg, t)

    expected = p0 + v0 * cfg.dt * 5
    assert np.allclose(world.state.positions[0], expected, atol=1e-3)


def test_random_movement_reproducible_and_bounded():
    cfg = make_config(num_drones=64, seed=123)
    s1 = Simulation(cfg)
    s2 = Simulation(cfg)
    s1.run(10)
    s2.run(10)
    assert np.array_equal(s1.world.state.positions, s2.world.state.positions)
    assert np.array_equal(s1.world.state.velocities, s2.world.state.velocities)
    # Velocities stay clipped to +/- max_speed.
    assert (np.abs(s1.world.state.velocities) <= cfg.max_speed + 1e-4).all()


def test_batched_policies_apply_per_group():
    cfg = make_config(num_drones=10, bounds_max=(1e9, 1e9, 1e9))
    world = World.create(cfg)
    # Half scripted, half random.
    world.state.movement_policy_ids[:5] = ScriptedMovementAlgorithm.policy_id
    world.state.movement_policy_ids[5:] = RandomMovementAlgorithm.policy_id
    scripted_v0 = world.state.velocities[:5].copy()

    ms = MovementSystem()
    rng = np.random.default_rng(1)
    ms.step(world.state, rng, cfg, 0)

    # Scripted drones keep their velocity exactly.
    assert np.array_equal(world.state.velocities[:5], scripted_v0)


# ------------------------------------------------------------- batched dispatch
class _CountingAlgorithm(ScriptedMovementAlgorithm):
    """A policy that records every ``idx`` batch it was called with."""

    policy_id = 99

    def __init__(self):
        self.calls = []

    def update_velocities(self, state, idx, rng, config, tick, context=None):
        self.calls.append(np.array(idx))


def test_mixed_policy_ids_are_grouped_and_each_policy_called_once():
    cfg = make_config(num_drones=9, bounds_max=(1e9, 1e9, 1e9))
    world = World.create(cfg)
    ids = np.array([99, 1, 99, 1, 1, 99, 99, 1, 99], dtype=np.int32)
    world.state.movement_policy_ids[:] = ids

    counting = _CountingAlgorithm()
    ms = MovementSystem(policies={99: counting, ScriptedMovementAlgorithm.policy_id: ScriptedMovementAlgorithm()})
    ms.step(world.state, np.random.default_rng(0), cfg, 0)

    assert len(counting.calls) == 1  # dispatched once for its whole batch, not per drone
    assert set(counting.calls[0].tolist()) == {0, 2, 5, 6, 8}


def test_inactive_drones_remain_unchanged():
    cfg = make_config(num_drones=4, bounds_max=(1e9, 1e9, 1e9))
    world = World.create(cfg)
    world.state.movement_policy_ids[:] = RandomMovementAlgorithm.policy_id
    world.state.active_mask[1] = False
    v_inactive_before = world.state.velocities[1].copy()
    p_inactive_before = world.state.positions[1].copy()

    ms = MovementSystem()
    ms.step(world.state, np.random.default_rng(0), cfg, 0)

    assert np.array_equal(world.state.velocities[1], v_inactive_before)
    assert np.array_equal(world.state.positions[1], p_inactive_before)


def test_unknown_policy_id_raises_explicitly():
    cfg = make_config(num_drones=3)
    world = World.create(cfg)
    world.state.movement_policy_ids[:] = [0, 42, 0]  # 42 is not registered

    ms = MovementSystem()  # default registry only has 0 (random) and 1 (scripted)
    with pytest.raises(ValueError, match="42"):
        ms.step(world.state, np.random.default_rng(0), cfg, 0)


# ------------------------------------------------------------ goal-directed
def _goal_world(cfg, positions, velocities, goals):
    n = len(positions)
    state = DroneState(
        positions=np.asarray(positions, dtype=np.float32),
        velocities=np.asarray(velocities, dtype=np.float32),
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=np.full(n, GoalDirectedMovementAlgorithm.policy_id, dtype=np.int32),
        goal_positions=np.asarray(goals, dtype=np.float32),
    )
    return World(config=cfg, state=state)


def test_goal_directed_distance_to_destination_decreases():
    cfg = make_config(num_drones=1, bounds_max=(1e9, 1e9, 1e9), max_accel=1.0, max_speed=3.0)
    world = _goal_world(cfg, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]])
    policy = GoalDirectedMovementAlgorithm()
    rng = np.random.default_rng(0)

    dist_prev = np.linalg.norm(world.state.goal_positions[0] - world.state.positions[0])
    for tick in range(10):
        idx = np.array([0])
        policy.update_velocities(world.state, idx, rng, cfg, tick)
        world.state.positions[idx] += world.state.velocities[idx] * cfg.dt
        dist = np.linalg.norm(world.state.goal_positions[0] - world.state.positions[0])
        assert dist < dist_prev + 1e-6
        dist_prev = dist


def test_goal_directed_respects_speed_and_accel_limits():
    cfg = make_config(num_drones=1, bounds_max=(1e9, 1e9, 1e9), max_accel=0.5, max_speed=2.0)
    world = _goal_world(cfg, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]])
    policy = GoalDirectedMovementAlgorithm()
    rng = np.random.default_rng(0)

    prev_speed = 0.0
    for tick in range(20):
        idx = np.array([0])
        v_before = world.state.velocities[0].copy()
        policy.update_velocities(world.state, idx, rng, cfg, tick)
        accel = np.linalg.norm(world.state.velocities[0] - v_before)
        assert accel <= cfg.max_accel + 1e-6
        speed = np.linalg.norm(world.state.velocities[0])
        assert speed <= cfg.max_speed + 1e-6
        world.state.positions[idx] += world.state.velocities[idx] * cfg.dt
        prev_speed = speed
    assert prev_speed > 0  # actually made progress, didn't just sit at zero


def test_goal_directed_tolerance_stops_the_drone():
    cfg = make_config(num_drones=1, bounds_max=(1e9, 1e9, 1e9), max_accel=5.0, max_speed=5.0, goal_tolerance=2.0)
    # Start already within tolerance of the goal.
    world = _goal_world(cfg, [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    policy = GoalDirectedMovementAlgorithm()
    policy.update_velocities(world.state, np.array([0]), np.random.default_rng(0), cfg, 0)
    assert np.allclose(world.state.velocities[0], 0.0)


def test_goal_directed_requires_goal_positions():
    cfg = make_config(num_drones=1, bounds_max=(1e9, 1e9, 1e9))
    world = World.create(cfg)
    world.state.movement_policy_ids[:] = GoalDirectedMovementAlgorithm.policy_id
    policy = GoalDirectedMovementAlgorithm()
    with pytest.raises(ValueError):
        policy.update_velocities(world.state, np.array([0]), np.random.default_rng(0), cfg, 0)


# ------------------------------------------------------------- local avoidance
def _context_for(state, cfg, pairs):
    predictor = TrajectoryPredictionService(cfg)
    prediction = predictor.predict(state, np.asarray(pairs, dtype=np.int64))
    return NeighborFeatureBuilder().build(state, prediction, state.goal_positions)


def test_local_avoidance_requires_context():
    cfg = make_config(num_drones=2, bounds_max=(1e9, 1e9, 1e9))
    world = _goal_world(
        cfg, [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]],
    )
    policy = LocalAvoidanceMovementAlgorithm()
    with pytest.raises(ValueError):
        policy.update_velocities(world.state, np.array([0, 1]), np.random.default_rng(0), cfg, 0, context=None)


def test_local_avoidance_head_on_generates_separating_correction():
    cfg = make_config(
        num_drones=2, bounds_max=(1e9, 1e9, 1e9), collision_radius=1.0, near_miss_radius=2.0,
        max_speed=5.0, max_accel=5.0, avoidance_max_accel=5.0, avoidance_strength=1.0,
        prediction_horizon=8.0,
    )
    # Small lateral offset so there is a direction to push along (a perfectly
    # aligned head-on pair has no lateral escape vector to correct into).
    positions = [[0.0, 0.0, 0.0], [10.0, 0.3, 0.0]]
    velocities = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    goals = [[1000.0, 0.0, 0.0], [-1000.0, 0.3, 0.0]]
    world = _goal_world(cfg, positions, velocities, goals)
    world.state.movement_policy_ids[:] = LocalAvoidanceMovementAlgorithm.policy_id

    context = _context_for(world.state, cfg, [[0, 1]])
    policy = LocalAvoidanceMovementAlgorithm()
    idx = np.array([0, 1])
    policy.update_velocities(world.state, idx, np.random.default_rng(0), cfg, 0, context=context)

    # Drone 0's velocity should now carry a component pushing it toward -y
    # (away from drone 1's +y offset) instead of pure +x.
    assert world.state.velocities[0, 1] < -1e-3
    assert world.state.velocities[1, 1] > 1e-3
    assert np.linalg.norm(world.state.velocities[0]) <= cfg.max_speed + 1e-4
    assert np.linalg.norm(world.state.velocities[1]) <= cfg.max_speed + 1e-4


def test_local_avoidance_crossing_paths_receive_correction():
    cfg = make_config(
        num_drones=2, bounds_max=(1e9, 1e9, 1e9), collision_radius=1.0, near_miss_radius=2.0,
        max_speed=5.0, max_accel=5.0, avoidance_max_accel=5.0, avoidance_strength=1.0,
        prediction_horizon=8.0,
    )
    positions = [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]
    velocities = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    goals = [[1000.0, 5.0, 0.0], [5.0, 1000.0, 0.0]]
    world = _goal_world(cfg, positions, velocities, goals)
    world.state.movement_policy_ids[:] = LocalAvoidanceMovementAlgorithm.policy_id

    context = _context_for(world.state, cfg, [[0, 1]])
    policy = LocalAvoidanceMovementAlgorithm()
    idx = np.array([0, 1])
    goal_only = GoalDirectedMovementAlgorithm()
    baseline_state = DroneState(
        positions=world.state.positions.copy(), velocities=world.state.velocities.copy(),
        active_mask=world.state.active_mask.copy(), movement_policy_ids=world.state.movement_policy_ids.copy(),
        goal_positions=world.state.goal_positions.copy(),
    )
    goal_only.update_velocities(baseline_state, idx, np.random.default_rng(0), cfg, 0)

    policy.update_velocities(world.state, idx, np.random.default_rng(0), cfg, 0, context=context)

    # The avoidance-adjusted velocity must differ from the pure goal-directed one.
    assert not np.allclose(world.state.velocities[idx], baseline_state.velocities[idx], atol=1e-4)


def test_local_avoidance_safe_parallel_drones_not_redirected():
    cfg = make_config(
        num_drones=2, bounds_max=(1e9, 1e9, 1e9), collision_radius=1.0, near_miss_radius=2.0,
        max_speed=5.0, max_accel=5.0, prediction_horizon=5.0,
    )
    far = cfg.near_miss_radius * 5.0
    positions = [[0.0, 0.0, 0.0], [0.0, far, 0.0]]
    velocities = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    goals = [[1000.0, 0.0, 0.0], [1000.0, far, 0.0]]
    world = _goal_world(cfg, positions, velocities, goals)
    world.state.movement_policy_ids[:] = LocalAvoidanceMovementAlgorithm.policy_id

    context = _context_for(world.state, cfg, [[0, 1]])
    policy = LocalAvoidanceMovementAlgorithm()
    idx = np.array([0, 1])
    policy.update_velocities(world.state, idx, np.random.default_rng(0), cfg, 0, context=context)

    # No meaningful lateral redirection: still moving essentially straight +x.
    assert abs(world.state.velocities[0, 1]) < 1e-3
    assert abs(world.state.velocities[1, 1]) < 1e-3


def test_local_avoidance_still_makes_goal_progress():
    cfg = make_config(
        num_drones=2, bounds_max=(1e9, 1e9, 1e9), collision_radius=1.0, near_miss_radius=2.0,
        max_speed=5.0, max_accel=5.0, avoidance_max_accel=2.0, avoidance_strength=0.5,
        prediction_horizon=6.0,
    )
    # A moderate crossing threat (not a worst-case dead-on collision) so the
    # bounded correction cannot cancel forward progress entirely.
    positions = [[0.0, 0.0, 0.0], [8.0, 1.0, 0.0]]
    velocities = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    goals = [[1000.0, 0.0, 0.0], [-1000.0, 1.0, 0.0]]
    world = _goal_world(cfg, positions, velocities, goals)
    world.state.movement_policy_ids[:] = LocalAvoidanceMovementAlgorithm.policy_id

    context = _context_for(world.state, cfg, [[0, 1]])
    policy = LocalAvoidanceMovementAlgorithm()
    idx = np.array([0, 1])
    policy.update_velocities(world.state, idx, np.random.default_rng(0), cfg, 0, context=context)

    # Drone 0 still makes net progress toward its goal (positive x-velocity
    # component), even though it has been perturbed off pure +x.
    assert world.state.velocities[0, 0] > 0.0
    assert world.state.velocities[1, 0] < 0.0


def test_local_avoidance_is_deterministic():
    cfg = make_config(
        num_drones=2, bounds_max=(1e9, 1e9, 1e9), collision_radius=1.0, near_miss_radius=2.0,
        max_speed=5.0, max_accel=5.0, avoidance_max_accel=5.0, avoidance_strength=1.0,
        prediction_horizon=8.0,
    )
    positions = [[0.0, 0.0, 0.0], [10.0, 0.3, 0.0]]
    velocities = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    goals = [[1000.0, 0.0, 0.0], [-1000.0, 0.3, 0.0]]

    def run(seed):
        # Independent, fresh state each time: avoidance/goal-directed use no
        # randomness, so the result must be identical regardless of rng seed.
        world = _goal_world(cfg, positions, velocities, goals)
        world.state.movement_policy_ids[:] = LocalAvoidanceMovementAlgorithm.policy_id
        context = _context_for(world.state, cfg, [[0, 1]])
        policy = LocalAvoidanceMovementAlgorithm()
        idx = np.array([0, 1])
        policy.update_velocities(world.state, idx, np.random.default_rng(seed), cfg, 0, context=context)
        return world.state.velocities.copy()

    assert np.array_equal(run(1), run(2))
