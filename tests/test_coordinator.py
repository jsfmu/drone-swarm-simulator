import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.coordinator import (
    DistributedConfig,
    DistributedCoordinator,
    PartitionLoadStats,
    TickCommitError,
)
from drone_sim.movement import (
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementSystem,
    ScriptedMovementAlgorithm,
)
from drone_sim.simulation import Simulation
from drone_sim.state import DroneState, World


def cfg(**kw):
    base = dict(
        num_drones=4,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(100.0, 100.0, 100.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        cell_size=2.0,
        seed=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _world(config, positions, velocities, policy_id, goal_positions=None):
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    n = positions.shape[0]
    state = DroneState(
        positions=positions.copy(),
        velocities=velocities.copy(),
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=np.full(n, policy_id, dtype=np.int32),
        goal_positions=None if goal_positions is None else np.asarray(goal_positions, dtype=np.float32).copy(),
    )
    return World(config=config, state=state)


def _pair_set(arr):
    return {(int(a), int(b)) for a, b in arr}


# ------------------------------------------------------ single-partition parity
def test_single_partition_matches_plain_simulation():
    """num_partitions=1 has no neighbours/ghosts/dedup at all -- it must be a
    faithful (bit-exact, for a non-RNG policy) reproduction of the plain
    Simulation path."""
    c = cfg(num_drones=12, cell_size=2.0)
    rng = np.random.default_rng(3)
    positions = rng.uniform(0, 100, size=(12, 3))
    goals = rng.uniform(0, 100, size=(12, 3))
    velocities = np.zeros((12, 3))

    plain_world = _world(c, positions, velocities, GoalDirectedMovementAlgorithm.policy_id, goals)
    dist_world = _world(c, positions, velocities, GoalDirectedMovementAlgorithm.policy_id, goals)

    plain_sim = Simulation(
        c,
        movement=MovementSystem(policies={GoalDirectedMovementAlgorithm.policy_id: GoalDirectedMovementAlgorithm()}),
        world=plain_world,
    )
    coord = DistributedCoordinator(
        c,
        DistributedConfig(num_workers=1, num_partitions=1),
        movement=MovementSystem(policies={GoalDirectedMovementAlgorithm.policy_id: GoalDirectedMovementAlgorithm()}),
        world=dist_world,
    )

    last_plain = last_dist = None
    for _ in range(10):
        last_plain = plain_sim.step()
        last_dist = coord.step()

    np.testing.assert_allclose(plain_sim.world.state.positions, coord.world.state.positions)
    np.testing.assert_allclose(plain_sim.world.state.velocities, coord.world.state.velocities)
    assert _pair_set(last_plain.collision_pairs) == _pair_set(last_dist.collision_pairs)
    assert _pair_set(last_plain.near_miss_pairs) == _pair_set(last_dist.near_miss_pairs)
    assert last_plain.num_candidate_pairs == last_dist.num_candidate_pairs


# ------------------------------------------------------- multi-partition parity
def _disjoint_pairs_world(c):
    """Two independent, already-colliding, zero-velocity drone pairs: one
    straddling the x=50 partition boundary, one entirely inside partition 0,
    far enough apart in y that they never share a grid cell/candidate pair
    with each other. No drone participates in more than one simultaneous
    collision, so CollisionResolutionEngine's sequential/order-sensitive
    resolution (a pre-existing property of the unmodified single-worker
    kernel, not something Phase 4 introduces) cannot make the outcome depend
    on partition count here.
    """
    positions = np.array(
        [
            [49.7, 10.0, 10.0],  # pair A, drone 0 -- partition 1 of 4 (or 0 of 1)
            [50.3, 10.0, 10.0],  # pair A, drone 1 -- partition 2 of 4 (or 0 of 1)
            [10.0, 50.0, 10.0],  # pair B, drone 2 -- partition 0
            [10.5, 50.0, 10.0],  # pair B, drone 3 -- partition 0
        ],
        dtype=np.float32,
    )
    velocities = np.zeros((4, 3), dtype=np.float32)
    return _world(c, positions, velocities, ScriptedMovementAlgorithm.policy_id)


def test_multi_partition_exact_agreement_with_single_partition():
    c = cfg()
    world_1 = _disjoint_pairs_world(c)
    world_4 = _disjoint_pairs_world(c)

    coord_1 = DistributedCoordinator(
        c, DistributedConfig(num_workers=1, num_partitions=1), world=world_1
    )
    coord_4 = DistributedCoordinator(
        c, DistributedConfig(num_workers=4, num_partitions=4), world=world_4
    )

    result_1 = coord_1.step()
    result_4 = coord_4.step()

    assert _pair_set(result_1.collision_pairs) == _pair_set(result_4.collision_pairs) == {(0, 1), (2, 3)}
    assert result_1.num_candidate_pairs == result_4.num_candidate_pairs
    np.testing.assert_allclose(coord_1.world.state.positions, coord_4.world.state.positions)
    np.testing.assert_allclose(coord_1.world.state.velocities, coord_4.world.state.velocities)


def test_process_executor_multi_partition_matches_single_partition():
    """Phase 5: the same multi-partition-vs-single-partition exact-agreement
    check as test_multi_partition_exact_agreement_with_single_partition, but
    with use_processes=True -- a real ProcessPoolExecutor must produce
    numerically identical results to the sequential/threaded paths."""
    c = cfg()
    world_1 = _disjoint_pairs_world(c)
    world_4 = _disjoint_pairs_world(c)

    coord_1 = DistributedCoordinator(
        c, DistributedConfig(num_workers=1, num_partitions=1, use_processes=True), world=world_1
    )
    coord_4 = DistributedCoordinator(
        c, DistributedConfig(num_workers=4, num_partitions=4, use_processes=True), world=world_4
    )
    try:
        result_1 = coord_1.step()
        result_4 = coord_4.step()
    finally:
        coord_1.shutdown()
        coord_4.shutdown()

    assert _pair_set(result_1.collision_pairs) == _pair_set(result_4.collision_pairs) == {(0, 1), (2, 3)}
    assert result_1.num_candidate_pairs == result_4.num_candidate_pairs
    np.testing.assert_allclose(coord_1.world.state.positions, coord_4.world.state.positions)
    np.testing.assert_allclose(coord_1.world.state.velocities, coord_4.world.state.velocities)


def test_use_threads_and_use_processes_mutually_exclusive_config_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DistributedConfig(use_threads=True, use_processes=True)


def test_coordinator_shutdown_is_a_no_op_for_default_sequential_pool():
    """shutdown() must be safe to call unconditionally, even when no process
    pool was ever created (the sequential/threaded default)."""
    c = cfg()
    coord = DistributedCoordinator(c, DistributedConfig(num_workers=2), world=_disjoint_pairs_world(c))
    coord.step()
    coord.shutdown()  # must not raise
    coord.shutdown()  # idempotent


def test_cross_partition_collision_deduplicated_not_double_counted():
    c = cfg()
    world = _disjoint_pairs_world(c)
    coord = DistributedCoordinator(c, DistributedConfig(num_workers=4, num_partitions=4), world=world)
    result = coord.step()

    pairs = [tuple(int(x) for x in p) for p in result.collision_pairs]
    assert pairs.count((0, 1)) == 1  # the cross-partition pair, not double-recorded
    assert pairs.count((2, 3)) == 1
    assert len(pairs) == 2


def test_deterministic_repeated_multi_worker_runs():
    c = cfg()
    world_a = _disjoint_pairs_world(c)
    world_b = _disjoint_pairs_world(c)
    coord_a = DistributedCoordinator(c, DistributedConfig(num_workers=3, num_partitions=4), world=world_a)
    coord_b = DistributedCoordinator(c, DistributedConfig(num_workers=3, num_partitions=4), world=world_b)

    for _ in range(5):
        ra = coord_a.step()
        rb = coord_b.step()
        assert _pair_set(ra.collision_pairs) == _pair_set(rb.collision_pairs)
        assert ra.num_candidate_pairs == rb.num_candidate_pairs

    np.testing.assert_array_equal(coord_a.world.state.positions, coord_b.world.state.positions)
    np.testing.assert_array_equal(coord_a.world.state.velocities, coord_b.world.state.velocities)


# ------------------------------------------------------------ ownership transfer
def test_ownership_transfer_across_boundary_during_run():
    c = cfg(num_drones=1, bounds_max=(100.0, 100.0, 100.0))
    positions = np.array([[45.0, 50.0, 50.0]], dtype=np.float32)
    velocities = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)  # crosses x=50 after 3 ticks
    world = _world(c, positions, velocities, ScriptedMovementAlgorithm.policy_id)
    coord = DistributedCoordinator(c, DistributedConfig(num_workers=2, num_partitions=2), world=world)

    assert coord._current_owners()[0] == 0
    for _ in range(3):
        coord.step()
    assert coord.world.state.positions[0, 0] == pytest.approx(51.0)
    assert coord._current_owners()[0] == 1


# ------------------------------------------------------------------ rebalancing
def test_rebalancing_moves_heaviest_partition_when_imbalanced():
    c = cfg(num_drones=1)
    world = _world(c, [[10.0, 10.0, 10.0]], [[0.0, 0.0, 0.0]], ScriptedMovementAlgorithm.policy_id)
    coord = DistributedCoordinator(
        c,
        DistributedConfig(num_workers=2, num_partitions=4, rebalance_imbalance_threshold=1.2),
        world=world,
    )
    # partitions 0,2 -> worker 0; partitions 1,3 -> worker 1 (round robin default)
    assert coord.partition_worker == {0: 0, 1: 1, 2: 0, 3: 1}

    coord.last_load_stats = [
        PartitionLoadStats(partition_id=0, owned_drone_count=100, ghost_drone_count=0, candidate_pair_count=500, tick_duration_s=0.01),
        PartitionLoadStats(partition_id=1, owned_drone_count=1, ghost_drone_count=0, candidate_pair_count=1, tick_duration_s=0.001),
        PartitionLoadStats(partition_id=2, owned_drone_count=1, ghost_drone_count=0, candidate_pair_count=1, tick_duration_s=0.001),
        PartitionLoadStats(partition_id=3, owned_drone_count=1, ghost_drone_count=0, candidate_pair_count=1, tick_duration_s=0.001),
    ]
    coord._maybe_rebalance()

    # partition 0 (worker 0's heaviest) should have moved to worker 1 (idlest).
    assert coord.partition_worker[0] == 1
    assert coord.reassignment_log[-1] == (0, 0, 1)


def test_rebalancing_does_nothing_when_balanced():
    c = cfg(num_drones=1)
    world = _world(c, [[10.0, 10.0, 10.0]], [[0.0, 0.0, 0.0]], ScriptedMovementAlgorithm.policy_id)
    coord = DistributedCoordinator(
        c, DistributedConfig(num_workers=2, num_partitions=2, rebalance_imbalance_threshold=1.5), world=world
    )
    before = dict(coord.partition_worker)
    coord.last_load_stats = [
        PartitionLoadStats(partition_id=0, owned_drone_count=10, ghost_drone_count=0, candidate_pair_count=10, tick_duration_s=0.001),
        PartitionLoadStats(partition_id=1, owned_drone_count=11, ghost_drone_count=0, candidate_pair_count=10, tick_duration_s=0.001),
    ]
    coord._maybe_rebalance()
    assert coord.partition_worker == before
    assert coord.reassignment_log == []


def test_rebalancing_runs_only_at_configured_interval_end_to_end():
    c = cfg(num_drones=6, seed=7)
    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 100, size=(6, 3))
    velocities = np.zeros((6, 3))
    world = _world(c, positions, velocities, ScriptedMovementAlgorithm.policy_id)
    coord = DistributedCoordinator(
        c, DistributedConfig(num_workers=2, num_partitions=2, rebalance_interval_ticks=3), world=world
    )
    for _ in range(2):
        coord.step()
    assert coord.reassignment_log == []  # not yet at the interval
    coord.step()  # tick 3: rebalance check runs (may or may not reassign)
    # No crash and ticking continues correctly either way.
    assert coord.clock.tick == 3


# --------------------------------------------------------------- fault recovery
def test_worker_failure_never_partially_commits():
    c = cfg(num_drones=2)
    world = _world(
        c, [[10.0, 10.0, 10.0], [80.0, 80.0, 80.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ScriptedMovementAlgorithm.policy_id,
    )
    coord = DistributedCoordinator(
        c, DistributedConfig(num_workers=1, num_partitions=1, worker_retry_limit=1), world=world
    )
    coord.set_fault_injector(lambda worker_id, tick, phase: True)  # always fails

    positions_before = coord.world.state.positions.copy()
    velocities_before = coord.world.state.velocities.copy()
    tick_before = coord.clock.tick

    with pytest.raises((TickCommitError, RuntimeError)):
        coord.step()

    np.testing.assert_array_equal(coord.world.state.positions, positions_before)
    np.testing.assert_array_equal(coord.world.state.velocities, velocities_before)
    assert coord.clock.tick == tick_before


def test_worker_failure_exhausting_retries_raises_tick_commit_error_cleanly():
    """With a spare healthy worker, a failure is reassigned rather than fatal
    -- but if the retry budget is exhausted before that reassigned attempt
    runs, the tick must fail cleanly (TickCommitError, not a crash) with the
    authoritative state still untouched."""
    c = cfg(num_drones=2)
    world = _world(
        c, [[10.0, 10.0, 10.0], [80.0, 80.0, 80.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ScriptedMovementAlgorithm.policy_id,
    )
    coord = DistributedCoordinator(
        c, DistributedConfig(num_workers=2, num_partitions=2, worker_retry_limit=1), world=world
    )
    coord.set_fault_injector(lambda worker_id, tick, phase: phase == "movement")  # any worker, movement phase

    positions_before = coord.world.state.positions.copy()

    with pytest.raises(TickCommitError):
        coord.step()

    np.testing.assert_array_equal(coord.world.state.positions, positions_before)
    assert coord.clock.tick == 0
    # The failing worker was still marked unhealthy and its partition reassigned,
    # even though the retry budget ran out before a healthy attempt could run.
    assert coord.reassignment_log


def test_successful_deterministic_retry_after_worker_failure():
    c = cfg(num_drones=2)
    world_flaky = _world(
        c, [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
        ScriptedMovementAlgorithm.policy_id,
    )
    world_clean = _world(
        c, [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
        ScriptedMovementAlgorithm.policy_id,
    )

    coord_flaky = DistributedCoordinator(
        c, DistributedConfig(num_workers=2, num_partitions=2, worker_retry_limit=2), world=world_flaky
    )
    calls = {"n": 0}

    def fault(worker_id, tick, phase):
        calls["n"] += 1
        return calls["n"] == 1  # fail exactly once, on the very first job attempted

    coord_flaky.set_fault_injector(fault)
    result_flaky = coord_flaky.step()

    coord_clean = DistributedCoordinator(
        c, DistributedConfig(num_workers=2, num_partitions=2, worker_retry_limit=2), world=world_clean
    )
    result_clean = coord_clean.step()

    assert coord_flaky.last_tick_attempts == 2
    np.testing.assert_allclose(coord_flaky.world.state.positions, coord_clean.world.state.positions)
    np.testing.assert_allclose(coord_flaky.world.state.velocities, coord_clean.world.state.velocities)
    assert _pair_set(result_flaky.collision_pairs) == _pair_set(result_clean.collision_pairs)


def test_all_workers_failed_raises_immediately():
    c = cfg(num_drones=1)
    world = _world(c, [[10.0, 10.0, 10.0]], [[0.0, 0.0, 0.0]], ScriptedMovementAlgorithm.policy_id)
    coord = DistributedCoordinator(
        c, DistributedConfig(num_workers=1, num_partitions=1, worker_retry_limit=3), world=world
    )
    coord.set_fault_injector(lambda worker_id, tick, phase: True)
    with pytest.raises((TickCommitError, RuntimeError)):
        coord.step()


# ------------------------------------------------------------------- validation
def test_requires_context_policy_is_rejected():
    c = cfg()
    with pytest.raises(NotImplementedError):
        DistributedCoordinator(
            c,
            DistributedConfig(num_workers=2, num_partitions=2),
            movement=MovementSystem(
                policies={LocalAvoidanceMovementAlgorithm.policy_id: LocalAvoidanceMovementAlgorithm()}
            ),
        )


def test_halo_distance_must_cover_interaction_radius():
    c = cfg(near_miss_radius=5.0, collision_radius=1.0, cell_size=5.0)
    with pytest.raises(ValueError):
        DistributedCoordinator(c, DistributedConfig(num_workers=2, num_partitions=2, halo_distance=0.5))


def test_default_config_is_single_worker_single_partition():
    dc = DistributedConfig()
    assert dc.num_workers == 1
    assert dc.num_partitions is None
