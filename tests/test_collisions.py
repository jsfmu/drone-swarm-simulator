import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.state import DroneState, World
from drone_sim.spatial_hash import SpatialHashGrid
from drone_sim.collisions import CollisionDetectionEngine, CollisionResolutionEngine
from drone_sim.simulation import Simulation


def cfg(n, world=20.0, coll=1.0, near=2.0, cell=None, seed=0):
    return SimulationConfig(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(world, world, world),
        collision_radius=coll,
        near_miss_radius=near,
        cell_size=cell,
        seed=seed,
    )


def _pair_set(arr):
    return {(int(a), int(b)) for a, b in arr}


def _detect_both(state, c):
    det = CollisionDetectionEngine(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    spatial = det.detect(state, grid)
    brute = det.detect_brute_force(state)
    return spatial, brute


# ----------------------------------------------------------------- equivalence
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("world,coll,near", [
    (15.0, 1.0, 2.0),
    (8.0, 1.0, 3.0),
    (10.0, 0.5, 1.5),
    (6.0, 2.0, 4.0),
])
def test_spatial_matches_brute_force(seed, world, coll, near):
    """Spatial-hash detection must equal the brute-force reference exactly."""
    c = cfg(250, world=world, coll=coll, near=near, seed=seed)
    state = DroneState.generate(c)
    spatial, brute = _detect_both(state, c)

    assert _pair_set(spatial.collision_pairs) == _pair_set(brute.collision_pairs)
    assert _pair_set(spatial.near_miss_pairs) == _pair_set(brute.near_miss_pairs)


def test_equivalence_across_a_full_run():
    """After many ticks of real dynamics, both detectors still agree."""
    c = cfg(150, world=12.0, coll=1.0, near=2.0, seed=99)
    sim = Simulation(c)
    det = CollisionDetectionEngine(c)
    for _ in range(30):
        sim.step()
        state = sim.world.state
        grid = SpatialHashGrid(c)
        grid.build(state.positions, state.active_indices())
        spatial = det.detect(state, grid)
        brute = det.detect_brute_force(state)
        assert _pair_set(spatial.collision_pairs) == _pair_set(brute.collision_pairs)
        assert _pair_set(spatial.near_miss_pairs) == _pair_set(brute.near_miss_pairs)


# --------------------------------------------------------------- exact scenario
def _manual_world(positions, c):
    positions = np.asarray(positions, dtype=np.float32)
    n = positions.shape[0]
    state = DroneState(
        positions=positions,
        velocities=np.zeros((n, 3), dtype=np.float32),
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=np.zeros(n, dtype=np.int32),
    )
    return state


def test_known_collision_and_near_miss_classification():
    c = cfg(0, world=100.0, coll=1.0, near=2.0)
    # pair (0,1): distance 0.5 -> collision
    # pair (2,3): distance 1.5 -> near miss
    # pair (4,5): distance 3.0 -> neither
    positions = [
        [10.0, 10.0, 10.0],
        [10.5, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [21.5, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [33.0, 30.0, 30.0],
    ]
    state = _manual_world(positions, c)
    det = CollisionDetectionEngine(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    res = det.detect(state, grid)

    assert _pair_set(res.collision_pairs) == {(0, 1)}
    assert _pair_set(res.near_miss_pairs) == {(2, 3)}
    # brute force agrees
    b = det.detect_brute_force(state)
    assert _pair_set(b.collision_pairs) == {(0, 1)}
    assert _pair_set(b.near_miss_pairs) == {(2, 3)}


def test_boundary_distances_classify_consistently():
    # distances exactly on the radii boundaries
    c = cfg(0, world=100.0, coll=1.0, near=2.0)
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],   # dist == collision_radius -> collision
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],  # dist == near_miss_radius -> near miss
    ]
    state = _manual_world(positions, c)
    det = CollisionDetectionEngine(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    res = det.detect(state, grid)
    assert _pair_set(res.collision_pairs) == {(0, 1)}
    assert _pair_set(res.near_miss_pairs) == {(2, 3)}


# ------------------------------------------------------------------- resolution
def test_head_on_collision_reverses_velocities():
    c = cfg(0, world=100.0, coll=1.0, near=2.0)
    positions = np.array([[10.0, 10.0, 10.0], [10.5, 10.0, 10.0]], dtype=np.float32)
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    state = DroneState(
        positions=positions,
        velocities=velocities,
        active_mask=np.ones(2, dtype=bool),
        movement_policy_ids=np.zeros(2, dtype=np.int32),
    )
    det = CollisionDetectionEngine(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    res = det.detect(state, grid)
    assert res.num_collisions == 1

    CollisionResolutionEngine(c).resolve(state, res)
    # Equal-mass head-on: velocities exchange -> reverse here.
    assert np.allclose(state.velocities[0], [-1.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(state.velocities[1], [1.0, 0.0, 0.0], atol=1e-4)
    # And they are separated to at least the contact distance.
    d = np.linalg.norm(state.positions[0] - state.positions[1])
    assert d >= c.collision_radius - 1e-4


def test_resolution_conserves_momentum_equal_mass():
    c = cfg(0, world=100.0, coll=1.0, near=2.0)
    positions = np.array([[5.0, 5.0, 5.0], [5.4, 5.2, 5.0]], dtype=np.float32)
    velocities = np.array([[2.0, -1.0, 0.5], [-0.5, 0.5, -1.0]], dtype=np.float32)
    state = DroneState(
        positions=positions,
        velocities=velocities,
        active_mask=np.ones(2, dtype=bool),
        movement_policy_ids=np.zeros(2, dtype=np.int32),
    )
    p_before = state.velocities.sum(axis=0).copy()
    ke_before = np.sum(state.velocities.astype(np.float64) ** 2)

    det = CollisionDetectionEngine(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    res = det.detect(state, grid)
    CollisionResolutionEngine(c).resolve(state, res)

    p_after = state.velocities.sum(axis=0)
    ke_after = np.sum(state.velocities.astype(np.float64) ** 2)
    assert np.allclose(p_before, p_after, atol=1e-3)   # momentum conserved
    assert np.isclose(ke_before, ke_after, atol=1e-2)  # elastic: KE conserved


def test_detection_result_events_materialize():
    c = cfg(0, world=100.0, coll=1.0, near=2.0)
    positions = [[10.0, 10.0, 10.0], [10.5, 10.0, 10.0], [20.0, 20.0, 20.0], [21.5, 20.0, 20.0]]
    state = _manual_world(positions, c)
    det = CollisionDetectionEngine(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    res = det.detect(state, grid)
    coll_events = res.collision_events(state, tick=5)
    near_events = res.near_miss_events(tick=5)
    assert len(coll_events) == 1 and coll_events[0].tick == 5
    assert coll_events[0].distance == pytest.approx(0.5, abs=1e-4)
    assert len(near_events) == 1 and near_events[0].min_distance == pytest.approx(1.5, abs=1e-4)
