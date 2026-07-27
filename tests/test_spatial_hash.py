import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.state import DroneState
from drone_sim.spatial_hash import SpatialHashGrid, _FORWARD_OFFSETS


def cfg(n, world=50.0, near=2.0, cell=None, seed=0):
    return SimulationConfig(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(world, world, world),
        collision_radius=near / 2,
        near_miss_radius=near,
        cell_size=cell,
        seed=seed,
    )


def _pair_set(arr):
    return {(int(a), int(b)) for a, b in arr}


def _all_pairs_within(positions, radius):
    """Brute-force set of unordered pairs within ``radius`` (inclusive)."""
    n = positions.shape[0]
    p = positions.astype(np.float64)
    out = set()
    for i in range(n):
        d = np.linalg.norm(p[i] - p, axis=1)
        for j in range(i + 1, n):
            if d[j] <= radius:
                out.add((i, j))
    return out


def test_forward_offsets_are_thirteen_and_antipodal_free():
    assert _FORWARD_OFFSETS.shape == (13, 3)
    s = {tuple(o) for o in _FORWARD_OFFSETS}
    for o in s:
        assert tuple(-np.array(o)) not in s


def test_candidate_pairs_are_unique():
    c = cfg(400, world=20.0, near=2.0, seed=3)  # fairly dense
    state = DroneState.generate(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    pairs = grid.candidate_pairs()
    # All i<j.
    assert (pairs[:, 0] < pairs[:, 1]).all()
    # No duplicates.
    s = _pair_set(pairs)
    assert len(s) == pairs.shape[0]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("world,near", [(20.0, 2.0), (10.0, 3.0), (30.0, 1.0)])
def test_candidates_are_superset_of_true_neighbors(seed, world, near):
    """The candidate set must contain every pair within the interaction radius."""
    c = cfg(300, world=world, near=near, seed=seed)
    state = DroneState.generate(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    cand = _pair_set(grid.candidate_pairs())

    truth = _all_pairs_within(state.positions, near)
    missing = truth - cand
    assert not missing, f"spatial hash missed {len(missing)} interacting pairs"


def test_cell_size_defaults_to_near_miss_radius():
    c = cfg(10)
    grid = SpatialHashGrid(c)
    assert grid.cell_size == c.near_miss_radius


def test_cell_size_smaller_than_radius_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(
            num_drones=10,
            near_miss_radius=2.0,
            collision_radius=1.0,
            cell_size=1.0,  # < near_miss_radius
        )


def test_empty_and_single_drone_grids():
    for n in (0, 1):
        c = cfg(n)
        state = DroneState.generate(c)
        grid = SpatialHashGrid(c)
        grid.build(state.positions, state.active_indices())
        assert grid.candidate_pairs().shape == (0, 2)


def test_occupancy_stats_matches_manual_count():
    c = cfg(400, world=20.0, near=2.0, seed=3)  # fairly dense -- some cells hold >1 drone
    state = DroneState.generate(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())

    coords = grid.cell_coords(state.positions[state.active_indices()])
    keys = coords[:, 0] + coords[:, 1] * grid.nx + coords[:, 2] * (grid.nx * grid.ny)
    _, counts = np.unique(keys, return_counts=True)

    occupied_cells, mean_occupancy, max_occupancy = grid.occupancy_stats()
    assert occupied_cells == counts.size
    assert max_occupancy == int(counts.max())
    assert mean_occupancy == pytest.approx(float(counts.mean()))
    # Every drone belongs to exactly one occupied cell.
    assert counts.sum() == state.active_indices().size


def test_occupancy_stats_empty_grid_is_zeroed():
    c = cfg(0)
    state = DroneState.generate(c)
    grid = SpatialHashGrid(c)
    grid.build(state.positions, state.active_indices())
    assert grid.occupancy_stats() == (0, 0.0, 0)


def test_occupancy_stats_requires_build_first():
    grid = SpatialHashGrid(cfg(10))
    with pytest.raises(RuntimeError):
        grid.occupancy_stats()
