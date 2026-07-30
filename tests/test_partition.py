import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.partition import PartitionGrid


def cfg(world=100.0, near=2.0, seed=0):
    return SimulationConfig(
        num_drones=10,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(world, world, world),
        collision_radius=near / 2,
        near_miss_radius=near,
        seed=seed,
    )


def test_every_coordinate_maps_to_exactly_one_partition():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=5)
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.0, 100.0, size=2000)
    owners = grid.owner_of(xs)
    assert owners.shape == xs.shape
    assert (owners >= 0).all() and (owners < 5).all()

    # Every partition covers a contiguous, non-overlapping slice: reconstruct
    # membership by brute-force interval check and compare.
    for part in grid.partitions:
        in_partition = (xs >= part.x_min) & (xs < part.x_max if part.partition_id < 4 else xs <= part.x_max)
        assert set(np.nonzero(in_partition)[0]) == set(np.nonzero(owners == part.partition_id)[0])


def test_partitions_are_contiguous_and_cover_full_world():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=4)
    parts = sorted(grid.partitions, key=lambda p: p.partition_id)
    assert parts[0].x_min == pytest.approx(0.0)
    assert parts[-1].x_max == pytest.approx(100.0)
    for a, b in zip(parts, parts[1:]):
        assert a.x_max == pytest.approx(b.x_min)


def test_owner_of_clips_out_of_range_values():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=4)
    owners = grid.owner_of(np.array([-50.0, 1000.0]))
    assert owners[0] == 0
    assert owners[1] == 3


def test_deterministic_partition_assignment():
    c = cfg(world=100.0)
    positions = np.random.default_rng(1).uniform(0, 100, size=(200, 3)).astype(np.float32)

    grid_a = PartitionGrid(c, num_partitions=6)
    grid_b = PartitionGrid(c, num_partitions=6)
    owners_a = grid_a.assign(positions)
    owners_b = grid_b.assign(positions)
    np.testing.assert_array_equal(owners_a, owners_b)

    # Repeated calls on the same instance are also stable.
    np.testing.assert_array_equal(owners_a, grid_a.assign(positions))


def test_single_partition_owns_everything():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=1)
    positions = np.array([[0.0, 0, 0], [50.0, 0, 0], [99.9, 0, 0]], dtype=np.float32)
    owners = grid.assign(positions)
    assert (owners == 0).all()
    assert grid.neighbors(0) == []


def test_neighbor_discovery():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=4)
    assert grid.neighbors(0) == [1]
    assert grid.neighbors(1) == [0, 2]
    assert grid.neighbors(2) == [1, 3]
    assert grid.neighbors(3) == [2]


def test_ownership_transfer_on_crossing():
    """A drone's owner is purely a function of its current position -- moving
    it across a boundary changes the answer with no extra bookkeeping."""
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=2)  # boundary at x=50
    pos_before = np.array([[49.0, 0, 0]], dtype=np.float32)
    pos_after = np.array([[51.0, 0, 0]], dtype=np.float32)
    assert grid.assign(pos_before)[0] == 0
    assert grid.assign(pos_after)[0] == 1


def test_ghost_export_indices_selects_only_near_boundary_drones():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=2)  # boundary at x=50, partitions [0,50) [50,100]
    halo = 5.0
    positions = np.array(
        [
            [10.0, 0, 0],  # partition 0, far from boundary -- not exported
            [48.0, 0, 0],  # partition 0, within halo of boundary -- exported to partition 1
            [49.9, 0, 0],  # partition 0, within halo -- exported to partition 1
        ],
        dtype=np.float32,
    )
    owned_idx = np.array([0, 1, 2], dtype=np.int64)
    exports = grid.ghost_export_indices(positions, owned_idx, partition_id=0, halo_distance=halo)
    assert set(exports.keys()) == {1}
    np.testing.assert_array_equal(np.sort(exports[1]), [1, 2])


def test_ghost_export_indices_empty_when_far_from_every_boundary():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=3)  # boundaries at x=33.33, 66.67
    positions = np.array([[50.0, 0, 0]], dtype=np.float32)  # middle partition, dead center
    owned_idx = np.array([0], dtype=np.int64)
    exports = grid.ghost_export_indices(positions, owned_idx, partition_id=1, halo_distance=2.0)
    assert exports == {}


def test_ghost_export_indices_no_neighbor_beyond_world_edge():
    c = cfg(world=100.0)
    grid = PartitionGrid(c, num_partitions=2)
    positions = np.array([[1.0, 0, 0]], dtype=np.float32)  # near the world's own edge, not an interior boundary
    owned_idx = np.array([0], dtype=np.int64)
    exports = grid.ghost_export_indices(positions, owned_idx, partition_id=0, halo_distance=5.0)
    assert exports == {}


def test_num_partitions_must_be_positive():
    c = cfg(world=100.0)
    with pytest.raises(ValueError):
        PartitionGrid(c, num_partitions=0)
