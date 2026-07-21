import numpy as np

from drone_sim.visualization import compute_density_grid, collision_marker_positions


def test_density_grid_shape_and_bounds():
    positions = np.array(
        [[0.0, 0.0, 0.0], [9.9, 9.9, 0.0], [5.0, 5.0, 0.0]], dtype=np.float32
    )
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([10.0, 10.0, 10.0])

    grid, xedges, yedges = compute_density_grid(positions, bounds_min, bounds_max, bins=10)

    assert grid.shape == (10, 10)
    assert xedges[0] == 0.0 and xedges[-1] == 10.0
    assert yedges[0] == 0.0 and yedges[-1] == 10.0
    assert grid.sum() == 3


def test_density_grid_counts_land_in_expected_cell():
    # A single drone near the origin should land in the bottom-left cell.
    positions = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([10.0, 10.0, 10.0])

    grid, _, _ = compute_density_grid(positions, bounds_min, bounds_max, bins=10)

    assert grid[0, 0] == 1
    assert grid.sum() == 1


def test_density_grid_ignores_z():
    # Two drones share x/y but differ in z; they must land in the same cell.
    positions = np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 99.0]], dtype=np.float32
    )
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([10.0, 10.0, 10.0])

    grid, _, _ = compute_density_grid(positions, bounds_min, bounds_max, bins=10)

    assert grid.sum() == 2
    assert grid.max() == 2


def test_collision_marker_positions_midpoint():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    pairs = np.array([[0, 1]], dtype=np.int64)

    markers = collision_marker_positions(positions, pairs)

    assert markers.shape == (1, 2)
    np.testing.assert_allclose(markers[0], [1.0, 2.0])


def test_collision_marker_positions_multiple_pairs():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [6.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    markers = collision_marker_positions(positions, pairs)

    assert markers.shape == (2, 2)
    np.testing.assert_allclose(markers[0], [1.0, 0.0])
    np.testing.assert_allclose(markers[1], [5.0, 5.0])


def test_collision_marker_positions_empty_pairs():
    positions = np.zeros((3, 3), dtype=np.float32)
    pairs = np.empty((0, 2), dtype=np.int64)

    markers = collision_marker_positions(positions, pairs)

    assert markers.shape == (0, 2)
