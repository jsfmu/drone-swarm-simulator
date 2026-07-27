import matplotlib

matplotlib.use("Agg")  # headless: must be set before RemoteSimulationViewer's first pyplot import

import numpy as np

from drone_sim.visualization import (
    RemoteSimulationViewer,
    compute_density_grid,
    collision_marker_positions,
)


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


def _attached_remote_viewer(**kwargs):
    """A RemoteSimulationViewer that never touches the network: attaching to
    an existing simulation_id skips create_simulation/start_simulation."""
    defaults = dict(viewport=(0.0, 100.0, 0.0, 100.0), x_bins=2, y_bins=2)
    defaults.update(kwargs)
    return RemoteSimulationViewer("http://example.invalid", simulation_id="test-sim", **defaults)


def test_join_url_carries_viewport_so_browser_can_match_it():
    viewer = _attached_remote_viewer(viewport=(1.0, 501.0, 2.0, 502.0))

    url = viewer.join_url()

    assert url.startswith("http://example.invalid/?")
    assert "simulation_id=test-sim" in url
    assert "x_min=1.0" in url
    assert "x_max=501.0" in url
    assert "y_min=2.0" in url
    assert "y_max=502.0" in url


def test_remote_viewer_shows_per_tick_marker_count_distinct_from_cumulative_total(monkeypatch):
    # Reproduces the exact confusion this was built to fix: a viewer polling
    # a shared simulation must show a per-tick count that's directly
    # comparable to index.html's "collision markers: N" line, not just the
    # ever-growing cumulative total (which looks wildly different from any
    # single-tick number by design -- see runtime.py's RunningMetrics).
    viewer = _attached_remote_viewer()
    fake_frame = {
        "status": "running",
        "tick": 42,
        "num_visible_drones": 10,
        "heatmap": {"counts": [[0, 1], [2, 0]]},
        "markers": [
            {"x": 5.0, "y": 5.0}, {"x": 6.0, "y": 6.0}, {"x": 7.0, "y": 7.0},
        ],
        "metrics": {"total_collisions": 500, "total_near_misses": 900, "ticks_per_second": 12.0},
    }
    monkeypatch.setattr(viewer._api, "get_frame", lambda *a, **k: fake_frame)

    viewer._poll_and_redraw()

    text = viewer.metrics_text.get_text()
    assert "collision markers: 3" in text
    assert "collisions: 500" in text
