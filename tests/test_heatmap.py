import time

import numpy as np
import pytest

from drone_sim.heatmap import MAX_BINS_PER_AXIS, HeatmapQuery, compute_heatmap
from drone_sim.snapshot import SimulationSnapshot
from drone_sim.viewport import ViewportQuery


def make_snapshot(positions):
    positions = np.asarray(positions, dtype=np.float32)
    n = positions.shape[0]
    velocities = np.zeros((n, 3), dtype=np.float32)
    drone_ids = np.arange(n, dtype=np.int64)
    id_to_row = np.arange(n, dtype=np.int64)
    empty_pairs = np.empty((0, 2), dtype=np.int64)
    empty_dist = np.empty(0, dtype=np.float64)
    return SimulationSnapshot(
        simulation_id="s",
        tick=3,
        time_s=3.0,
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([100.0, 100.0, 100.0]),
        drone_ids=drone_ids,
        positions=positions,
        velocities=velocities,
        id_to_row=id_to_row,
        collision_pairs=empty_pairs,
        collision_distances=empty_dist,
        near_miss_pairs=empty_pairs,
        near_miss_distances=empty_dist,
        num_active_drones=n,
        metrics={},
        captured_at=time.time(),
    )


def test_count_sum_equals_visible_drone_count():
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 100, size=(500, 3)).astype(np.float32)
    snap = make_snapshot(positions)
    query = HeatmapQuery(viewport=ViewportQuery(x_min=0, x_max=100, y_min=0, y_max=100), x_bins=10, y_bins=10)
    result = compute_heatmap(snap, query)
    assert int(result.counts.sum()) == result.num_drones_included == 500


def test_known_positions_land_in_expected_bins():
    positions = [[5, 5, 0], [95, 95, 0], [5, 95, 0]]
    snap = make_snapshot(positions)
    query = HeatmapQuery(viewport=ViewportQuery(x_min=0, x_max=100, y_min=0, y_max=100), x_bins=10, y_bins=10)
    result = compute_heatmap(snap, query)
    assert result.counts[0, 0] == 1     # (5,5)  -> bottom-left bin
    assert result.counts[9, 9] == 1     # (95,95) -> top-right bin
    assert result.counts[9, 0] == 1     # (5,95)  -> top-left bin
    assert result.max_density == 1


def test_empty_input_produces_valid_zero_count_grid():
    snap = make_snapshot(np.empty((0, 3), dtype=np.float32))
    query = HeatmapQuery(viewport=ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10), x_bins=4, y_bins=4)
    result = compute_heatmap(snap, query)
    assert result.counts.shape == (4, 4)
    assert int(result.counts.sum()) == 0
    assert result.max_density == 0
    assert result.x_edges.shape[0] == 5
    assert result.y_edges.shape[0] == 5


def test_viewport_range_determines_edges():
    positions = [[5, 5, 0]]
    snap = make_snapshot(positions)
    query = HeatmapQuery(viewport=ViewportQuery(x_min=20, x_max=40, y_min=-10, y_max=10), x_bins=2, y_bins=2)
    result = compute_heatmap(snap, query)
    assert result.x_edges[0] == 20 and result.x_edges[-1] == 40
    assert result.y_edges[0] == -10 and result.y_edges[-1] == 10
    # The single drone is outside [20,40]x[-10,10], so nothing is counted.
    assert int(result.counts.sum()) == 0


def test_invalid_bin_counts_are_rejected():
    vp = ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10)
    with pytest.raises(ValueError):
        HeatmapQuery(viewport=vp, x_bins=0, y_bins=10)
    with pytest.raises(ValueError):
        HeatmapQuery(viewport=vp, x_bins=10, y_bins=-1)


def test_large_bin_count_requests_are_bounded():
    vp = ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10)
    with pytest.raises(ValueError):
        HeatmapQuery(viewport=vp, x_bins=MAX_BINS_PER_AXIS + 1, y_bins=10)


def test_heatmap_at_100k_positions_no_python_loop():
    rng = np.random.default_rng(2)
    positions = rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32)
    snap = make_snapshot(positions)
    query = HeatmapQuery(viewport=ViewportQuery(x_min=0, x_max=1000, y_min=0, y_max=1000), x_bins=100, y_bins=100)

    t0 = time.perf_counter()
    result = compute_heatmap(snap, query)
    elapsed = time.perf_counter() - t0

    assert int(result.counts.sum()) == 100_000
    assert elapsed < 2.0
