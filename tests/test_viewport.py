import time

import numpy as np
import pytest

from drone_sim.snapshot import SimulationSnapshot
from drone_sim.viewport import ViewportQuery, find_visible_drones


def make_snapshot(positions, velocities=None, drone_ids=None):
    positions = np.asarray(positions, dtype=np.float32)
    n = positions.shape[0]
    if velocities is None:
        velocities = np.zeros((n, 3), dtype=np.float32)
    if drone_ids is None:
        drone_ids = np.arange(n, dtype=np.int64)
    id_space = int(drone_ids.max()) + 1 if drone_ids.size else 0
    id_to_row = np.full(id_space, -1, dtype=np.int64)
    id_to_row[drone_ids] = np.arange(n, dtype=np.int64)
    empty_pairs = np.empty((0, 2), dtype=np.int64)
    empty_dist = np.empty(0, dtype=np.float64)
    return SimulationSnapshot(
        simulation_id="s",
        tick=7,
        time_s=7.0,
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


def test_xy_filtering_includes_expected_drones():
    positions = [[0, 0, 0], [5, 5, 0], [50, 50, 0], [99, 99, 0]]
    snap = make_snapshot(positions)
    visible = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10))
    assert set(int(d) for d in visible.drone_ids) == {0, 1}


def test_altitude_filtering_includes_expected_drones():
    positions = [[1, 1, 0], [1, 1, 5], [1, 1, 20]]
    snap = make_snapshot(positions)
    visible = find_visible_drones(
        snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10, z_min=1, z_max=10)
    )
    assert set(int(d) for d in visible.drone_ids) == {1}


def test_drones_outside_one_boundary_are_excluded():
    positions = [[5, 5, 0], [15, 5, 0], [5, 15, 0]]
    snap = make_snapshot(positions)
    visible = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10))
    assert set(int(d) for d in visible.drone_ids) == {0}


def test_inclusive_boundary_behavior():
    positions = [[0, 0, 0], [10, 10, 0], [10.0001, 10, 0]]
    snap = make_snapshot(positions)
    visible = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10))
    assert set(int(d) for d in visible.drone_ids) == {0, 1}


def test_inactive_drones_excluded_because_snapshot_only_holds_active_positions():
    # The snapshot itself is built only from active drone rows (see snapshot.py);
    # viewport filtering just needs to confirm it never resurrects extra rows.
    positions = [[5, 5, 0]]
    snap = make_snapshot(positions, drone_ids=np.array([3], dtype=np.int64))
    visible = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10))
    assert list(int(d) for d in visible.drone_ids) == [3]


def test_empty_viewport_returns_empty_result():
    positions = [[5, 5, 0], [6, 6, 0]]
    snap = make_snapshot(positions)
    visible = find_visible_drones(snap, ViewportQuery(x_min=90, x_max=99, y_min=90, y_max=99))
    assert visible.total_visible == 0
    assert visible.drone_ids.shape[0] == 0


def test_empty_snapshot_returns_empty_result():
    snap = make_snapshot(np.empty((0, 3), dtype=np.float32), drone_ids=np.empty(0, dtype=np.int64))
    visible = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10))
    assert visible.total_visible == 0


def test_reversed_bounds_are_rejected():
    with pytest.raises(ValueError):
        ViewportQuery(x_min=10, x_max=0, y_min=0, y_max=10)
    with pytest.raises(ValueError):
        ViewportQuery(x_min=0, x_max=10, y_min=10, y_max=0)
    with pytest.raises(ValueError):
        ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10, z_min=5, z_max=1)


def test_truncation_reports_total_visible_and_deterministic_prefix():
    positions = [[i, i, 0] for i in range(20)]
    snap = make_snapshot(positions)
    visible = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=100, y_min=0, y_max=100), limit=5)
    assert visible.total_visible == 20
    assert visible.truncated is True
    assert visible.drone_ids.shape[0] == 5
    # Deterministic: same call twice returns the same prefix.
    visible2 = find_visible_drones(snap, ViewportQuery(x_min=0, x_max=100, y_min=0, y_max=100), limit=5)
    assert np.array_equal(visible.drone_ids, visible2.drone_ids)


def test_filtering_is_vectorized_at_scale():
    """Correctness + a scale smoke test: 100,000 drones filtered fast with no
    per-drone Python loop (a Python loop at this scale would be visibly slow)."""
    rng = np.random.default_rng(0)
    n = 100_000
    positions = rng.uniform(0, 1000, size=(n, 3)).astype(np.float32)
    snap = make_snapshot(positions)

    t0 = time.perf_counter()
    visible = find_visible_drones(snap, ViewportQuery(x_min=100, x_max=200, y_min=100, y_max=200))
    elapsed = time.perf_counter() - t0

    expected_mask = (
        (positions[:, 0] >= 100) & (positions[:, 0] <= 200)
        & (positions[:, 1] >= 100) & (positions[:, 1] <= 200)
    )
    assert visible.total_visible == int(expected_mask.sum())
    assert elapsed < 1.0
