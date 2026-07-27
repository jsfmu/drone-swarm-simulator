import time

import numpy as np

from drone_sim.collision_queries import query_collision_markers
from drone_sim.snapshot import SimulationSnapshot
from drone_sim.viewport import ViewportQuery


def make_snapshot(positions, velocities, collision_pairs, collision_distances):
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    n = positions.shape[0]
    drone_ids = np.arange(n, dtype=np.int64)
    id_to_row = np.arange(n, dtype=np.int64)
    empty_pairs = np.empty((0, 2), dtype=np.int64)
    empty_dist = np.empty(0, dtype=np.float64)
    return SimulationSnapshot(
        simulation_id="s",
        tick=42,
        time_s=42.0,
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([100.0, 100.0, 100.0]),
        drone_ids=drone_ids,
        positions=positions,
        velocities=velocities,
        id_to_row=id_to_row,
        collision_pairs=np.asarray(collision_pairs, dtype=np.int64),
        collision_distances=np.asarray(collision_distances, dtype=np.float64),
        near_miss_pairs=empty_pairs,
        near_miss_distances=empty_dist,
        num_active_drones=n,
        metrics={},
        captured_at=time.time(),
    )


def test_marker_drone_ids_match_collision_pair():
    positions = [[0, 0, 0], [2, 0, 0]]
    velocities = [[0, 0, 0], [0, 0, 0]]
    snap = make_snapshot(positions, velocities, [[0, 1]], [2.0])
    markers = query_collision_markers(snap)
    assert len(markers) == 1
    assert (markers[0].drone_a, markers[0].drone_b) == (0, 1)


def test_marker_tick_matches_snapshot():
    positions = [[0, 0, 0], [2, 0, 0]]
    velocities = [[0, 0, 0], [0, 0, 0]]
    snap = make_snapshot(positions, velocities, [[0, 1]], [2.0])
    markers = query_collision_markers(snap)
    assert markers[0].tick == snap.tick == 42


def test_marker_midpoint_is_correct():
    positions = [[0, 0, 0], [4, 6, 8]]
    velocities = [[0, 0, 0], [0, 0, 0]]
    snap = make_snapshot(positions, velocities, [[0, 1]], [10.77])
    markers = query_collision_markers(snap)
    m = markers[0]
    assert (m.x, m.y, m.z) == (2.0, 3.0, 4.0)


def test_marker_distance_is_correct():
    positions = [[0, 0, 0], [3, 4, 0]]  # distance 5
    velocities = [[0, 0, 0], [0, 0, 0]]
    snap = make_snapshot(positions, velocities, [[0, 1]], [5.0])
    markers = query_collision_markers(snap)
    assert markers[0].distance == 5.0


def test_relative_speed_is_correct():
    positions = [[0, 0, 0], [1, 0, 0]]
    velocities = [[3, 0, 0], [0, 4, 0]]  # relative vector (3,-4,0), magnitude 5
    snap = make_snapshot(positions, velocities, [[0, 1]], [1.0])
    markers = query_collision_markers(snap)
    assert markers[0].relative_speed == 5.0


def test_viewport_filtering_excludes_collisions_outside_area():
    positions = [[0, 0, 0], [2, 0, 0], [90, 90, 0], [92, 90, 0]]
    velocities = [[0, 0, 0]] * 4
    snap = make_snapshot(positions, velocities, [[0, 1], [2, 3]], [2.0, 2.0])
    markers = query_collision_markers(snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10))
    assert len(markers) == 1
    assert (markers[0].drone_a, markers[0].drone_b) == (0, 1)


def test_altitude_filtering_works():
    positions = [[0, 0, 0], [2, 0, 0], [0, 0, 50], [2, 0, 50]]
    velocities = [[0, 0, 0]] * 4
    snap = make_snapshot(positions, velocities, [[0, 1], [2, 3]], [2.0, 2.0])
    markers = query_collision_markers(
        snap, ViewportQuery(x_min=0, x_max=10, y_min=0, y_max=10, z_min=40, z_max=60)
    )
    assert len(markers) == 1
    assert (markers[0].drone_a, markers[0].drone_b) == (2, 3)


def test_no_duplicate_reversed_pairs():
    positions = [[0, 0, 0], [2, 0, 0]]
    velocities = [[0, 0, 0], [0, 0, 0]]
    # Canonical i<j pair only, as produced by CollisionDetectionEngine/SpatialHashGrid.
    snap = make_snapshot(positions, velocities, [[0, 1]], [2.0])
    markers = query_collision_markers(snap)
    seen = {(m.drone_a, m.drone_b) for m in markers}
    assert len(markers) == len(seen) == 1


def test_empty_collisions_returns_empty_list():
    positions = [[0, 0, 0]]
    velocities = [[0, 0, 0]]
    snap = make_snapshot(positions, velocities, np.empty((0, 2), dtype=np.int64), np.empty(0))
    assert query_collision_markers(snap) == []
