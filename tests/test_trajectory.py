import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.state import DroneState
from drone_sim.trajectory import PredictedRisk, TrajectoryPredictionService


def cfg(**kw):
    base = dict(
        num_drones=0,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1000.0, 1000.0, 1000.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        prediction_horizon=10.0,
        seed=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _state(positions, velocities):
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    n = positions.shape[0]
    return DroneState(
        positions=positions,
        velocities=velocities,
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=np.zeros(n, dtype=np.int32),
    )


def test_head_on_pair_produces_expected_closest_approach_time():
    c = cfg()
    # i at x=0 moving +1; j at x=10 moving -1. Closest approach (distance 0)
    # at t = 5 (relative closing speed 2, gap 10).
    state = _state([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    result = TrajectoryPredictionService(c).predict(state, np.array([[0, 1]], dtype=np.int64))

    assert result.time_to_closest_approach[0] == pytest.approx(5.0, abs=1e-6)
    assert result.predicted_separation[0] == pytest.approx(0.0, abs=1e-6)
    assert result.risk[0] == PredictedRisk.PREDICTED_COLLISION


def test_parallel_pair_produces_no_false_predicted_collision():
    c = cfg()
    # Same velocity, offset by 5 in y (near_miss_radius=2) -> never converges.
    state = _state([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = TrajectoryPredictionService(c).predict(state, np.array([[0, 1]], dtype=np.int64))

    assert result.risk[0] == PredictedRisk.NOT_CLOSING_OR_OUTSIDE_HORIZON
    assert result.predicted_separation[0] == pytest.approx(5.0, abs=1e-6)


def test_zero_relative_velocity_does_not_produce_nan_or_inf():
    c = cfg()
    # Identical velocities -> zero relative velocity -> would divide by zero
    # without the guard.
    state = _state([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], [[3.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
    result = TrajectoryPredictionService(c).predict(state, np.array([[0, 1]], dtype=np.int64))

    assert np.isfinite(result.time_to_closest_approach).all()
    assert np.isfinite(result.predicted_separation).all()
    assert result.time_to_closest_approach[0] == 0.0
    assert result.predicted_separation[0] == pytest.approx(1.5, abs=1e-6)


def test_predicted_separation_matches_analytical_case():
    c = cfg()
    # i stationary at origin; j moving perpendicular to the line joining them
    # (pure "flyby"): starts at (10, -20, 0) moving +y at speed 4.
    # Closest approach happens when the y-component of relative position is 0,
    # i.e. after 20/4 = 5 seconds, at which point separation = 10 (pure x gap).
    state = _state([[0.0, 0.0, 0.0], [10.0, -20.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    result = TrajectoryPredictionService(c).predict(state, np.array([[0, 1]], dtype=np.int64))

    assert result.time_to_closest_approach[0] == pytest.approx(5.0, abs=1e-6)
    assert result.predicted_separation[0] == pytest.approx(10.0, abs=1e-6)
    assert result.risk[0] == PredictedRisk.CURRENTLY_SAFE


def test_prediction_horizon_clips_time_to_closest_approach():
    c = cfg(prediction_horizon=2.0)
    # Same head-on pair as above but the true closest approach (t=5) is well
    # beyond the horizon (2.0).
    state = _state([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    result = TrajectoryPredictionService(c).predict(state, np.array([[0, 1]], dtype=np.int64))

    assert result.time_to_closest_approach[0] == pytest.approx(2.0, abs=1e-6)
    assert result.risk[0] == PredictedRisk.NOT_CLOSING_OR_OUTSIDE_HORIZON


def test_output_shapes_and_dtypes():
    c = cfg()
    n = 6
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, size=(n, 3))
    velocities = rng.uniform(-2, 2, size=(n, 3))
    state = _state(positions, velocities)
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)

    result = TrajectoryPredictionService(c).predict(state, pairs)

    assert result.pairs.shape == (5, 2)
    assert result.time_to_closest_approach.shape == (5,)
    assert result.predicted_separation.shape == (5,)
    assert result.risk.shape == (5,)
    assert result.time_to_closest_approach.dtype == np.float64
    assert result.predicted_separation.dtype == np.float64
    assert result.risk.dtype == np.int8
    assert result.num_pairs == 5


def test_empty_candidate_pairs_returns_empty_arrays():
    c = cfg()
    state = _state([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    result = TrajectoryPredictionService(c).predict(state, np.empty((0, 2), dtype=np.int64))

    assert result.pairs.shape == (0, 2)
    assert result.time_to_closest_approach.shape == (0,)
    assert result.predicted_separation.shape == (0,)
    assert result.risk.shape == (0,)
