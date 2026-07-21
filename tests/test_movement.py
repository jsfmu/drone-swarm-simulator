import numpy as np
import pytest

from drone_sim.config import SimulationConfig, BoundaryMode
from drone_sim.state import DroneState, World
from drone_sim.movement import (
    MovementSystem,
    ScriptedMovementAlgorithm,
    RandomMovementAlgorithm,
)
from drone_sim.simulation import Simulation


def make_config(**kw):
    base = dict(
        num_drones=200,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(100.0, 100.0, 100.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        dt=1.0,
        seed=7,
    )
    base.update(kw)
    return SimulationConfig(**base)


def test_generation_is_reproducible():
    cfg = make_config()
    a = DroneState.generate(cfg)
    b = DroneState.generate(cfg)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.velocities, b.velocities)


def test_generation_within_bounds():
    cfg = make_config(num_drones=1000)
    s = DroneState.generate(cfg)
    assert (s.positions >= cfg.bounds_min_arr).all()
    assert (s.positions <= cfg.bounds_max_arr).all()


def test_state_dtypes_and_shapes():
    cfg = make_config(num_drones=50)
    s = DroneState.generate(cfg)
    assert s.positions.shape == (50, 3)
    assert s.positions.dtype == np.float32
    assert s.velocities.dtype == np.float32
    assert s.active_mask.dtype == np.bool_
    assert s.active_mask.all()


def test_scripted_movement_is_constant_velocity():
    # A single scripted drone should move linearly by v*dt each tick.
    cfg = make_config(num_drones=1, bounds_max=(1e9, 1e9, 1e9), dt=0.5)
    world = World.create(cfg)
    world.state.movement_policy_ids[:] = ScriptedMovementAlgorithm.policy_id
    v0 = world.state.velocities[0].copy()
    p0 = world.state.positions[0].copy()

    ms = MovementSystem()
    rng = np.random.default_rng(0)
    for t in range(5):
        ms.step(world.state, rng, cfg, t)

    expected = p0 + v0 * cfg.dt * 5
    assert np.allclose(world.state.positions[0], expected, atol=1e-3)


def test_random_movement_reproducible_and_bounded():
    cfg = make_config(num_drones=64, seed=123)
    s1 = Simulation(cfg)
    s2 = Simulation(cfg)
    s1.run(10)
    s2.run(10)
    assert np.array_equal(s1.world.state.positions, s2.world.state.positions)
    assert np.array_equal(s1.world.state.velocities, s2.world.state.velocities)
    # Velocities stay clipped to +/- max_speed.
    assert (np.abs(s1.world.state.velocities) <= cfg.max_speed + 1e-4).all()


def test_batched_policies_apply_per_group():
    cfg = make_config(num_drones=10, bounds_max=(1e9, 1e9, 1e9))
    world = World.create(cfg)
    # Half scripted, half random.
    world.state.movement_policy_ids[:5] = ScriptedMovementAlgorithm.policy_id
    world.state.movement_policy_ids[5:] = RandomMovementAlgorithm.policy_id
    scripted_v0 = world.state.velocities[:5].copy()

    ms = MovementSystem()
    rng = np.random.default_rng(1)
    ms.step(world.state, rng, cfg, 0)

    # Scripted drones keep their velocity exactly.
    assert np.array_equal(world.state.velocities[:5], scripted_v0)
