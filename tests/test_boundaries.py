import numpy as np

from drone_sim.config import SimulationConfig, BoundaryMode
from drone_sim.state import DroneState, World
from drone_sim.boundaries import BoundaryManager
from drone_sim.simulation import Simulation


def base_config(**kw):
    d = dict(
        num_drones=3,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(10.0, 10.0, 10.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        seed=1,
    )
    d.update(kw)
    return SimulationConfig(**d)


def _world_with(positions, velocities, cfg):
    n = positions.shape[0]
    state = DroneState(
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        active_mask=np.ones(n, dtype=bool),
        movement_policy_ids=np.zeros(n, dtype=np.int32),
    )
    return World(config=cfg, state=state)


def test_reflect_clamps_position_and_flips_velocity():
    cfg = base_config(boundary_mode=BoundaryMode.REFLECT)
    pos = np.array([[-5.0, 5.0, 5.0], [5.0, 15.0, 5.0], [5.0, 5.0, 5.0]])
    vel = np.array([[-2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    world = _world_with(pos, vel, cfg)

    BoundaryManager().apply(world)

    # Drone 0 clamped to x=0 with x-velocity flipped.
    assert world.state.positions[0, 0] == 0.0
    assert world.state.velocities[0, 0] == 2.0
    # Drone 1 clamped to y=10 with y-velocity flipped.
    assert world.state.positions[1, 1] == 10.0
    assert world.state.velocities[1, 1] == -3.0
    # Drone 2 untouched.
    assert np.array_equal(world.state.positions[2], np.array([5, 5, 5], dtype=np.float32))
    assert np.array_equal(world.state.velocities[2], np.array([1, 1, 1], dtype=np.float32))


def test_clamp_mode_zeros_velocity():
    cfg = base_config(boundary_mode=BoundaryMode.CLAMP)
    pos = np.array([[-1.0, 5.0, 5.0]])
    vel = np.array([[-4.0, 0.0, 0.0]])
    world = _world_with(pos, vel, cfg)

    BoundaryManager().apply(world)
    assert world.state.positions[0, 0] == 0.0
    assert world.state.velocities[0, 0] == 0.0


def test_drones_stay_in_bounds_over_a_run():
    cfg = SimulationConfig(
        num_drones=500,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(50.0, 50.0, 50.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        max_speed=8.0,
        seed=42,
    )
    sim = Simulation(cfg)
    sim.run(50)
    pos = sim.world.state.positions
    assert (pos >= cfg.bounds_min_arr - 1e-3).all()
    assert (pos <= cfg.bounds_max_arr + 1e-3).all()
