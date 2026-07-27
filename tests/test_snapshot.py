import numpy as np

from drone_sim.config import SimulationConfig
from drone_sim.simulation import Simulation
from drone_sim.snapshot import build_snapshot


def cfg(n, world=20.0, coll=1.0, near=2.0, seed=0):
    return SimulationConfig(
        num_drones=n,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(world, world, world),
        collision_radius=coll,
        near_miss_radius=near,
        seed=seed,
    )


def test_snapshot_tick_is_correct():
    sim = Simulation(cfg(20))
    sim.step()
    sim.step()
    snap = build_snapshot("sim-1", sim, None, sim.metrics.summary())
    assert snap.tick == 2


def test_snapshot_does_not_alias_mutable_arrays():
    sim = Simulation(cfg(20))
    result = sim.step()
    snap = build_snapshot("sim-1", sim, result, sim.metrics.summary())

    original = snap.positions.copy()
    sim.world.state.positions[:] = 0.0  # mutate the live state directly
    assert np.array_equal(snap.positions, original)


def test_advancing_simulation_does_not_mutate_older_snapshot():
    sim = Simulation(cfg(20))
    sim.step()
    snap1 = build_snapshot("sim-1", sim, None, sim.metrics.summary())
    positions_before = snap1.positions.copy()

    for _ in range(5):
        sim.step()

    assert np.array_equal(snap1.positions, positions_before)
    assert snap1.tick == 1  # unchanged even though sim.clock.tick advanced


def test_positions_collisions_and_metrics_belong_to_same_tick():
    sim = Simulation(cfg(30, world=6.0, coll=1.0, near=1.5))
    result = None
    for _ in range(5):
        result = sim.step()
    snap = build_snapshot("sim-1", sim, result, sim.metrics.summary())

    assert snap.tick == sim.clock.tick
    assert snap.metrics["num_ticks"] == 5
    # Collision pairs captured are exactly the ones from the tick that produced `result`.
    assert np.array_equal(snap.collision_pairs, result.collision_pairs)


def test_snapshot_arrays_are_internally_consistent():
    """positions/velocities/drone_ids stay row-aligned (same length) -- the
    property that matters for a single build_snapshot() call. True
    concurrent-read safety (many readers while a background thread steps)
    is guarded by SimulationRuntime's lock and is covered in test_runtime.py.
    """
    sim = Simulation(cfg(50, world=10.0))
    sim.step()
    snap = build_snapshot("sim-1", sim, None, sim.metrics.summary())
    assert snap.positions.shape[0] == snap.velocities.shape[0] == snap.drone_ids.shape[0]
    assert snap.positions.shape[0] == snap.num_active_drones


def test_build_snapshot_does_not_compute_metrics_itself():
    """build_snapshot() must not call sim.metrics.summary() (or otherwise scan
    the tick history) -- that is the O(ticks-so-far) cost that caused the
    Phase 3A browser runtime to slow down over a session while the Matplotlib
    viewer (which never calls summary()) stayed fast. Passing metrics={}
    while sim has real history recorded proves build_snapshot() trusts the
    caller's dict rather than recomputing it.
    """
    sim = Simulation(cfg(20))
    for _ in range(10):
        sim.step()
    snap = build_snapshot("sim-1", sim, None, {})
    assert snap.metrics == {}
