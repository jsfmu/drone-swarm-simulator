import numpy as np
import pytest

from drone_sim.config import SimulationConfig
from drone_sim.scenarios import (
    SCENARIOS,
    converging_group,
    crossing_paths,
    head_on_collision,
    near_miss,
    parallel_safe,
    rare_collision_background,
    stationary_obstacle,
)
from drone_sim.simulation import Simulation

NUM_TICKS = 14  # comfortably straddles the _CLOSE_TICKS=5 meeting point


def cfg(**kw):
    base = dict(
        num_drones=0,
        bounds_min=(-500.0, -500.0, -500.0),
        bounds_max=(500.0, 500.0, 500.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        cell_size=2.0,
        dt=1.0,
        seed=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _pair_set(arr):
    return {(int(a), int(b)) for a, b in arr}


def _run_and_collect(world, config, num_ticks):
    """Run a scenario's world and return the union of collision/near-miss
    pairs seen across every tick."""
    sim = Simulation(config, world=world)
    all_collisions = set()
    all_near_misses = set()
    for _ in range(num_ticks):
        result = sim.step()
        all_collisions |= _pair_set(result.collision_pairs)
        all_near_misses |= _pair_set(result.near_miss_pairs)
    return all_collisions, all_near_misses


@pytest.mark.parametrize(
    "factory",
    [head_on_collision, crossing_paths, near_miss, parallel_safe, stationary_obstacle, converging_group],
)
def test_scenarios_are_seed_reproducible(factory):
    config = cfg()
    a = factory(config)
    b = factory(config)
    assert np.array_equal(a.world.state.positions, b.world.state.positions)
    assert np.array_equal(a.world.state.velocities, b.world.state.velocities)
    assert a.expected_collision_pairs == b.expected_collision_pairs
    assert a.expected_near_miss_pairs == b.expected_near_miss_pairs


def test_rare_collision_background_is_seed_reproducible():
    config = cfg(num_drones=0, seed=5)
    a = rare_collision_background(config, num_background=30, seed=5)
    b = rare_collision_background(config, num_background=30, seed=5)
    assert np.array_equal(a.world.state.positions, b.world.state.positions)
    assert np.array_equal(a.world.state.velocities, b.world.state.velocities)
    assert np.array_equal(a.world.state.goal_positions, b.world.state.goal_positions)


@pytest.mark.parametrize("factory", [head_on_collision, crossing_paths, stationary_obstacle])
def test_injected_collision_pairs_are_detected(factory):
    config = cfg()
    scenario = factory(config)
    collisions, near_misses = _run_and_collect(scenario.world, config, NUM_TICKS)

    for pair in scenario.expected_collision_pairs:
        assert pair in collisions, f"expected collision {pair} was never detected"


def test_converging_group_collisions_are_detected():
    config = cfg()
    scenario = converging_group(config, num_drones=5)
    collisions, _ = _run_and_collect(scenario.world, config, NUM_TICKS)

    # All drones meet near the center at ~the same tick: every expected pair
    # should show up as a collision at some point.
    assert set(scenario.expected_collision_pairs) <= collisions


def test_injected_near_miss_is_distinguished_from_collision():
    config = cfg()
    scenario = near_miss(config)
    collisions, near_misses = _run_and_collect(scenario.world, config, NUM_TICKS)

    for pair in scenario.expected_near_miss_pairs:
        assert pair in near_misses, f"expected near miss {pair} was never detected"
        assert pair not in collisions, f"near-miss pair {pair} incorrectly registered as a collision"


def test_safe_control_remains_safe():
    config = cfg()
    scenario = parallel_safe(config)
    collisions, near_misses = _run_and_collect(scenario.world, config, NUM_TICKS)

    assert collisions == set()
    assert near_misses == set()


def test_rare_collision_background_ground_truth_holds():
    config = cfg(seed=1)
    scenario = rare_collision_background(
        config, num_background=50, num_injected_collisions=2, num_injected_near_misses=2, seed=1
    )
    collisions, near_misses = _run_and_collect(scenario.world, config, NUM_TICKS)

    for pair in scenario.expected_collision_pairs:
        assert pair in collisions
    for pair in scenario.expected_near_miss_pairs:
        assert pair in near_misses
        assert pair not in collisions


def test_scenarios_registry_contains_all_seven():
    assert set(SCENARIOS) == {
        "head_on_collision",
        "crossing_paths",
        "near_miss",
        "parallel_safe",
        "stationary_obstacle",
        "converging_group",
        "rare_collision_background",
    }
