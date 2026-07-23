"""Tests for benchmarks/benchmark_avoidance.py's setup helpers and a tiny
end-to-end invocation. Imports the benchmark module directly (it is a script,
not part of the installed drone_sim package), matching how the script itself
resolves drone_sim via sys.path.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_DIR = _REPO_ROOT / "benchmarks"
sys.path.insert(0, str(_BENCH_DIR))

import benchmarks.benchmark_avoidance as ba  # noqa: E402
from drone_sim.movement import GoalDirectedMovementAlgorithm, LocalAvoidanceMovementAlgorithm  # noqa: E402


def test_cloned_worlds_are_identical_except_policy_id():
    config = ba.build_config(200, seed=7)
    template = ba.make_template_world(config)

    world_gd = ba.clone_world_for_policy(template, GoalDirectedMovementAlgorithm.policy_id)
    world_la = ba.clone_world_for_policy(template, LocalAvoidanceMovementAlgorithm.policy_id)

    assert np.array_equal(world_gd.state.positions, world_la.state.positions)
    assert np.array_equal(world_gd.state.velocities, world_la.state.velocities)
    assert np.array_equal(world_gd.state.active_mask, world_la.state.active_mask)
    assert np.array_equal(world_gd.state.goal_positions, world_la.state.goal_positions)
    assert np.all(world_gd.state.movement_policy_ids == GoalDirectedMovementAlgorithm.policy_id)
    assert np.all(world_la.state.movement_policy_ids == LocalAvoidanceMovementAlgorithm.policy_id)

    # And both match the (untouched) template itself.
    assert np.array_equal(world_gd.state.positions, template.state.positions)
    assert np.array_equal(world_la.state.positions, template.state.positions)


def test_cloning_does_not_mutate_the_template():
    config = ba.build_config(100, seed=3)
    template = ba.make_template_world(config)
    positions_before = template.state.positions.copy()

    world = ba.clone_world_for_policy(template, LocalAvoidanceMovementAlgorithm.policy_id)
    world.state.positions[:] = 0.0  # mutate the clone
    world.state.movement_policy_ids[:] = 999

    assert np.array_equal(template.state.positions, positions_before)
    assert not np.all(template.state.movement_policy_ids == 999)


def test_template_world_is_seed_reproducible():
    config_a = ba.build_config(100, seed=11)
    config_b = ba.build_config(100, seed=11)
    world_a = ba.make_template_world(config_a)
    world_b = ba.make_template_world(config_b)

    assert np.array_equal(world_a.state.positions, world_b.state.positions)
    assert np.array_equal(world_a.state.velocities, world_b.state.velocities)
    assert np.array_equal(world_a.state.goal_positions, world_b.state.goal_positions)


def test_run_policy_benchmark_completes_and_reports_expected_fields():
    config = ba.build_config(40, seed=1)
    template = ba.make_template_world(config)
    result = ba.run_policy_benchmark(
        config, template, "local_avoidance", LocalAvoidanceMovementAlgorithm.policy_id,
        LocalAvoidanceMovementAlgorithm, ticks=2, warmup=1,
    )
    assert result.drones == 40
    assert result.mean_ms >= 0
    assert result.unique_collision_events >= 0
    assert result.collision_pair_ticks >= result.unique_collision_events - 1e-9 or result.unique_collision_events == 0
    assert 0.0 <= result.destination_completion_rate <= 1.0
    assert 0.0 <= result.stationary_percentage <= 1.0


def test_stage_profile_marks_goal_directed_stages_skipped():
    config = ba.build_config(40, seed=1)
    template = ba.make_template_world(config)
    profile = ba.run_stage_profile(
        config, template, GoalDirectedMovementAlgorithm.policy_id, GoalDirectedMovementAlgorithm, ticks=2, warmup=1,
    )
    assert profile.context_stages_skipped is True
    assert profile.pre_grid_ns == 0
    assert profile.prediction_ns == 0


def test_memory_footprint_is_positive_and_consistent():
    config = ba.build_config(60, seed=1)
    template = ba.make_template_world(config)
    mem = ba.measure_memory_footprint(config, template)

    assert mem.drone_state_bytes > 0
    assert mem.total_tracked_bytes == (
        mem.drone_state_bytes + mem.candidate_pair_bytes + mem.context_bytes + mem.prediction_bytes
    )


def test_tiny_benchmark_invocation_completes_successfully():
    """A full CLI invocation with tiny parameters must exit cleanly and
    produce output, without asserting anything about wall-clock speed."""
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH_DIR / "benchmark_avoidance.py"),
            "--sizes", "30", "--ticks", "2", "--warmup", "1", "--seeds", "1", "2",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "goal_directed" in proc.stdout
    assert "local_avoidance" in proc.stdout
    assert "Slowdown ratio" in proc.stdout
