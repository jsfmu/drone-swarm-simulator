import json

import numpy as np
import pytest

from drone_sim.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from drone_sim.config import SimulationConfig
from drone_sim.movement import (
    GoalDirectedMovementAlgorithm,
    LocalAvoidanceMovementAlgorithm,
    MovementSystem,
)
from drone_sim.scenarios import head_on_collision
from drone_sim.simulation import Simulation
from drone_sim.state import World


def make_config(**kw):
    base = dict(
        num_drones=40,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(50.0, 50.0, 50.0),
        collision_radius=1.0,
        near_miss_radius=2.0,
        cell_size=2.0,
        seed=5,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _goal_world(config, policy_id, rng_seed=1):
    world = World.create(config)
    center = (config.bounds_min_arr + config.bounds_max_arr) / 2.0
    world.state.goal_positions = (
        2.0 * center[None, :] - world.state.positions.astype(np.float64)
    ).astype(np.float32)
    world.state.movement_policy_ids[:] = policy_id
    return world


def _goal_directed_sim(config):
    algo = GoalDirectedMovementAlgorithm()
    ms = MovementSystem(policies={algo.policy_id: algo})
    world = _goal_world(config, algo.policy_id)
    return Simulation(config, movement=ms, world=world), ms


def _local_avoidance_sim(config):
    algo = LocalAvoidanceMovementAlgorithm()
    ms = MovementSystem(policies={algo.policy_id: algo})
    world = _goal_world(config, algo.policy_id)
    return Simulation(config, movement=ms, world=world), ms


# ------------------------------------------------------------- basic round-trip
def test_round_trip_preserves_tick_time_and_state(tmp_path):
    config = make_config()
    sim, ms = _goal_directed_sim(config)
    for _ in range(4):
        sim.step()

    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)
    resumed = load_checkpoint(ckpt, movement=ms)

    assert resumed.clock.tick == sim.clock.tick == 4
    assert resumed.clock.time_s == pytest.approx(sim.clock.time_s)
    assert np.array_equal(resumed.world.state.positions, sim.world.state.positions)
    assert np.array_equal(resumed.world.state.velocities, sim.world.state.velocities)
    assert np.array_equal(resumed.world.state.active_mask, sim.world.state.active_mask)
    assert np.array_equal(resumed.world.state.movement_policy_ids, sim.world.state.movement_policy_ids)
    assert np.array_equal(resumed.world.state.goal_positions, sim.world.state.goal_positions)
    # A freshly-loaded checkpoint must never carry over metrics history.
    assert resumed.metrics.ticks == []


def test_round_trip_without_goal_positions(tmp_path):
    """Phase 1 default (Random/Scripted) worlds never set goal_positions."""
    config = make_config()
    sim = Simulation(config)  # default MovementSystem: Random+Scripted
    sim.step()

    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)
    resumed = load_checkpoint(ckpt)

    assert resumed.world.state.goal_positions is None
    assert np.array_equal(resumed.world.state.positions, sim.world.state.positions)


def test_round_trip_preserves_inactive_drones(tmp_path):
    config = make_config()
    sim, ms = _goal_directed_sim(config)
    sim.world.state.active_mask[0] = False
    sim.world.state.active_mask[7] = False
    for _ in range(3):
        sim.step()

    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)
    resumed = load_checkpoint(ckpt, movement=ms)

    assert not resumed.world.state.active_mask[0]
    assert not resumed.world.state.active_mask[7]
    assert np.array_equal(resumed.world.state.active_mask, sim.world.state.active_mask)


# --------------------------------------------------------- deterministic resume
def test_deterministic_resume_matches_uninterrupted_run_goal_directed(tmp_path):
    """Core Phase 5 acceptance criterion: run N ticks -> checkpoint -> M more
    ticks, versus separately loading the checkpoint and running M ticks --
    authoritative state must match exactly."""
    config = make_config()
    sim, ms = _goal_directed_sim(config)

    for _ in range(6):
        sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    for _ in range(5):
        sim.step()
    uninterrupted_positions = sim.world.state.positions.copy()
    uninterrupted_velocities = sim.world.state.velocities.copy()
    uninterrupted_tick = sim.clock.tick

    resumed = load_checkpoint(ckpt, movement=ms)
    for _ in range(5):
        resumed.step()

    assert resumed.clock.tick == uninterrupted_tick
    assert np.array_equal(resumed.world.state.positions, uninterrupted_positions)
    assert np.array_equal(resumed.world.state.velocities, uninterrupted_velocities)


def test_deterministic_resume_matches_uninterrupted_run_local_avoidance(tmp_path):
    """Same equivalence check, but for the context-requiring
    LocalAvoidanceMovementAlgorithm path (pre-movement grid, trajectory
    prediction, MovementContext) -- not just the simpler goal-directed one."""
    config = make_config()
    sim, ms = _local_avoidance_sim(config)

    for _ in range(4):
        sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    for _ in range(4):
        sim.step()
    uninterrupted_positions = sim.world.state.positions.copy()

    resumed = load_checkpoint(ckpt, movement=ms)
    for _ in range(4):
        resumed.step()

    assert np.array_equal(resumed.world.state.positions, uninterrupted_positions)


def test_deterministic_resume_with_random_policy_matches_rng_stream(tmp_path):
    """RandomMovementAlgorithm consumes the movement RNG every tick -- this
    only reproduces post-resume if the RNG's bit-generator state (not just
    config.seed) is captured and restored exactly."""
    config = make_config(num_drones=15)
    sim = Simulation(config)  # default registry includes RandomMovementAlgorithm (id 0)

    for _ in range(3):
        sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    for _ in range(6):
        sim.step()
    uninterrupted_velocities = sim.world.state.velocities.copy()

    resumed = load_checkpoint(ckpt)
    for _ in range(6):
        resumed.step()

    assert np.array_equal(resumed.world.state.velocities, uninterrupted_velocities)


def test_deterministic_resume_with_controlled_collision_scenario(tmp_path):
    """A known collision-course scenario (head_on_collision: two drones
    guaranteed to collide exactly 5 ticks in) must still collide identically
    whether observed in one continuous run or resumed from a checkpoint,
    checkpointing right before the collision tick."""
    config = make_config(num_drones=2, bounds_max=(200.0, 200.0, 200.0))
    scenario = head_on_collision(config)
    original_velocities = scenario.world.state.velocities.copy()
    sim = Simulation(config, world=scenario.world)

    for _ in range(3):  # checkpoint before the collision (collision is at tick 5)
        sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    for _ in range(4):
        sim.step()
    uninterrupted_positions = sim.world.state.positions.copy()
    uninterrupted_velocities = sim.world.state.velocities.copy()
    # Sanity: the scenario actually produced a collision response (resolution
    # changes velocities away from the two drones' original closing values).
    assert not np.array_equal(uninterrupted_velocities, original_velocities)

    resumed = load_checkpoint(ckpt)
    for _ in range(4):
        resumed.step()

    assert np.array_equal(resumed.world.state.positions, uninterrupted_positions)
    assert np.array_equal(resumed.world.state.velocities, uninterrupted_velocities)


# -------------------------------------------------------------------- validation
def test_validate_checkpoint_returns_metadata(tmp_path):
    config = make_config()
    sim, _ = _goal_directed_sim(config)
    sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    meta = validate_checkpoint(ckpt, expected_num_drones=config.num_drones)
    assert meta["tick"] == 1
    assert meta["schema_version"] == CHECKPOINT_SCHEMA_VERSION


def test_validate_checkpoint_rejects_drone_count_mismatch(tmp_path):
    config = make_config()
    sim, _ = _goal_directed_sim(config)
    sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    with pytest.raises(CheckpointError, match="num_drones"):
        validate_checkpoint(ckpt, expected_num_drones=config.num_drones + 1)


def test_validate_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        validate_checkpoint(tmp_path / "does_not_exist.npz")


def test_validate_checkpoint_rejects_schema_version_mismatch(tmp_path):
    config = make_config(num_drones=5)
    sim, _ = _goal_directed_sim(config)
    sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    with np.load(ckpt, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        arrays = {k: data[k] for k in data.files if k != "meta_json"}
    meta["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    np.savez(ckpt, meta_json=np.array(json.dumps(meta)), **arrays)

    with pytest.raises(CheckpointError, match="schema_version"):
        validate_checkpoint(ckpt)


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda p: p.write_bytes(p.read_bytes()[: p.stat().st_size // 2]), id="truncated"),
        pytest.param(lambda p: p.write_bytes(b"not a checkpoint at all" * 10), id="garbage"),
    ],
)
def test_validate_checkpoint_rejects_corrupted_file(tmp_path, corrupt):
    config = make_config(num_drones=5)
    sim, _ = _goal_directed_sim(config)
    sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    corrupt(ckpt)

    with pytest.raises(CheckpointError):
        validate_checkpoint(ckpt)


def test_validate_checkpoint_rejects_npz_missing_meta_json(tmp_path):
    ckpt = tmp_path / "not_a_checkpoint.npz"
    np.savez(ckpt, some_array=np.array([1, 2, 3]))

    with pytest.raises(CheckpointError, match="meta_json"):
        validate_checkpoint(ckpt)


def test_load_checkpoint_never_starts_background_execution(tmp_path):
    """load_checkpoint returns a plain Simulation -- nothing here should ever
    spin up a thread or otherwise begin advancing on its own."""
    config = make_config(num_drones=5)
    sim, _ = _goal_directed_sim(config)
    sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)

    resumed = load_checkpoint(ckpt)
    tick_before = resumed.clock.tick
    import time as _time

    _time.sleep(0.05)
    assert resumed.clock.tick == tick_before  # nothing advanced it in the background


# --------------------------------------------------------------------- atomicity
def test_atomic_write_leaves_previous_checkpoint_untouched_on_failure(tmp_path, monkeypatch):
    config = make_config(num_drones=5)
    sim, _ = _goal_directed_sim(config)
    sim.step()
    ckpt = tmp_path / "sim.ckpt.npz"
    save_checkpoint(sim, ckpt)
    original_bytes = ckpt.read_bytes()

    sim.step()

    import numpy as _np

    def _boom(*a, **k):
        raise RuntimeError("simulated failure mid-save")

    monkeypatch.setattr(_np, "savez", _boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        save_checkpoint(sim, ckpt)

    # The original, valid checkpoint must be exactly as it was -- a failed
    # write must never leave a partial file at the destination path.
    assert ckpt.read_bytes() == original_bytes
    # And no leftover temp file in the destination directory.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".checkpoint_tmp_")]
    assert leftovers == []


def test_save_checkpoint_creates_parent_directories(tmp_path):
    config = make_config(num_drones=5)
    sim, _ = _goal_directed_sim(config)
    sim.step()
    nested = tmp_path / "a" / "b" / "c" / "sim.ckpt.npz"
    save_checkpoint(sim, nested)
    assert nested.exists()
    validate_checkpoint(nested, expected_num_drones=5)
