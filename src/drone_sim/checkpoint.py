"""Versioned simulation checkpointing (Phase 5).

A checkpoint captures enough authoritative state to deterministically resume
a plain :class:`~drone_sim.simulation.Simulation`: config, tick/time, drone
arrays (positions, velocities, active_mask, movement_policy_ids,
goal_positions), and the movement RNG's exact bit-generator state (so a
resumed run's random draws continue the same stream, not restart from the
seed -- see :meth:`~drone_sim.simulation.SimulationEngine.get_rng_state`).

Deliberately NOT persisted (see README.md's Phase 5 section for the reasoning):

* Locks, threads, sockets, worker/process handles -- ``SimulationRuntime``/
  ``WorkerPool``/``DistributedCoordinator`` machinery is never touched here.
  A caller resumes a plain ``Simulation`` and may re-wrap it in a fresh
  runtime/coordinator afterward if it wants one running in the background.
* The full per-tick ``MetricsCollector`` history -- diagnostic and unbounded
  (grows every tick), not authoritative simulation state (``RunningMetrics``
  already treats tick history the same way — see ``runtime.py``). A resumed
  ``Simulation`` starts with a fresh, empty metrics log, exactly like a
  brand-new ``Simulation`` does; only ``clock.tick``/``clock.time_s`` (which
  *are* authoritative) carry over.
* Transient spatial-hash/candidate-pair structures -- rebuilt from positions
  on the very next tick, never authoritative.
* Movement-policy *objects* -- ``MovementSystem``/its policies are
  constructor arguments the caller supplies to :func:`load_checkpoint`, the
  same as ``Simulation(config, movement=...)`` already requires. Every
  current policy (see ``movement.py``) holds no per-instance mutable state
  beyond constructor-time constants, so once ``movement_policy_ids`` (already
  part of ``DroneState``) is restored there is nothing policy-specific left
  to serialize.

Format: one ``.npz`` (zip) container -- NumPy's own array format, not pickle.
Reads use ``allow_pickle=False``, so a corrupted or maliciously crafted file
cannot execute arbitrary code; :func:`validate_checkpoint` checks the schema
version plus every array's shape/dtype/config-compatibility explicitly before
:func:`load_checkpoint` builds anything. Writes are atomic: the full archive
is built in a temp file in the same directory, then ``os.replace()``d into
place, so a crash or concurrent reader mid-write can only ever observe the
old (still-valid) file or the fully-written new one -- never a partial file
at the destination path.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import BoundaryMode, SimulationConfig
from .movement import MovementSystem
from .simulation import Simulation, SimulationClock
from .state import DroneState, World

#: Bump this and add a migration/rejection path in validate_checkpoint()
#: whenever the on-disk shape changes -- never silently reinterpret an old
#: checkpoint under a new layout.
CHECKPOINT_SCHEMA_VERSION = 1

PathLike = Union[str, Path]


class CheckpointError(Exception):
    """Raised for any invalid, corrupt, or incompatible checkpoint file."""


def _config_to_dict(config: SimulationConfig) -> dict:
    return {
        "num_drones": config.num_drones,
        "bounds_min": list(config.bounds_min),
        "bounds_max": list(config.bounds_max),
        "collision_radius": config.collision_radius,
        "near_miss_radius": config.near_miss_radius,
        "cell_size": config.cell_size,
        "dt": config.dt,
        "seed": config.seed,
        "max_speed": config.max_speed,
        "boundary_mode": config.boundary_mode.value,
        "max_accel": config.max_accel,
        "avoidance_max_accel": config.avoidance_max_accel,
        "avoidance_strength": config.avoidance_strength,
        "goal_tolerance": config.goal_tolerance,
        "prediction_horizon": config.prediction_horizon,
    }


def _config_from_dict(d: dict) -> SimulationConfig:
    return SimulationConfig(
        num_drones=d["num_drones"],
        bounds_min=tuple(d["bounds_min"]),
        bounds_max=tuple(d["bounds_max"]),
        collision_radius=d["collision_radius"],
        near_miss_radius=d["near_miss_radius"],
        cell_size=d["cell_size"],
        dt=d["dt"],
        seed=d["seed"],
        max_speed=d["max_speed"],
        boundary_mode=BoundaryMode(d["boundary_mode"]),
        max_accel=d["max_accel"],
        avoidance_max_accel=d["avoidance_max_accel"],
        avoidance_strength=d["avoidance_strength"],
        goal_tolerance=d["goal_tolerance"],
        prediction_horizon=d["prediction_horizon"],
    )


def save_checkpoint(sim: Simulation, path: PathLike) -> None:
    """Write an atomic, versioned checkpoint of ``sim`` to ``path``.

    Builds the full ``.npz`` content in a temp file next to ``path`` first,
    then ``os.replace()``s it into place. ``os.replace`` is atomic on both
    POSIX and Windows when source and destination are on the same volume
    (guaranteed here: the temp file is created in ``path``'s own parent
    directory) -- a crash or concurrent reader mid-write can never observe a
    half-written file at ``path`` itself; on any failure the temp file is
    removed and ``path`` is left exactly as it was.
    """
    path = Path(path)
    state = sim.world.state
    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": _config_to_dict(sim.config),
        "tick": sim.clock.tick,
        "time_s": sim.clock.time_s,
        "rng_state": sim.engine.get_rng_state(),
        "has_goal_positions": state.goal_positions is not None,
    }

    arrays = {
        "positions": state.positions,
        "velocities": state.velocities,
        "active_mask": state.active_mask,
        "movement_policy_ids": state.movement_policy_ids,
    }
    if state.goal_positions is not None:
        arrays["goal_positions"] = state.goal_positions

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".checkpoint_tmp_", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        # meta_json travels inside the same archive as a 0-d unicode-string
        # array (NOT a pickled object) so allow_pickle=False can still read it.
        np.savez(tmp_path, meta_json=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def validate_checkpoint(path: PathLike, expected_num_drones: Optional[int] = None) -> dict:
    """Load and validate a checkpoint's metadata/array shapes without
    constructing a :class:`Simulation`.

    Raises :class:`CheckpointError` with a specific, actionable message on any
    problem (missing file, wrong schema version, missing/mis-shaped array,
    drone-count mismatch, or a file that isn't a checkpoint at all); returns
    the parsed metadata dict on success.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint file not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "meta_json" not in data.files:
                raise CheckpointError(f"{path}: missing meta_json — not a drone_sim checkpoint")
            meta = json.loads(str(data["meta_json"]))
            version = meta.get("schema_version")
            if version != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointError(
                    f"{path}: unsupported checkpoint schema_version {version!r} "
                    f"(this drone_sim build supports {CHECKPOINT_SCHEMA_VERSION})"
                )
            cfg_dict = meta["config"]
            n = cfg_dict["num_drones"]
            if expected_num_drones is not None and n != expected_num_drones:
                raise CheckpointError(
                    f"{path}: checkpoint has num_drones={n}, expected {expected_num_drones}"
                )

            required = ("positions", "velocities", "active_mask", "movement_policy_ids")
            for key in required:
                if key not in data.files:
                    raise CheckpointError(f"{path}: missing required array {key!r}")

            positions = data["positions"]
            velocities = data["velocities"]
            active_mask = data["active_mask"]
            policy_ids = data["movement_policy_ids"]
            if positions.shape != (n, 3):
                raise CheckpointError(f"{path}: positions shape {positions.shape} != ({n}, 3)")
            if velocities.shape != (n, 3):
                raise CheckpointError(f"{path}: velocities shape {velocities.shape} != ({n}, 3)")
            if active_mask.shape != (n,):
                raise CheckpointError(f"{path}: active_mask shape {active_mask.shape} != ({n},)")
            if policy_ids.shape != (n,):
                raise CheckpointError(f"{path}: movement_policy_ids shape {policy_ids.shape} != ({n},)")

            if meta.get("has_goal_positions"):
                if "goal_positions" not in data.files:
                    raise CheckpointError(f"{path}: meta says has_goal_positions but the array is missing")
                if data["goal_positions"].shape != (n, 3):
                    raise CheckpointError(f"{path}: goal_positions shape {data['goal_positions'].shape} != ({n}, 3)")

            return meta
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001 - any corruption becomes a clear CheckpointError
        raise CheckpointError(f"{path}: could not be read as a checkpoint ({exc!r})") from exc


def load_checkpoint(path: PathLike, movement: Optional[MovementSystem] = None) -> Simulation:
    """Load a checkpoint into a fresh, non-running :class:`Simulation`.

    Never starts any background execution — the caller decides whether to
    wrap the returned ``Simulation`` in a ``SimulationRuntime``/
    ``DistributedCoordinator`` afterward. ``movement`` mirrors
    ``Simulation``'s own constructor: pass the same ``MovementSystem`` the
    checkpointed run used if it wasn't the Phase 1 default (Random/Scripted),
    so ``movement_policy_ids`` resolve to the same policies.
    """
    path = Path(path)
    meta = validate_checkpoint(path)
    config = _config_from_dict(meta["config"])

    with np.load(path, allow_pickle=False) as data:
        state = DroneState(
            positions=np.array(data["positions"], dtype=np.float32),
            velocities=np.array(data["velocities"], dtype=np.float32),
            active_mask=np.array(data["active_mask"], dtype=bool),
            movement_policy_ids=np.array(data["movement_policy_ids"], dtype=np.int32),
            goal_positions=(
                np.array(data["goal_positions"], dtype=np.float32)
                if meta.get("has_goal_positions") else None
            ),
        )

    world = World(config=config, state=state)
    sim = Simulation(config, movement=movement, world=world)
    sim.clock = SimulationClock(dt=config.dt, tick=meta["tick"], time_s=meta["time_s"])
    sim.engine.set_rng_state(meta["rng_state"])
    return sim
