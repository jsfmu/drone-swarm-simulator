"""Read-only per-tick snapshot of a :class:`~drone_sim.simulation.Simulation`.

A :class:`SimulationSnapshot` captures everything visualization queries need
from one *completed* tick. It is the only point where visualization code is
allowed to touch ``Simulation``/``DroneState`` internals -- ``viewport.py``,
``heatmap.py``, and ``collision_queries.py`` only ever read a snapshot, never
the live simulation, so they can never observe a tick half-applied.

Building a snapshot copies exactly the active-drone rows it needs (NumPy
fancy indexing already allocates a new array, so no explicit ``.copy()`` is
needed there) and never mutates or aliases ``DroneState.positions`` /
``velocities``. Once built, a snapshot is frozen and remains correct forever,
even after the simulation advances further ticks.

``metrics`` is passed in by the caller rather than computed here.
``build_snapshot`` used to call ``sim.metrics.summary()`` directly, which
rebuilds NumPy arrays from the simulation's *entire* tick history and sorts
them for percentiles every single call -- an unbounded O(ticks-so-far) cost
paid on every tick. ``SimulationRuntime`` now maintains its own O(1)/bounded
``RunningMetrics`` accumulator (see ``runtime.py``) and passes its cheap
summary in here instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .collisions import DetectionResult
from .simulation import Simulation


@dataclass(frozen=True)
class SimulationSnapshot:
    """Immutable visualization-ready view of one completed simulation tick."""

    simulation_id: str
    tick: int
    time_s: float
    bounds_min: np.ndarray  # (3,) float64
    bounds_max: np.ndarray  # (3,) float64

    drone_ids: np.ndarray  # (M,) int64 -- active drone ids this tick, ascending
    positions: np.ndarray  # (M, 3) float32, row-aligned with drone_ids
    velocities: np.ndarray  # (M, 3) float32, row-aligned with drone_ids
    # Maps a drone id (0..N-1, full population) to its row in positions/
    # velocities above, or -1 if that drone was not active this tick.
    # Lets collision-marker queries look up a drone's position by id without
    # a Python loop, since collision pairs reference the full drone id space.
    id_to_row: np.ndarray  # (N,) int64

    collision_pairs: np.ndarray  # (K, 2) int64, i<j
    collision_distances: np.ndarray  # (K,) float64
    near_miss_pairs: np.ndarray  # (L, 2) int64, i<j
    near_miss_distances: np.ndarray  # (L,) float64

    num_active_drones: int
    metrics: Dict[str, float]
    captured_at: float  # time.time() wall clock; staleness display only


def build_snapshot(
    simulation_id: str,
    sim: Simulation,
    last_result: DetectionResult | None,
    metrics: Dict[str, float],
) -> SimulationSnapshot:
    """Capture a :class:`SimulationSnapshot` of ``sim``'s current tick.

    ``last_result`` is the :class:`DetectionResult` returned by the
    ``Simulation.step()`` call that produced the current tick (``None`` before
    any tick has run). It is not recomputed here -- this function never calls
    detection itself, it only copies out the already-computed result.

    ``metrics`` is a plain dict (typically ``RunningMetrics.summary()``) --
    this function does not compute it, so it stays cheap regardless of how
    long the simulation has been running (see module docstring).
    """
    world = sim.world
    state = world.state

    active = state.active_indices()
    positions = state.positions[active]  # fancy indexing -> already a copy
    velocities = state.velocities[active]

    n = state.num_drones
    id_to_row = np.full(n, -1, dtype=np.int64)
    id_to_row[active] = np.arange(active.size, dtype=np.int64)

    if last_result is not None:
        collision_pairs = last_result.collision_pairs
        collision_distances = last_result.collision_distances
        near_miss_pairs = last_result.near_miss_pairs
        near_miss_distances = last_result.near_miss_distances
    else:
        collision_pairs = np.empty((0, 2), dtype=np.int64)
        collision_distances = np.empty(0, dtype=np.float64)
        near_miss_pairs = np.empty((0, 2), dtype=np.int64)
        near_miss_distances = np.empty(0, dtype=np.float64)

    return SimulationSnapshot(
        simulation_id=simulation_id,
        tick=sim.clock.tick,
        time_s=sim.clock.time_s,
        bounds_min=world.bounds_min,
        bounds_max=world.bounds_max,
        drone_ids=active.astype(np.int64),
        positions=positions,
        velocities=velocities,
        id_to_row=id_to_row,
        collision_pairs=collision_pairs,
        collision_distances=collision_distances,
        near_miss_pairs=near_miss_pairs,
        near_miss_distances=near_miss_distances,
        num_active_drones=int(active.size),
        metrics=metrics,
        captured_at=time.time(),
    )
