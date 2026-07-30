"""Deterministic spatial partitioning of the simulation world (Phase 4).

The world is divided into non-overlapping slabs along the X axis. A drone's
owning partition is a pure function of its current X position, so ownership
never needs to be transmitted or persisted -- it is recomputed each tick from
the (single, authoritative) position array.

One-dimensional slab partitioning, rather than a full 3-D grid, is a
deliberate scope choice: it makes owner lookup, neighbor discovery, and
halo/ghost-boundary selection exact and O(1)-per-drone with no ambiguity
(every interior partition has exactly two neighbours; the two end partitions
have exactly one), while still satisfying every Phase 4 spatial-partition
requirement (owner-of-coordinate, drone assignment, neighbour discovery,
boundary-proximity queries, ownership transfer on crossing). A full 3-D grid
is a possible future refinement, not required by this phase's scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass(frozen=True)
class Partition:
    """One non-overlapping X-axis slab of the world."""

    partition_id: int
    x_min: float
    x_max: float


class PartitionGrid:
    """Assigns drones to partitions and answers neighbour/boundary queries.

    All positions are assumed to already lie within ``config.bounds_min_arr``/
    ``bounds_max_arr`` on the X axis (``BoundaryManager`` guarantees this after
    every tick); :meth:`owner_of` defensively clips out-of-range values rather
    than raising, so a stale/pre-boundary position never produces an invalid
    partition id.
    """

    def __init__(self, config: SimulationConfig, num_partitions: int) -> None:
        if num_partitions < 1:
            raise ValueError("num_partitions must be >= 1")

        lo = float(config.bounds_min_arr[0])
        hi = float(config.bounds_max_arr[0])
        if not hi > lo:
            raise ValueError("world bounds_max[0] must be greater than bounds_min[0]")

        self.config = config
        self.num_partitions = num_partitions
        edges = np.linspace(lo, hi, num_partitions + 1)
        self._edges = edges
        self._lo = lo
        self._hi = hi
        self.partitions = tuple(
            Partition(partition_id=i, x_min=float(edges[i]), x_max=float(edges[i + 1]))
            for i in range(num_partitions)
        )

    # --------------------------------------------------------------- lookup
    def owner_of(self, x: np.ndarray) -> np.ndarray:
        """Vectorised owning-partition id for each X coordinate in ``x``.

        Partition ``k`` covers ``[edges[k], edges[k+1])``, except the last
        partition, which is closed on the right (covers the world's
        ``bounds_max`` boundary itself).
        """
        xc = np.clip(np.asarray(x, dtype=np.float64), self._lo, self._hi)
        idx = np.searchsorted(self._edges, xc, side="right") - 1
        np.clip(idx, 0, self.num_partitions - 1, out=idx)
        return idx.astype(np.int32)

    def assign(self, positions: np.ndarray) -> np.ndarray:
        """Owning-partition id for every row of an ``(N, 3)`` position array."""
        return self.owner_of(positions[:, 0])

    # ------------------------------------------------------------ topology
    def neighbors(self, partition_id: int) -> list[int]:
        """Adjacent partition ids (at most two: left and right)."""
        out = []
        if partition_id > 0:
            out.append(partition_id - 1)
        if partition_id < self.num_partitions - 1:
            out.append(partition_id + 1)
        return out

    # ------------------------------------------------------------- ghosts
    def ghost_export_indices(
        self,
        positions: np.ndarray,
        owned_indices: np.ndarray,
        partition_id: int,
        halo_distance: float,
    ) -> dict[int, np.ndarray]:
        """Which of ``owned_indices`` (all owned by ``partition_id``) are
        close enough to a shared boundary to matter for a neighbour's
        collision detection.

        Returns ``{neighbor_partition_id: subset_of_owned_indices}``. A drone
        can appear in at most two entries (a partition has at most two
        neighbours) if it is close to both boundaries of a narrow slab.
        """
        if owned_indices.size == 0:
            return {}

        part = self.partitions[partition_id]
        x = positions[owned_indices, 0]
        out: dict[int, np.ndarray] = {}

        left_id = partition_id - 1
        if left_id >= 0:
            near_left = owned_indices[x <= part.x_min + halo_distance]
            if near_left.size:
                out[left_id] = near_left

        right_id = partition_id + 1
        if right_id < self.num_partitions:
            near_right = owned_indices[x >= part.x_max - halo_distance]
            if near_right.size:
                out[right_id] = near_right

        return out
