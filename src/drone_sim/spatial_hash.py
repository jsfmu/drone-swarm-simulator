"""Uniform three-dimensional spatial hash.

The world is divided into uniform XYZ cells of edge ``cell_size``. Each drone
is only compared with drones in its own cell and the 26 neighbouring cells.
Because ``cell_size >= near_miss_radius`` (enforced in the config), any two
drones within an interaction radius fall in the same or adjacent cells, so no
interacting pair is ever missed.

Candidate pairs are **unique**: each unordered pair is produced at most once
per tick. This is achieved by scanning, for every cell, only itself (i<j) plus
13 "forward" neighbour offsets — one representative from each of the 13
antipodal neighbour pairs.
"""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def _forward_offsets() -> np.ndarray:
    """The 13 neighbour offsets that pick one cell from each antipodal pair.

    Together with their negations these cover all 26 neighbours; no offset in
    the set is the negation of another, which guarantees each unordered pair of
    distinct cells is visited exactly once.
    """
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                # Keep the lexicographically-positive half (ordered by z,y,x).
                if (dz > 0) or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0):
                    offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=np.int64)  # shape (13, 3)


_FORWARD_OFFSETS = _forward_offsets()

#: Phase 5 optimization (see benchmarks/phase5_results/optimization_notes.md):
#: build() additionally fills a dense cell -> unique-cell-index lookup array
#: when the world isn't too sparse, so candidate_pairs() can replace an
#: O(log occupied_cells) np.searchsorted() per forward offset with an O(1)
#: fancy-index gather. Measured 1.4x-1.7x faster at this project's designed
#: density (~64 cells/drone, see benchmark_simulation.py/benchmark_avoidance.py)
#: and scales better at 100,000 drones than at 1,000. The two guards below
#: exist because the same trick measured as low as 0.03x-0.6x (a *regression*)
#: on a much sparser world: filling a mostly-empty dense array costs
#: O(total_cells) up front, which dominates once total_cells grows far past
#: occupied_cells. Both conditions are checked fresh every build() call (cheap:
#: a couple of scalar comparisons), so a config that happens to be sparse
#: automatically falls back to the exact pre-optimization searchsorted path in
#: candidate_pairs() -- same output either way, verified in
#: test_spatial_hash.py's parametrized dense-vs-sparse equivalence tests.
_DENSE_LOOKUP_MAX_RATIO = 128.0
_DENSE_LOOKUP_MAX_CELLS = 20_000_000


class SpatialHashGrid:
    """Assigns drones to cells and yields unique neighbour candidate pairs."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.cell_size = config.effective_cell_size
        lo = config.bounds_min_arr
        hi = config.bounds_max_arr
        # Number of cells per axis (at least 1). A drone exactly on ``hi`` is
        # clipped into the last cell.
        dims = np.ceil((hi - lo) / self.cell_size).astype(np.int64)
        self.dims = np.maximum(dims, 1)
        self.nx, self.ny, self.nz = (int(self.dims[0]), int(self.dims[1]), int(self.dims[2]))
        self._lo = lo
        self._plane = self.nx * self.ny  # stride for z in the linear key
        self._total_cells = self.nx * self.ny * self.nz

        # Populated by build().
        self._built = False
        self._unique_keys: np.ndarray | None = None
        self._group_start: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        self._sorted_drones: np.ndarray | None = None
        self._ucx = self._ucy = self._ucz = None
        #: Dense cell -> unique-cell-index lookup, or ``None`` when this
        #: build's world is too sparse for it to be worthwhile (see above).
        self._lookup: np.ndarray | None = None

    # ------------------------------------------------------------------ build
    def cell_coords(self, positions: np.ndarray) -> np.ndarray:
        """Integer cell coordinates for ``positions`` (M, 3) -> (M, 3)."""
        c = np.floor((positions - self._lo) / self.cell_size).astype(np.int64)
        np.clip(c, 0, self.dims - 1, out=c)
        return c

    def _linear_key(self, coords: np.ndarray) -> np.ndarray:
        return coords[:, 0] + coords[:, 1] * self.nx + coords[:, 2] * self._plane

    def build(self, positions: np.ndarray, active_indices: np.ndarray) -> None:
        """Index the given active drones by cell.

        ``positions`` is the full (N, 3) array; ``active_indices`` selects the
        rows to index. Stores a compact CSR-style grouping of drone ids by
        occupied cell.
        """
        self._built = True
        if active_indices.size == 0:
            self._unique_keys = np.empty(0, dtype=np.int64)
            self._group_start = np.empty(0, dtype=np.int64)
            self._counts = np.empty(0, dtype=np.int64)
            self._sorted_drones = np.empty(0, dtype=np.int64)
            self._ucx = self._ucy = self._ucz = np.empty(0, dtype=np.int64)
            self._lookup = None
            return

        coords = self.cell_coords(positions[active_indices])
        keys = self._linear_key(coords)

        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        self._sorted_drones = active_indices[order].astype(np.int64)

        # Unique keys/group starts/counts directly from the already-sorted
        # array (O(n)) instead of np.unique(..., return_index=True,
        # return_counts=True), which internally re-sorts its input with no
        # "already sorted" fast path -- a measured, redundant O(n log n) on
        # data argsort() just produced (see benchmarks/phase5_results).
        is_new = np.empty(sorted_keys.shape[0], dtype=bool)
        is_new[0] = True
        np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=is_new[1:])
        group_start = np.flatnonzero(is_new)
        unique_keys = sorted_keys[group_start]
        counts = np.diff(group_start, append=sorted_keys.shape[0])

        self._unique_keys = unique_keys
        self._group_start = group_start.astype(np.int64)
        self._counts = counts.astype(np.int64)

        # Decode unique cell coordinates for neighbour arithmetic.
        self._ucz = unique_keys // self._plane
        rem = unique_keys % self._plane
        self._ucy = rem // self.nx
        self._ucx = rem % self.nx

        # Dense cell -> unique-cell-index lookup, only when the world isn't
        # too sparse for it to pay off (see the module-level constants'
        # docstring). Guards are cheap scalar checks, re-evaluated every
        # build() so a config's actual occupancy -- not a fixed assumption --
        # decides the strategy each tick.
        occupied = unique_keys.shape[0]
        if (
            self._total_cells <= _DENSE_LOOKUP_MAX_CELLS
            and self._total_cells <= _DENSE_LOOKUP_MAX_RATIO * max(occupied, 1)
        ):
            lookup = np.full(self._total_cells, -1, dtype=np.int32)
            lookup[unique_keys] = np.arange(occupied, dtype=np.int32)
            self._lookup = lookup
        else:
            self._lookup = None

    # ----------------------------------------------------------- diagnostics
    def occupancy_stats(self) -> tuple[int, float, int]:
        """``(occupied_cells, mean_occupancy, max_occupancy)`` from the last
        :meth:`build`. Cheap: reads the already-computed per-cell ``_counts``
        array (one entry per *occupied* cell, not per drone or per total
        cell), never rescans positions. For diagnostics/benchmarking only.
        """
        if not self._built:
            raise RuntimeError("call build() before occupancy_stats()")
        counts = self._counts
        if counts is None or counts.size == 0:
            return 0, 0.0, 0
        return int(counts.size), float(counts.mean()), int(counts.max())

    # -------------------------------------------------------- candidate pairs
    def candidate_pairs(self) -> np.ndarray:
        """Return unique unordered candidate pairs as an (K, 2) int64 array.

        Each row ``(i, j)`` satisfies ``i < j`` and appears at most once.
        """
        if not self._built:
            raise RuntimeError("call build() before candidate_pairs()")

        uk = self._unique_keys
        if uk is None or uk.size == 0:
            return np.empty((0, 2), dtype=np.int64)

        gs = self._group_start
        cnt = self._counts
        sd = self._sorted_drones

        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []

        # --- within-cell pairs (offset 0,0,0): i<j inside each occupied cell.
        multi = np.nonzero(cnt >= 2)[0]
        for ci in multi:
            drones = sd[gs[ci] : gs[ci] + cnt[ci]]
            ii, jj = np.triu_indices(drones.size, k=1)
            left_parts.append(drones[ii])
            right_parts.append(drones[jj])

        # --- cross-cell pairs (13 forward neighbour offsets), vectorised.
        ucx, ucy, ucz = self._ucx, self._ucy, self._ucz
        lookup = self._lookup
        for dx, dy, dz in _FORWARD_OFFSETS:
            ncx = ucx + dx
            ncy = ucy + dy
            ncz = ucz + dz
            valid = (
                (ncx >= 0) & (ncx < self.nx)
                & (ncy >= 0) & (ncy < self.ny)
                & (ncz >= 0) & (ncz < self.nz)
            )
            if not valid.any():
                continue

            src_cells = np.nonzero(valid)[0]
            nkey = ncx[src_cells] + ncy[src_cells] * self.nx + ncz[src_cells] * self._plane

            if lookup is not None:
                # O(1) gather -- see build()'s dense-lookup docstring. Same
                # result as the searchsorted branch below, just faster
                # whenever this build's world was dense enough to earn it.
                pos = lookup[nkey].astype(np.int64)
                exists = pos >= 0
            else:
                pos = np.searchsorted(uk, nkey)
                in_range = pos < uk.size
                exists = np.zeros(pos.shape, dtype=bool)
                exists[in_range] = uk[pos[in_range]] == nkey[in_range]
            if not exists.any():
                continue

            s_idx = src_cells[exists]      # index into unique cells (source)
            t_idx = pos[exists]            # index into unique cells (target neighbour)

            a = cnt[s_idx]                 # source group sizes
            b = cnt[t_idx]                 # target group sizes
            total = a * b
            P = int(total.sum())
            if P == 0:
                continue

            # Block cartesian product across all matched cell pairs at once.
            mp = np.repeat(np.arange(s_idx.size), total)
            start = np.cumsum(total) - total
            k = np.arange(P) - start[mp]
            bb = b[mp]
            la = k // bb                   # local offset within source group
            lb = k % bb                    # local offset within target group
            left = sd[gs[s_idx][mp] + la]
            right = sd[gs[t_idx][mp] + lb]
            left_parts.append(left)
            right_parts.append(right)

        if not left_parts:
            return np.empty((0, 2), dtype=np.int64)

        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)
        # Canonical (min, max) ordering so every pair is (i<j).
        i = np.minimum(left, right)
        j = np.maximum(left, right)
        return np.stack([i, j], axis=1)
