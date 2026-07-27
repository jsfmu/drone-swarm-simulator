"""Vectorized 2D density heatmap over a viewport's visible drone positions.

Uses ``numpy.histogram2d`` exactly like the existing debug viewer's
``visualization.compute_density_grid`` (no per-drone Python loop), but over
an arbitrary requested viewport rather than the whole world, and with
independently configurable X/Y bin counts -- which is why it lives here as
its own small function rather than being forced through the viewer's helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .snapshot import SimulationSnapshot
from .viewport import ViewportQuery, find_visible_drones

#: Hard upper bound on requested bins per axis, to reject pathological
#: requests (e.g. ``x_bins=10_000_000``) before any allocation happens.
MAX_BINS_PER_AXIS = 2_000


@dataclass(frozen=True)
class HeatmapQuery:
    viewport: ViewportQuery
    x_bins: int = 50
    y_bins: int = 50

    def __post_init__(self) -> None:
        if self.x_bins < 1 or self.y_bins < 1:
            raise ValueError("x_bins and y_bins must be >= 1")
        if self.x_bins > MAX_BINS_PER_AXIS or self.y_bins > MAX_BINS_PER_AXIS:
            raise ValueError(
                f"x_bins/y_bins must be <= {MAX_BINS_PER_AXIS} (got {self.x_bins}, {self.y_bins})"
            )


@dataclass(frozen=True)
class HeatmapResult:
    """A ready-to-render density grid plus enough metadata to position cells."""

    tick: int
    counts: np.ndarray  # (y_bins, x_bins) int64 -- row=y, col=x (imshow origin="lower" convention)
    x_edges: np.ndarray  # (x_bins + 1,) float64
    y_edges: np.ndarray  # (y_bins + 1,) float64
    x_bins: int
    y_bins: int
    max_density: int
    num_drones_included: int


def compute_heatmap(snapshot: SimulationSnapshot, query: HeatmapQuery) -> HeatmapResult:
    """Bin the drones visible in ``query.viewport`` into an X/Y density grid.

    The histogram range is exactly the requested viewport bounds, so
    ``sum(counts) == num_drones_included`` (every visible drone lands in some
    bin) and the caller can map bin indices back to world coordinates using
    ``x_edges``/``y_edges``. An empty viewport returns an all-zero grid with
    edges still spanning the requested bounds.
    """
    visible = find_visible_drones(snapshot, query.viewport)
    x_range = (query.viewport.x_min, query.viewport.x_max)
    y_range = (query.viewport.y_min, query.viewport.y_max)

    if visible.positions.shape[0] == 0:
        counts = np.zeros((query.y_bins, query.x_bins), dtype=np.int64)
        x_edges = np.linspace(x_range[0], x_range[1], query.x_bins + 1)
        y_edges = np.linspace(y_range[0], y_range[1], query.y_bins + 1)
        return HeatmapResult(
            tick=snapshot.tick,
            counts=counts,
            x_edges=x_edges,
            y_edges=y_edges,
            x_bins=query.x_bins,
            y_bins=query.y_bins,
            max_density=0,
            num_drones_included=0,
        )

    x = visible.positions[:, 0]
    y = visible.positions[:, 1]
    grid, x_edges, y_edges = np.histogram2d(
        x, y, bins=[query.x_bins, query.y_bins], range=[x_range, y_range]
    )
    counts = grid.T.astype(np.int64)

    return HeatmapResult(
        tick=snapshot.tick,
        counts=counts,
        x_edges=x_edges,
        y_edges=y_edges,
        x_bins=query.x_bins,
        y_bins=query.y_bins,
        max_density=int(counts.max()),
        num_drones_included=int(visible.positions.shape[0]),
    )
