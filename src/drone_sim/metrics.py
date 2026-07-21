"""Simulation metrics.

Records per-tick timing and event counts, and provides aggregate statistics
for benchmarking: tick latency, throughput (ticks/sec), candidate pairs,
collisions, and near misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TickMetrics:
    tick: int
    tick_time_s: float
    candidate_pairs: int
    collisions: int
    near_misses: int
    active_drones: int


@dataclass
class MetricsCollector:
    """Accumulates :class:`TickMetrics` and summarises them."""

    ticks: List[TickMetrics] = field(default_factory=list)

    def record(self, tm: TickMetrics) -> None:
        self.ticks.append(tm)

    def reset(self) -> None:
        self.ticks.clear()

    def summary(self) -> Dict[str, float]:
        if not self.ticks:
            return {}
        times = np.array([t.tick_time_s for t in self.ticks], dtype=np.float64)
        cand = np.array([t.candidate_pairs for t in self.ticks], dtype=np.int64)
        coll = np.array([t.collisions for t in self.ticks], dtype=np.int64)
        near = np.array([t.near_misses for t in self.ticks], dtype=np.int64)
        total_time = float(times.sum())
        n = len(self.ticks)
        return {
            "num_ticks": n,
            "total_time_s": total_time,
            "mean_tick_ms": float(times.mean() * 1e3),
            "median_tick_ms": float(np.median(times) * 1e3),
            "p95_tick_ms": float(np.percentile(times, 95) * 1e3),
            "min_tick_ms": float(times.min() * 1e3),
            "max_tick_ms": float(times.max() * 1e3),
            "ticks_per_second": float(n / total_time) if total_time > 0 else float("inf"),
            "mean_candidate_pairs": float(cand.mean()),
            "total_candidate_pairs": int(cand.sum()),
            "total_collisions": int(coll.sum()),
            "total_near_misses": int(near.sum()),
        }
