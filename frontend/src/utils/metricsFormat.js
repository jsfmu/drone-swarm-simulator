// Formats one dashboard frame's metrics/timings into ordered {label, value}
// rows for MetricsPanel. Pure and DOM-free so label/value correctness (and,
// importantly, that per-tick and cumulative collision counts are DISTINCT
// rows with unambiguous labels -- never merged into one number) is testable
// directly.
export function formatMetrics(frameMeta) {
  if (!frameMeta) return [];
  const m = frameMeta.metrics || {};
  const t = frameMeta.timings || {};

  return [
    { key: "tick", label: "tick", value: frameMeta.tick },
    { key: "status", label: "status", value: frameMeta.status },
    { key: "visible_drones", label: "active drones (viewport)", value: frameMeta.num_visible_drones },
    { key: "mean_tick_ms", label: "mean tick time (ms)", value: fmt(m.mean_tick_ms) },
    { key: "ticks_per_second", label: "ticks / second", value: fmt(m.ticks_per_second, 1) },
    { key: "candidate_pairs", label: "candidate pairs (mean/tick)", value: fmt(m.mean_candidate_pairs, 1) },
    { key: "markers_this_tick", label: "collision markers (this tick)", value: frameMeta.markerCount ?? 0 },
    { key: "collisions_cumulative", label: "collisions (cumulative)", value: m.total_collisions ?? 0 },
    { key: "near_misses_cumulative", label: "near misses (cumulative)", value: m.total_near_misses ?? 0 },
    { key: "snapshot_build_ms", label: "snapshot build (ms)", value: fmt(t.snapshot_build_ms) },
    { key: "heatmap_ms", label: "heatmap query (ms)", value: fmt(t.heatmap_ms) },
    { key: "collisions_ms", label: "collision query (ms)", value: fmt(t.collisions_ms) },
    { key: "serialization_ms", label: "serialization (ms)", value: fmt(t.serialization_ms) },
    { key: "generation_ms", label: "frame generation (ms)", value: fmt(t.generation_ms) },
    { key: "seq", label: "stream seq", value: frameMeta.seq },
  ];
}

function fmt(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}
