import { describe, it, expect } from "vitest";
import { formatMetrics } from "../utils/metricsFormat.js";

const sampleFrameMeta = {
  tick: 42,
  status: "running",
  num_visible_drones: 1234,
  markerCount: 3,
  metrics: {
    mean_tick_ms: 5.4321,
    ticks_per_second: 184.9,
    mean_candidate_pairs: 210.7,
    total_collisions: 146824,
    total_near_misses: 5012,
  },
  timings: {
    snapshot_build_ms: 0.334,
    heatmap_ms: 0.19,
    collisions_ms: 0.04,
    serialization_ms: 0.75,
    generation_ms: 1.5,
  },
  seq: 17,
};

describe("formatMetrics", () => {
  it("returns an empty list when there is no frame yet", () => {
    expect(formatMetrics(null)).toEqual([]);
  });

  it("includes a row for every required metrics-panel field", () => {
    const rows = formatMetrics(sampleFrameMeta);
    const keys = rows.map((r) => r.key);
    for (const required of [
      "tick", "status", "visible_drones", "mean_tick_ms", "ticks_per_second",
      "candidate_pairs", "markers_this_tick", "collisions_cumulative",
      "near_misses_cumulative", "snapshot_build_ms", "heatmap_ms",
      "collisions_ms", "serialization_ms", "generation_ms", "seq",
    ]) {
      expect(keys).toContain(required);
    }
  });

  it("keeps this-tick collision markers and cumulative collisions as DISTINCT rows/values", () => {
    const rows = formatMetrics(sampleFrameMeta);
    const thisTick = rows.find((r) => r.key === "markers_this_tick");
    const cumulative = rows.find((r) => r.key === "collisions_cumulative");
    expect(thisTick.label).not.toBe(cumulative.label);
    expect(thisTick.value).toBe(3);
    expect(cumulative.value).toBe(146824);
  });

  it("formats millisecond timing fields to 3 decimal places", () => {
    const rows = formatMetrics(sampleFrameMeta);
    const snapshotRow = rows.find((r) => r.key === "snapshot_build_ms");
    expect(snapshotRow.value).toBe("0.334");
  });

  it("formats ticks_per_second to 1 decimal place", () => {
    const rows = formatMetrics(sampleFrameMeta);
    const row = rows.find((r) => r.key === "ticks_per_second");
    expect(row.value).toBe("184.9");
  });

  it("falls back to '-' for missing numeric fields instead of throwing", () => {
    const rows = formatMetrics({ tick: 1, status: "created", metrics: {}, timings: {} });
    const row = rows.find((r) => r.key === "mean_tick_ms");
    expect(row.value).toBe("-");
  });
});
