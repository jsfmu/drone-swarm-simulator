import { describe, it, expect } from "vitest";
import { formatDistributedMetrics, formatPerPartitionLoad, formatServiceHealth } from "../utils/groupedMetrics.js";

const SAMPLE_DISTRIBUTED = {
  tick: 42,
  num_workers: 4,
  num_partitions: 4,
  healthy_worker_count: 4,
  unhealthy_worker_count: 0,
  last_tick_attempts: 1,
  total_reassignments: 2,
  reassignments_this_tick: 0,
  ghost_drone_count_last_tick: 120,
  owned_drone_count_last_tick: 5000,
  candidate_pair_count_last_tick: 830,
  per_partition_load: [
    { partition_id: 0, owned_drone_count: 1250, ghost_drone_count: 30, candidate_pair_count: 200, tick_duration_s: 0.012 },
    { partition_id: 1, owned_drone_count: 1240, ghost_drone_count: 28, candidate_pair_count: 210, tick_duration_s: 0.011 },
  ],
};

describe("formatDistributedMetrics", () => {
  it("returns an empty list when distributed metrics are absent (single-process simulation)", () => {
    expect(formatDistributedMetrics(null)).toEqual([]);
    expect(formatDistributedMetrics(undefined)).toEqual([]);
  });

  it("formats every field from a real metrics_summary() payload", () => {
    const rows = formatDistributedMetrics(SAMPLE_DISTRIBUTED);
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r.value]));
    expect(byKey.num_workers).toBe(4);
    expect(byKey.num_partitions).toBe(4);
    expect(byKey.healthy_workers).toBe(4);
    expect(byKey.unhealthy_workers).toBe(0);
    expect(byKey.total_reassignments).toBe(2);
    expect(byKey.owned_drones).toBe(5000);
    expect(byKey.ghost_drones).toBe(120);
    expect(byKey.candidate_pairs).toBe(830);
  });

  it("shows a placeholder instead of throwing when an individual field is missing", () => {
    const rows = formatDistributedMetrics({ num_workers: 2 });
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r.value]));
    expect(byKey.num_partitions).toBe("-");
    expect(byKey.healthy_workers).toBe("-");
    // Fields with a meaningful zero-default (never "unknown") stay 0, not "-".
    expect(byKey.total_reassignments).toBe(0);
    expect(byKey.unhealthy_workers).toBe(0);
  });
});

describe("formatPerPartitionLoad", () => {
  it("returns an empty list when unavailable", () => {
    expect(formatPerPartitionLoad(null)).toEqual([]);
    expect(formatPerPartitionLoad({})).toEqual([]);
  });

  it("maps each partition's load, converting tick_duration_s to milliseconds", () => {
    const rows = formatPerPartitionLoad(SAMPLE_DISTRIBUTED);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toEqual({ partitionId: 0, owned: 1250, ghost: 30, candidatePairs: 200, tickMs: "12.00" });
  });
});

describe("formatServiceHealth", () => {
  it("shows unreachable/placeholder values when nothing has been polled yet", () => {
    const rows = formatServiceHealth(null, null, null);
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r.value]));
    expect(byKey.backend_health).toBe("unreachable");
    expect(byKey.readiness).toBe("-");
    expect(byKey.process_uptime).toBe("-");
    expect(byKey.process_rss).toBe("-");
  });

  it("reports unreachable readiness distinctly from a not_ready backend", () => {
    const unreachable = formatServiceHealth(null, { reachable: false, status: "unreachable" }, null);
    expect(Object.fromEntries(unreachable.map((r) => [r.key, r.value])).readiness).toBe("unreachable");

    const notReady = formatServiceHealth(null, { reachable: true, status: "not_ready" }, null);
    expect(Object.fromEntries(notReady.map((r) => [r.key, r.value])).readiness).toBe("not_ready");
  });

  it("formats a full global metrics payload", () => {
    const globalMetrics = {
      process: { uptime_s: 3725, resident_set_size_bytes: 128_000_000 },
      api: { request_count: 340, mean_request_latency_ms: 2.5 },
      streaming: { total_active_stream_consumers: 2, frames_published_total: 900, frames_superseded_total: 10 },
    };
    const rows = formatServiceHealth(globalMetrics, { reachable: true, status: "ready" }, { status: "ok" });
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r.value]));
    expect(byKey.backend_health).toBe("ok");
    expect(byKey.readiness).toBe("ready");
    expect(byKey.process_uptime).toBe("1h 2m");
    expect(byKey.process_rss).toBe("128.0 MB");
    expect(byKey.api_request_count).toBe(340);
    expect(byKey.api_latency).toBe("2.50");
    expect(byKey.stream_consumers).toBe(2);
    expect(byKey.frames_published).toBe(900);
    expect(byKey.frames_superseded).toBe(10);
  });
});
