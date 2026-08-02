// Formatters for the Phase 5 distributed-execution and service-health
// panels. Deliberately separate from metricsFormat.js (which formats the
// existing per-tick SSE frameMeta, untouched here) -- these read from the
// polled GET /simulations/{id}/metrics (distributed_metrics) and GET
// /metrics + /health + /ready responses instead. Every function is tolerant
// of missing/undefined input: a metric the backend didn't return is shown as
// unavailable ("-"), never fabricated, and never throws.
function fmt(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function fmtBytes(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value / 1e6).toFixed(1)} MB`;
}

function fmtDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "-";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${rem}s`;
  return `${rem}s`;
}

// distributed is MetricsResponse.distributed_metrics -- null/undefined for a
// single-process simulation (absence, not a zeroed object, is how the
// backend marks "not distributed"; see coordinator.metrics_summary()).
export function formatDistributedMetrics(distributed) {
  if (!distributed) return [];
  return [
    { key: "num_workers", label: "workers", value: distributed.num_workers ?? "-" },
    { key: "num_partitions", label: "partitions", value: distributed.num_partitions ?? "-" },
    { key: "healthy_workers", label: "healthy workers", value: distributed.healthy_worker_count ?? "-" },
    { key: "unhealthy_workers", label: "unhealthy workers", value: distributed.unhealthy_worker_count ?? 0 },
    { key: "last_tick_attempts", label: "last tick attempts", value: distributed.last_tick_attempts ?? "-" },
    { key: "total_reassignments", label: "reassignments (total)", value: distributed.total_reassignments ?? 0 },
    {
      key: "reassignments_this_tick",
      label: "reassignments (this tick)",
      value: distributed.reassignments_this_tick ?? 0,
    },
    { key: "owned_drones", label: "owned drones (last tick)", value: distributed.owned_drone_count_last_tick ?? "-" },
    { key: "ghost_drones", label: "ghost drones (last tick)", value: distributed.ghost_drone_count_last_tick ?? "-" },
    {
      key: "candidate_pairs",
      label: "candidate pairs (last tick)",
      value: distributed.candidate_pair_count_last_tick ?? "-",
    },
  ];
}

// Per-partition load table rows -- distributed.per_partition_load is an
// array of {partition_id, owned_drone_count, ghost_drone_count,
// candidate_pair_count, tick_duration_s}. Returns [] when unavailable so a
// caller can render "no data yet" without a length check on undefined.
export function formatPerPartitionLoad(distributed) {
  const rows = distributed?.per_partition_load;
  if (!Array.isArray(rows)) return [];
  return rows.map((r) => ({
    partitionId: r.partition_id,
    owned: r.owned_drone_count,
    ghost: r.ghost_drone_count,
    candidatePairs: r.candidate_pair_count,
    tickMs: fmt(r.tick_duration_s * 1e3, 2),
  }));
}

// globalMetrics is GET /metrics's full body (process/api/streaming +
// per-simulation entries); ready/health are api.getReady()/api.getHealth()'s
// resolved bodies. Any of the three may be null (not yet polled, or the
// backend was unreachable) -- every row degrades to "-"/"unreachable"
// instead of throwing.
export function formatServiceHealth(globalMetrics, ready, health) {
  const process = globalMetrics?.process;
  const api = globalMetrics?.api;
  const streaming = globalMetrics?.streaming;
  return [
    { key: "backend_health", label: "backend health", value: health ? "ok" : "unreachable" },
    {
      key: "readiness",
      label: "readiness",
      value: ready?.reachable === false ? "unreachable" : ready?.status ?? "-",
    },
    { key: "process_uptime", label: "process uptime", value: fmtDuration(process?.uptime_s) },
    { key: "process_rss", label: "process memory (RSS)", value: fmtBytes(process?.resident_set_size_bytes) },
    { key: "api_request_count", label: "API requests served", value: api?.request_count ?? "-" },
    { key: "api_latency", label: "API mean latency (ms)", value: fmt(api?.mean_request_latency_ms, 2) },
    {
      key: "stream_consumers",
      label: "active stream consumers",
      value: streaming?.total_active_stream_consumers ?? "-",
    },
    { key: "frames_published", label: "frames published (total)", value: streaming?.frames_published_total ?? "-" },
    {
      key: "frames_superseded",
      label: "frames superseded (total)",
      value: streaming?.frames_superseded_total ?? "-",
    },
  ];
}
