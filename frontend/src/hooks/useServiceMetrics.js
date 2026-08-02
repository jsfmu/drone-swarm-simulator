import { useEffect, useRef, useState } from "react";
import * as api from "../api.js";

// Distributed-execution and service-health data is deliberately NOT part of
// useSimulationStream's SSE frame -- the stream's own "metrics" key is
// RunningMetrics.summary() only (see metricsFormat.js), never
// distributed_metrics/process/api/streaming globals (see monitoring.py).
// This polls the three extra endpoints that carry that data on a bounded,
// slow interval so it never competes with the simulation's own tick
// throughput -- 3s by default, an order of magnitude slower than the 8Hz
// stream, since worker/partition health and process RSS don't need
// per-tick freshness the way the heatmap does.
const DEFAULT_INTERVAL_MS = 3000;

export const initialServiceMetricsState = {
  globalMetrics: null,
  distributedMetrics: null,
  health: null,
  ready: null,
  lastPolledAt: null,
};

export function useServiceMetrics({ apiBase, simulationId, intervalMs = DEFAULT_INTERVAL_MS }) {
  const [state, setState] = useState(initialServiceMetricsState);
  const inFlightRef = useRef(false);

  useEffect(() => {
    if (!apiBase) return undefined;
    let cancelled = false;

    async function poll() {
      // Never overlap requests: a slow/hung poll must not pile up a queue of
      // duplicate in-flight requests against the backend.
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        const [globalMetrics, ready, health] = await Promise.all([
          api.getGlobalMetrics(apiBase).catch(() => null),
          api.getReady(apiBase).catch(() => null),
          api.getHealth(apiBase).catch(() => null),
        ]);
        let distributedMetrics = null;
        if (simulationId) {
          distributedMetrics = await api
            .getMetrics(apiBase, simulationId)
            .then((m) => m.distributed_metrics ?? null)
            .catch(() => null);
        }
        if (!cancelled) {
          setState({ globalMetrics, distributedMetrics, health, ready, lastPolledAt: Date.now() });
        }
      } finally {
        inFlightRef.current = false;
      }
    }

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [apiBase, simulationId, intervalMs]);

  return state;
}
