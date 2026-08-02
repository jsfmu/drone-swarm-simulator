import { formatDistributedMetrics, formatPerPartitionLoad } from "../utils/groupedMetrics.js";

// Worker/partition health, rebalances, and per-partition load -- sourced
// from DistributedCoordinator.metrics_summary() via GET
// /simulations/{id}/metrics's distributed_metrics field (polled, not
// streamed -- the SSE frame's own "metrics" key never includes this). Renders
// a clear "not a distributed simulation" placeholder rather than an empty
// table when distributedMetrics is null, so its absence reads as
// informative, not broken.
export default function DistributedPanel({ distributedMetrics }) {
  if (!distributedMetrics) {
    return (
      <div className="distributed-panel distributed-panel--empty">
        Not a distributed simulation. Select "Distributed" under Execution mode and create a new simulation to see
        worker/partition metrics here.
      </div>
    );
  }

  const rows = formatDistributedMetrics(distributedMetrics);
  const partitionRows = formatPerPartitionLoad(distributedMetrics);

  return (
    <div className="distributed-panel">
      <table className="distributed-panel__summary">
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              <td className="distributed-panel__label">{row.label}</td>
              <td className="distributed-panel__value">{String(row.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {partitionRows.length ? (
        <table className="distributed-panel__partitions">
          <thead>
            <tr>
              <th>partition</th>
              <th>owned</th>
              <th>ghost</th>
              <th>pairs</th>
              <th>tick (ms)</th>
            </tr>
          </thead>
          <tbody>
            {partitionRows.map((r) => (
              <tr key={r.partitionId}>
                <td>{r.partitionId}</td>
                <td>{r.owned}</td>
                <td>{r.ghost}</td>
                <td>{r.candidatePairs}</td>
                <td>{r.tickMs}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </div>
  );
}
