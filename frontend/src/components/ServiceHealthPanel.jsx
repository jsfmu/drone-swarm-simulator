import { formatServiceHealth } from "../utils/groupedMetrics.js";

// Process/API/streaming health -- sourced from GET /health, GET /ready, and
// the process/api/streaming sections of the global GET /metrics (polled at a
// low, bounded interval; see useServiceMetrics.js). Every row already
// degrades to "-"/"unreachable" via formatServiceHealth when a piece is
// missing, so this never needs its own empty-state branch.
export default function ServiceHealthPanel({ globalMetrics, ready, health }) {
  const rows = formatServiceHealth(globalMetrics, ready, health);
  return (
    <table className="service-health-panel">
      <tbody>
        {rows.map((row) => (
          <tr key={row.key}>
            <td className="service-health-panel__label">{row.label}</td>
            <td className="service-health-panel__value">{String(row.value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
