import { formatMetrics } from "../utils/metricsFormat.js";

export default function MetricsPanel({ frameMeta }) {
  const rows = formatMetrics(frameMeta);
  if (!rows.length) {
    return <div className="metrics-panel metrics-panel--empty">No data yet.</div>;
  }
  return (
    <table className="metrics-panel">
      <tbody>
        {rows.map((row) => (
          <tr key={row.key}>
            <td className="metrics-panel__label">{row.label}</td>
            <td className="metrics-panel__value">{String(row.value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
