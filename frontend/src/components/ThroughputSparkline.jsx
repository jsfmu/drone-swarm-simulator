import { buildSparklinePath } from "../utils/sparkline.js";

const WIDTH = 180;
const HEIGHT = 40;

// Compact recent-history indicator for ticks/second -- hand-rolled inline
// SVG (see sparkline.js), matching this project's existing "no charting
// library" approach (HeatmapCanvas is hand-rolled <canvas> for the same
// reason; frontend/package.json has no chart dependency). Renders nothing
// but a flat baseline until enough history has accumulated, rather than
// fabricating a shape from zero data points.
export default function ThroughputSparkline({ history, currentValue, label = "ticks / second" }) {
  const path = buildSparklinePath(history, WIDTH, HEIGHT);
  return (
    <div className="throughput-sparkline">
      <div className="throughput-sparkline__header">
        <span className="throughput-sparkline__label">{label}</span>
        <span className="throughput-sparkline__value">
          {typeof currentValue === "number" ? currentValue.toFixed(1) : "-"}
        </span>
      </div>
      <svg
        className="throughput-sparkline__svg"
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${label} recent history`}
      >
        {path ? <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" /> : null}
      </svg>
    </div>
  );
}
