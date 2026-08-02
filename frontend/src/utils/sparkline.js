// Pure helpers for a small throughput history sparkline. No charting
// library involved -- this project has none (see frontend/package.json) and
// a bounded point history doesn't need one. Mirrors canvas.js's DOM-free,
// unit-testable style: these functions only take/return numbers and plain
// arrays, never touch a <canvas>/<svg> element directly.

export const DEFAULT_HISTORY_LENGTH = 60;

// Immutable append-and-trim, the same bounded-window idea as runtime.py's
// RunningMetrics.recent_tick_times_ms (a deque(maxlen=RECENT_WINDOW)) -- kept
// a plain array here since it only ever feeds the two functions below.
export function pushHistory(history, value, maxLen = DEFAULT_HISTORY_LENGTH) {
  if (typeof value !== "number" || !Number.isFinite(value)) return history;
  const next = [...history, value];
  return next.length > maxLen ? next.slice(next.length - maxLen) : next;
}

// Maps a value history onto point coordinates within a width x height box,
// y-flipped (SVG/canvas y grows downward; a sparkline should grow upward)
// and normalized to the series' own min/max -- not a fixed scale, since a
// throughput series' useful range varies enormously by drone count. A flat
// (or single-point) series renders as a horizontal mid-line rather than
// dividing by zero.
export function buildSparklinePoints(history, width, height, padding = 2) {
  if (!history || history.length === 0) return [];
  const min = Math.min(...history);
  const max = Math.max(...history);
  const span = max - min;
  const innerW = Math.max(width - padding * 2, 1);
  const innerH = Math.max(height - padding * 2, 1);
  const n = history.length;
  return history.map((v, i) => {
    const x = n === 1 ? padding : padding + (i / (n - 1)) * innerW;
    const t = span === 0 ? 0.5 : (v - min) / span;
    const y = padding + (1 - t) * innerH;
    return { x, y };
  });
}

// SVG path `d` attribute for buildSparklinePoints' output -- "" (no <path>
// rendered) when there's no history yet.
export function buildSparklinePath(history, width, height, padding = 2) {
  const points = buildSparklinePoints(history, width, height, padding);
  if (points.length === 0) return "";
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
}
