// Pure coordinate math shared by the heatmap grid and collision-marker
// overlay -- kept dependency-free from the DOM/canvas so it is unit
// testable without a browser or a canvas mock.

const DEFAULT_MARGIN = { left: 56, right: 12, top: 12, bottom: 38 };

export function computeLayout(canvasWidth, canvasHeight, margin = {}) {
  const m = { ...DEFAULT_MARGIN, ...margin };
  return {
    margin: m,
    plotWidth: Math.max(canvasWidth - m.left - m.right, 1),
    plotHeight: Math.max(canvasHeight - m.top - m.bottom, 1),
  };
}

function span(min, max) {
  return max - min || 1;
}

// World (x, y) -> canvas pixel (x, y). Canvas y grows downward; world y is
// flipped so y_min draws at the bottom of the plot area, matching both the
// Matplotlib viewer and the existing Phase 3A static/index.html renderer.
export function worldToCanvas(x, y, viewport, layout) {
  const spanX = span(viewport.x_min, viewport.x_max);
  const spanY = span(viewport.y_min, viewport.y_max);
  const { margin, plotWidth, plotHeight } = layout;
  return {
    x: margin.left + ((x - viewport.x_min) / spanX) * plotWidth,
    y: margin.top + plotHeight - ((y - viewport.y_min) / spanY) * plotHeight,
  };
}

export function canvasToWorld(px, py, viewport, layout) {
  const spanX = span(viewport.x_min, viewport.x_max);
  const spanY = span(viewport.y_min, viewport.y_max);
  const { margin, plotWidth, plotHeight } = layout;
  return {
    x: viewport.x_min + ((px - margin.left) / plotWidth) * spanX,
    y: viewport.y_min + ((margin.top + plotHeight - py) / plotHeight) * spanY,
  };
}
