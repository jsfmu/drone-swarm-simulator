import { infernoColor } from "./color.js";

// Turns one /stream frame's `heatmap` block into a flat list of drawable
// rectangles, entirely independent of any canvas context -- HeatmapCanvas
// just iterates this list and calls ctx.fillRect/ctx.fillStyle per entry.
// Kept pure so the binning-to-pixel math is unit testable without a DOM.
export function buildHeatmapCells(heatmap, layout) {
  const { x_bins, y_bins, counts, max_density } = heatmap;
  if (!x_bins || !y_bins || !counts || !counts.length) return [];

  const cellW = layout.plotWidth / x_bins;
  const cellH = layout.plotHeight / y_bins;
  const max = Math.max(max_density, 1);
  const cells = [];

  for (let row = 0; row < y_bins; row++) {
    for (let col = 0; col < x_bins; col++) {
      const count = counts[row][col];
      const t = count / max;
      // Canvas y grows downward; flip so row 0 (y_min) draws at the bottom.
      const y = layout.margin.top + layout.plotHeight - (row + 1) * cellH;
      cells.push({
        row,
        col,
        count,
        x: layout.margin.left + col * cellW,
        y,
        w: cellW + 0.5,
        h: cellH + 0.5,
        color: infernoColor(t),
      });
    }
  }
  return cells;
}
