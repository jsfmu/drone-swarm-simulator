import { describe, it, expect } from "vitest";
import { computeLayout } from "../utils/canvas.js";
import { buildHeatmapCells } from "../utils/heatmapDraw.js";
import { infernoColor } from "../utils/color.js";

describe("buildHeatmapCells", () => {
  const layout = computeLayout(600, 400);

  it("produces one cell per bin (x_bins * y_bins)", () => {
    const heatmap = { x_bins: 2, y_bins: 3, max_density: 5, counts: [[0, 1], [2, 3], [4, 5]] };
    const cells = buildHeatmapCells(heatmap, layout);
    expect(cells).toHaveLength(6);
  });

  it("colors the max-density cell with infernoColor(1)", () => {
    const heatmap = { x_bins: 2, y_bins: 1, max_density: 10, counts: [[0, 10]] };
    const cells = buildHeatmapCells(heatmap, layout);
    const maxCell = cells.find((c) => c.count === 10);
    expect(maxCell.color).toBe(infernoColor(1));
  });

  it("colors a zero-count cell with infernoColor(0)", () => {
    const heatmap = { x_bins: 2, y_bins: 1, max_density: 10, counts: [[0, 10]] };
    const cells = buildHeatmapCells(heatmap, layout);
    const zeroCell = cells.find((c) => c.count === 0);
    expect(zeroCell.color).toBe(infernoColor(0));
  });

  it("places row 0 (y_min) at the bottom of the plot area", () => {
    const heatmap = { x_bins: 1, y_bins: 2, max_density: 1, counts: [[1], [0]] };
    const cells = buildHeatmapCells(heatmap, layout);
    const row0 = cells.find((c) => c.row === 0);
    const row1 = cells.find((c) => c.row === 1);
    // Canvas y grows downward, so row 0 (the bottom of the world) must have
    // a LARGER canvas y than row 1 (the top of the world).
    expect(row0.y).toBeGreaterThan(row1.y);
  });

  it("returns an empty list for a heatmap with no bins", () => {
    expect(buildHeatmapCells({ x_bins: 0, y_bins: 0, max_density: 0, counts: [] }, layout)).toEqual([]);
  });
});

describe("infernoColor", () => {
  it("clamps out-of-range inputs", () => {
    expect(infernoColor(-1)).toBe(infernoColor(0));
    expect(infernoColor(2)).toBe(infernoColor(1));
  });

  it("returns a valid rgb() string", () => {
    expect(infernoColor(0.5)).toMatch(/^rgb\(\d+,\d+,\d+\)$/);
  });
});
