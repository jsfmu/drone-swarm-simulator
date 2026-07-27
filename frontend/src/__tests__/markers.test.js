import { describe, it, expect } from "vitest";
import { computeLayout } from "../utils/canvas.js";
import { markerCanvasPositions } from "../utils/markers.js";

describe("markerCanvasPositions", () => {
  const viewport = { x_min: 0, x_max: 100, y_min: 0, y_max: 100 };
  const layout = computeLayout(600, 400);

  it("places a marker at the world center at the canvas plot-area center", () => {
    const markers = [{ x: 50, y: 50, drone_a: 1, drone_b: 2, distance: 0.5 }];
    const [pos] = markerCanvasPositions(markers, viewport, layout);
    expect(pos.x).toBeCloseTo(layout.margin.left + layout.plotWidth / 2, 5);
    expect(pos.y).toBeCloseTo(layout.margin.top + layout.plotHeight / 2, 5);
  });

  it("preserves marker identity fields without inferring anything new", () => {
    const markers = [{ x: 10, y: 10, drone_a: 3, drone_b: 7, distance: 0.9 }];
    const [pos] = markerCanvasPositions(markers, viewport, layout);
    expect(pos.drone_a).toBe(3);
    expect(pos.drone_b).toBe(7);
    expect(pos.distance).toBe(0.9);
  });

  it("maps an empty marker list to an empty list", () => {
    expect(markerCanvasPositions([], viewport, layout)).toEqual([]);
  });

  it("produces one canvas position per input marker, in order", () => {
    const markers = [
      { x: 0, y: 0, drone_a: 1, drone_b: 2, distance: 0.1 },
      { x: 100, y: 100, drone_a: 3, drone_b: 4, distance: 0.2 },
    ];
    const positions = markerCanvasPositions(markers, viewport, layout);
    expect(positions).toHaveLength(2);
    expect(positions[0].drone_a).toBe(1);
    expect(positions[1].drone_a).toBe(3);
  });
});
