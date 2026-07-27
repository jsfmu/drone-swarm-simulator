import { describe, it, expect } from "vitest";
import { computeLayout, worldToCanvas, canvasToWorld } from "../utils/canvas.js";

describe("computeLayout", () => {
  it("subtracts margins from the canvas size", () => {
    const layout = computeLayout(600, 400);
    expect(layout.plotWidth).toBe(600 - 56 - 12);
    expect(layout.plotHeight).toBe(400 - 12 - 38);
  });

  it("never returns a non-positive plot size", () => {
    const layout = computeLayout(10, 10);
    expect(layout.plotWidth).toBeGreaterThan(0);
    expect(layout.plotHeight).toBeGreaterThan(0);
  });
});

describe("worldToCanvas / canvasToWorld", () => {
  const viewport = { x_min: 0, x_max: 100, y_min: 0, y_max: 100 };
  const layout = computeLayout(600, 400);

  it("maps the viewport's bottom-left world corner to the plot area's bottom-left pixel", () => {
    const p = worldToCanvas(0, 0, viewport, layout);
    expect(p.x).toBeCloseTo(layout.margin.left, 5);
    expect(p.y).toBeCloseTo(layout.margin.top + layout.plotHeight, 5);
  });

  it("maps the viewport's top-right world corner to the plot area's top-right pixel", () => {
    const p = worldToCanvas(100, 100, viewport, layout);
    expect(p.x).toBeCloseTo(layout.margin.left + layout.plotWidth, 5);
    expect(p.y).toBeCloseTo(layout.margin.top, 5);
  });

  it("maps the center of the world to the center of the plot area", () => {
    const p = worldToCanvas(50, 50, viewport, layout);
    expect(p.x).toBeCloseTo(layout.margin.left + layout.plotWidth / 2, 5);
    expect(p.y).toBeCloseTo(layout.margin.top + layout.plotHeight / 2, 5);
  });

  it("is the exact inverse of canvasToWorld", () => {
    const original = { x: 37.5, y: 82.1 };
    const canvasPos = worldToCanvas(original.x, original.y, viewport, layout);
    const roundTrip = canvasToWorld(canvasPos.x, canvasPos.y, viewport, layout);
    expect(roundTrip.x).toBeCloseTo(original.x, 5);
    expect(roundTrip.y).toBeCloseTo(original.y, 5);
  });
});
