import { describe, it, expect } from "vitest";
import { pushHistory, buildSparklinePoints, buildSparklinePath, DEFAULT_HISTORY_LENGTH } from "../utils/sparkline.js";

describe("pushHistory", () => {
  it("appends a value to an empty history", () => {
    expect(pushHistory([], 5)).toEqual([5]);
  });

  it("keeps appending in order", () => {
    expect(pushHistory([1, 2], 3)).toEqual([1, 2, 3]);
  });

  it("trims from the front once the max length is exceeded", () => {
    const history = [1, 2, 3];
    expect(pushHistory(history, 4, 3)).toEqual([2, 3, 4]);
  });

  it("never grows past DEFAULT_HISTORY_LENGTH by default", () => {
    let history = [];
    for (let i = 0; i < DEFAULT_HISTORY_LENGTH + 10; i++) {
      history = pushHistory(history, i);
    }
    expect(history.length).toBe(DEFAULT_HISTORY_LENGTH);
    expect(history[history.length - 1]).toBe(DEFAULT_HISTORY_LENGTH + 9);
  });

  it("ignores non-finite values instead of corrupting the history", () => {
    expect(pushHistory([1, 2], NaN)).toEqual([1, 2]);
    expect(pushHistory([1, 2], undefined)).toEqual([1, 2]);
    expect(pushHistory([1, 2], "not a number")).toEqual([1, 2]);
  });
});

describe("buildSparklinePoints", () => {
  it("returns no points for an empty history", () => {
    expect(buildSparklinePoints([], 100, 40)).toEqual([]);
  });

  it("places a single point at the left edge, vertically centered", () => {
    const points = buildSparklinePoints([5], 100, 40, 2);
    expect(points).toHaveLength(1);
    expect(points[0].x).toBe(2);
    expect(points[0].y).toBeCloseTo(20, 5);
  });

  it("renders a flat series as a horizontal mid-line, never dividing by zero", () => {
    const points = buildSparklinePoints([7, 7, 7], 100, 40, 0);
    expect(points.every((p) => Math.abs(p.y - 20) < 1e-6)).toBe(true);
  });

  it("maps the minimum value to the bottom and the maximum to the top", () => {
    const points = buildSparklinePoints([0, 10], 100, 40, 0);
    const [low, high] = points;
    expect(low.y).toBeGreaterThan(high.y);
  });

  it("spaces x coordinates evenly across the width", () => {
    const points = buildSparklinePoints([1, 2, 3, 4], 90, 40, 0);
    expect(points[0].x).toBe(0);
    expect(points[3].x).toBe(90);
    expect(points[1].x).toBeCloseTo(30, 5);
  });
});

describe("buildSparklinePath", () => {
  it("returns an empty string when there's no history", () => {
    expect(buildSparklinePath([], 100, 40)).toBe("");
  });

  it("starts with M and continues with L segments", () => {
    const path = buildSparklinePath([1, 2, 3], 100, 40);
    expect(path.startsWith("M")).toBe(true);
    expect(path.match(/L/g)).toHaveLength(2);
  });
});
