import { describe, it, expect } from "vitest";
import { buildCreateSimulationRequest, buildStreamQuery } from "../utils/requestBuilder.js";

describe("buildCreateSimulationRequest", () => {
  const baseForm = { numDrones: 500, seed: 3, xMin: 0, xMax: 200, yMin: 0, yMax: 100 };

  it("computes bounds_max from the x/y min/max fields", () => {
    const body = buildCreateSimulationRequest(baseForm);
    expect(body.num_drones).toBe(500);
    expect(body.seed).toBe(3);
    expect(body.bounds_max[0]).toBe(200);
    expect(body.bounds_max[1]).toBe(100);
  });

  it("omits policy when unset", () => {
    const body = buildCreateSimulationRequest(baseForm);
    expect(body).not.toHaveProperty("policy");
  });

  it("includes policy when set to goal_directed", () => {
    const body = buildCreateSimulationRequest({ ...baseForm, policy: "goal_directed" });
    expect(body.policy).toBe("goal_directed");
  });

  it("includes policy when set to local_avoidance", () => {
    const body = buildCreateSimulationRequest({ ...baseForm, policy: "local_avoidance" });
    expect(body.policy).toBe("local_avoidance");
  });

  it("omits scenario when unset but includes it when set", () => {
    expect(buildCreateSimulationRequest(baseForm)).not.toHaveProperty("scenario");
    const body = buildCreateSimulationRequest({ ...baseForm, scenario: "near_miss" });
    expect(body.scenario).toBe("near_miss");
  });

  it("never sends a bounds dimension below 1", () => {
    const body = buildCreateSimulationRequest({ ...baseForm, xMin: 10, xMax: 10 });
    expect(body.bounds_max[0]).toBeGreaterThanOrEqual(1);
  });
});

describe("buildStreamQuery", () => {
  it("defaults x_bins/y_bins/hz when unset", () => {
    const q = buildStreamQuery({ xMax: 100, yMax: 100 });
    expect(q.x_bins).toBe(60);
    expect(q.y_bins).toBe(60);
    expect(q.hz).toBe(8);
  });

  it("passes through explicit values", () => {
    const q = buildStreamQuery({ xMax: 100, yMax: 100, xBins: 30, yBins: 40, hz: 12 });
    expect(q.x_bins).toBe(30);
    expect(q.y_bins).toBe(40);
    expect(q.hz).toBe(12);
  });
});
