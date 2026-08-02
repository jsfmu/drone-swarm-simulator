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

  it("sends distributed:false and no other execution fields in local mode", () => {
    const body = buildCreateSimulationRequest({ ...baseForm, executionMode: "local" });
    expect(body.distributed).toBe(false);
    expect(body).not.toHaveProperty("num_workers");
    expect(body).not.toHaveProperty("num_partitions");
    expect(body).not.toHaveProperty("executor");
  });

  it("sends distributed:true plus num_workers/executor in distributed mode", () => {
    const body = buildCreateSimulationRequest({
      ...baseForm,
      executionMode: "distributed",
      numWorkers: 4,
      executor: "processes",
    });
    expect(body.distributed).toBe(true);
    expect(body.num_workers).toBe(4);
    expect(body.executor).toBe("processes");
    expect(body).not.toHaveProperty("num_partitions");
  });

  it("includes num_partitions only when explicitly set in distributed mode", () => {
    const body = buildCreateSimulationRequest({
      ...baseForm,
      executionMode: "distributed",
      numWorkers: 2,
      numPartitions: 8,
    });
    expect(body.num_partitions).toBe(8);
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
