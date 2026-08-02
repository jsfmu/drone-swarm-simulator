import { describe, it, expect } from "vitest";
import {
  isDistributed,
  buildExecutionModeFields,
  formatExecutionModeBadge,
  shouldShowDistributedFields,
} from "../utils/executionMode.js";

describe("isDistributed / shouldShowDistributedFields", () => {
  it("is true only for the literal 'distributed' mode", () => {
    expect(isDistributed("distributed")).toBe(true);
    expect(isDistributed("local")).toBe(false);
    expect(isDistributed(undefined)).toBe(false);
    expect(isDistributed("")).toBe(false);
  });

  it("shouldShowDistributedFields mirrors isDistributed", () => {
    expect(shouldShowDistributedFields("distributed")).toBe(true);
    expect(shouldShowDistributedFields("local")).toBe(false);
  });
});

describe("buildExecutionModeFields", () => {
  it("returns only distributed:false in local mode", () => {
    const fields = buildExecutionModeFields({ executionMode: "local" });
    expect(fields).toEqual({ distributed: false });
  });

  it("includes num_workers and executor in distributed mode", () => {
    const fields = buildExecutionModeFields({
      executionMode: "distributed",
      numWorkers: 6,
      executor: "threads",
    });
    expect(fields.distributed).toBe(true);
    expect(fields.num_workers).toBe(6);
    expect(fields.executor).toBe("threads");
  });

  it("defaults num_workers to 1 and executor to sequential when unset", () => {
    const fields = buildExecutionModeFields({ executionMode: "distributed" });
    expect(fields.num_workers).toBe(1);
    expect(fields.executor).toBe("sequential");
  });

  it("omits num_partitions when blank, includes it when a value is given", () => {
    const withoutPartitions = buildExecutionModeFields({ executionMode: "distributed", numPartitions: "" });
    expect(withoutPartitions).not.toHaveProperty("num_partitions");

    const withPartitions = buildExecutionModeFields({ executionMode: "distributed", numPartitions: "5" });
    expect(withPartitions.num_partitions).toBe(5);
  });
});

describe("formatExecutionModeBadge", () => {
  it("shows a neutral label when there is no simulation", () => {
    expect(formatExecutionModeBadge(null)).toEqual({ label: "NO SIMULATION", mode: "none" });
  });

  it("shows LOCAL for a single_process simulation", () => {
    expect(formatExecutionModeBadge({ execution_mode: "single_process" })).toEqual({
      label: "LOCAL",
      mode: "local",
    });
  });

  it("shows worker count for a distributed simulation", () => {
    const badge = formatExecutionModeBadge({ execution_mode: "distributed", num_workers: 4 });
    expect(badge.mode).toBe("distributed");
    expect(badge.label).toContain("4");
    expect(badge.label).toContain("DISTRIBUTED");
  });

  it("still shows DISTRIBUTED when num_workers is missing", () => {
    const badge = formatExecutionModeBadge({ execution_mode: "distributed" });
    expect(badge.label).toBe("DISTRIBUTED");
  });
});
