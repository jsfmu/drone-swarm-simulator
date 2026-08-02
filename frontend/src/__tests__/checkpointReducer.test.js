import { describe, it, expect } from "vitest";
import { checkpointReducer, initialCheckpointState, isCheckpointBusy } from "../utils/checkpointReducer.js";

describe("checkpointReducer", () => {
  it("starts idle with no prior save/load", () => {
    expect(initialCheckpointState.saveStatus).toBe("idle");
    expect(initialCheckpointState.loadStatus).toBe("idle");
    expect(initialCheckpointState.lastSaved).toBeNull();
    expect(initialCheckpointState.lastLoaded).toBeNull();
  });

  it("transitions to saving on save_start and clears any prior error", () => {
    const withError = { ...initialCheckpointState, error: "old error" };
    const next = checkpointReducer(withError, { type: "save_start" });
    expect(next.saveStatus).toBe("saving");
    expect(next.error).toBeNull();
  });

  it("records the result and marks success on save_success", () => {
    const saving = checkpointReducer(initialCheckpointState, { type: "save_start" });
    const result = { name: "demo", tick: 12, num_drones: 500, size_bytes: 4096, saved_at: "2026-01-01T00:00:00Z" };
    const next = checkpointReducer(saving, { type: "save_success", result });
    expect(next.saveStatus).toBe("success");
    expect(next.lastSaved).toEqual(result);
    expect(next.error).toBeNull();
  });

  it("keeps the error message on save_error", () => {
    const saving = checkpointReducer(initialCheckpointState, { type: "save_start" });
    const next = checkpointReducer(saving, { type: "save_error", error: "disk full" });
    expect(next.saveStatus).toBe("error");
    expect(next.error).toBe("disk full");
  });

  it("transitions to loading on load_start", () => {
    const next = checkpointReducer(initialCheckpointState, { type: "load_start" });
    expect(next.loadStatus).toBe("loading");
  });

  it("records the result and marks success on load_success", () => {
    const loading = checkpointReducer(initialCheckpointState, { type: "load_start" });
    const result = { name: "demo", tick: 12, num_drones: 500, status: "paused" };
    const next = checkpointReducer(loading, { type: "load_success", result });
    expect(next.loadStatus).toBe("success");
    expect(next.lastLoaded).toEqual(result);
  });

  it("keeps the error message on load_error", () => {
    const loading = checkpointReducer(initialCheckpointState, { type: "load_start" });
    const next = checkpointReducer(loading, { type: "load_error", error: "checkpoint not found" });
    expect(next.loadStatus).toBe("error");
    expect(next.error).toBe("checkpoint not found");
  });

  it("save and load statuses are independent of each other", () => {
    let state = checkpointReducer(initialCheckpointState, { type: "save_start" });
    state = checkpointReducer(state, { type: "load_start" });
    expect(state.saveStatus).toBe("saving");
    expect(state.loadStatus).toBe("loading");
  });

  it("resets fully regardless of prior state", () => {
    const dirty = { saveStatus: "error", loadStatus: "success", lastSaved: {}, lastLoaded: {}, error: "x" };
    const next = checkpointReducer(dirty, { type: "reset" });
    expect(next).toEqual(initialCheckpointState);
  });

  it("ignores unknown action types", () => {
    const next = checkpointReducer(initialCheckpointState, { type: "not_a_real_action" });
    expect(next).toBe(initialCheckpointState);
  });
});

describe("isCheckpointBusy", () => {
  it("is false when idle", () => {
    expect(isCheckpointBusy(initialCheckpointState)).toBe(false);
  });

  it("is true while saving", () => {
    expect(isCheckpointBusy({ ...initialCheckpointState, saveStatus: "saving" })).toBe(true);
  });

  it("is true while loading", () => {
    expect(isCheckpointBusy({ ...initialCheckpointState, loadStatus: "loading" })).toBe(true);
  });

  it("is false after success or error", () => {
    expect(isCheckpointBusy({ ...initialCheckpointState, saveStatus: "success" })).toBe(false);
    expect(isCheckpointBusy({ ...initialCheckpointState, loadStatus: "error" })).toBe(false);
  });
});
