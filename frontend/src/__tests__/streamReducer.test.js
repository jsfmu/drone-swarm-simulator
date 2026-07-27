import { describe, it, expect } from "vitest";
import { streamReducer, initialStreamState } from "../utils/streamReducer.js";

describe("streamReducer", () => {
  it("starts idle", () => {
    expect(initialStreamState.status).toBe("idle");
  });

  it("transitions idle -> connecting on connect", () => {
    const next = streamReducer(initialStreamState, { type: "connect" });
    expect(next.status).toBe("connecting");
  });

  it("transitions connecting -> open on open", () => {
    const connecting = streamReducer(initialStreamState, { type: "connect" });
    const open = streamReducer(connecting, { type: "open" });
    expect(open.status).toBe("open");
  });

  it("stores the frame and stays open on message", () => {
    const open = { ...initialStreamState, status: "open" };
    const next = streamReducer(open, { type: "message", frame: { tick: 5 }, receivedAt: 123 });
    expect(next.status).toBe("open");
    expect(next.lastFrame).toEqual({ tick: 5 });
    expect(next.lastEventAt).toBe(123);
  });

  it("transitions open -> error on error, keeping the error message", () => {
    const open = { ...initialStreamState, status: "open" };
    const next = streamReducer(open, { type: "error", error: "boom" });
    expect(next.status).toBe("error");
    expect(next.lastError).toBe("boom");
  });

  it("transitions open -> closed on closed, keeping the reason", () => {
    const open = { ...initialStreamState, status: "open" };
    const next = streamReducer(open, { type: "closed", reason: "simulation_deleted" });
    expect(next.status).toBe("closed");
    expect(next.lastError).toBe("simulation_deleted");
  });

  it("resets fully to idle on disconnect regardless of prior state", () => {
    const open = { status: "open", lastFrame: { tick: 99 }, lastError: null, lastEventAt: 456 };
    const next = streamReducer(open, { type: "disconnect" });
    expect(next).toEqual(initialStreamState);
  });

  it("ignores unknown action types", () => {
    const open = { ...initialStreamState, status: "open" };
    const next = streamReducer(open, { type: "not_a_real_action" });
    expect(next).toBe(open);
  });
});
