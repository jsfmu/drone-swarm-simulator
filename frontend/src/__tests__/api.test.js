import { afterEach, describe, it, expect, vi } from "vitest";
import * as api from "../api.js";

const BASE = "http://127.0.0.1:8000";

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

function textResponse(status, text) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => {
      throw new Error("not json");
    },
    text: async () => text,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("request() error handling (via createSimulation/getStatus)", () => {
  it("resolves with the parsed JSON body on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, { simulation_id: "abc123", status: "created" })));
    const body = await api.getStatus(BASE, "abc123");
    expect(body.simulation_id).toBe("abc123");
  });

  it("resolves null on a 204 response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, status: 204, json: async () => ({}), text: async () => "" })
    );
    const body = await api.deleteSimulation(BASE, "abc123");
    expect(body).toBeNull();
  });

  it("surfaces FastAPI's {detail: string} on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(404, { detail: "unknown simulation_id 'x'" })));
    await expect(api.getStatus(BASE, "x")).rejects.toThrow(/unknown simulation_id 'x'/);
  });

  it("surfaces a Pydantic validation array as a joined message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(422, { detail: [{ msg: "field required", loc: ["body", "name"] }] })
      )
    );
    await expect(api.saveCheckpoint(BASE, "x", "")).rejects.toThrow(/field required/);
  });

  it("falls back to the raw body text when the error response isn't JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(textResponse(500, "internal server error")));
    await expect(api.getStatus(BASE, "x")).rejects.toThrow(/internal server error/);
  });

  it("produces a readable message when the network itself fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.getStatus(BASE, "x")).rejects.toThrow(/network error/);
  });

  it("attaches the HTTP status to the thrown error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(409, { detail: "already running" })));
    try {
      await api.startSimulation(BASE, "x");
      throw new Error("expected startSimulation to throw");
    } catch (err) {
      expect(err.status).toBe(409);
    }
  });
});

describe("deleteSimulation", () => {
  it("swallows a network failure rather than rejecting (best-effort delete)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.deleteSimulation(BASE, "x")).resolves.toBeUndefined();
  });

  it("swallows a 404 (already gone) rather than rejecting", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(404, { detail: "unknown simulation_id" })));
    await expect(api.deleteSimulation(BASE, "x")).resolves.toBeUndefined();
  });
});

describe("checkpoint endpoints", () => {
  it("saveCheckpoint POSTs the name to /simulations/{id}/checkpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, { simulation_id: "s1", name: "demo", tick: 3, num_drones: 50, size_bytes: 100, saved_at: "t" })
    );
    vi.stubGlobal("fetch", fetchMock);
    const result = await api.saveCheckpoint(BASE, "s1", "demo");
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/simulations/s1/checkpoint`,
      expect.objectContaining({ method: "POST", body: JSON.stringify({ name: "demo" }) })
    );
    expect(result.tick).toBe(3);
  });

  it("loadCheckpoint POSTs the name to /simulations/{id}/checkpoint/load", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, { simulation_id: "s1", name: "demo", tick: 3, num_drones: 50, status: "paused" })
    );
    vi.stubGlobal("fetch", fetchMock);
    await api.loadCheckpoint(BASE, "s1", "demo");
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/simulations/s1/checkpoint/load`,
      expect.objectContaining({ method: "POST", body: JSON.stringify({ name: "demo" }) })
    );
  });

  it("listCheckpoints GETs /checkpoints", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { checkpoints: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const result = await api.listCheckpoints(BASE);
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/checkpoints`, undefined);
    expect(result.checkpoints).toEqual([]);
  });
});

describe("metrics/health endpoints", () => {
  it("getMetrics GETs the per-simulation metrics endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { simulation_id: "s1", tick: 1, metrics: {} }));
    vi.stubGlobal("fetch", fetchMock);
    await api.getMetrics(BASE, "s1");
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/simulations/s1/metrics`, undefined);
  });

  it("getGlobalMetrics GETs /metrics", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { simulations: {}, total_simulations: 0 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.getGlobalMetrics(BASE);
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/metrics`, undefined);
  });

  it("getHealth GETs /health", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { status: "ok", uptime_s: 1.2 }));
    vi.stubGlobal("fetch", fetchMock);
    const result = await api.getHealth(BASE);
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/health`, undefined);
    expect(result.status).toBe("ok");
  });
});

describe("getReady", () => {
  it("reports ready when the backend returns 200", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, { status: "ready" })));
    const result = await api.getReady(BASE);
    expect(result).toEqual({ ok: true, reachable: true, status: "ready" });
  });

  it("reports not_ready (not an error) when the backend returns 503", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(503, { status: "not_ready" })));
    const result = await api.getReady(BASE);
    expect(result.ok).toBe(false);
    expect(result.reachable).toBe(true);
    expect(result.status).toBe("not_ready");
  });

  it("reports unreachable (never throws) when the network fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const result = await api.getReady(BASE);
    expect(result).toEqual({ ok: false, reachable: false, status: "unreachable" });
  });
});
