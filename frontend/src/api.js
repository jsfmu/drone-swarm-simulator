// Thin REST wrapper over the Phase 3A/3B FastAPI backend. Mirrors
// src/drone_sim/api_client.py's endpoint set (the stdlib-Python client used
// by the Matplotlib --remote viewer) so both clients speak the identical
// contract to the same server.
const JSON_HEADERS = { "Content-Type": "application/json" };

// FastAPI's standard error body is {"detail": "..."} (HTTPException) or a
// Pydantic validation array -- prefer surfacing that over a raw JSON dump,
// but never throw while trying (a non-JSON body, e.g. a proxy's HTML error
// page, must still produce a readable message instead of crashing here).
function extractDetail(bodyText) {
  try {
    const parsed = JSON.parse(bodyText);
    if (typeof parsed?.detail === "string") return parsed.detail;
    if (Array.isArray(parsed?.detail)) {
      return parsed.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
  } catch {
    // not JSON -- fall through to the raw text below
  }
  return bodyText;
}

async function request(base, path, options) {
  let res;
  try {
    res = await fetch(`${base}${path}`, options);
  } catch (err) {
    // Network failure / server unreachable (connection refused, DNS, CORS
    // preflight rejection) -- fetch() rejects rather than resolving with a
    // response, so this needs its own message distinct from an HTTP error.
    throw new Error(`network error calling ${path}: ${err.message || err}`);
  }
  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    const detail = extractDetail(bodyText) || res.statusText || "request failed";
    const error = new Error(`${detail} (${res.status} ${path})`);
    error.status = res.status;
    throw error;
  }
  if (res.status === 204) return null;
  return res.json();
}

export function createSimulation(base, body) {
  return request(base, "/simulations", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export function getStatus(base, id) {
  return request(base, `/simulations/${id}`);
}

export function startSimulation(base, id) {
  return request(base, `/simulations/${id}/start`, { method: "POST" });
}

export function pauseSimulation(base, id) {
  return request(base, `/simulations/${id}/pause`, { method: "POST" });
}

export function resumeSimulation(base, id) {
  return request(base, `/simulations/${id}/resume`, { method: "POST" });
}

export function stepSimulation(base, id) {
  return request(base, `/simulations/${id}/step`, { method: "POST" });
}

export function resetSimulation(base, id) {
  return request(base, `/simulations/${id}/reset`, { method: "POST" });
}

export function deleteSimulation(base, id) {
  // Best-effort, mirrors static/index.html's stopSimulationIfAny: a 404
  // (already gone) or network hiccup must never block creating a replacement.
  return request(base, `/simulations/${id}`, { method: "DELETE" }).catch(() => {});
}

// Phase 5 monitoring/checkpoint additions below. Same request() pattern as
// every call above -- non-2xx throws, callers decide how to surface it.

export function getMetrics(base, id) {
  return request(base, `/simulations/${id}/metrics`);
}

// Global, process-wide (not per-simulation): process uptime/RSS, API request
// count/latency, streaming counters, and every tracked simulation's own
// entry (including its "distributed" key when applicable). See
// drone_sim.api.monitoring's /metrics for the exact shape.
export function getGlobalMetrics(base) {
  return request(base, "/metrics");
}

export function getHealth(base) {
  return request(base, "/health");
}

// /ready intentionally returns 503 while not-ready -- that is a meaningful,
// expected response here, not a transport failure, so this resolves to the
// parsed body either way instead of throwing on the 503 case. A genuine
// network failure (server down) also resolves (never throws), since this is
// polled on a timer and one unreachable tick must not become an unhandled
// rejection -- see formatServiceHealth() for how "unreachable" is displayed.
export async function getReady(base) {
  try {
    const res = await fetch(`${base}/ready`);
    const body = await res.json().catch(() => ({ status: "unknown" }));
    return { ok: res.ok, reachable: true, ...body };
  } catch {
    return { ok: false, reachable: false, status: "unreachable" };
  }
}

export function saveCheckpoint(base, id, name) {
  return request(base, `/simulations/${id}/checkpoint`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name }),
  });
}

export function loadCheckpoint(base, id, name) {
  return request(base, `/simulations/${id}/checkpoint/load`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name }),
  });
}

export function listCheckpoints(base) {
  return request(base, "/checkpoints");
}
