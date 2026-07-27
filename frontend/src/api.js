// Thin REST wrapper over the Phase 3A/3B FastAPI backend. Mirrors
// src/drone_sim/api_client.py's endpoint set (the stdlib-Python client used
// by the Matplotlib --remote viewer) so both clients speak the identical
// contract to the same server.
const JSON_HEADERS = { "Content-Type": "application/json" };

async function request(base, path, options) {
  const res = await fetch(`${base}${path}`, options);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${path}: ${body}`);
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
