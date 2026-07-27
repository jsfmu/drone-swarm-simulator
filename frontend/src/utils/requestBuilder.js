// Builds the POST /simulations request body from the SimulationControls +
// PolicyControls form state. Kept pure so "does picking a policy/scenario
// actually configure the request" is testable without rendering anything.
export function buildCreateSimulationRequest(form) {
  const xMin = Number(form.xMin ?? 0);
  const yMin = Number(form.yMin ?? 0);
  const width = Number(form.xMax) - xMin;
  const height = Number(form.yMax) - yMin;
  const depth = Number(form.zMax ?? 100) - Number(form.zMin ?? 0);

  const body = {
    num_drones: Number(form.numDrones),
    bounds_max: [Math.max(width, 1), Math.max(height, 1), Math.max(depth, 1)],
    seed: Number(form.seed ?? 0),
  };
  if (form.dt !== undefined && form.dt !== "") body.dt = Number(form.dt);
  if (form.maxSpeed !== undefined && form.maxSpeed !== "") body.max_speed = Number(form.maxSpeed);
  if (form.collisionRadius !== undefined && form.collisionRadius !== "") {
    body.collision_radius = Number(form.collisionRadius);
  }
  if (form.nearMissRadius !== undefined && form.nearMissRadius !== "") {
    body.near_miss_radius = Number(form.nearMissRadius);
  }
  // policy/scenario are Phase 3B additions to CreateSimulationRequest --
  // omitted entirely (not sent as null/"") when unset, reproducing the
  // Phase 3A default (RandomMovementAlgorithm, DroneState.generate()).
  if (form.policy) body.policy = form.policy;
  if (form.scenario) body.scenario = form.scenario;
  return body;
}

export function buildStreamQuery(viewport) {
  return {
    x_min: viewport.xMin ?? 0,
    x_max: viewport.xMax,
    y_min: viewport.yMin ?? 0,
    y_max: viewport.yMax,
    x_bins: viewport.xBins ?? 60,
    y_bins: viewport.yBins ?? 60,
    hz: viewport.hz ?? 8,
  };
}
