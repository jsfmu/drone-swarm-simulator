// Pure helpers for the local/distributed execution-mode selector. Kept
// separate from requestBuilder.js's buildCreateSimulationRequest so "should
// the distributed fields show" / "what does the badge say" are testable
// independent of the full create-simulation payload shape.

export function isDistributed(executionMode) {
  return executionMode === "distributed";
}

// CreateSimulationRequest fields consumed only when distributed=true --
// omitted entirely in local mode so the payload states clear intent rather
// than relying on the backend's documented-but-implicit "silently ignored
// otherwise" behavior (see models.py's CreateSimulationRequest docstring).
export function buildExecutionModeFields(execForm) {
  const distributed = isDistributed(execForm?.executionMode);
  const body = { distributed };
  if (!distributed) return body;

  body.num_workers = Number(execForm.numWorkers || 1);
  if (execForm.numPartitions !== undefined && execForm.numPartitions !== "" && execForm.numPartitions !== null) {
    body.num_partitions = Number(execForm.numPartitions);
  }
  body.executor = execForm.executor || "sequential";
  return body;
}

// Badge shown near simulation status/id. `status` is whatever
// SimulationStatusResponse-shaped object the dashboard currently has
// (execution_mode/num_workers) -- kept a pure function of that response so
// it needs no React state of its own.
export function formatExecutionModeBadge(status) {
  if (!status) return { label: "NO SIMULATION", mode: "none" };
  if (status.execution_mode === "distributed") {
    const workers = status.num_workers;
    return {
      label: workers ? `DISTRIBUTED · ${workers} WORKER${workers === 1 ? "" : "S"}` : "DISTRIBUTED",
      mode: "distributed",
    };
  }
  return { label: "LOCAL", mode: "local" };
}

// requestBuilder.js's buildCreateSimulationRequest already omits fields the
// user left blank; this only decides which *inputs* the ExecutionModeControls
// form should render, so the executor/num_workers/num_partitions fields never
// even exist in the DOM while "local" is selected (not just disabled).
export function shouldShowDistributedFields(executionMode) {
  return isDistributed(executionMode);
}
