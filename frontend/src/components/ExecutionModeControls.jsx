import { shouldShowDistributedFields } from "../utils/executionMode.js";

const EXECUTORS = [
  { value: "sequential", label: "sequential (no thread/process pool)" },
  { value: "threads", label: "threads (measured: no benefit, GIL-bound)" },
  { value: "processes", label: "processes (measured 1.2x-1.8x at 4-8 workers)" },
];

/**
 * Selects local vs. distributed execution for the NEXT created simulation --
 * mirrors PolicyControls' "applied on next create" contract exactly: this
 * never mutates a simulation that already exists (there is no "change mode
 * of a running simulation" backend capability to call even if it tried).
 * Fields here map directly onto CreateSimulationRequest's Phase 5 additions
 * (distributed/num_workers/num_partitions/executor) -- see
 * executionMode.js's buildExecutionModeFields.
 */
export default function ExecutionModeControls({ execForm, onChange }) {
  const showDistributed = shouldShowDistributedFields(execForm.executionMode);
  return (
    <fieldset className="execution-mode-controls">
      <legend>Execution mode (applied on next create)</legend>
      <div className="execution-mode-controls__toggle">
        <label>
          <input
            type="radio"
            name="executionMode"
            value="local"
            checked={execForm.executionMode !== "distributed"}
            onChange={() => onChange({ executionMode: "local" })}
          />
          Local (single process)
        </label>
        <label>
          <input
            type="radio"
            name="executionMode"
            value="distributed"
            checked={execForm.executionMode === "distributed"}
            onChange={() => onChange({ executionMode: "distributed" })}
          />
          Distributed (coordinator + worker pool)
        </label>
      </div>
      {showDistributed ? (
        <div className="execution-mode-controls__fields">
          <label className="execution-mode-controls__field">
            workers
            <input
              type="number"
              min="1"
              max="32"
              value={execForm.numWorkers ?? 1}
              onChange={(e) => onChange({ numWorkers: e.target.value })}
            />
          </label>
          <label className="execution-mode-controls__field">
            partitions (blank = same as workers)
            <input
              type="number"
              min="1"
              value={execForm.numPartitions ?? ""}
              onChange={(e) => onChange({ numPartitions: e.target.value })}
            />
          </label>
          <label className="execution-mode-controls__field">
            executor
            <select value={execForm.executor ?? "sequential"} onChange={(e) => onChange({ executor: e.target.value })}>
              {EXECUTORS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <div className="execution-mode-controls__hint">
            distributed + local_avoidance policy is rejected by the backend (needs cross-partition context exchange
            not yet implemented) -- pick goal_directed or the random-walk default.
          </div>
        </div>
      ) : null}
    </fieldset>
  );
}
