const FIELDS = [
  ["numDrones", "num_drones"],
  ["seed", "seed"],
  ["xMin", "x_min"],
  ["xMax", "x_max"],
  ["yMin", "y_min"],
  ["yMax", "y_max"],
  ["zMin", "z_min (optional)"],
  ["zMax", "z_max (optional)"],
  ["xBins", "heatmap x_bins"],
  ["yBins", "heatmap y_bins"],
  ["hz", "stream rate (Hz)"],
];

export default function SimulationControls({
  form,
  onFieldChange,
  onCreate,
  onStart,
  onPauseResume,
  onStep,
  onReset,
  onDelete,
  isPaused,
  hasSimulation,
  isCreating,
}) {
  return (
    <div className="simulation-controls">
      <div className="simulation-controls__fields">
        {FIELDS.map(([key, label]) => (
          <label key={key} className="simulation-controls__field">
            {label}
            <input
              type="number"
              value={form[key] ?? ""}
              onChange={(e) => onFieldChange({ [key]: e.target.value })}
            />
          </label>
        ))}
      </div>
      <div className="simulation-controls__buttons">
        <button onClick={onCreate} disabled={isCreating}>
          {isCreating ? "Creating…" : "Create / New simulation"}
        </button>
        <button onClick={onStart} disabled={!hasSimulation}>
          Start
        </button>
        <button onClick={onPauseResume} disabled={!hasSimulation}>
          {isPaused ? "Resume" : "Pause"}
        </button>
        <button onClick={onStep} disabled={!hasSimulation}>
          Step
        </button>
        <button onClick={onReset} disabled={!hasSimulation}>
          Reset
        </button>
        <button onClick={onDelete} disabled={!hasSimulation}>
          Delete
        </button>
      </div>
    </div>
  );
}
