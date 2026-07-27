const POLICIES = [
  { value: "", label: "Random walk (Phase 1 default, no avoidance)" },
  { value: "goal_directed", label: "GoalDirectedMovementAlgorithm (no-avoidance baseline)" },
  { value: "local_avoidance", label: "LocalAvoidanceMovementAlgorithm (avoidance)" },
];

const SCENARIOS = [
  { value: "", label: "(none -- random background traffic)" },
  { value: "head_on_collision", label: "head_on_collision" },
  { value: "crossing_paths", label: "crossing_paths" },
  { value: "near_miss", label: "near_miss" },
  { value: "parallel_safe", label: "parallel_safe" },
  { value: "stationary_obstacle", label: "stationary_obstacle" },
  { value: "converging_group", label: "converging_group" },
  { value: "rare_collision_background", label: "rare_collision_background" },
];

/**
 * Selects an EXISTING movement policy / scenario factory (see
 * GoalDirectedMovementAlgorithm / LocalAvoidanceMovementAlgorithm /
 * scenarios.py) -- never modifies or reimplements either. Applies on the
 * next "Create simulation" click; changing it does not affect a simulation
 * already running.
 */
export default function PolicyControls({ policy, scenario, onChange, runningLabel }) {
  return (
    <fieldset className="policy-controls">
      <legend>Movement policy &amp; scenario (applied on next create)</legend>
      <label className="policy-controls__field">
        Policy
        <select value={policy} onChange={(e) => onChange({ policy: e.target.value })}>
          {POLICIES.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      <label className="policy-controls__field">
        Scenario
        <select value={scenario} onChange={(e) => onChange({ scenario: e.target.value })}>
          {SCENARIOS.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
      {runningLabel ? <div className="policy-controls__running">Running: {runningLabel}</div> : null}
    </fieldset>
  );
}
