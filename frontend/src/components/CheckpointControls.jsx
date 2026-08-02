import { isCheckpointBusy } from "../utils/checkpointReducer.js";

/**
 * Presentational checkpoint save/load UI -- follows this dashboard's
 * existing convention (SimulationControls/PolicyControls) of owning no
 * fetch calls itself; SimulationDashboard owns the api.js calls and
 * checkpointReducer dispatch, this only renders that state and reports user
 * intent via callbacks. Backed entirely by the Phase 5 checkpoint HTTP
 * endpoints (POST .../checkpoint, POST .../checkpoint/load, GET
 * /checkpoints) -- no new persistence layer, just a UI over what
 * checkpoint.py already implements.
 */
export default function CheckpointControls({
  name,
  onNameChange,
  onSave,
  onLoad,
  checkpoints,
  state,
  hasSimulation,
  isRunning,
  isDistributed,
}) {
  const busy = isCheckpointBusy(state);
  const saveDisabled = !hasSimulation || busy || isDistributed;
  const loadDisabled = !hasSimulation || busy || isRunning || isDistributed || !name;

  return (
    <div className="checkpoint-controls">
      {isDistributed ? (
        <div className="checkpoint-controls__hint">
          Checkpointing is only supported for local (single-process) simulations -- the checkpoint format captures a
          plain Simulation's RNG state, which a DistributedCoordinator doesn't have a single instance of.
        </div>
      ) : null}
      {isRunning ? (
        <div className="checkpoint-controls__hint">Pause the simulation before loading a checkpoint.</div>
      ) : null}

      <label className="checkpoint-controls__field">
        checkpoint name
        <input
          type="text"
          value={name}
          placeholder="e.g. before-rebalance"
          onChange={(e) => onNameChange(e.target.value)}
          disabled={busy}
        />
      </label>

      <div className="checkpoint-controls__buttons">
        <button onClick={onSave} disabled={saveDisabled}>
          {state.saveStatus === "saving" ? "Saving…" : "Save checkpoint"}
        </button>
        <button onClick={onLoad} disabled={loadDisabled}>
          {state.loadStatus === "loading" ? "Loading…" : "Load checkpoint"}
        </button>
      </div>

      {checkpoints && checkpoints.length ? (
        <div className="checkpoint-controls__list">
          <div className="checkpoint-controls__list-label">available checkpoints</div>
          <ul>
            {checkpoints.map((c) => (
              <li key={c.name}>
                <button className="checkpoint-controls__list-item" onClick={() => onNameChange(c.name)} disabled={busy}>
                  {c.name} <span className="checkpoint-controls__list-meta">tick {c.tick} · {c.num_drones} drones</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {state.lastSaved ? (
        <div className="checkpoint-controls__feedback checkpoint-controls__feedback--success">
          Saved "{state.lastSaved.name}" at tick {state.lastSaved.tick} ({state.lastSaved.num_drones} drones,{" "}
          {(state.lastSaved.size_bytes / 1e6).toFixed(2)} MB)
        </div>
      ) : null}
      {state.lastLoaded ? (
        <div className="checkpoint-controls__feedback checkpoint-controls__feedback--success">
          Loaded "{state.lastLoaded.name}" -- restored to tick {state.lastLoaded.tick} ({state.lastLoaded.num_drones}{" "}
          drones), status: {state.lastLoaded.status}
        </div>
      ) : null}
      {state.error ? <div className="checkpoint-controls__feedback checkpoint-controls__feedback--error">{state.error}</div> : null}
    </div>
  );
}
