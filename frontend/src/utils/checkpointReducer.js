// Save/load state machine for CheckpointControls, factored out as a pure
// reducer (same idea as streamReducer.js for useSimulationStream) so
// disabled-while-in-flight / success / failure transitions are unit-testable
// without rendering anything.
export const initialCheckpointState = {
  saveStatus: "idle", // idle | saving | success | error
  loadStatus: "idle", // idle | loading | success | error
  lastSaved: null, // { name, tick, num_drones, size_bytes, saved_at }
  lastLoaded: null, // { name, tick, num_drones, status }
  error: null,
};

export function checkpointReducer(state, action) {
  switch (action.type) {
    case "save_start":
      return { ...state, saveStatus: "saving", error: null };
    case "save_success":
      return { ...state, saveStatus: "success", lastSaved: action.result, error: null };
    case "save_error":
      return { ...state, saveStatus: "error", error: action.error ?? "checkpoint save failed" };
    case "load_start":
      return { ...state, loadStatus: "loading", error: null };
    case "load_success":
      return { ...state, loadStatus: "success", lastLoaded: action.result, error: null };
    case "load_error":
      return { ...state, loadStatus: "error", error: action.error ?? "checkpoint load failed" };
    case "reset":
      return { ...initialCheckpointState };
    default:
      return state;
  }
}

// A request is in flight -- used to disable both buttons (not just the one
// that started it) so a save and a load can never race each other against
// the same simulation_id.
export function isCheckpointBusy(state) {
  return state.saveStatus === "saving" || state.loadStatus === "loading";
}
