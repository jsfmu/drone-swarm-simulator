// Connection-state machine for useSimulationStream, factored out as a pure
// reducer so its transitions are unit-testable without a real EventSource.
export const initialStreamState = {
  status: "idle", // idle | connecting | open | error | closed
  lastFrame: null,
  lastError: null,
  lastEventAt: null,
};

export function streamReducer(state, action) {
  switch (action.type) {
    case "connect":
      return { ...state, status: "connecting", lastError: null };
    case "open":
      return { ...state, status: "open" };
    case "message":
      return {
        ...state,
        status: "open",
        lastFrame: action.frame,
        lastEventAt: action.receivedAt ?? Date.now(),
      };
    case "error":
      return { ...state, status: "error", lastError: action.error ?? "stream error" };
    case "closed":
      return { ...state, status: "closed", lastError: action.reason ?? null };
    case "disconnect":
      return { ...initialStreamState };
    default:
      return state;
  }
}
