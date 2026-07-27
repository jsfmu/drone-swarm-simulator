import { useEffect, useReducer, useRef } from "react";
import { streamReducer, initialStreamState } from "../utils/streamReducer.js";
import { buildStreamQuery } from "../utils/requestBuilder.js";

/**
 * Opens (and cleans up) a GET /simulations/{id}/stream EventSource.
 *
 * Two different things happen on every frame, deliberately kept separate:
 *  - `onFrame(frame)` (a ref-stable callback) gets the FULL frame, including
 *    the heatmap grid and marker list, synchronously -- so a caller (see
 *    SimulationViewport) can draw it straight onto a canvas via a ref/
 *    imperative handle, entirely outside React state/rerenders.
 *  - the reducer state this hook returns only ever holds a small `frameMeta`
 *    object (tick/status/metrics/timings/seq -- no heatmap counts, no
 *    markers), so components reading it (MetricsPanel, ConnectionStatus)
 *    rerender on a small object, not the full streamed payload.
 *
 * Reconnects automatically whenever simulationId/viewport/hz change (a new
 * EventSource is opened; the previous one is closed in the effect cleanup).
 */
export function useSimulationStream({ apiBase, simulationId, viewport, hz, onFrame }) {
  const [state, dispatch] = useReducer(streamReducer, initialStreamState);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const { x_min, x_max, y_min, y_max, x_bins, y_bins } = buildStreamQuery({ ...viewport, hz });

  useEffect(() => {
    if (!simulationId || !apiBase) return undefined;

    dispatch({ type: "connect" });
    const qs = new URLSearchParams({
      x_min, x_max, y_min, y_max, x_bins, y_bins, hz: hz ?? 8,
    });
    const url = `${apiBase}/simulations/${simulationId}/stream?${qs.toString()}`;
    const source = new EventSource(url);

    source.onopen = () => dispatch({ type: "open" });

    source.onmessage = (evt) => {
      let frame;
      try {
        frame = JSON.parse(evt.data);
      } catch {
        return; // malformed frame: skip, keep the connection open
      }
      const frameMeta = {
        tick: frame.tick,
        status: frame.status,
        num_visible_drones: frame.num_visible_drones,
        markerCount: frame.markers ? frame.markers.length : 0,
        metrics: frame.metrics,
        timings: frame.timings,
        seq: frame.seq,
        server_time: frame.server_time,
      };
      dispatch({ type: "message", frame: frameMeta, receivedAt: Date.now() });
      onFrameRef.current?.(frame);
    };

    source.addEventListener("closed", (evt) => {
      let reason;
      try {
        reason = JSON.parse(evt.data).reason;
      } catch {
        reason = "closed";
      }
      dispatch({ type: "closed", reason });
      source.close();
    });

    source.onerror = () => {
      dispatch({ type: "error", error: "stream connection error" });
    };

    return () => {
      source.close();
      dispatch({ type: "disconnect" });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase, simulationId, x_min, x_max, y_min, y_max, x_bins, y_bins, hz]);

  return state;
}
