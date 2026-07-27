import { useCallback, useRef, useState } from "react";
import * as api from "../api.js";
import { buildCreateSimulationRequest } from "../utils/requestBuilder.js";
import { useSimulationStream } from "../hooks/useSimulationStream.js";
import SimulationControls from "./SimulationControls.jsx";
import PolicyControls from "./PolicyControls.jsx";
import SimulationViewport from "./SimulationViewport.jsx";
import MetricsPanel from "./MetricsPanel.jsx";
import CollisionSummary from "./CollisionSummary.jsx";
import ConnectionStatus from "./ConnectionStatus.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const DEFAULT_FORM = {
  numDrones: 2000,
  seed: 0,
  xMin: 0, xMax: 500, yMin: 0, yMax: 500, zMin: "", zMax: "",
  xBins: 60, yBins: 60, hz: 8,
};

export default function SimulationDashboard() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [policy, setPolicy] = useState("");
  const [scenario, setScenario] = useState("");
  const [simulationId, setSimulationId] = useState(null);
  const [runningLabel, setRunningLabel] = useState("");
  const [isPaused, setIsPaused] = useState(true);
  const [error, setError] = useState(null);

  const viewportRef = useRef(null);

  const handleFieldChange = useCallback((patch) => setForm((f) => ({ ...f, ...patch })), []);
  const handlePolicyChange = useCallback((patch) => {
    if ("policy" in patch) setPolicy(patch.policy);
    if ("scenario" in patch) setScenario(patch.scenario);
  }, []);

  const handleCreate = useCallback(async () => {
    try {
      // A replacement simulation must stop the previous dashboard-owned one
      // first -- otherwise every "Create" click leaks another background
      // thread forever (see README's "Orphaned runtime threads" section,
      // the same bug static/index.html's createSimulation() fixes).
      if (simulationId) await api.deleteSimulation(API_BASE, simulationId);

      const body = buildCreateSimulationRequest({ ...form, policy, scenario });
      const status = await api.createSimulation(API_BASE, body);
      setSimulationId(status.simulation_id);
      setIsPaused(true);
      setRunningLabel(
        [policy || "random_walk", scenario ? `scenario=${scenario}` : null].filter(Boolean).join(", ")
      );
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, [form, policy, scenario, simulationId]);

  const handleStart = useCallback(async () => {
    if (!simulationId) return;
    try {
      await api.startSimulation(API_BASE, simulationId);
      setIsPaused(false);
    } catch (err) {
      setError(String(err));
    }
  }, [simulationId]);

  const handlePauseResume = useCallback(async () => {
    if (!simulationId) return;
    try {
      if (isPaused) {
        await api.resumeSimulation(API_BASE, simulationId);
      } else {
        await api.pauseSimulation(API_BASE, simulationId);
      }
      setIsPaused((p) => !p);
    } catch (err) {
      setError(String(err));
    }
  }, [simulationId, isPaused]);

  const handleStep = useCallback(async () => {
    if (!simulationId) return;
    try {
      if (!isPaused) {
        await api.pauseSimulation(API_BASE, simulationId);
        setIsPaused(true);
      }
      await api.stepSimulation(API_BASE, simulationId);
    } catch (err) {
      setError(String(err));
    }
  }, [simulationId, isPaused]);

  const handleReset = useCallback(async () => {
    if (!simulationId) return;
    try {
      await api.resetSimulation(API_BASE, simulationId);
      setIsPaused(true);
    } catch (err) {
      setError(String(err));
    }
  }, [simulationId]);

  const handleDelete = useCallback(async () => {
    if (!simulationId) return;
    await api.deleteSimulation(API_BASE, simulationId);
    setSimulationId(null);
    setRunningLabel("");
  }, [simulationId]);

  const viewport = {
    x_min: Number(form.xMin || 0),
    x_max: Number(form.xMax),
    y_min: Number(form.yMin || 0),
    y_max: Number(form.yMax),
    x_bins: Number(form.xBins || 60),
    y_bins: Number(form.yBins || 60),
  };

  const handleFrame = useCallback((frame) => {
    viewportRef.current?.drawFrame(frame, viewport);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.xMin, form.xMax, form.yMin, form.yMax]);

  const streamState = useSimulationStream({
    apiBase: API_BASE,
    simulationId,
    viewport: {
      xMin: viewport.x_min, xMax: viewport.x_max, yMin: viewport.y_min, yMax: viewport.y_max,
      xBins: viewport.x_bins, yBins: viewport.y_bins,
    },
    hz: Number(form.hz || 8),
    onFrame: handleFrame,
  });

  return (
    <div className="simulation-dashboard">
      <h1>Drone Collision Simulator — Dashboard</h1>
      <SimulationControls
        form={form}
        onFieldChange={handleFieldChange}
        onCreate={handleCreate}
        onStart={handleStart}
        onPauseResume={handlePauseResume}
        onStep={handleStep}
        onReset={handleReset}
        onDelete={handleDelete}
        isPaused={isPaused}
        hasSimulation={Boolean(simulationId)}
      />
      <PolicyControls policy={policy} scenario={scenario} onChange={handlePolicyChange} runningLabel={runningLabel} />
      {error ? <div className="simulation-dashboard__error">{error}</div> : null}
      <div className="simulation-dashboard__body">
        <SimulationViewport ref={viewportRef} />
        <div className="simulation-dashboard__side">
          <ConnectionStatus
            status={simulationId ? streamState.status : "idle"}
            lastError={streamState.lastError}
            lastEventAt={streamState.lastEventAt}
          />
          <CollisionSummary frameMeta={streamState.lastFrame} />
          <MetricsPanel frameMeta={streamState.lastFrame} />
        </div>
      </div>
    </div>
  );
}
