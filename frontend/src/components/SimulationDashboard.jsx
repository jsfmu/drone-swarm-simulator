import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import * as api from "../api.js";
import { buildCreateSimulationRequest } from "../utils/requestBuilder.js";
import { checkpointReducer, initialCheckpointState } from "../utils/checkpointReducer.js";
import { pushHistory } from "../utils/sparkline.js";
import { useSimulationStream } from "../hooks/useSimulationStream.js";
import { useServiceMetrics } from "../hooks/useServiceMetrics.js";
import SimulationControls from "./SimulationControls.jsx";
import PolicyControls from "./PolicyControls.jsx";
import ExecutionModeControls from "./ExecutionModeControls.jsx";
import ExecutionModeBadge from "./ExecutionModeBadge.jsx";
import SimulationViewport from "./SimulationViewport.jsx";
import MetricsPanel from "./MetricsPanel.jsx";
import CollisionSummary from "./CollisionSummary.jsx";
import ConnectionStatus from "./ConnectionStatus.jsx";
import ThroughputSparkline from "./ThroughputSparkline.jsx";
import DistributedPanel from "./DistributedPanel.jsx";
import ServiceHealthPanel from "./ServiceHealthPanel.jsx";
import CheckpointControls from "./CheckpointControls.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const DEFAULT_FORM = {
  numDrones: 2000,
  seed: 0,
  xMin: 0, xMax: 500, yMin: 0, yMax: 500, zMin: "", zMax: "",
  xBins: 60, yBins: 60, hz: 8,
};

const DEFAULT_EXEC_FORM = {
  executionMode: "local",
  numWorkers: 4,
  numPartitions: "",
  executor: "processes",
};

export default function SimulationDashboard() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [policy, setPolicy] = useState("");
  const [scenario, setScenario] = useState("");
  const [execForm, setExecForm] = useState(DEFAULT_EXEC_FORM);
  const [simulationId, setSimulationId] = useState(null);
  const [simStatus, setSimStatus] = useState(null);
  const [runningLabel, setRunningLabel] = useState("");
  const [isPaused, setIsPaused] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState(null);
  const [tickRateHistory, setTickRateHistory] = useState([]);

  const [checkpointName, setCheckpointName] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpointState, dispatchCheckpoint] = useReducer(checkpointReducer, initialCheckpointState);

  const viewportRef = useRef(null);

  const handleFieldChange = useCallback((patch) => setForm((f) => ({ ...f, ...patch })), []);
  const handlePolicyChange = useCallback((patch) => {
    if ("policy" in patch) setPolicy(patch.policy);
    if ("scenario" in patch) setScenario(patch.scenario);
  }, []);
  const handleExecFormChange = useCallback((patch) => setExecForm((f) => ({ ...f, ...patch })), []);

  const refreshCheckpointList = useCallback(async () => {
    try {
      const resp = await api.listCheckpoints(API_BASE);
      setCheckpoints(resp.checkpoints || []);
    } catch {
      // Best-effort: a listing failure must never block Save/Load themselves.
    }
  }, []);

  useEffect(() => {
    refreshCheckpointList();
  }, [refreshCheckpointList]);

  const handleCreate = useCallback(async () => {
    if (isCreating) return; // prevent duplicate submissions on a double-click
    setIsCreating(true);
    try {
      // A replacement simulation must stop the previous dashboard-owned one
      // first -- otherwise every "Create" click leaks another background
      // thread forever (see README's "Orphaned runtime threads" section,
      // the same bug static/index.html's createSimulation() fixes).
      if (simulationId) await api.deleteSimulation(API_BASE, simulationId);

      const body = buildCreateSimulationRequest({ ...form, policy, scenario, ...execForm });
      const status = await api.createSimulation(API_BASE, body);
      setSimulationId(status.simulation_id);
      setSimStatus(status);
      setIsPaused(true);
      setTickRateHistory([]);
      dispatchCheckpoint({ type: "reset" });
      setRunningLabel(
        [policy || "random_walk", scenario ? `scenario=${scenario}` : null].filter(Boolean).join(", ")
      );
      setError(null);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setIsCreating(false);
    }
  }, [form, policy, scenario, execForm, simulationId, isCreating]);

  const handleStart = useCallback(async () => {
    if (!simulationId) return;
    try {
      await api.startSimulation(API_BASE, simulationId);
      setIsPaused(false);
    } catch (err) {
      setError(String(err.message || err));
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
      setError(String(err.message || err));
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
      setError(String(err.message || err));
    }
  }, [simulationId, isPaused]);

  const handleReset = useCallback(async () => {
    if (!simulationId) return;
    try {
      await api.resetSimulation(API_BASE, simulationId);
      setIsPaused(true);
      setTickRateHistory([]);
    } catch (err) {
      setError(String(err.message || err));
    }
  }, [simulationId]);

  const handleDelete = useCallback(async () => {
    if (!simulationId) return;
    await api.deleteSimulation(API_BASE, simulationId);
    setSimulationId(null);
    setSimStatus(null);
    setRunningLabel("");
    setTickRateHistory([]);
    dispatchCheckpoint({ type: "reset" });
  }, [simulationId]);

  const handleSaveCheckpoint = useCallback(async () => {
    if (!simulationId || !checkpointName) return;
    dispatchCheckpoint({ type: "save_start" });
    try {
      const result = await api.saveCheckpoint(API_BASE, simulationId, checkpointName);
      dispatchCheckpoint({ type: "save_success", result });
      refreshCheckpointList();
    } catch (err) {
      dispatchCheckpoint({ type: "save_error", error: String(err.message || err) });
    }
  }, [simulationId, checkpointName, refreshCheckpointList]);

  const handleLoadCheckpoint = useCallback(async () => {
    if (!simulationId || !checkpointName) return;
    dispatchCheckpoint({ type: "load_start" });
    try {
      const result = await api.loadCheckpoint(API_BASE, simulationId, checkpointName);
      dispatchCheckpoint({ type: "load_success", result });
      // load_checkpoint() always leaves the runtime PAUSED (see runtime.py's
      // docstring) -- reflect that immediately rather than waiting on the
      // next lifecycle call to reveal it. Tick history before/after a load
      // can jump backward, so a continuous sparkline spanning that jump
      // would misrepresent recent throughput; start it fresh instead.
      setIsPaused(true);
      setTickRateHistory([]);
      const status = await api.getStatus(API_BASE, simulationId);
      setSimStatus(status);
    } catch (err) {
      dispatchCheckpoint({ type: "load_error", error: String(err.message || err) });
    }
  }, [simulationId, checkpointName]);

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

  // Sparkline history comes from the already-connected SSE stream (up to
  // 8/s) rather than a separate poll -- ticks_per_second is already part of
  // every frame's metrics (see metricsFormat.js), so this needs no new
  // network traffic at all.
  useEffect(() => {
    const tps = streamState.lastFrame?.metrics?.ticks_per_second;
    if (typeof tps === "number" && Number.isFinite(tps)) {
      setTickRateHistory((h) => pushHistory(h, tps));
    }
  }, [streamState.lastFrame]);

  const serviceMetrics = useServiceMetrics({ apiBase: API_BASE, simulationId });

  const isDistributed = simStatus?.execution_mode === "distributed";

  return (
    <div className="simulation-dashboard">
      <div className="simulation-dashboard__topbar">
        <h1>Drone Collision Simulator — Dashboard</h1>
        <div className="simulation-dashboard__topbar-status">
          <ConnectionStatus
            status={simulationId ? streamState.status : "idle"}
            lastError={streamState.lastError}
            lastEventAt={streamState.lastEventAt}
          />
          <span className="simulation-dashboard__sim-id">
            {simulationId ? `sim: ${simulationId}` : "no simulation"}
          </span>
          <ExecutionModeBadge status={simStatus} />
        </div>
      </div>

      {error ? <div className="simulation-dashboard__error">{error}</div> : null}

      <section className="simulation-dashboard__section" aria-label="Configuration">
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
          isCreating={isCreating}
        />
        <div className="simulation-dashboard__config-row">
          <PolicyControls policy={policy} scenario={scenario} onChange={handlePolicyChange} runningLabel={runningLabel} />
          <ExecutionModeControls execForm={execForm} onChange={handleExecFormChange} />
        </div>
      </section>

      <div className="simulation-dashboard__body">
        <div className="simulation-dashboard__main">
          <SimulationViewport ref={viewportRef} />
          <div className="simulation-dashboard__metrics-row">
            <div className="simulation-dashboard__card">
              <h2>Simulation &amp; performance</h2>
              <CollisionSummary frameMeta={streamState.lastFrame} />
              <MetricsPanel frameMeta={streamState.lastFrame} />
            </div>
            <div className="simulation-dashboard__card">
              <h2>Throughput</h2>
              <ThroughputSparkline
                history={tickRateHistory}
                currentValue={streamState.lastFrame?.metrics?.ticks_per_second}
              />
            </div>
          </div>
        </div>

        <div className="simulation-dashboard__side">
          <div className="simulation-dashboard__card">
            <h2>Distributed execution</h2>
            <DistributedPanel distributedMetrics={serviceMetrics.distributedMetrics} />
          </div>
          <div className="simulation-dashboard__card">
            <h2>Checkpoint management</h2>
            <CheckpointControls
              name={checkpointName}
              onNameChange={setCheckpointName}
              onSave={handleSaveCheckpoint}
              onLoad={handleLoadCheckpoint}
              checkpoints={checkpoints}
              state={checkpointState}
              hasSimulation={Boolean(simulationId)}
              isRunning={!isPaused}
              isDistributed={isDistributed}
            />
          </div>
          <div className="simulation-dashboard__card">
            <h2>Service health</h2>
            <ServiceHealthPanel
              globalMetrics={serviceMetrics.globalMetrics}
              ready={serviceMetrics.ready}
              health={serviceMetrics.health}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
