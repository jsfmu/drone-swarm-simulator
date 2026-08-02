import { formatExecutionModeBadge } from "../utils/executionMode.js";

// Shown near the simulation id/status in the top bar so it's immediately
// obvious whether the active simulation is running through the local
// SimulationEngine or a DistributedCoordinator -- not just visible if you
// happen to open the distributed-details panel.
export default function ExecutionModeBadge({ status }) {
  const { label, mode } = formatExecutionModeBadge(status);
  return <span className={`execution-mode-badge execution-mode-badge--${mode}`}>{label}</span>;
}
