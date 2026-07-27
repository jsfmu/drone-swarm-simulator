const LABELS = {
  idle: "idle",
  connecting: "connecting…",
  open: "connected",
  error: "connection error",
  closed: "closed",
};

const COLORS = {
  idle: "#898781",
  connecting: "#fb9a06",
  open: "#3ecf6a",
  error: "#ff3b3b",
  closed: "#898781",
};

export default function ConnectionStatus({ status, lastError, lastEventAt }) {
  const label = LABELS[status] ?? status;
  const color = COLORS[status] ?? "#898781";
  return (
    <div className="connection-status">
      <span className="connection-status__dot" style={{ backgroundColor: color }} />
      <span>{label}</span>
      {status === "error" && lastError ? <span className="connection-status__detail">({lastError})</span> : null}
      {status === "closed" && lastError ? <span className="connection-status__detail">({lastError})</span> : null}
      {lastEventAt ? (
        <span className="connection-status__detail">last update: {new Date(lastEventAt).toLocaleTimeString()}</span>
      ) : null}
    </div>
  );
}
