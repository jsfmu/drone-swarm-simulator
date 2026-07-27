// Deliberately shows "this tick" and "cumulative" as two separate, clearly
// labeled numbers -- README's "Collision-marker semantics" section documents
// why these can look wildly different (a handful per tick vs. hundreds of
// thousands cumulative) and that this is expected, not a bug, as long as
// both numbers are shown with unambiguous labels.
export default function CollisionSummary({ frameMeta }) {
  if (!frameMeta) return null;
  const m = frameMeta.metrics || {};
  return (
    <div className="collision-summary">
      <div>
        <span className="collision-summary__label">collision markers (this tick):</span>{" "}
        <span className="collision-summary__value">{frameMeta.markerCount ?? 0}</span>
      </div>
      <div>
        <span className="collision-summary__label">collisions (cumulative):</span>{" "}
        <span className="collision-summary__value">{m.total_collisions ?? 0}</span>
      </div>
      <div>
        <span className="collision-summary__label">near misses (cumulative):</span>{" "}
        <span className="collision-summary__value">{m.total_near_misses ?? 0}</span>
      </div>
    </div>
  );
}
