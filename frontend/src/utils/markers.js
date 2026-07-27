import { worldToCanvas } from "./canvas.js";

// Collision markers come only from the backend's already-classified
// `frame.markers` array (see collision_queries.py) -- this module only
// converts their world-space midpoint into canvas pixels for drawing. It
// never infers or recomputes a collision from rendered positions.
export function markerCanvasPositions(markers, viewport, layout) {
  return markers.map((m) => {
    const { x, y } = worldToCanvas(m.x, m.y, viewport, layout);
    return { x, y, drone_a: m.drone_a, drone_b: m.drone_b, distance: m.distance };
  });
}
