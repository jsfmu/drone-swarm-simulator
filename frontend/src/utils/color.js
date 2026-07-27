// Approximates Matplotlib's "inferno" colormap (visualization.py's
// cmap="inferno" and static/index.html's own copy of this same ramp) so the
// dashboard, the Matplotlib debug viewer, and the Phase 3A browser page all
// read as one visual language for density instead of three unrelated scales.
const INFERNO_STOPS = [
  [0.0, 0x00, 0x00, 0x04],
  [0.13, 0x1b, 0x0c, 0x42],
  [0.25, 0x4b, 0x0c, 0x6b],
  [0.38, 0x78, 0x1c, 0x6d],
  [0.5, 0xa5, 0x2c, 0x60],
  [0.63, 0xcf, 0x44, 0x46],
  [0.75, 0xed, 0x69, 0x25],
  [0.88, 0xfb, 0x9a, 0x06],
  [1.0, 0xfc, 0xff, 0xa4],
];

export function infernoColor(t) {
  const c = Math.min(1, Math.max(0, t));
  for (let i = 0; i < INFERNO_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = INFERNO_STOPS[i];
    const [t1, r1, g1, b1] = INFERNO_STOPS[i + 1];
    if (c >= t0 && c <= t1) {
      const f = (c - t0) / (t1 - t0 || 1);
      const r = Math.round(r0 + (r1 - r0) * f);
      const g = Math.round(g0 + (g1 - g0) * f);
      const b = Math.round(b0 + (b1 - b0) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  const [, r, g, b] = INFERNO_STOPS[INFERNO_STOPS.length - 1];
  return `rgb(${r},${g},${b})`;
}
