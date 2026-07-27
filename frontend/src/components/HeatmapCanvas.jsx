import { forwardRef, useImperativeHandle, useRef } from "react";
import { computeLayout, worldToCanvas } from "../utils/canvas.js";
import { buildHeatmapCells } from "../utils/heatmapDraw.js";
import { markerCanvasPositions } from "../utils/markers.js";

/**
 * Owns the actual <canvas> element and exposes an imperative `drawFrame()`.
 *
 * Deliberately NOT driven by React state/props on every frame: the parent
 * (SimulationViewport) calls `ref.current.drawFrame(frame, viewport)`
 * directly from the SSE `onmessage` handler, so a canvas redraw happens
 * exactly once per streamed frame and never triggers (or is triggered by) a
 * React rerender. `viewport` is only used for its bounds (for the
 * world<->canvas transform and axis ticks); heatmap cell geometry itself
 * comes entirely from `frame.heatmap`.
 */
const HeatmapCanvas = forwardRef(function HeatmapCanvas({ width, height }, ref) {
  const canvasRef = useRef(null);
  const lastRef = useRef(null); // { frame, viewport } -- for resize-redraw

  useImperativeHandle(ref, () => ({
    drawFrame(frame, viewport) {
      lastRef.current = { frame, viewport };
      draw(canvasRef.current, frame, viewport);
    },
    redrawLast() {
      if (lastRef.current) draw(canvasRef.current, lastRef.current.frame, lastRef.current.viewport);
    },
  }));

  return <canvas ref={canvasRef} width={width} height={height} style={{ display: "block" }} />;
});

export default HeatmapCanvas;

function draw(canvas, frame, viewport) {
  if (!canvas || !frame || !viewport) return;
  const ctx = canvas.getContext("2d");
  const layout = computeLayout(canvas.width, canvas.height);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (frame.heatmap && frame.num_visible_drones > 0) {
    for (const cell of buildHeatmapCells(frame.heatmap, layout)) {
      ctx.fillStyle = cell.color;
      ctx.fillRect(cell.x, cell.y, cell.w, cell.h);
    }
  }

  drawAxes(ctx, viewport, layout);

  if (frame.markers && frame.markers.length) {
    ctx.strokeStyle = "#ff3b3b";
    ctx.lineWidth = 2;
    for (const m of markerCanvasPositions(frame.markers, viewport, layout)) {
      ctx.beginPath();
      ctx.moveTo(m.x - 5, m.y - 5);
      ctx.lineTo(m.x + 5, m.y + 5);
      ctx.moveTo(m.x + 5, m.y - 5);
      ctx.lineTo(m.x - 5, m.y + 5);
      ctx.stroke();
    }
  }
}

function axisTicks(min, max, n = 5) {
  const ticks = [];
  for (let i = 0; i < n; i++) ticks.push(min + (i * (max - min)) / (n - 1));
  return ticks;
}

function fmtTick(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

function drawAxes(ctx, viewport, layout) {
  const { margin, plotWidth, plotHeight } = layout;
  const plotBottom = margin.top + plotHeight;
  const plotRight = margin.left + plotWidth;

  ctx.strokeStyle = "#383835";
  ctx.fillStyle = "#898781";
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(margin.left, plotBottom + 0.5);
  ctx.lineTo(plotRight, plotBottom + 0.5);
  ctx.moveTo(margin.left - 0.5, margin.top);
  ctx.lineTo(margin.left - 0.5, plotBottom);
  ctx.stroke();

  const xTicks = axisTicks(viewport.x_min ?? 0, viewport.x_max ?? 1);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  xTicks.forEach((v, i) => {
    const { x } = worldToCanvas(v, viewport.y_min ?? 0, viewport, layout);
    ctx.beginPath();
    ctx.moveTo(x, plotBottom);
    ctx.lineTo(x, plotBottom + 5);
    ctx.stroke();
    ctx.fillText(fmtTick(v), x, plotBottom + 8);
  });

  const yTicks = axisTicks(viewport.y_min ?? 0, viewport.y_max ?? 1);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  yTicks.forEach((v) => {
    const { y } = worldToCanvas(viewport.x_min ?? 0, v, viewport, layout);
    ctx.beginPath();
    ctx.moveTo(margin.left - 5, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    ctx.fillText(fmtTick(v), margin.left - 8, y);
  });
}
