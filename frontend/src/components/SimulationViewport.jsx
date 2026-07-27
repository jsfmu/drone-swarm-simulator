import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import HeatmapCanvas from "./HeatmapCanvas.jsx";

/**
 * Sizing host for HeatmapCanvas: measures its container with a
 * ResizeObserver and redraws the last frame after a resize (canvas content
 * is cleared whenever its width/height attributes change, so a resize needs
 * an explicit redraw, not just a CSS-level stretch).
 */
const SimulationViewport = forwardRef(function SimulationViewport(_props, ref) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [size, setSize] = useState({ width: 640, height: 480 });

  useImperativeHandle(ref, () => ({
    drawFrame(frame, viewport) {
      canvasRef.current?.drawFrame(frame, viewport);
    },
  }));

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setSize({ width: Math.round(width), height: Math.round(width * 0.75) });
        }
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    // Canvas width/height changed -> its bitmap was cleared; repaint whatever
    // we last drew instead of leaving a blank canvas until the next frame.
    canvasRef.current?.redrawLast();
  }, [size.width, size.height]);

  return (
    <div ref={containerRef} style={{ width: "100%", maxWidth: 720 }}>
      <HeatmapCanvas ref={canvasRef} width={size.width} height={size.height} />
    </div>
  );
});

export default SimulationViewport;
