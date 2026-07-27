import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 3B dashboard. Talks to the Phase 3A/3B FastAPI backend
// (uvicorn drone_sim.api.app:app, default http://127.0.0.1:8000) over
// plain REST + SSE -- see src/api.js. No proxy is configured by default;
// set VITE_API_BASE_URL if the backend runs somewhere other than
// http://127.0.0.1:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  test: {
    environment: "node",
    include: ["src/__tests__/**/*.test.js"],
  },
});
