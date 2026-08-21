/**
 * Express application assembly. Kept separate from the listener so tests can
 * import the app without binding a port.
 */
import cors from "cors";
import express from "express";

import { CORS_ORIGINS } from "./config.js";
import healthRoutes from "./routes/health.js";
import jobRoutes from "./routes/jobs.js";
import storageRoutes from "./routes/storage.js";
import uploadRoutes from "./routes/uploads.js";

export function createApp() {
  const app = express();

  app.use(express.json({ limit: "1mb" }));
  app.use(cors({ origin: CORS_ORIGINS }));

  app.use("/api", healthRoutes);
  app.use("/api", uploadRoutes);
  app.use("/api", jobRoutes);
  app.use("/api", storageRoutes);

  app.get("/", (_req, res) =>
    res.json({ service: "tomo-piv-gateway", health: "/api/health" })
  );

  app.use((_req, res) => res.status(404).json({ error: "not found" }));

  // An axios failure carries the upstream status and detail; surface those
  // rather than flattening everything into a 500.
  app.use((err, _req, res, _next) => {
    const upstream = err.response;
    const status = err.status || upstream?.status || 500;
    const detail = upstream?.data?.detail || err.message || "internal error";
    if (status >= 500) console.error("[error]", err);
    res.status(status).json({ error: typeof detail === "string" ? detail : JSON.stringify(detail) });
  });

  return app;
}
