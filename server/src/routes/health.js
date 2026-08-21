import { Router } from "express";

import { PORT, STEPS } from "../config.js";
import * as jobStore from "../jobs.js";
import * as modal from "../modal.js";
import * as r2 from "../r2.js";
import { route } from "./_helpers.js";

const router = Router();

/**
 * One call the UI can make to find out which of the three external services
 * are actually wired up -- so a misconfigured secret shows as a red badge
 * instead of a job failing halfway through a billed GPU minute.
 */
router.get(
  "/health",
  route(async (_req, res) => {
    const [r2Status, modalStatus] = await Promise.all([r2.health(), modal.health()]);
    res.json({
      status: "ok",
      gateway: `express on :${PORT}`,
      r2: r2Status,
      redis: jobStore.redisHealth(),
      modal: modalStatus,
      steps: STEPS,
    });
  })
);

export default router;
