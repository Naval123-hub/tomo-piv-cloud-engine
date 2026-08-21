import { Router } from "express";

import { BadRequest, URL_TTL_SECONDS, contentTypeFor } from "../config.js";
import * as r2 from "../r2.js";
import * as storage from "../storage.js";
import { route } from "./_helpers.js";

const router = Router();

/**
 * POST /api/upload-url
 * body: { run_name, filenames: [...], project_key?, target?: "run"|"calibration" }
 *
 * Returns presigned PUT URLs so the browser streams raw frames straight into
 * R2. Keeping multi-GB uploads out of this process is the whole point: the
 * gateway stays a small, stateless dispatcher.
 */
router.post(
  "/upload-url",
  route(async (req, res) => {
    if (!r2.isConfigured()) {
      return res.status(503).json({ error: "Cloudflare R2 is not configured on this server" });
    }

    const { run_name: runName, project_key: projectKey = "default", target = "run", filenames } = req.body || {};

    if (!Array.isArray(filenames) || filenames.length === 0) {
      throw new BadRequest("filenames must be a non-empty array");
    }
    if (target === "run" && !runName) {
      throw new BadRequest("run_name is required when target is 'run'");
    }

    // Quota gate, enforced here rather than only in the UI: a signed PUT is a
    // capability, and one already handed out would still work after the button
    // greyed out. Calibration is exempt -- it is ~8 KB, every run in the
    // project needs it, and refusing it to reclaim that space would strand
    // runs whose frames are already in the bucket.
    const usage = await storage.usage();
    if (target === "run" && usage.uploads_blocked) {
      return res.status(507).json({
        error: usage.message,
        usage,
      });
    }

    const base = target === "calibration" ? r2.calibrationPrefix(projectKey) : r2.inputPrefix(projectKey, runName);

    const planned = filenames.map((raw) => {
      const name = String(raw);
      // A signed PUT is write access to the bucket, so the key must be pinned
      // inside the intended prefix -- reject traversal and absolute paths.
      if (name.includes("..") || name.startsWith("/") || name.includes("\\")) {
        throw new BadRequest(`illegal filename: ${name}`);
      }
      return { filename: name, key: `${base}/${name}`, contentType: contentTypeFor(name) };
    });

    const uploads = await Promise.all(
      planned.map(async (item) => ({ ...item, url: await r2.signUpload(item.key) }))
    );

    // The new objects land outside this process, so the cached bucket total is
    // stale the moment these URLs are used. Drop it rather than let the gauge
    // under-report until the TTL lapses.
    storage.invalidateUsage();

    res.json({ bucket: r2.bucket, prefix: base, expires_in: URL_TTL_SECONDS, uploads, usage });
  })
);

export default router;
