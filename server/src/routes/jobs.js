import { Router } from "express";

import { BadRequest, STEPS, TERMINAL_STATUSES, URL_TTL_SECONDS } from "../config.js";
import * as jobStore from "../jobs.js";
import * as cache from "../manifest.js";
import * as modal from "../modal.js";
import { buildParams } from "../params.js";
import * as r2 from "../r2.js";
import { route } from "./_helpers.js";

const router = Router();

/**
 * POST /api/jobs/submit
 * body: { step, run_name, project_key?, has_t2?, noise_floor_pct?,
 *         mart_iterations?, z_range?: [min,max], dt_us?, window_xy?,
 *         force?: true }
 *
 * Checks R2 for an identical completed run before dispatching. A hit returns
 * 200 with `from_cache: true` and signed download links, and no GPU container
 * is started at all. `force: true` skips the check and always recomputes.
 *
 * Two things this cache deliberately does not do:
 *
 *   - It does not reason about *upstream* steps. Each step is keyed on its own
 *     parameters, so re-running Step 3 with a new noise floor does not by
 *     itself invalidate a stored Step 6. In practice the console sends the
 *     full parameter set with every submit, so any slider change misses every
 *     step's cache -- conservative, occasionally wasteful, never wrong in the
 *     direction that matters.
 *   - It does not verify the *inputs* are unchanged. Re-uploading raw frames
 *     under an existing run name and resubmitting will serve the old result;
 *     delete the run's outputs (or pass force) after replacing its frames.
 */
router.post(
  "/jobs/submit",
  route(async (req, res) => {
    const { step, run_name: runName, project_key: projectKey = "default", force } = req.body || {};
    if (!runName) throw new BadRequest("run_name is required");
    if (!STEPS.includes(step)) throw new BadRequest(`step must be one of: ${STEPS.join(", ")}`);

    const params = buildParams(req.body);

    // Before the Modal check, not after: a cached result is serveable even
    // when the GPU backend is unreachable or unconfigured.
    if (r2.isConfigured() && !force) {
      const found = await cache.lookup({ projectKey, runName, step, params });
      if (found.hit) {
        console.log("[submit] %s %s -> cache hit (%d file(s), no GPU dispatch)",
                    step, runName, found.artifacts.length);
        return res.status(200).json(cache.presentHit(found));
      }
      console.log("[submit] %s %s -> cache miss: %s", step, runName, found.reason);
      req.cacheMiss = found.reason;
    }

    if (!modal.isConfigured()) {
      return res.status(503).json({ error: "MODAL_API_URL is not configured on this server" });
    }

    const { data, status } = await modal.submitJob({ runName, step, projectKey, params });

    console.log("[submit] %s %s -> job %s", step, runName, data.job_id);
    res.status(status === 202 ? 202 : 200).json({
      job_id: data.job_id,
      status: data.status,
      step: data.step,
      run_name: runName,
      from_cache: false,
      cache_miss_reason: req.cacheMiss || null,
      // Echoed back so the client can show exactly what the pipeline received.
      resolved_params: params,
    });
  })
);

router.get(
  "/jobs",
  route(async (req, res) => {
    const limit = Math.min(200, Math.max(1, Number(req.query.limit || 25)));
    const { source, records } = await jobStore.listJobs(limit);
    res.json({ source, jobs: records.map((record) => jobStore.presentJob(record, { logTail: 0 })) });
  })
);

router.get(
  "/jobs/:id/status",
  route(async (req, res) => {
    const record = await jobStore.readJob(req.params.id);
    if (!record) return res.status(404).json({ error: `unknown job ${req.params.id}` });
    res.json(jobStore.presentJob(record, { logTail: Number(req.query.log_tail || 200) }));
  })
);

/** Incremental log tail -- the client polls with the count it already holds. */
router.get(
  "/jobs/:id/logs",
  route(async (req, res) => {
    const record = await jobStore.readJob(req.params.id);
    if (!record) return res.status(404).json({ error: `unknown job ${req.params.id}` });
    const since = Math.max(0, Number(req.query.since || 0));
    const logs = Array.isArray(record.logs) ? record.logs : [];
    res.json({
      job_id: record.job_id,
      status: record.status,
      next_offset: logs.length,
      lines: logs.slice(since),
    });
  })
);

/**
 * GET /api/jobs/:id/download                      -> every artifact, freshly signed
 * GET /api/jobs/:id/download?key=<objectKey>      -> just that one
 * GET /api/jobs/:id/download?key=<k>&redirect=1   -> 302 at the file
 */
router.get(
  "/jobs/:id/download",
  route(async (req, res) => {
    if (!r2.isConfigured()) {
      return res.status(503).json({ error: "Cloudflare R2 is not configured on this server" });
    }

    const record = await jobStore.readJob(req.params.id);
    if (!record) return res.status(404).json({ error: `unknown job ${req.params.id}` });

    const artifacts = record.artifacts || [];
    if (artifacts.length === 0) {
      return res.json({
        job_id: record.job_id,
        status: record.status,
        artifacts: [],
        detail: TERMINAL_STATUSES.includes(record.status)
          ? "this job produced no artifacts"
          : "artifacts appear once the job finishes",
      });
    }

    const requestedKey = req.query.key;
    if (requestedKey) {
      // Only sign keys that genuinely belong to this job. Without this check
      // the endpoint would presign any object in the bucket for anyone who can
      // guess a job id.
      const artifact = artifacts.find((a) => a.key === requestedKey);
      if (!artifact) return res.status(404).json({ error: "that key is not an artifact of this job" });

      const url = await r2.signDownload(artifact.key, artifact.name);
      if (req.query.redirect) return res.redirect(302, url);
      return res.json({ ...artifact, url, expires_in: URL_TTL_SECONDS });
    }

    const signed = await Promise.all(
      artifacts.map(async (a) => ({ ...a, url: await r2.signDownload(a.key, a.name) }))
    );

    // Grouped by the kind labels project_layout.ARTIFACT_KINDS assigns, so the
    // client can render sections without knowing that taxonomy itself.
    const byKind = signed.reduce((acc, a) => {
      (acc[a.kind] ||= []).push(a);
      return acc;
    }, {});

    res.json({
      job_id: record.job_id,
      run_name: record.run_name,
      status: record.status,
      expires_in: URL_TTL_SECONDS,
      total_bytes: signed.reduce((sum, a) => sum + (a.size || 0), 0),
      artifacts: signed,
      by_kind: byKind,
    });
  })
);

router.delete(
  "/jobs/:id",
  route(async (req, res) => {
    if (!modal.isConfigured()) return res.status(503).json({ error: "MODAL_API_URL is not configured" });
    res.json(await modal.cancelJob(req.params.id));
  })
);

/** Runs already uploaded to R2, for the client's run picker. */
router.get(
  "/runs",
  route(async (req, res) => {
    if (!modal.isConfigured()) return res.json({ runs: [] });
    res.json(await modal.listRuns(req.query.project_key || "default"));
  })
);

export default router;
