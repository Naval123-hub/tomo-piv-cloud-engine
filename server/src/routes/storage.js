import { Router } from "express";

import { URL_TTL_SECONDS } from "../config.js";
import * as r2 from "../r2.js";
import * as storage from "../storage.js";
import { route } from "./_helpers.js";

const router = Router();

/** Every route here needs the bucket; say so once, in one voice. */
const requireR2 = (_req, res, next) =>
  r2.isConfigured()
    ? next()
    : res.status(503).json({ error: "Cloudflare R2 is not configured on this server" });

router.use("/storage", requireR2);

/**
 * GET /api/storage/list           -> everything under inputs/ and outputs/
 * GET /api/storage/list?prefix=…  -> one subtree
 *
 * Folders come back with rollups and their file lists, plus a per-run summary
 * spanning both roots, so the console can render the whole manager without a
 * request per folder.
 */
router.get(
  "/storage/list",
  route(async (req, res) => {
    const listing = await storage.browse(req.query.prefix || "");
    res.json({ ...listing, usage: await storage.usage() });
  })
);

/**
 * GET /api/storage/usage[?fresh=1]
 *
 * Bucket total against the 10 GB free tier. Cheap to poll: the total is cached
 * briefly and invalidated on any delete this process performs. `fresh=1`
 * forces a recount, which the client uses right after an upload finishes.
 */
router.get(
  "/storage/usage",
  route(async (req, res) => {
    res.json(await storage.usage({ fresh: Boolean(req.query.fresh) }));
  })
);

/**
 * GET /api/storage/download?key=…              -> { url } to sign client-side
 * GET /api/storage/download?key=…&redirect=1   -> 302, browser saves the file
 * GET /api/storage/download?key=…&inline=1     -> 302, browser renders it
 *
 * The inline form is what the image viewer puts in `<img src>`. Redirecting
 * per load rather than handing out a batch of signed URLs means the viewer can
 * never show broken images because its links quietly expired -- every load is
 * freshly signed, and signing is an offline computation, so this costs no R2
 * round trip.
 *
 * The key is checked against the two prefixes the pipeline owns. Without that
 * this endpoint would presign *any* object in the bucket for anyone who can
 * reach the gateway.
 */
router.get(
  "/storage/download",
  route(async (req, res) => {
    const key = storage.assertInsideRoot(req.query.key, "key");
    const name = key.slice(key.lastIndexOf("/") + 1);
    const inline = Boolean(req.query.inline);

    const url = await r2.signDownload(key, name, URL_TTL_SECONDS, inline ? "inline" : "attachment");
    if (inline || req.query.redirect) return res.redirect(302, url);
    res.json({ key, name, url, expires_in: URL_TTL_SECONDS });
  })
);

/**
 * DELETE /api/storage/delete
 * body: { key }                            -> one object
 *       { prefix }                         -> one folder, recursively
 *       { prefixes: [...] }                -> several folders (legacy cleanup)
 *       { run_name, project_key? }         -> a run's inputs AND outputs
 *       { prefix: "outputs/", force: true } -> a whole root
 *
 * Deleting is the only way back under quota, so it is a first-class operation
 * rather than something to do in the Cloudflare console. The guardrails live
 * in storage.js: targets must sit under inputs/ or outputs/, and clearing an
 * entire root needs an explicit force.
 */
router.delete(
  "/storage/delete",
  route(async (req, res) => {
    const { key, prefix, prefixes, run_name: runName, project_key: projectKey, force } = req.body || {};
    const result = await storage.remove({
      key, prefix, prefixes, runName, projectKey, force: Boolean(force),
    });

    console.log(
      "[storage] deleted %d object(s), freed %s from %s",
      result.deleted,
      `${storage.toGB(result.freed_bytes).toFixed(3)} GB`,
      result.targets.join(", ")
    );
    res.json(result);
  })
);

export default router;
