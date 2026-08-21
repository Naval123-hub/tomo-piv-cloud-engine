/**
 * Environment and shared constants.
 *
 * Loaded once, at import, so every other module reads settled values instead
 * of poking at `process.env` and getting a different answer depending on when
 * it was called.
 */
import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// Look for .env next to package.json (3d_piv/server/.env), then fall back to
// the repo root so a single shared .env can serve the Python side too.
dotenv.config({ path: path.resolve(here, "../.env") });
dotenv.config({ path: path.resolve(here, "../../../.env") });

export const PORT = Number(process.env.PORT || 5000);

export const MODAL_API_URL = (process.env.MODAL_API_URL || "").replace(/\/$/, "");
export const MODAL_API_TOKEN = process.env.API_AUTH_TOKEN || "";

export const URL_TTL_SECONDS = Number(process.env.URL_TTL_SECONDS || 3600);

export const CORS_ORIGINS = (process.env.CORS_ORIGIN || "http://localhost:5173,http://localhost:3000")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

export const R2 = {
  endpoint: process.env.CLOUDFLARE_R2_ENDPOINT,
  accessKey: process.env.CLOUDFLARE_R2_ACCESS_KEY,
  secretKey: process.env.CLOUDFLARE_R2_SECRET_KEY,
  bucket: process.env.CLOUDFLARE_R2_BUCKET,
};
export const r2Configured = Boolean(R2.endpoint && R2.accessKey && R2.secretKey && R2.bucket);

export const REDIS_URL = process.env.REDIS_URL || process.env.UPSTASH_REDIS_URL || "";

/** Must match JobStore.namespace in cloud_io.py -- this server reads records
 *  the Python worker writes; it does not own that schema. */
export const REDIS_NAMESPACE = process.env.REDIS_NAMESPACE || "tomopiv:jobs";

/**
 * Storage quota.
 *
 * Cloudflare's free tier is 10 GB of stored objects, and Cloudflare bills in
 * decimal GB (1e9 bytes), not GiB -- so these are decimal on purpose and the
 * UI must format them the same way, or the dashboard and the invoice will
 * disagree by 7%.
 *
 * The warning threshold is where the gateway starts refusing *dataset*
 * uploads. Calibration uploads stay allowed: it is one ~8 KB file that every
 * run in a project depends on, and blocking it to save 8 KB out of 10 GB would
 * strand runs that are otherwise ready to go.
 */
export const STORAGE_LIMIT_BYTES = Number(process.env.R2_STORAGE_LIMIT_BYTES || 10e9);
export const STORAGE_WARN_BYTES = Number(process.env.R2_STORAGE_WARN_BYTES || 9e9);

/** How long a bucket-usage total is reused before being recomputed. Computing
 *  it means listing every object, and the UI polls it. */
export const USAGE_CACHE_MS = Number(process.env.R2_USAGE_CACHE_MS || 30_000);

/** Ceiling on a single listing, so one call can never pull an unbounded
 *  response into memory. Reported honestly as `truncated` when hit. */
export const MAX_LIST_KEYS = Number(process.env.R2_MAX_LIST_KEYS || 20_000);

/** The only two prefixes this gateway will list or delete. Everything the
 *  pipeline writes lives under one of them, so anything else is a mistake or
 *  an attack. */
export const STORAGE_ROOTS = ["inputs/", "outputs/"];

/** Completion marker written by the worker (cloud_io.MANIFEST_NAME). */
export const MANIFEST_NAME = "manifest.json";

/**
 * Extensions worth handing back as download links on a cache hit. The
 * manifest's artifact list is already the pipeline's own definition of "an
 * output", so this is a second filter rather than the primary one -- it keeps
 * a future artifact kind from arriving in the download grid unannounced.
 */
export const DOWNLOAD_EXTENSIONS = ["vtk", "tif", "tiff", "csv", "jpg", "jpeg", "png", "npy"];

/** Step keys accepted by the Python worker (pipeline/steps.py). */
export const STEPS = ["step1", "step2", "step3", "steps45", "step6", "full"];

export const TERMINAL_STATUSES = ["succeeded", "failed", "cancelled"];

/**
 * Content types that matter for "click the link and get a usable file".
 * .vtk and .npy are deliberately octet-stream so browsers download them
 * rather than trying to render them; ParaView ignores the MIME type.
 */
const CONTENT_TYPES = {
  vtk: "application/octet-stream",
  npy: "application/octet-stream",
  ims: "application/octet-stream",
  xml: "application/xml",
  tif: "image/tiff",
  tiff: "image/tiff",
  csv: "text/csv",
  json: "application/json",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
};

export const contentTypeFor = (name) =>
  CONTENT_TYPES[String(name).split(".").pop().toLowerCase()] || "application/octet-stream";

/** Thrown by the parameter layer; the error middleware turns it into a 400. */
export class BadRequest extends Error {
  constructor(message) {
    super(message);
    this.status = 400;
  }
}
