/**
 * Job records: reading them, and turning them into something the UI can show.
 *
 * The Python worker owns the schema (JobStore / JobRecord in cloud_io.py) and
 * writes every record to Redis. This module reads that store directly, which
 * is why polling a running job costs no GPU container time. If Redis is not
 * configured -- or the worker fell back to modal.Dict -- it asks the Modal API
 * instead, so the endpoint behaves the same either way.
 */
import Redis from "ioredis";

import { REDIS_NAMESPACE, REDIS_URL, TERMINAL_STATUSES } from "./config.js";
import * as modal from "./modal.js";

let redis = null;
let redisReady = false;

if (REDIS_URL) {
  redis = new Redis(REDIS_URL, {
    lazyConnect: true,
    maxRetriesPerRequest: 2,
    // Upstash requires TLS; a rediss:// URL turns that on automatically.
    retryStrategy: (times) => Math.min(times * 200, 2000),
  });
  redis.on("ready", () => {
    redisReady = true;
    console.log("[redis] connected");
  });
  redis.on("error", (err) => {
    redisReady = false;
    console.warn("[redis] %s -- falling back to the Modal API for job status", err.message);
  });
  redis.connect().catch(() => {});
} else {
  console.warn("[redis] REDIS_URL not set -- job status will be proxied from Modal");
}

export const redisHealth = () => ({
  ok: redisReady,
  detail: redisReady ? "connected" : REDIS_URL ? "connecting/unavailable" : "not configured",
});

const jobKey = (jobId) => `${REDIS_NAMESPACE}:${jobId}`;
const indexKey = () => `${REDIS_NAMESPACE}:index`;

/** Reads one record: Redis first, then Modal. Returns null if unknown. */
export async function readJob(jobId) {
  if (redis && redisReady) {
    try {
      const raw = await redis.get(jobKey(jobId));
      if (raw) return JSON.parse(raw);
    } catch (err) {
      console.warn("[redis] read failed for %s: %s", jobId, err.message);
    }
  }
  if (!modal.isConfigured()) return null;
  try {
    return await modal.getJob(jobId);
  } catch (err) {
    if (err.response?.status === 404) return null;
    throw err;
  }
}

export async function listJobs(limit) {
  // The index is a sorted set scored by creation time (JobStore._write).
  if (redis && redisReady) {
    const ids = await redis.zrevrange(indexKey(), 0, limit - 1);
    const raws = ids.length ? await redis.mget(ids.map(jobKey)) : [];
    return { source: "redis", records: raws.filter(Boolean).map((raw) => JSON.parse(raw)) };
  }
  if (!modal.isConfigured()) return { source: "none", records: [] };
  const data = await modal.listJobs(limit);
  return { source: "modal", records: data.jobs || [] };
}

/**
 * Progress markers.
 *
 * The pipeline reports no percentage of its own, but it does log a distinctive
 * banner as each step starts, and a `MART iter n/N` line per reconstruction
 * pass. Reading those gives an honest coarse estimate -- it is a progress
 * indicator, not a measurement, and the UI labels it as such.
 */
const STEP_MARKERS = [
  { marker: "=== Step 1", weight: 0.08 },
  { marker: "=== Step 2", weight: 0.22 },
  { marker: "=== Step 3", weight: 0.35 },
  { marker: "=== Steps 4-5", weight: 0.45 },
  { marker: "=== Step 6", weight: 0.85 },
];
const MART_SPAN_START = 0.45;
const MART_SPAN_END = 0.85;

export function deriveProgress(record) {
  if (!record) return 0;
  if (record.status === "succeeded") return 100;
  if (record.status === "queued") return 2;
  if (TERMINAL_STATUSES.includes(record.status)) return record.progress ?? 0;

  const logs = Array.isArray(record.logs) ? record.logs : [];
  const text = logs.join("\n");

  let progress = 0.05;
  let reached = null;
  for (const { marker, weight } of STEP_MARKERS) {
    if (text.includes(marker)) {
      progress = weight;
      reached = marker;
    }
  }

  // Inside MART the per-iteration log gives real sub-step resolution.
  if (reached === "=== Steps 4-5") {
    const iterations = [...text.matchAll(/MART iter (\d+)\/(\d+)/g)];
    if (iterations.length) {
      const [, done, total] = iterations[iterations.length - 1];
      progress = MART_SPAN_START + (MART_SPAN_END - MART_SPAN_START) * Math.min(1, Number(done) / Number(total));
    }
  }

  // A single-step job is finished when that one step is, so it should not sit
  // at a full-pipeline weight.
  if (record.step && record.step !== "full") progress = Math.max(progress, 0.5);

  return Math.round(Math.min(0.97, progress) * 100);
}

/** Shapes a record for the client, hiding the store's internals. */
export function presentJob(record, { logTail = 200 } = {}) {
  const logs = Array.isArray(record.logs) ? record.logs : [];
  return {
    job_id: record.job_id,
    run_name: record.run_name,
    step: record.step,
    status: record.status,
    message: record.message || "",
    error: record.error || null,
    progress: deriveProgress(record),
    done: TERMINAL_STATUSES.includes(record.status),
    params: record.params || {},
    created_at: record.created_at,
    started_at: record.started_at,
    finished_at: record.finished_at,
    duration_s: record.started_at
      ? Number(((record.finished_at || Date.now() / 1000) - record.started_at).toFixed(1))
      : 0,
    log_count: logs.length,
    logs: logTail > 0 ? logs.slice(-logTail) : [],
    artifact_count: (record.artifacts || []).length,
  };
}
