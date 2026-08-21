/**
 * HTTP client for the Python GPU backend (app_modal.py's FastAPI surface).
 * This gateway contacts Modal only to START or CANCEL work -- job polling is
 * served from Redis, so a running job costs no container time to observe.
 */
import axios from "axios";

import { MODAL_API_TOKEN, MODAL_API_URL } from "./config.js";

export const isConfigured = () => Boolean(MODAL_API_URL);

export const client = axios.create({
  baseURL: MODAL_API_URL || undefined,
  timeout: 30000,
  headers: MODAL_API_TOKEN ? { Authorization: `Bearer ${MODAL_API_TOKEN}` } : {},
});

export async function health() {
  if (!MODAL_API_URL) return { ok: false, detail: "MODAL_API_URL not set" };
  try {
    const { data } = await client.get("/health", { timeout: 8000 });
    return { ok: true, detail: `gpu ${data.gpu}, job store ${data.job_store}` };
  } catch (err) {
    return { ok: false, detail: err.message };
  }
}

export async function submitJob({ runName, step, projectKey, params }) {
  const { data, status } = await client.post("/jobs", {
    run_name: runName,
    step,
    project_key: projectKey,
    params,
  });
  return { data, status };
}

export const cancelJob = (jobId) =>
  client.delete(`/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data);

export const getJob = (jobId) =>
  client.get(`/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data);

export const listJobs = (limit) =>
  client.get("/jobs", { params: { limit } }).then((r) => r.data);

export const listRuns = (projectKey) =>
  client.get("/runs", { params: { project_key: projectKey } }).then((r) => r.data);
