/**
 * The result cache.
 *
 * Every successful run writes `manifest.json` into its output folder
 * (`outputs/{project}/{run}/{step}/`) recording the exact hyperparameters that
 * produced the files beside it. Before dispatching a job, the gateway reads
 * that one small object and, if the parameters match and the files are still
 * there, hands back download links instead of buying another GPU minute.
 *
 * Three design notes worth keeping:
 *
 * **No Redis involved.** R2 is the source of truth for "has this already been
 * computed". Job records expire after a week; results do not, so a cache that
 * depended on Redis would forget answers that are still sitting in the bucket.
 *
 * **The comparison is structural, not a hash.** A fingerprint would have to be
 * computed identically in Python and JavaScript, and float formatting does not
 * survive that round trip -- `json.dumps(1.0)` is `"1.0"`, `JSON.stringify(1.0)`
 * is `"1"`. Comparing the parsed values sidesteps the whole problem, and it
 * yields a usable diff instead of a yes/no.
 *
 * **Misses are cheap; false hits are not.** Every ambiguous case -- no
 * manifest, an unreadable one, a missing file, an empty artifact list --
 * resolves to "recompute". The only path that returns `hit: true` is one where
 * every recorded file was found in the bucket.
 */
import { DOWNLOAD_EXTENSIONS, URL_TTL_SECONDS } from "./config.js";
import * as r2 from "./r2.js";

const extOf = (name) => String(name).split(".").pop().toLowerCase();

/** Structural equality over JSON values (objects compared key-set-wise, not by
 *  insertion order). */
export function deepEqual(a, b) {
  if (a === b) return true;
  if (a === null || b === null || a === undefined || b === undefined) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) &&
      Array.isArray(b) &&
      a.length === b.length &&
      a.every((item, i) => deepEqual(item, b[i]))
    );
  }
  if (typeof a === "object" && typeof b === "object") {
    const keys = Object.keys(a);
    return (
      keys.length === Object.keys(b).length &&
      keys.every((key) => Object.hasOwn(b, key) && deepEqual(a[key], b[key]))
    );
  }
  return false;
}

/**
 * Which parameters differ, so a miss can say *why* rather than just "no".
 * A parameter present on one side only counts as changed -- omitting one means
 * the pipeline default applies, which is a different run from pinning it.
 */
export function diffParams(cached = {}, requested = {}) {
  const names = new Set([...Object.keys(cached || {}), ...Object.keys(requested || {})]);
  const changed = [];
  for (const name of [...names].sort()) {
    if (!deepEqual(cached?.[name], requested?.[name])) {
      changed.push({
        param: name,
        cached: cached?.[name] ?? null,
        requested: requested?.[name] ?? null,
      });
    }
  }
  return changed;
}

/**
 * Looks for a reusable result.
 *
 * Resolves to `{ hit: false, reason }` or `{ hit: true, manifest, artifacts }`
 * where each artifact carries a freshly signed download URL. Never throws for
 * an ordinary miss; a genuine R2 failure does propagate, because silently
 * treating an outage as a cache miss would double-bill every job during it.
 */
export async function lookup({ projectKey, runName, step, params }) {
  const prefix = r2.outputPrefix(projectKey, runName, step);
  const manifest = await r2.getJson(r2.manifestKey(projectKey, runName, step));

  if (!manifest) {
    return { hit: false, reason: `no completed run stored at ${prefix}/`, prefix };
  }

  const changed = diffParams(manifest.params, params);
  if (changed.length > 0) {
    const names = changed.map((c) => c.param);
    return {
      hit: false,
      prefix,
      reason:
        `parameters differ from the stored run (${names.slice(0, 3).join(", ")}` +
        `${names.length > 3 ? `, +${names.length - 3} more` : ""})`,
      changed,
    };
  }

  // The manifest's artifact list is the pipeline's own definition of "an
  // output" (project_layout.ARTIFACT_KINDS); this filter is a second gate so a
  // future artifact kind cannot arrive in the download grid unannounced.
  const recorded = (manifest.artifacts || []).filter((a) =>
    DOWNLOAD_EXTENSIONS.includes(extOf(a.name))
  );
  if (recorded.length === 0) {
    return { hit: false, prefix, reason: "the stored run recorded no downloadable output" };
  }

  // A manifest is a claim about the bucket, not proof: files can be deleted by
  // the storage manager, or by a lifecycle rule, long after it was written.
  const { objects } = await r2.listObjects(`${prefix}/`);
  const present = new Map(objects.map((obj) => [obj.key, obj.size]));
  const missing = recorded.filter((a) => !present.has(a.key));
  if (missing.length > 0) {
    return {
      hit: false,
      prefix,
      reason: `${missing.length} of ${recorded.length} stored file(s) are no longer in the bucket`,
      missing: missing.slice(0, 10).map((a) => a.key),
    };
  }

  const artifacts = await Promise.all(
    recorded.map(async (artifact) => ({
      ...artifact,
      size: present.get(artifact.key) ?? artifact.size ?? 0,
      url: await r2.signDownload(artifact.key, artifact.name),
    }))
  );

  return { hit: true, prefix, manifest, artifacts };
}

/** Shapes a hit into the same download payload `/api/jobs/:id/download`
 *  returns, so the client renders a cached result with no special casing. */
export function presentHit({ manifest, artifacts, prefix }) {
  const byKind = artifacts.reduce((acc, artifact) => {
    (acc[artifact.kind || "Other"] ||= []).push(artifact);
    return acc;
  }, {});

  return {
    from_cache: true,
    // The job that originally produced these files. It may have expired out of
    // Redis; the UI shows it as provenance, not as something to poll.
    job_id: manifest.job_id || null,
    status: "succeeded",
    done: true,
    progress: 100,
    step: manifest.step,
    run_name: manifest.run_name,
    project_key: manifest.project_key,
    prefix,
    cached_at: manifest.completed_at || null,
    original_duration_s: manifest.duration_s ?? null,
    resolved_params: manifest.params || {},
    expires_in: URL_TTL_SECONDS,
    total_bytes: artifacts.reduce((sum, a) => sum + (a.size || 0), 0),
    artifacts,
    by_kind: byKind,
  };
}
