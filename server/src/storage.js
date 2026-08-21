/**
 * Bucket browsing, usage accounting and deletion — the model behind the
 * console's Cloud Storage tab.
 *
 * R2's free tier is 10 GB and a single raw acquisition run is around 700 MB,
 * so this is not housekeeping: filling the bucket is roughly a dozen uploads
 * away, and the failure mode without a gauge is a PUT that 4xx's halfway
 * through a multi-GB upload with no explanation.
 */
import {
  BadRequest,
  MANIFEST_NAME,
  STEPS,
  STORAGE_LIMIT_BYTES,
  STORAGE_ROOTS,
  STORAGE_WARN_BYTES,
  USAGE_CACHE_MS,
} from "./config.js";
import * as r2 from "./r2.js";

/** Decimal GB, matching how Cloudflare meters and bills storage. */
export const toGB = (bytes) => bytes / 1e9;
const formatGB = (bytes) => `${toGB(bytes).toFixed(2)} GB`;

/* ------------------------------------------------------------------ *
 * Usage
 * ------------------------------------------------------------------ */

// Computing usage means listing every object in the bucket, and the console
// polls it, so the result is reused briefly. Anything that changes the total
// from this process invalidates it immediately, so the number is only ever
// stale with respect to writes made elsewhere.
let usageCache = null;

export const invalidateUsage = () => {
  usageCache = null;
};

export async function usage({ fresh = false } = {}) {
  const now = Date.now();
  if (!fresh && usageCache && now - usageCache.at < USAGE_CACHE_MS) return usageCache.value;

  const { objects, truncated } = await r2.listObjects("");
  const bytes = objects.reduce((sum, obj) => sum + obj.size, 0);
  const blocked = bytes >= STORAGE_WARN_BYTES;

  const value = {
    bytes,
    object_count: objects.length,
    truncated,
    limit_bytes: STORAGE_LIMIT_BYTES,
    warn_bytes: STORAGE_WARN_BYTES,
    gb: Number(toGB(bytes).toFixed(2)),
    limit_gb: Number(toGB(STORAGE_LIMIT_BYTES).toFixed(2)),
    warn_gb: Number(toGB(STORAGE_WARN_BYTES).toFixed(2)),
    percent: Math.min(100, Math.round((bytes / STORAGE_LIMIT_BYTES) * 100)),
    over_threshold: blocked,
    // Datasets only. A calibration file is ~8 KB and every run in the project
    // depends on it -- refusing it to reclaim 8 KB out of 10 GB would strand
    // runs whose frames are already uploaded.
    uploads_blocked: blocked,
    message: blocked
      ? `Storage Quota Alert: Bucket usage is ${formatGB(bytes)} / ${formatGB(STORAGE_LIMIT_BYTES)}. ` +
        "Delete files before uploading new runs."
      : null,
    checked_at: new Date(now).toISOString(),
  };

  usageCache = { at: now, value };
  return value;
}

/* ------------------------------------------------------------------ *
 * Browsing
 * ------------------------------------------------------------------ */

/**
 * Reads meaning out of a folder path. The layout is fixed by cloud_io.py:
 *
 *   inputs/{project}/{run}/…                 raw acquisition frames
 *   inputs/{project}/Properties/Calibration/ per-project calibration
 *   outputs/{project}/{run}/{step}/{group}/  results of one step
 *
 * so a folder can be labelled and attributed to a run without any extra
 * metadata call.
 *
 * Outputs written before the cache existed used `outputs/{run}/{job_id}/…`,
 * which has the same shape and would otherwise be misread as a project called
 * `{run}` containing a run called `{job_id}`. The fourth segment settles it:
 * in the current layout it is always a step name. Those folders are marked
 * `legacy` -- they can never produce a cache hit, so the only useful thing to
 * do with them is see how much space they occupy and delete them.
 */
function describe(folder) {
  const [root, project, third, fourth] = folder.split("/");
  const base = { project_key: project || null, run_name: null, step: null, legacy: false };

  if (root === "inputs") {
    if (third === "Properties") return { ...base, label: `${project} · calibration`, calibration: true };
    return { ...base, run_name: third || null, label: [project, third].filter(Boolean).join(" · ") };
  }

  if (root === "outputs") {
    if (!STEPS.includes(fourth)) {
      return {
        ...base,
        project_key: null,
        legacy: true,
        label: `${[project, third].filter(Boolean).join(" · ")} — legacy layout`,
      };
    }
    return {
      ...base,
      run_name: third || null,
      step: fourth,
      label: [project, third, fourth].filter(Boolean).join(" · "),
    };
  }

  return { ...base, label: folder };
}

/** True when an object key sits in a current-layout output folder. */
function isCurrentOutput(parts) {
  return parts[0] === "outputs" && STEPS.includes(parts[3]);
}

function groupIntoFolders(objects) {
  const folders = new Map();

  for (const object of objects) {
    const cut = object.key.lastIndexOf("/");
    const folder = cut === -1 ? "" : object.key.slice(0, cut);
    if (!folders.has(folder)) {
      folders.set(folder, { prefix: folder, ...describe(folder), bytes: 0, files: [] });
    }
    const entry = folders.get(folder);
    entry.bytes += object.size;
    entry.files.push({
      key: object.key,
      name: object.key.slice(cut + 1),
      size: object.size,
      last_modified: object.last_modified,
      // The completion marker is what makes the folder a cache entry; the UI
      // flags it rather than letting it look like just another output file.
      manifest: object.key.slice(cut + 1) === MANIFEST_NAME,
    });
  }

  return [...folders.values()]
    .map((folder) => ({ ...folder, object_count: folder.files.length }))
    .sort((a, b) => a.prefix.localeCompare(b.prefix));
}

/** Per-run rollup spanning both roots, so "Delete run" can show what it will
 *  actually reclaim before the user commits to it. Legacy output folders are
 *  left out: their second segment is a run name and their third a job id, so
 *  counting them would invent runs that never existed. */
function summarizeRuns(objects) {
  const runs = new Map();

  for (const object of objects) {
    const parts = object.key.split("/");
    const [root, project, third] = parts;
    if (!project || !third || third === "Properties") continue;
    if (root === "outputs" && !isCurrentOutput(parts)) continue;
    if (root !== "inputs" && root !== "outputs") continue;

    const id = `${project}/${third}`;
    if (!runs.has(id)) {
      runs.set(id, {
        project_key: project,
        run_name: third,
        input_bytes: 0,
        output_bytes: 0,
        bytes: 0,
        object_count: 0,
      });
    }
    const run = runs.get(id);
    run[root === "inputs" ? "input_bytes" : "output_bytes"] += object.size;
    run.bytes += object.size;
    run.object_count += 1;
  }

  return [...runs.values()].sort((a, b) => b.bytes - a.bytes);
}

/** What the pre-cache output layout is still costing, and the exact prefixes
 *  to pass to `remove` to reclaim it. */
function summarizeLegacy(objects) {
  const prefixes = new Map();
  let bytes = 0;
  let objectCount = 0;

  for (const object of objects) {
    const parts = object.key.split("/");
    if (parts[0] !== "outputs" || isCurrentOutput(parts)) continue;
    // One entry per outputs/{run}/{job_id}, which is what a legacy job wrote.
    const prefix = parts.slice(0, 3).join("/");
    prefixes.set(prefix, (prefixes.get(prefix) || 0) + object.size);
    bytes += object.size;
    objectCount += 1;
  }

  return {
    bytes,
    object_count: objectCount,
    prefixes: [...prefixes.entries()]
      .map(([prefix, size]) => ({ prefix, bytes: size }))
      .sort((a, b) => b.bytes - a.bytes),
  };
}

/** Rejects anything outside the two prefixes the pipeline owns. Without this,
 *  a `prefix` of `""` is a request to list — or delete — the entire bucket. */
export function assertInsideRoot(target, field = "key") {
  if (typeof target !== "string" || !target.trim()) {
    throw new BadRequest(`${field} must be a non-empty string`);
  }
  if (target.includes("..")) throw new BadRequest(`${field} may not contain '..'`);
  if (!STORAGE_ROOTS.some((root) => target.startsWith(root))) {
    throw new BadRequest(`${field} must start with ${STORAGE_ROOTS.join(" or ")}`);
  }
  return target;
}

/**
 * Everything under `inputs/` and `outputs/`, or under one prefix if given.
 * Returns folders with rollups plus their file lists, which is what a file
 * manager needs in a single round trip.
 */
export async function browse(prefix = "") {
  const roots = prefix ? [assertInsideRoot(prefix, "prefix")] : STORAGE_ROOTS;
  const sections = [];
  let all = [];
  let truncated = false;

  for (const root of roots) {
    const { objects, truncated: cut } = await r2.listObjects(root);
    truncated = truncated || cut;
    all = all.concat(objects);
    sections.push({
      id: root.replace(/\/$/, ""),
      prefix: root,
      object_count: objects.length,
      bytes: objects.reduce((sum, obj) => sum + obj.size, 0),
      folders: groupIntoFolders(objects),
    });
  }

  return {
    bucket: r2.bucket,
    prefix: prefix || null,
    truncated,
    object_count: all.length,
    total_bytes: all.reduce((sum, obj) => sum + obj.size, 0),
    sections,
    runs: summarizeRuns(all),
    legacy: summarizeLegacy(all),
  };
}

/* ------------------------------------------------------------------ *
 * Deletion
 * ------------------------------------------------------------------ */

/**
 * Deletes one object, one folder, or every trace of a run.
 *
 * Targets are resolved to a concrete key list first, which costs one LIST but
 * buys three things: the reply can state how much space was actually
 * reclaimed, a prefix that matches nothing is reported as such instead of
 * succeeding vacuously, and a single `key` that does not exist cannot be
 * confused with one that did.
 *
 * Deleting a run removes its outputs as well as its inputs, so the next
 * submit is a genuine recompute rather than a cache hit against results whose
 * source frames are gone.
 */
export async function remove({ key, prefix, prefixes, projectKey, runName, force = false } = {}) {
  const targets = [];
  const asFolder = (raw, field) => {
    const normalized = assertInsideRoot(raw, field).replace(/\/+$/, "");
    if (STORAGE_ROOTS.includes(`${normalized}/`) && !force) {
      throw new BadRequest(
        `deleting all of '${normalized}/' needs force: true -- that is the whole ${normalized} tree`
      );
    }
    return `${normalized}/`;
  };

  if (key) {
    assertInsideRoot(key, "key");
    targets.push({ kind: "file", prefix: key, exact: true });
  } else if (Array.isArray(prefixes) && prefixes.length > 0) {
    // Batch form, used for the "clean up legacy outputs" action: those folders
    // have no single common prefix that does not also cover current results.
    for (const each of prefixes) targets.push({ kind: "folder", prefix: asFolder(each, "prefixes[]") });
  } else if (prefix) {
    targets.push({ kind: "folder", prefix: asFolder(prefix, "prefix") });
  } else if (runName) {
    const project = projectKey || "default";
    if (runName.includes("/") || runName.includes("..")) {
      throw new BadRequest(`illegal run_name: ${runName}`);
    }
    targets.push({ kind: "run-inputs", prefix: `inputs/${project}/${runName}/` });
    targets.push({ kind: "run-outputs", prefix: `outputs/${project}/${runName}/` });
  } else {
    throw new BadRequest("provide one of: key, prefix, prefixes, or run_name");
  }

  const seen = new Set();
  const matched = [];
  for (const target of targets) {
    const { objects } = await r2.listObjects(target.prefix);
    for (const object of target.exact ? objects.filter((o) => o.key === target.prefix) : objects) {
      // Overlapping targets must not be double-counted in `freed_bytes`.
      if (seen.has(object.key)) continue;
      seen.add(object.key);
      matched.push(object);
    }
  }

  if (matched.length === 0) {
    return {
      deleted: 0,
      freed_bytes: 0,
      errors: [],
      targets: targets.map((t) => t.prefix),
      detail: "nothing matched -- already deleted?",
    };
  }

  const freed = matched.reduce((sum, obj) => sum + obj.size, 0);
  const { deleted, errors } = await r2.deleteKeys(matched.map((obj) => obj.key));
  invalidateUsage();

  return {
    deleted,
    freed_bytes: freed,
    errors,
    targets: targets.map((t) => t.prefix),
    usage: await usage({ fresh: true }),
  };
}
