/**
 * Tests for the result cache and the storage manager.
 *
 *   npm test        (server/)
 *
 * No test framework and no network: `node:test` ships with Node, and the S3
 * client's `send` is replaced with an in-memory bucket. Presigning is left
 * real, because it is an offline SigV4 computation -- so these tests still
 * prove the URLs are actually produced.
 *
 * The fake pages at three keys per response on purpose. Continuation-token
 * loops are the classic place an object lister quietly returns the first page
 * and calls it the whole bucket.
 */
import assert from "node:assert/strict";
import { after, before, beforeEach, describe, it } from "node:test";

process.env.CLOUDFLARE_R2_ENDPOINT = "https://account.r2.cloudflarestorage.com";
process.env.CLOUDFLARE_R2_ACCESS_KEY = "test-access-key";
process.env.CLOUDFLARE_R2_SECRET_KEY = "test-secret-key";
process.env.CLOUDFLARE_R2_BUCKET = "tomo-piv-test";
process.env.R2_STORAGE_LIMIT_BYTES = "10000000000"; // 10 GB
process.env.R2_STORAGE_WARN_BYTES = "9000000000"; //  9 GB
process.env.R2_USAGE_CACHE_MS = "0"; // never reuse a total inside a test

// Empty, not deleted. config.js loads server/.env at import, and dotenv fills
// in any key that is *absent* -- so deleting these would hand the tests the
// developer's real Upstash and Modal endpoints. An ioredis client pointed at a
// live Redis also keeps the event loop open, which hangs the runner rather
// than failing it.
process.env.MODAL_API_URL = "";
process.env.REDIS_URL = "";
process.env.UPSTASH_REDIS_URL = "";

const { createApp } = await import("../src/app.js");
const cache = await import("../src/manifest.js");
const r2 = await import("../src/r2.js");
const storage = await import("../src/storage.js");

/* ------------------------------------------------------------------ *
 * In-memory bucket
 * ------------------------------------------------------------------ */

const PAGE_SIZE = 3;
let bucket = new Map();
const realSend = r2.client.send.bind(r2.client);

const put = (key, size, body = null) => bucket.set(key, { size, body });

function fakeSend(command) {
  const name = command.constructor.name;
  const input = command.input;

  if (name === "ListObjectsV2Command") {
    const keys = [...bucket.keys()].filter((k) => k.startsWith(input.Prefix || "")).sort();
    const start = input.ContinuationToken ? keys.indexOf(input.ContinuationToken) + 1 : 0;
    const page = keys.slice(start, start + PAGE_SIZE);
    const last = page.at(-1);
    return Promise.resolve({
      Contents: page.map((key) => ({
        Key: key,
        Size: bucket.get(key).size,
        LastModified: new Date("2026-08-21T00:00:00Z"),
      })),
      IsTruncated: keys.indexOf(last) < keys.length - 1,
      NextContinuationToken: last,
    });
  }

  if (name === "GetObjectCommand") {
    const entry = bucket.get(input.Key);
    if (!entry) {
      const err = new Error("NoSuchKey");
      err.name = "NoSuchKey";
      return Promise.reject(err);
    }
    return Promise.resolve({ Body: { transformToString: async () => entry.body } });
  }

  if (name === "DeleteObjectsCommand") {
    const deleted = [];
    for (const { Key } of input.Delete.Objects) if (bucket.delete(Key)) deleted.push({ Key });
    return Promise.resolve({ Deleted: deleted, Errors: [] });
  }

  if (name === "HeadBucketCommand") return Promise.resolve({});

  return Promise.reject(new Error(`unexpected command: ${name}`));
}

/** The parameter dict params.js builds from the console's default settings. */
const PARAMS = {
  HAS_T2: true,
  NOISE_FLOOR_PERCENTILE: 99.3,
  CROSS_CORR_WINDOW_XY: 32,
  CROSS_CORR_WINDOW_XY_FINE: 16,
  CROSS_CORR_DT_US: 50,
  VOLUME_Z_MIN_MM: -6,
  VOLUME_Z_MAX_MM: 6,
  mart_iterations: 15,
};

const OUT = "outputs/default/RUN_A/step6";

function seedCompletedRun(params = PARAMS) {
  put(`${OUT}/Volumes/vector_00.vtk`, 4_000_000);
  put(`${OUT}/Volumes/vector_00.csv`, 120_000);
  put(`${OUT}/Volumes/volume_00_MIP.jpg`, 90_000);
  put(
    `${OUT}/manifest.json`,
    512,
    JSON.stringify({
      schema: 1,
      job_id: "20260821-101500-abcd1234",
      project_key: "default",
      run_name: "RUN_A",
      step: "step6",
      params,
      artifacts: [
        { key: `${OUT}/Volumes/vector_00.vtk`, name: "vector_00.vtk", kind: "Vector fields (VTK)", size: 4_000_000 },
        { key: `${OUT}/Volumes/vector_00.csv`, name: "vector_00.csv", kind: "Vector tables (CSV)", size: 120_000 },
        { key: `${OUT}/Volumes/volume_00_MIP.jpg`, name: "volume_00_MIP.jpg", kind: "Volume previews (JPG)", size: 90_000 },
      ],
      total_bytes: 4_210_000,
      duration_s: 412.5,
      completed_at: "2026-08-21T10:15:00Z",
    })
  );
}

before(() => {
  r2.client.send = fakeSend;
});
after(() => {
  r2.client.send = realSend;
});
beforeEach(() => {
  bucket = new Map();
  storage.invalidateUsage();
});

/* ------------------------------------------------------------------ *
 * Object listing
 * ------------------------------------------------------------------ */

describe("listObjects", () => {
  it("follows continuation tokens to the end of the bucket", async () => {
    for (let i = 0; i < 10; i += 1) put(`inputs/default/RUN_A/Frame${String(i).padStart(2, "0")}-0.ims`, 1000);

    const { objects, truncated } = await r2.listObjects("inputs/");

    assert.equal(objects.length, 10, "must page past the first 3 results");
    assert.equal(truncated, false);
    assert.equal(objects[0].size, 1000);
  });

  it("stops at the cap and says so rather than throwing", async () => {
    for (let i = 0; i < 10; i += 1) put(`inputs/default/RUN_A/f${i}.ims`, 1);

    const { objects, truncated } = await r2.listObjects("inputs/", { limit: 4 });

    assert.equal(objects.length, 4);
    assert.equal(truncated, true);
  });
});

/* ------------------------------------------------------------------ *
 * Parameter comparison
 * ------------------------------------------------------------------ */

describe("diffParams", () => {
  it("is insensitive to key order", () => {
    assert.deepEqual(cache.diffParams({ a: 1, b: 2 }, { b: 2, a: 1 }), []);
  });

  it("treats an omitted parameter as a change", () => {
    // Omitting one means the pipeline default applies, which is a different
    // run from pinning it -- even if the values happen to coincide today.
    const changed = cache.diffParams({ a: 1, b: 2 }, { a: 1 });
    assert.equal(changed.length, 1);
    assert.equal(changed[0].param, "b");
    assert.equal(changed[0].requested, null);
  });

  it("compares nested arrays element-wise", () => {
    assert.deepEqual(cache.diffParams({ burst: [0, 1, 2] }, { burst: [0, 1, 2] }), []);
    assert.equal(cache.diffParams({ burst: [0, 1, 2] }, { burst: [0, 1] }).length, 1);
  });

  it("survives the Python -> JSON -> JS float round trip", () => {
    // json.dumps(1.0) is "1.0"; JSON.parse gives 1. Same number, and the
    // structural compare must not care about the formatting difference.
    assert.deepEqual(cache.diffParams(JSON.parse('{"dt": 1.0}'), { dt: 1 }), []);
    assert.deepEqual(cache.diffParams(JSON.parse('{"p": 99.3}'), { p: 99.3 }), []);
  });
});

/* ------------------------------------------------------------------ *
 * The cache
 * ------------------------------------------------------------------ */

describe("cache lookup", () => {
  const args = { projectKey: "default", runName: "RUN_A", step: "step6", params: PARAMS };

  it("hits on identical parameters and signs every recorded file", async () => {
    seedCompletedRun();

    const found = await cache.lookup(args);

    assert.equal(found.hit, true);
    assert.equal(found.artifacts.length, 3);
    for (const artifact of found.artifacts) {
      assert.match(artifact.url, /^https:\/\/account\.r2\.cloudflarestorage\.com\//);
      assert.match(artifact.url, /X-Amz-Signature=/);
    }

    const payload = cache.presentHit(found);
    assert.equal(payload.from_cache, true);
    assert.equal(payload.status, "succeeded");
    assert.equal(payload.done, true);
    assert.equal(payload.progress, 100);
    assert.equal(payload.job_id, "20260821-101500-abcd1234");
    assert.equal(payload.total_bytes, 4_210_000);
    assert.deepEqual(Object.keys(payload.by_kind).sort(), [
      "Vector fields (VTK)",
      "Vector tables (CSV)",
      "Volume previews (JPG)",
    ]);
  });

  it("misses when nothing has been computed yet", async () => {
    const found = await cache.lookup(args);
    assert.equal(found.hit, false);
    assert.match(found.reason, /no completed run stored at outputs\/default\/RUN_A\/step6\//);
  });

  it("misses when a parameter changed, and names which one", async () => {
    seedCompletedRun();

    const found = await cache.lookup({ ...args, params: { ...PARAMS, mart_iterations: 20 } });

    assert.equal(found.hit, false);
    assert.match(found.reason, /mart_iterations/);
    assert.deepEqual(found.changed, [{ param: "mart_iterations", cached: 15, requested: 20 }]);
  });

  it("misses when a recorded file has been deleted from the bucket", async () => {
    seedCompletedRun();
    bucket.delete(`${OUT}/Volumes/vector_00.vtk`);

    const found = await cache.lookup(args);

    assert.equal(found.hit, false);
    assert.match(found.reason, /1 of 3 stored file\(s\) are no longer in the bucket/);
  });

  it("misses when the stored run produced nothing downloadable", async () => {
    put(`${OUT}/manifest.json`, 100, JSON.stringify({ params: PARAMS, artifacts: [] }));

    const found = await cache.lookup(args);

    assert.equal(found.hit, false);
    assert.match(found.reason, /no downloadable output/);
  });

  it("misses for a different run or step at the same parameters", async () => {
    seedCompletedRun();

    assert.equal((await cache.lookup({ ...args, runName: "RUN_B" })).hit, false);
    assert.equal((await cache.lookup({ ...args, step: "steps45" })).hit, false);
  });
});

/* ------------------------------------------------------------------ *
 * Usage and quota
 * ------------------------------------------------------------------ */

describe("usage", () => {
  it("reports headroom below the threshold with no alert", async () => {
    put("inputs/default/RUN_A/Frame00-0.ims", 2e9);

    const result = await storage.usage({ fresh: true });

    assert.equal(result.bytes, 2e9);
    assert.equal(result.gb, 2);
    assert.equal(result.limit_gb, 10);
    assert.equal(result.percent, 20);
    assert.equal(result.over_threshold, false);
    assert.equal(result.uploads_blocked, false);
    assert.equal(result.message, null);
  });

  it("raises the alert at exactly 9 GB, not a byte later", async () => {
    put("inputs/default/RUN_A/Frame00-0.ims", 9e9);

    const result = await storage.usage({ fresh: true });

    assert.equal(result.over_threshold, true);
    assert.equal(result.uploads_blocked, true);
    assert.match(result.message, /^Storage Quota Alert: Bucket usage is 9\.00 GB \/ 10\.00 GB\./);
    assert.match(result.message, /Delete files before uploading new runs\.$/);
  });

  it("sums across both roots, not just inputs", async () => {
    put("inputs/default/RUN_A/Frame00-0.ims", 4e9);
    put("outputs/default/RUN_A/step6/Volumes/vector_00.vtk", 5.5e9);

    assert.equal((await storage.usage({ fresh: true })).uploads_blocked, true);
  });
});

/* ------------------------------------------------------------------ *
 * Browsing
 * ------------------------------------------------------------------ */

describe("browse", () => {
  beforeEach(() => {
    put("inputs/default/RUN_A/Frame00-0.ims", 4_000_000);
    put("inputs/default/RUN_A/Frame00-1.ims", 4_000_000);
    put("inputs/default/Properties/Calibration/Calibration.xml", 8_000);
    put("outputs/default/RUN_A/step6/Volumes/vector_00.vtk", 4_000_000);
    put("outputs/default/RUN_A/step6/manifest.json", 512);
  });

  it("groups objects into folders with rollups", async () => {
    const listing = await storage.browse();

    assert.deepEqual(listing.sections.map((s) => s.id), ["inputs", "outputs"]);

    const folders = listing.sections[0].folders;
    const run = folders.find((f) => f.prefix === "inputs/default/RUN_A");
    assert.equal(run.object_count, 2);
    assert.equal(run.bytes, 8_000_000);
    assert.equal(run.run_name, "RUN_A");
    assert.equal(run.files[0].name, "Frame00-0.ims");
  });

  it("recognises the calibration folder and the cache manifest", async () => {
    const listing = await storage.browse();

    const calibration = listing.sections[0].folders.find((f) => f.calibration);
    assert.equal(calibration.prefix, "inputs/default/Properties/Calibration");
    assert.equal(calibration.run_name, null);

    const outputs = listing.sections[1].folders.find((f) => f.prefix === "outputs/default/RUN_A/step6");
    assert.equal(outputs.step, "step6");
    assert.equal(outputs.files.find((f) => f.name === "manifest.json").manifest, true);
  });

  it("rolls a run up across both roots", async () => {
    const listing = await storage.browse();

    assert.equal(listing.runs.length, 1);
    const [run] = listing.runs;
    assert.equal(run.run_name, "RUN_A");
    assert.equal(run.input_bytes, 8_000_000);
    assert.equal(run.output_bytes, 4_000_512);
    // Calibration is per project, not per run, so it is not attributed to one.
    assert.equal(run.bytes, 12_000_512);
  });

  it("does not mistake the pre-cache output layout for a project and run", async () => {
    // outputs/{run}/{job_id}/{group} has the same shape as the current
    // outputs/{project}/{run}/{step}, so without the step check this folder
    // would be reported as a run called "20260821-062240-84a6d3f0".
    put("outputs/RUN_A/20260821-062240-84a6d3f0/volumes/vector_00.vtk", 3_000_000);

    const listing = await storage.browse();
    const folder = listing.sections[1].folders.find((f) => f.legacy);

    assert.equal(folder.prefix, "outputs/RUN_A/20260821-062240-84a6d3f0/volumes");
    assert.equal(folder.run_name, null);
    assert.equal(folder.step, null);
    assert.match(folder.label, /legacy layout/);

    assert.equal(
      listing.runs.some((r) => r.run_name.startsWith("20260821-")),
      false,
      "a job id must never appear as a run"
    );

    assert.equal(listing.legacy.bytes, 3_000_000);
    assert.equal(listing.legacy.object_count, 1);
    assert.deepEqual(listing.legacy.prefixes, [
      { prefix: "outputs/RUN_A/20260821-062240-84a6d3f0", bytes: 3_000_000 },
    ]);
  });

  it("counts a current-layout output as a run, not as legacy", async () => {
    const listing = await storage.browse();

    assert.equal(listing.legacy.bytes, 0);
    assert.equal(listing.runs[0].output_bytes, 4_000_512);
  });

  it("refuses a prefix outside inputs/ and outputs/", async () => {
    // Traversal is rejected before the root check, so it reports the more
    // specific reason -- both are refusals, and the order is the safe one.
    await assert.rejects(() => storage.browse("../"), /may not contain '\.\.'/);
    await assert.rejects(() => storage.browse("secrets/"), /must start with inputs\/ or outputs\//);
    await assert.rejects(() => storage.browse("inputsX/"), /must start with inputs\/ or outputs\//);
  });
});

/* ------------------------------------------------------------------ *
 * Deletion
 * ------------------------------------------------------------------ */

describe("remove", () => {
  beforeEach(() => {
    put("inputs/default/RUN_A/Frame00-0.ims", 4_000_000);
    put("inputs/default/RUN_A/Frame00-1.ims", 4_000_000);
    put("inputs/default/RUN_B/Frame00-0.ims", 4_000_000);
    put("outputs/default/RUN_A/step6/Volumes/vector_00.vtk", 1_000_000);
    put("outputs/default/RUN_A/step6/manifest.json", 512);
  });

  it("deletes a single file and reports what it freed", async () => {
    const result = await storage.remove({ key: "inputs/default/RUN_A/Frame00-0.ims" });

    assert.equal(result.deleted, 1);
    assert.equal(result.freed_bytes, 4_000_000);
    assert.equal(bucket.has("inputs/default/RUN_A/Frame00-0.ims"), false);
    assert.equal(bucket.has("inputs/default/RUN_A/Frame00-1.ims"), true);
  });

  it("deletes a folder recursively without touching a sibling", async () => {
    const result = await storage.remove({ prefix: "inputs/default/RUN_A" });

    assert.equal(result.deleted, 2);
    assert.equal(bucket.has("inputs/default/RUN_B/Frame00-0.ims"), true);
  });

  it("does not let a prefix bleed into a similarly named sibling", async () => {
    put("inputs/default/RUN_A_2/Frame00-0.ims", 1);

    await storage.remove({ prefix: "inputs/default/RUN_A" });

    assert.equal(bucket.has("inputs/default/RUN_A_2/Frame00-0.ims"), true);
  });

  it("deletes a run's inputs and outputs together", async () => {
    const result = await storage.remove({ runName: "RUN_A", projectKey: "default" });

    assert.equal(result.deleted, 4);
    assert.deepEqual(
      [...bucket.keys()].sort(),
      ["inputs/default/RUN_B/Frame00-0.ims"]
    );
  });

  it("leaves no cached result behind when a run is deleted", async () => {
    // The whole point: a deleted run must not be servable from cache
    // afterwards, or the next submit would return links to files that no
    // longer exist.
    await storage.remove({ runName: "RUN_A", projectKey: "default" });

    const found = await cache.lookup({
      projectKey: "default",
      runName: "RUN_A",
      step: "step6",
      params: PARAMS,
    });
    assert.equal(found.hit, false);
  });

  it("deletes several folders in one call without double-counting overlap", async () => {
    const result = await storage.remove({
      prefixes: [
        "inputs/default/RUN_A",
        "outputs/default/RUN_A/step6",
        "inputs/default/RUN_A/", // same target again, trailing slash
      ],
    });

    assert.equal(result.deleted, 4);
    assert.equal(result.freed_bytes, 9_000_512, "an overlapping target must not be counted twice");
    assert.deepEqual([...bucket.keys()], ["inputs/default/RUN_B/Frame00-0.ims"]);
  });

  it("applies the same guardrails to every entry in a batch", async () => {
    await assert.rejects(
      () => storage.remove({ prefixes: ["inputs/default/RUN_A", "secrets/"] }),
      /must start with inputs\/ or outputs\//
    );
    assert.equal(bucket.size, 5, "a bad entry must abort the whole batch");
  });

  it("reports a no-op instead of pretending to succeed", async () => {
    const result = await storage.remove({ prefix: "inputs/default/NOPE" });

    assert.equal(result.deleted, 0);
    assert.match(result.detail, /nothing matched/);
  });

  it("refuses targets that could empty the bucket", async () => {
    await assert.rejects(() => storage.remove({}), /provide one of/);
    await assert.rejects(() => storage.remove({ prefix: "" }), /provide one of/);
    await assert.rejects(() => storage.remove({ prefix: "/" }), /must start with/);
    await assert.rejects(() => storage.remove({ prefix: "inputs/../" }), /may not contain/);
    await assert.rejects(() => storage.remove({ prefix: "inputs/" }), /needs force: true/);
    await assert.rejects(() => storage.remove({ runName: "../other" }), /illegal run_name/);

    assert.equal(bucket.size, 5, "no guardrail may delete anything");
  });

  it("clears a whole root when force is given", async () => {
    const result = await storage.remove({ prefix: "outputs/", force: true });

    assert.equal(result.deleted, 2);
    assert.equal([...bucket.keys()].every((k) => k.startsWith("inputs/")), true);
  });
});

/* ------------------------------------------------------------------ *
 * HTTP surface
 * ------------------------------------------------------------------ */

describe("HTTP routes", () => {
  let server;
  let base;

  before(async () => {
    server = createApp().listen(0);
    await new Promise((resolve) => server.once("listening", resolve));
    base = `http://127.0.0.1:${server.address().port}`;
  });
  after(() => server.close());

  const request = async (path, options) => {
    const response = await fetch(base + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    return { status: response.status, body: await response.json() };
  };

  const submit = (overrides = {}) =>
    request("/api/jobs/submit", {
      method: "POST",
      body: JSON.stringify({
        step: "step6",
        run_name: "RUN_A",
        project_key: "default",
        has_t2: true,
        noise_floor_pct: 99.3,
        mart_iterations: 15,
        z_range: [-6, 6],
        dt_us: 50,
        window_xy: 32,
        ...overrides,
      }),
    });

  it("serves a cache hit without Modal configured at all", async () => {
    seedCompletedRun();

    const { status, body } = await submit();

    // MODAL_API_URL is unset in this process, so reaching the dispatcher at
    // all would be a 503. A 200 proves no GPU dispatch was attempted.
    assert.equal(status, 200);
    assert.equal(body.from_cache, true);
    assert.equal(body.artifacts.length, 3);
    assert.equal(body.prefix, OUT);
  });

  it("falls through to dispatch when the parameters differ", async () => {
    seedCompletedRun();

    const { status, body } = await submit({ mart_iterations: 20 });

    assert.equal(status, 503);
    assert.match(body.error, /MODAL_API_URL is not configured/);
  });

  it("bypasses the cache when force is set", async () => {
    seedCompletedRun();

    const { status } = await submit({ force: true });

    assert.equal(status, 503, "force must skip the lookup and go to the dispatcher");
  });

  it("still validates parameters before consulting the cache", async () => {
    seedCompletedRun();

    const { status, body } = await submit({ z_range: [6, -6] });

    assert.equal(status, 400);
    assert.match(body.error, /z_range is empty/);
  });

  it("refuses dataset uploads past the quota but still allows calibration", async () => {
    put("inputs/default/RUN_A/big.ims", 9.4e9);

    const dataset = await request("/api/upload-url", {
      method: "POST",
      body: JSON.stringify({ run_name: "RUN_B", filenames: ["Frame00-0.ims"] }),
    });
    assert.equal(dataset.status, 507);
    assert.match(dataset.body.error, /Storage Quota Alert/);

    const calibration = await request("/api/upload-url", {
      method: "POST",
      body: JSON.stringify({ target: "calibration", filenames: ["Calibration/Calibration.xml"] }),
    });
    assert.equal(calibration.status, 200);
    assert.equal(calibration.body.uploads.length, 1);
  });

  it("lists the bucket and deletes through the API", async () => {
    put("inputs/default/RUN_A/Frame00-0.ims", 4_000_000);
    put("outputs/default/RUN_A/step6/Volumes/vector_00.vtk", 1_000_000);

    const listed = await request("/api/storage/list");
    assert.equal(listed.status, 200);
    assert.equal(listed.body.object_count, 2);
    assert.equal(listed.body.total_bytes, 5_000_000);
    assert.equal(listed.body.usage.gb, 0.01);

    const deleted = await request("/api/storage/delete", {
      method: "DELETE",
      body: JSON.stringify({ run_name: "RUN_A" }),
    });
    assert.equal(deleted.status, 200);
    assert.equal(deleted.body.deleted, 2);
    assert.equal(deleted.body.freed_bytes, 5_000_000);

    assert.equal((await request("/api/storage/list")).body.object_count, 0);
  });

  it("signs a stored object for download, and inline for the viewer", async () => {
    put("outputs/default/RUN_A/step3/drop/Camera1_t1.jpg", 1_000_000);
    const key = "outputs/default/RUN_A/step3/drop/Camera1_t1.jpg";

    const { status, body } = await request(`/api/storage/download?key=${encodeURIComponent(key)}`);
    assert.equal(status, 200);
    assert.equal(body.name, "Camera1_t1.jpg");
    assert.match(body.url, /X-Amz-Signature=/);
    // The disposition is inside the signature, so it must appear in the URL.
    assert.match(decodeURIComponent(body.url), /attachment; filename="Camera1_t1\.jpg"/);

    const inline = await fetch(
      `${base}/api/storage/download?key=${encodeURIComponent(key)}&inline=1`,
      { redirect: "manual" }
    );
    assert.equal(inline.status, 302);
    assert.match(decodeURIComponent(inline.headers.get("location")), /inline; filename="Camera1_t1\.jpg"/);
  });

  it("will not presign an object outside the pipeline's own prefixes", async () => {
    // Without this the endpoint hands out a readable URL for anything in the
    // bucket to anyone who can reach the gateway.
    for (const key of ["secrets/creds.json", "../etc/passwd", ""]) {
      const { status } = await request(`/api/storage/download?key=${encodeURIComponent(key)}`);
      assert.equal(status, 400, `key ${JSON.stringify(key)} must be refused`);
    }
  });

  it("turns a bad delete target into a 400, not a 500", async () => {
    const { status, body } = await request("/api/storage/delete", {
      method: "DELETE",
      body: JSON.stringify({ prefix: "secrets/" }),
    });

    assert.equal(status, 400);
    assert.match(body.error, /must start with inputs\/ or outputs\//);
  });
});
