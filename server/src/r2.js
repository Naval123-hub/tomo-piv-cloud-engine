/**
 * Cloudflare R2 (S3-compatible) client and URL signing.
 *
 * R2 quirks encoded once here so no route has to think about them:
 *   - the region is always the literal string "auto";
 *   - path-style addressing is required, because R2 does not serve
 *     virtual-host style bucket subdomains on the S3 API endpoint;
 *   - presigning is an offline computation, so it never needs the network.
 */
import {
  DeleteObjectsCommand,
  GetObjectCommand,
  HeadBucketCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

import { MANIFEST_NAME, MAX_LIST_KEYS, R2, URL_TTL_SECONDS, contentTypeFor, r2Configured } from "./config.js";

export const client = r2Configured
  ? new S3Client({
      region: "auto",
      endpoint: R2.endpoint,
      forcePathStyle: true,
      credentials: { accessKeyId: R2.accessKey, secretAccessKey: R2.secretKey },
    })
  : null;

export const isConfigured = () => client !== null;
export const bucket = R2.bucket;

/** Liveness probe for /api/health. */
export async function health() {
  if (!client) return { ok: false, detail: "not configured" };
  try {
    await client.send(new HeadBucketCommand({ Bucket: bucket }));
    return { ok: true, detail: `bucket '${bucket}' reachable` };
  } catch (err) {
    // Surface the real cause -- a bad key, a wrong endpoint host and a missing
    // bucket all fail here and look identical without it.
    return { ok: false, detail: `${err.name}: ${err.message}`.slice(0, 200) };
  }
}

/**
 * Download link. ResponseContentDisposition makes the browser save the file
 * under its real name instead of rendering it inline.
 *
 * `disposition: "inline"` is the exception, for the image viewer: the same
 * object then renders in place rather than downloading. Both are signed the
 * same way -- the header is part of the signature, which is why it has to be
 * chosen here rather than by the caller adding a query parameter.
 */
export function signDownload(key, filename, expiresIn = URL_TTL_SECONDS, disposition = "attachment") {
  return getSignedUrl(
    client,
    new GetObjectCommand({
      Bucket: bucket,
      Key: key,
      ResponseContentDisposition: `${disposition}; filename="${filename}"`,
      ResponseContentType: contentTypeFor(filename),
    }),
    { expiresIn }
  );
}

/**
 * Upload link. The client MUST send this exact Content-Type header on the PUT
 * or the signature will not validate.
 */
export function signUpload(key, expiresIn = URL_TTL_SECONDS) {
  return getSignedUrl(
    client,
    new PutObjectCommand({ Bucket: bucket, Key: key, ContentType: contentTypeFor(key) }),
    { expiresIn }
  );
}

/* ------------------------------------------------------------------ *
 * Raw object operations
 * ------------------------------------------------------------------ */

/**
 * Every object under `prefix`, following continuation tokens.
 *
 * Returns `{ objects, truncated }` rather than throwing at the cap: a bucket
 * larger than expected should degrade to "here is what we could list" with a
 * visible flag, not to an error page.
 */
export async function listObjects(prefix = "", { limit = MAX_LIST_KEYS } = {}) {
  const objects = [];
  let token;
  let truncated = false;

  do {
    const page = await client.send(
      new ListObjectsV2Command({ Bucket: bucket, Prefix: prefix, ContinuationToken: token })
    );
    for (const item of page.Contents || []) {
      if (objects.length >= limit) {
        truncated = true;
        break;
      }
      objects.push({
        key: item.Key,
        size: item.Size ?? 0,
        last_modified: item.LastModified ? new Date(item.LastModified).toISOString() : null,
      });
    }
    token = !truncated && page.IsTruncated ? page.NextContinuationToken : undefined;
  } while (token);

  return { objects, truncated };
}

/** Reads and parses a JSON object. A missing key is `null`, not an error --
 *  "no manifest here" is the normal, expected answer on a cache miss. */
export async function getJson(key) {
  try {
    const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    return JSON.parse(await response.Body.transformToString());
  } catch (err) {
    if (err.name === "NoSuchKey" || err.$metadata?.httpStatusCode === 404) return null;
    throw err;
  }
}

/** Deletes objects in batches of 1000 -- the S3 DeleteObjects limit, which R2
 *  enforces too. Per-key failures come back in `errors` instead of aborting
 *  the whole delete. */
export async function deleteKeys(keys) {
  const unique = [...new Set(keys.filter(Boolean))];
  let deleted = 0;
  const errors = [];

  for (let start = 0; start < unique.length; start += 1000) {
    const batch = unique.slice(start, start + 1000);
    const response = await client.send(
      new DeleteObjectsCommand({
        Bucket: bucket,
        Delete: { Objects: batch.map((Key) => ({ Key })), Quiet: false },
      })
    );
    deleted += (response.Deleted || []).length;
    for (const err of response.Errors || []) {
      errors.push({ key: err.Key, message: err.Message || err.Code || "delete failed" });
    }
  }

  return { deleted, errors, requested: unique.length };
}

/* ------------------------------------------------------------------ *
 * Key conventions
 * ------------------------------------------------------------------ *
 * These come from cloud_io.py -- the GPU worker reads inputs from and writes
 * outputs to exactly these prefixes, so they are not free to drift on one
 * side only. `outputPrefix` in particular is the whole basis of the result
 * cache: no job id in the path is what lets this gateway find an earlier
 * result before it dispatches a new one. */

export const inputPrefix = (projectKey, runName) => `inputs/${projectKey}/${runName}`;
export const calibrationPrefix = (projectKey) => `inputs/${projectKey}/Properties`;
export const outputPrefix = (projectKey, runName, step) => `outputs/${projectKey}/${runName}/${step}`;
export const manifestKey = (projectKey, runName, step) =>
  `${outputPrefix(projectKey, runName, step)}/${MANIFEST_NAME}`;
