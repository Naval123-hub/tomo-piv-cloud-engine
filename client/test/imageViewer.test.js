/**
 * Tests for the Cloud Storage image viewer's model.
 *
 *   npm test        (client/)
 *
 * Folder shapes here are the real ones the gateway's /api/storage/list returns
 * for this bucket.
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  MAX_ZOOM,
  MIN_ZOOM,
  clampZoom,
  downloadUrl,
  imagesIn,
  inlineUrl,
  isImage,
  stepZoom,
  wrapIndex,
} from "../src/imageViewer.js";

const file = (name, size = 1_000_000) => ({
  key: `outputs/default/RUN_A/full/drop/${name}`,
  name,
  size,
  last_modified: "2026-08-21T11:14:00.000Z",
});

const FOLDER = {
  prefix: "outputs/default/RUN_A/full/drop",
  files: [
    file("Camera1_t1.jpg"),
    file("Camera1_t1_detections.csv", 4_000),
    file("Camera1_t2.jpg"),
    file("Camera2_t1.jpg"),
    file("manifest.json", 512),
  ],
};

describe("isImage", () => {
  it("accepts what the pipeline actually writes", () => {
    for (const name of ["Camera1_t1.jpg", "volume_b000_t1_MIP.jpg", "shot.PNG", "x.jpeg"]) {
      assert.equal(isImage(name), true, name);
    }
  });

  it("rejects everything else, so Open never appears on a volume", () => {
    for (const name of ["vector.vtk", "volume.tif", "d.csv", "manifest.json", "Frame0-0.ims", "noext"]) {
      assert.equal(isImage(name), false, name);
    }
  });
});

describe("imagesIn", () => {
  it("keeps only images and preserves listing order", () => {
    assert.deepEqual(imagesIn(FOLDER).map((f) => f.name), [
      "Camera1_t1.jpg",
      "Camera1_t2.jpg",
      "Camera2_t1.jpg",
    ]);
  });

  it("is empty rather than undefined for a folder with no images", () => {
    assert.deepEqual(imagesIn({ files: [file("v.vtk")] }), []);
    assert.deepEqual(imagesIn({}), []);
    assert.deepEqual(imagesIn(undefined), []);
  });
});

describe("wrapIndex", () => {
  it("steps forward and back", () => {
    assert.equal(wrapIndex(0, 1, 3), 1);
    assert.equal(wrapIndex(2, -1, 3), 1);
  });

  it("wraps at both ends", () => {
    assert.equal(wrapIndex(2, 1, 3), 0, "past the last image returns to the first");
    assert.equal(wrapIndex(0, -1, 3), 2, "before the first returns to the last");
  });

  it("survives a single-image folder", () => {
    assert.equal(wrapIndex(0, 1, 1), 0);
    assert.equal(wrapIndex(0, -1, 1), 0);
  });

  it("returns 0 for an empty set instead of NaN", () => {
    assert.equal(wrapIndex(0, 1, 0), 0);
    assert.equal(wrapIndex(3, -1, undefined), 0);
  });
});

describe("zoom", () => {
  it("clamps to the usable range", () => {
    assert.equal(clampZoom(0.2), MIN_ZOOM);
    assert.equal(clampZoom(999), MAX_ZOOM);
    assert.equal(clampZoom(2), 2);
  });

  it("steps along the ladder and stops at the ends", () => {
    assert.equal(stepZoom(1, 1), 1.5);
    assert.equal(stepZoom(1.5, 1), 2);
    assert.equal(stepZoom(MAX_ZOOM, 1), MAX_ZOOM);
    assert.equal(stepZoom(2, -1), 1.5);
    assert.equal(stepZoom(MIN_ZOOM, -1), MIN_ZOOM);
  });

  it("snaps back onto the ladder from a continuous ctrl-wheel value", () => {
    // 2.3x came from pinching; the buttons must still land on real stops
    // rather than drifting off by whatever the gesture left behind.
    assert.equal(stepZoom(2.3, 1), 3);
    assert.equal(stepZoom(2.3, -1), 2);
  });
});

describe("urls", () => {
  const key = "outputs/default/RUN_A/full/drop/Camera1_t1.jpg";

  it("routes through the gateway, not a stored presigned link", () => {
    // Re-signing per load is what keeps a viewer left open past the one-hour
    // expiry working instead of filling with broken images.
    assert.equal(inlineUrl(key).startsWith("/api/storage/download?key="), true);
    assert.match(inlineUrl(key), /&inline=1$/);
    assert.match(downloadUrl(key), /&redirect=1$/);
  });

  it("encodes keys with characters that would break the query string", () => {
    const awkward = "inputs/default/RUN A/Frame 0+1&x.jpg";
    assert.equal(inlineUrl(awkward).includes("RUN A"), false);
    assert.equal(inlineUrl(awkward).includes("&x.jpg"), false);
    assert.equal(
      decodeURIComponent(new URL(inlineUrl(awkward), "http://x").searchParams.get("key")),
      awkward
    );
  });
});
