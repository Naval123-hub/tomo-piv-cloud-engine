/**
 * The model behind the Cloud Storage image viewer.
 *
 * Kept apart from App.jsx because this is the part with real behaviour rather
 * than markup: which files are viewable, how navigation wraps, and how zoom
 * steps. Off-by-one in any of those shows the wrong picture under the right
 * caption, which is worse than showing nothing — so it is unit-tested.
 *
 * The viewer is deliberately scoped to **one folder**. A folder in this bucket
 * is already a meaningful set — `converted_jpg/` is every raw frame of a run,
 * `drop/` is every detection overlay — so "the images next to this one" needs
 * no cross-folder model to be useful.
 */

export const extOf = (name) => String(name).split(".").pop().toLowerCase();

export const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "gif", "webp", "bmp"];

export const isImage = (name) => IMAGE_EXTENSIONS.includes(extOf(name));

/** The viewable files of a folder, in the order the listing already put them
 *  (the gateway sorts by key, so `Camera1_t1` … `Camera4_t2` is natural). */
export function imagesIn(folder) {
  return (folder?.files || []).filter((file) => isImage(file.name));
}

/**
 * Moves `index` by `delta`, wrapping at both ends.
 *
 * Wrapping is right for a folder: the set is small and closed, and stopping
 * dead at the last frame reads as a broken control when the position counter
 * is right there saying 8 / 8. Returns 0 for an empty set rather than NaN.
 */
export function wrapIndex(index, delta, length) {
  if (!Number.isFinite(length) || length <= 0) return 0;
  return (((index + delta) % length) + length) % length;
}

/** Zoom stops. 1 is fit-to-pane; 8x is enough to inspect a single droplet
 *  centroid on a 2448x2048 frame. */
export const ZOOM_STEPS = [1, 1.5, 2, 3, 4, 6, 8];
export const MIN_ZOOM = ZOOM_STEPS[0];
export const MAX_ZOOM = ZOOM_STEPS[ZOOM_STEPS.length - 1];

export const clampZoom = (value) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));

/** Next stop in `direction` (+1 in, -1 out). Snaps onto the ladder from any
 *  continuous value, so the buttons still behave after a pinch/ctrl-wheel. */
export function stepZoom(current, direction) {
  const zoom = clampZoom(current);
  if (direction > 0) return ZOOM_STEPS.find((step) => step > zoom + 1e-6) ?? MAX_ZOOM;
  return [...ZOOM_STEPS].reverse().find((step) => step < zoom - 1e-6) ?? MIN_ZOOM;
}

/**
 * The URL that renders a stored object in place.
 *
 * Goes through the gateway rather than a presigned R2 link held in state: the
 * gateway re-signs on every load, so a viewer left open past the one-hour
 * expiry keeps working instead of filling with broken images.
 */
// Empty in dev, where vite proxies /api; set for a production build served
// from a different host than the gateway. Matches App.jsx.
const API = import.meta.env?.VITE_API_URL || "";

export const inlineUrl = (key) =>
  `${API}/api/storage/download?key=${encodeURIComponent(key)}&inline=1`;

/** Same object, but as a download the browser saves under its real name. */
export const downloadUrl = (key) =>
  `${API}/api/storage/download?key=${encodeURIComponent(key)}&redirect=1`;
