/**
 * Translation between the client's friendly parameter names and the pipeline's
 * module-level constants (pipeline/project_layout.py).
 *
 * This is the single place that mapping lives. Bounds mirror
 * project_layout.TUNABLES so an out-of-range value is rejected here, in
 * milliseconds, rather than after a GPU container cold start.
 */
import { BadRequest } from "./config.js";

const PARAM_MAP = {
  has_t2: { target: "HAS_T2", type: "bool" },
  noise_floor_pct: { target: "NOISE_FLOOR_PERCENTILE", type: "float", min: 80, max: 99.99 },
  window_xy: { target: "CROSS_CORR_WINDOW_XY", type: "int", min: 8, max: 512 },
  window_xy_fine: { target: "CROSS_CORR_WINDOW_XY_FINE", type: "int", min: 8, max: 512 },
  window_z: { target: "CROSS_CORR_WINDOW_Z", type: "int", min: 4, max: 256 },
  dt_us: { target: "CROSS_CORR_DT_US", type: "float", min: 0.01, max: 100000 },
  max_particle_px: { target: "MAX_PARTICLE_PX", type: "int", min: 0, max: 64 },
  crop_top_px: { target: "CROP_TOP_PX", type: "int", min: 0, max: 1024 },
  crop_bottom_px: { target: "CROP_BOTTOM_PX", type: "int", min: 0, max: 1024 },
  crop_left_px: { target: "CROP_LEFT_PX", type: "int", min: 0, max: 1216 },
  crop_right_px: { target: "CROP_RIGHT_PX", type: "int", min: 0, max: 1216 },
  burst_indices: { target: "BURST_FRAME_INDICES", type: "raw" },
  burst_index_for_volume: { target: "BURST_FRAME_INDEX_FOR_VOLUME", type: "raw" },
  // Runtime arguments to the step functions -- passed through unchanged.
  mart_iterations: { target: "mart_iterations", type: "int", min: 1, max: 200 },
  overlap_frac: { target: "overlap_frac", type: "float", min: 0, max: 0.85 },
  include_step6: { target: "include_step6", type: "bool" },
};

/**
 * Reverse index, so the pipeline's own constant names work on the wire too.
 *
 * The console sends the friendly `crop_left_px`, but `CROP_LEFT_PX` is what
 * project_layout.py actually calls it and therefore what anyone reading the
 * pipeline reaches for first. Accepting both costs nothing and both go through
 * the identical bounds check below -- an alias must never be a way around
 * validation.
 */
const BY_TARGET = new Map(
  Object.entries(PARAM_MAP).map(([name, spec]) => [spec.target, { name, spec }])
);

function coerce(name, spec, value) {
  if (spec.type === "raw") return value;
  if (spec.type === "bool") {
    if (typeof value === "string") return ["1", "true", "yes", "on"].includes(value.toLowerCase());
    return Boolean(value);
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    throw new BadRequest(`${name} must be a number, got ${JSON.stringify(value)}`);
  }
  if (spec.min !== undefined && num < spec.min) {
    throw new BadRequest(`${name}=${num} is below the minimum ${spec.min}`);
  }
  if (spec.max !== undefined && num > spec.max) {
    throw new BadRequest(`${name}=${num} is above the maximum ${spec.max}`);
  }
  return spec.type === "int" ? Math.round(num) : num;
}

/** Builds the `params` object the FastAPI worker expects. */
export function buildParams(body = {}) {
  const params = {};

  for (const [name, spec] of Object.entries(PARAM_MAP)) {
    if (body[name] === undefined || body[name] === null) continue;
    params[spec.target] = coerce(name, spec, body[name]);
  }

  // Uppercase aliases, applied second so an explicit friendly name always wins
  // if a caller somehow sends both spellings.
  for (const [target, { spec }] of BY_TARGET) {
    if (params[target] !== undefined) continue;
    if (body[target] === undefined || body[target] === null) continue;
    params[target] = coerce(target, spec, body[target]);
  }

  // z_range is the one non-1:1 mapping: the client sends a [min, max] pair,
  // the pipeline holds two separate constants.
  if (body.z_range !== undefined) {
    const range = Array.isArray(body.z_range) ? body.z_range : [body.z_range?.min, body.z_range?.max];
    const [zMin, zMax] = range.map(Number);
    if (!Number.isFinite(zMin) || !Number.isFinite(zMax)) {
      throw new BadRequest("z_range must be [min_mm, max_mm]");
    }
    if (zMin >= zMax) {
      throw new BadRequest(`z_range is empty: min (${zMin}) must be below max (${zMax})`);
    }
    params.VOLUME_Z_MIN_MM = zMin;
    params.VOLUME_Z_MAX_MM = zMax;
  }

  // The second correlation pass is a window-OFFSET refinement: it subdivides
  // the coarse window, so it must be smaller. project_layout.py ships a
  // coarse/fine pair (32 / 128) inherited from two different project presets,
  // which is backwards -- so when the client sends only window_xy we derive a
  // sane fine window rather than letting that stale default apply. An explicit
  // window_xy_fine always wins.
  if (params.CROSS_CORR_WINDOW_XY !== undefined && params.CROSS_CORR_WINDOW_XY_FINE === undefined) {
    params.CROSS_CORR_WINDOW_XY_FINE = Math.max(8, Math.round(params.CROSS_CORR_WINDOW_XY / 2));
  }
  if (
    params.CROSS_CORR_WINDOW_XY_FINE !== undefined &&
    params.CROSS_CORR_WINDOW_XY !== undefined &&
    params.CROSS_CORR_WINDOW_XY_FINE > params.CROSS_CORR_WINDOW_XY
  ) {
    throw new BadRequest(
      `window_xy_fine (${params.CROSS_CORR_WINDOW_XY_FINE}) must not exceed window_xy ` +
        `(${params.CROSS_CORR_WINDOW_XY}) -- the fine pass refines the coarse one`
    );
  }

  return params;
}
