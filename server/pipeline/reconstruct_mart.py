"""Steps 4-5: build the MART voxel geometry and reconstruct Volume A (t1) and
Volume B (t2) from the 4-camera filtered images, using the existing DaVis
pinhole calibration. Saves .npy, a .tif stack, and a max-intensity-projection
JPEG per volume."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from pipeline import ims_io
from pipeline.calibration import parse_calibration, parse_pixels_per_mm
from pipeline.gpu_utils import get_xp, to_numpy
from pipeline.mart import run_mart
from pipeline.project_layout import (
    CALIBRATION_XML, DEFAULT_RUN_NAME, HAS_T2, burst_indices, get_run_layout, image_stem, volume_burst_indices,
)
from pipeline.voxel_grid import Z_MAX_MM, Z_MIN_MM, build_voxel_grid, estimate_world_xy_bounds

log = logging.getLogger(__name__)

N_ITERATIONS = 15
RELAXATION = 1.0


def _load_timestep_images(layout, timestep: int, width: int, height: int, xp, burst_index: int) -> dict[int, "np.ndarray"]:
    images = {}
    for camera in (1, 2, 3, 4):
        stem = image_stem(camera, timestep, burst_index)
        ims_path = layout.filtered_dir / f"{stem}.ims"
        if not ims_path.exists():
            raise FileNotFoundError(
                f"missing Step 2 output {ims_path}; run enhance.py first (make sure burst_index={burst_index} "
                f"is included in BURST_FRAME_INDICES in project_layout.py before running Steps 1-3)"
            )
        decoder_id = ims_io.read_decoder_id(layout.filtered_dir / f"{stem}-decoder.xml")
        img12 = ims_io.load_ims_image(ims_path, width, height, decoder_id=decoder_id, burst_index=0)
        img_norm = (img12.astype(np.float32) / ims_io.MAX_12BIT)
        images[camera] = xp.asarray(img_norm)
    return images


def _save_volume(volume, grid, out_stem, volumes_dir) -> None:
    vol_np = to_numpy(volume)
    np.save(volumes_dir / f"{out_stem}.npy", vol_np)
    tifffile.imwrite(volumes_dir / f"{out_stem}.tif", vol_np, dtype=np.float32)

    mip = vol_np.max(axis=0)
    lo, hi = np.percentile(mip, [1, 99.5])
    hi = max(hi, lo + 1e-6)
    mip8 = np.clip((mip - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(mip8).save(volumes_dir / f"{out_stem}_MIP.jpg", quality=95)

    log.info("Saved %s: shape=%s min=%.4f max=%.4f mean=%.4f", out_stem, vol_np.shape,
              float(vol_np.min()), float(vol_np.max()), float(vol_np.mean()))


def _prepare_calibration_and_grid(layout):
    """Shared by both entry points below: parses calibration at the actual
    image resolution and builds the voxel grid from ALL detections currently
    sitting in drop_dir (every burst instant Steps 1-3 have processed so far,
    not just the one(s) about to be reconstructed). This is deliberate, not
    an oversight: it's what keeps every volume built for this run -- no
    matter which burst instant, and even across separate script invocations
    -- on the exact same voxel grid (same origin, same pitch), which cross-
    correlation requires to be meaningful. Re-running Steps 1-3 for more
    instants after some volumes already exist will shift this grid on the
    next build, so build all the volumes you want to compare in one sitting.
    """
    header_path = layout.run_dir / "Frame0-0.ims"
    width, height = ims_io.read_frame_header(header_path)

    cameras = parse_calibration(CALIBRATION_XML, image_width=width, image_height=height)
    ideal_pitch = 1.0 / parse_pixels_per_mm(CALIBRATION_XML, image_width=width)
    log.info("Parsed calibration for %d cameras at image size %dx%d. Ideal voxel pitch ~ %.4f mm",
              len(cameras), width, height, ideal_pitch)

    detection_csvs = sorted(layout.drop_dir.glob("*_detections.csv"))
    x_min, x_max, y_min, y_max = estimate_world_xy_bounds(cameras, detection_csvs)
    grid = build_voxel_grid(
        x_bounds=(x_min, x_max), y_bounds=(y_min, y_max), z_bounds=(Z_MIN_MM, Z_MAX_MM),
        ideal_pitch_mm=ideal_pitch,
    )

    grid_info = {
        "nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
        "x0_mm": grid.x0, "y0_mm": grid.y0, "z0_mm": grid.z0,
        "pitch_mm": grid.pitch, "units": "mm",
        "note": "volume array axis order is (z, y, x); world coord of voxel [k,j,i] "
                "is (x0+(i+0.5)*pitch, y0+(j+0.5)*pitch, z0+(k+0.5)*pitch)",
    }
    with open(layout.volumes_dir / "volume_grid.json", "w") as f:
        json.dump(grid_info, f, indent=2)

    return cameras, grid, width, height


def volume_stem(burst_index: int, timestep: int | None = None) -> str:
    """Volume filename stem, named by which burst instant it came from and
    (for double-pulse) which pulse:
      double-pulse instant 5: volume_stem(5, 1) -> "volume_b005_t1"
                               volume_stem(5, 2) -> "volume_b005_t2"
      single-pulse instant 5: volume_stem(5)    -> "volume_b005"
    Naming by burst instant (not a fixed "A"/"B") matters because different
    instants must NOT share a filename -- reconstructing instant 5 and later
    instant 10 has to produce two distinct volumes on disk, not silently
    overwrite each other."""
    base = f"volume_b{burst_index:03d}"
    return base if timestep is None else f"{base}_t{timestep}"


def _check_burst_processed(burst_index: int) -> None:
    if burst_index not in burst_indices():
        raise ValueError(
            f"burst_index={burst_index} was not processed by Steps 1-3 (BURST_FRAME_INDICES="
            f"{burst_indices()} in project_layout.py) -- add it there and re-run enhance.py first."
        )


def _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu,
                       timestep: int, burst_index: int, n_iterations: int, stem: str) -> None:
    log.info("Reconstructing %s (burst instant %d, t%d)...", stem, burst_index, timestep)
    images = _load_timestep_images(layout, timestep, width, height, xp, burst_index)
    t0 = time.time()
    volume = run_mart(cameras, images, grid, xp, using_gpu, n_iterations=n_iterations, relaxation=RELAXATION)
    log.info("%s MART reconstruction took %.1fs", stem, time.time() - t0)
    _save_volume(volume, grid, stem, layout.volumes_dir)


def build_volume_if_missing(run_name: str, burst_index: int, n_iterations: int = N_ITERATIONS) -> Path:
    """Single-pulse: builds+saves the one-exposure volume for burst_index if
    it doesn't already exist; otherwise does nothing and returns the
    existing path. Used by cross_correlate.py for HAS_T2=0 runs (arbitrary
    X/Y burst-instant correlation instead of a real t1/t2 pair)."""
    layout = get_run_layout(run_name)
    layout.ensure_dirs()
    stem = volume_stem(burst_index)
    npy_path = layout.volumes_dir / f"{stem}.npy"
    if npy_path.exists():
        log.info("%s already exists, skipping reconstruction", npy_path.name)
        return npy_path

    _check_burst_processed(burst_index)
    cameras, grid, width, height = _prepare_calibration_and_grid(layout)
    xp, using_gpu = get_xp(prefer_gpu=True)
    log.info("Compute backend: %s", "GPU (CuPy)" if using_gpu else "CPU (numpy)")
    _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu, 1, burst_index, n_iterations, stem)
    return npy_path


def build_double_pulse_volumes_if_missing(
    run_name: str, burst_index: int, n_iterations: int = N_ITERATIONS
) -> tuple[Path, Path]:
    """Double-pulse: builds+saves the t1 and t2 volumes for burst_index,
    each skipped individually if already present. Mirrors
    build_volume_if_missing above but for a real t1/t2 pulse pair."""
    layout = get_run_layout(run_name)
    layout.ensure_dirs()
    stem_t1, stem_t2 = volume_stem(burst_index, 1), volume_stem(burst_index, 2)
    path_t1 = layout.volumes_dir / f"{stem_t1}.npy"
    path_t2 = layout.volumes_dir / f"{stem_t2}.npy"
    if path_t1.exists() and path_t2.exists():
        log.info("%s and %s already exist, skipping reconstruction", path_t1.name, path_t2.name)
        return path_t1, path_t2

    _check_burst_processed(burst_index)
    cameras, grid, width, height = _prepare_calibration_and_grid(layout)
    xp, using_gpu = get_xp(prefer_gpu=True)
    log.info("Compute backend: %s", "GPU (CuPy)" if using_gpu else "CPU (numpy)")

    if path_t1.exists():
        log.info("%s already exists, skipping", path_t1.name)
    else:
        _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu, 1, burst_index, n_iterations, stem_t1)
    if path_t2.exists():
        log.info("%s already exists, skipping", path_t2.name)
    else:
        _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu, 2, burst_index, n_iterations, stem_t2)
    return path_t1, path_t2


def run_steps_4_5(run_name: str = DEFAULT_RUN_NAME, n_iterations: int = N_ITERATIONS,
                   burst_index=None) -> None:
    """Direct/explicit CLI entry point -- unlike the build_*_if_missing()
    helpers above (used for on-demand auto-build from cross_correlate.py),
    this ALWAYS reconstructs, even if a volume for an instant already
    exists, since an explicit `python -m pipeline.reconstruct_mart` call is
    a deliberate request to (re)build, e.g. after changing enhance.py.

    burst_index: single int, list, or range of instants to build volume(s)
    for; None (default) uses BURST_FRAME_INDEX_FOR_VOLUME from
    project_layout.py (itself an int/list/range). All instants requested in
    one call share the same voxel grid (see _prepare_calibration_and_grid),
    which is what makes any two of the resulting volumes comparable."""
    if burst_index is None:
        targets = volume_burst_indices()
    elif isinstance(burst_index, int):
        targets = [burst_index]
    else:
        targets = list(burst_index)

    for idx in targets:
        _check_burst_processed(idx)

    layout = get_run_layout(run_name)
    layout.ensure_dirs()
    log.info("Reconstructing volume(s) for burst instant(s) %s (HAS_T2=%d)", targets, HAS_T2)

    cameras, grid, width, height = _prepare_calibration_and_grid(layout)
    xp, using_gpu = get_xp(prefer_gpu=True)
    log.info("Compute backend: %s", "GPU (CuPy)" if using_gpu else "CPU (numpy)")

    for idx in targets:
        if HAS_T2:
            _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu,
                               1, idx, n_iterations, volume_stem(idx, 1))
            _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu,
                               2, idx, n_iterations, volume_stem(idx, 2))
        else:
            _build_one_volume(layout, cameras, grid, width, height, xp, using_gpu,
                               1, idx, n_iterations, volume_stem(idx))

    if not HAS_T2:
        log.info("HAS_T2=0 (single-pulse) -- only one exposure per instant exists; for cross-correlation "
                  "use cross_correlate.py directly with CROSS_CORR_BURST_INDEX_X/Y in project_layout.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Steps 4-5: MART tomographic reconstruction")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--burst-index", type=int, nargs="+", default=None,
                         help="which processed burst instant(s) to reconstruct, space-separated for multiple "
                              "(each must be in BURST_FRAME_INDICES); defaults to BURST_FRAME_INDEX_FOR_VOLUME "
                              "in project_layout.py")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_steps_4_5(args.run, args.iterations, args.burst_index)


if __name__ == "__main__":
    main()
