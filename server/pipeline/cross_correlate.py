"""Step 6 (optional): 3D cross-correlation of Volume A (t1) / Volume B (t2) to
get a directional velocity vector field, following standard Tomo-PIV practice.

Method (see references below):
  1. Split both volumes into a regular grid of overlapping 3D interrogation
     windows (same location in A and B).
  2. For each window pair, compute the 3D cross-correlation via FFT:
         R = IFFT( FFT(subA) * conj(FFT(subB)) )
     (Wiener-Khinchin theorem; this is the standard fast way to do the
     correlation sum without an O(N^2) direct loop.)
  3. The integer-voxel peak of R gives the most likely local displacement.
  4. A 3-point Gaussian estimator is fit through the peak and its immediate
     neighbours *independently along each axis* to refine the displacement to
     sub-voxel precision (Willert & Gharib 1991; standard in PIV texts).
  5. Windows with negligible intensity (no droplets) or a peak sitting on the
     window edge (displacement too large to be captured) or a poor peak-to-
     second-peak ratio (Keane & Adrian 1992 detectability criterion) are
     flagged invalid rather than silently trusted.
  6. Displacement (voxels) -> physical displacement (mm, via the voxel pitch)
     -> velocity (mm/s, dividing by the true inter-pulse time dt).

References:
  - Elsinga, Scarano, Wieneke, van Oudheusden (2006), "Tomographic particle
    image velocimetry", Experiments in Fluids 41:933-947 -- source of the
    overall MART-then-3D-cross-correlation Tomo-PIV pipeline this follows.
  - Raffel, Willert, Scarano, Kahler, Wereley, Kompenhans, "Particle Image
    Velocimetry: A Practical Guide" (3rd ed.) -- FFT correlation + Gaussian
    sub-pixel/sub-voxel peak estimator, and the peak-ratio validation metric.
  - Keane & Adrian (1992), "Theory of cross-correlation analysis of PIV
    images", Applied Scientific Research 49:191-215 -- detectability/peak-
    ratio criterion used here for validation.

IMPORTANT CAVEAT ON dt: the true laser inter-pulse time was not found in
human-readable form anywhere in this DaVis project's exported settings. The
Timing XML contains a raw value (dt=50000000, type RTE::TimeSpan/int64) whose
tick unit is undocumented; 50,000,000 as *picoseconds* = 50us is used here as
the best-supported default (consistent in order of magnitude with the 40us
camera exposure time found in the frame metadata), but this is an inference,
not a confirmed fact. Displacement (mm) and direction are unaffected by this
uncertainty -- only the absolute speed (mm/s) scales with it. Override with
--dt-us if you can confirm the real value from DaVis directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.project_layout import (
    CROSS_CORR_BURST_INDEX_X, CROSS_CORR_BURST_INDEX_Y, CROSS_CORR_DT_US, CROSS_CORR_WINDOW_XY,
    CROSS_CORR_WINDOW_XY_FINE, CROSS_CORR_WINDOW_Z, DEFAULT_RUN_NAME, HAS_T2, get_run_layout, volume_burst_indices,
)

log = logging.getLogger(__name__)

# WINDOW_XY/WINDOW_Z/ASSUMED_DT_US now live in project_layout.py as
# CROSS_CORR_WINDOW_XY / CROSS_CORR_WINDOW_Z / CROSS_CORR_DT_US -- these are
# the most project-specific values in the whole pipeline (they depend on the
# real droplet displacement and recording rate, both of which must be
# re-measured per project, not guessed) so they're set alongside the other
# per-project knobs instead of buried here. See the comment there for how
# to re-derive them, and the measured numbers behind the current defaults.
WINDOW_XY = CROSS_CORR_WINDOW_XY
WINDOW_XY_FINE = CROSS_CORR_WINDOW_XY_FINE
WINDOW_Z = CROSS_CORR_WINDOW_Z
OVERLAP_FRAC = 0.5    # 50% overlap between neighbouring windows (standard PIV default)
ENERGY_MIN_FRAC = 0.0005  # skip windows whose peak intensity is this tiny a fraction of the volume max
PEAK_RATIO_MIN = 1.2   # Keane & Adrian detectability threshold: primary/secondary peak
ASSUMED_DT_US = CROSS_CORR_DT_US

# Universal Outlier Detection (Westerweel & Scarano 2005): normalizes each
# vector's deviation from its local neighbourhood median by a robust estimate
# of local spread, giving one threshold that works regardless of the local
# flow's absolute noise level. 2.0 is their recommended default for
# reasonably clean data (their paper documents 2-4 depending on how much
# spurious data you're willing to tolerate); UOD_EPSILON_VOX is the paper's
# suggested small floor added to the normalizing spread so near-uniform
# neighbourhoods (~0 local variation) don't produce a divide-by-near-zero
# blowup that flags everything.
UOD_THRESHOLD = 2.0
UOD_EPSILON_VOX = 0.1
UOD_MIN_NEIGHBOURS = 3


@dataclass
class VectorSample:
    x_mm: float
    y_mm: float
    z_mm: float
    dx_mm: float
    dy_mm: float
    dz_mm: float
    peak_ratio: float
    valid: bool


def _gaussian_subpixel_1d(r_minus: float, r_zero: float, r_plus: float) -> float:
    """3-point Gaussian peak estimator (Willert & Gharib 1991). Returns a
    sub-voxel offset in [-1, 1] to add to the integer peak location."""
    if r_minus <= 0 or r_zero <= 0 or r_plus <= 0:
        return 0.0
    denom = np.log(r_minus) - 2.0 * np.log(r_zero) + np.log(r_plus)
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (np.log(r_minus) - np.log(r_plus)) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0
    return float(offset)


def _cross_correlate_window(subA: np.ndarray, subB: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Returns (displacement_voxels[dz,dy,dx], peak_ratio, on_edge)."""
    fa = np.fft.rfftn(subA)
    fb = np.fft.rfftn(subB)
    # conj(fa)*fb (not fa*conj(fb)) so the peak lands at +shift when B = A
    # shifted by +shift -- i.e. displacement is "where did A's content go in B".
    r = np.fft.irfftn(np.conj(fa) * fb, s=subA.shape, axes=(0, 1, 2))
    r = np.fft.fftshift(r)

    shape = np.array(r.shape)
    center = shape // 2

    peak_idx = np.array(np.unravel_index(np.argmax(r), r.shape))
    peak_val = r[tuple(peak_idx)]

    on_edge = np.any(peak_idx == 0) or np.any(peak_idx == shape - 1)

    # peak-ratio validation: mask a small neighbourhood around the primary
    # peak, the next-highest value elsewhere is the "secondary peak".
    r_masked = r.copy()
    lo = np.maximum(peak_idx - 2, 0)
    hi = np.minimum(peak_idx + 3, shape)
    r_masked[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = -np.inf
    secondary_val = r_masked.max()
    peak_ratio = float(peak_val / secondary_val) if secondary_val > 0 else float("inf")

    sub_offset = np.zeros(3)
    if not on_edge:
        for axis in range(3):
            idx_minus = peak_idx.copy(); idx_minus[axis] -= 1
            idx_plus = peak_idx.copy(); idx_plus[axis] += 1
            sub_offset[axis] = _gaussian_subpixel_1d(
                float(r[tuple(idx_minus)]), float(peak_val), float(r[tuple(idx_plus)])
            )

    displacement = (peak_idx.astype(np.float64) + sub_offset) - center
    return displacement, peak_ratio, bool(on_edge)


def _correlate_pass(
    volA: np.ndarray, volB: np.ndarray, window_xy: int, window_z: int, overlap_frac: float,
    energy_threshold: float, offset_lookup=None,
) -> tuple[list, list[int], list[int], list[int], int]:
    """Runs one correlation pass over a regular grid of interrogation windows.

    offset_lookup(center_z, center_y, center_x) -> (dz,dy,dx) voxels or None
    lets this pass pre-shift its B-window using a coarser pass's result
    before correlating (iterative/multigrid PIV window-offset technique --
    Scarano & Riethmuller 2000): the coarse pass captures the bulk
    displacement, so a fine pass using it as a predictor only has to resolve
    the small residual, which both increases spatial resolution (smaller
    windows fit) and improves SNR (a smaller true displacement is easier to
    pin down precisely). The applied integer pre-shift is added back to the
    residual sub-voxel result, so the returned displacement is always the
    TOTAL displacement, not just the residual.

    Returns (grid, z_starts, y_starts, x_starts, window_z) where grid[zk][yk][xk]
    is a dict {disp_vox, peak_ratio, on_edge, zi, yi, xi} or None (skipped --
    empty/no signal in either sub-window)."""
    nz, ny, nx = volA.shape
    window_z = min(window_z, nz)
    step_xy = max(1, int(round(window_xy * (1 - overlap_frac))))
    step_z = max(1, int(round(window_z * (1 - overlap_frac))))
    z_starts = list(range(0, max(nz - window_z, 0) + 1, step_z))
    y_starts = list(range(0, max(ny - window_xy, 0) + 1, step_xy))
    x_starts = list(range(0, max(nx - window_xy, 0) + 1, step_xy))

    grid: list = [[[None] * len(x_starts) for _ in range(len(y_starts))] for _ in range(len(z_starts))]

    for zk, zi in enumerate(z_starts):
        for yk, yi in enumerate(y_starts):
            for xk, xi in enumerate(x_starts):
                bzi, byi, bxi = zi, yi, xi
                if offset_lookup is not None:
                    cz = zi + window_z / 2.0
                    cy = yi + window_xy / 2.0
                    cx = xi + window_xy / 2.0
                    predicted = offset_lookup(cz, cy, cx)
                    if predicted is not None:
                        bzi = int(min(max(round(zi + predicted[0]), 0), nz - window_z))
                        byi = int(min(max(round(yi + predicted[1]), 0), ny - window_xy))
                        bxi = int(min(max(round(xi + predicted[2]), 0), nx - window_xy))

                subA = volA[zi:zi + window_z, yi:yi + window_xy, xi:xi + window_xy]
                subB = volB[bzi:bzi + window_z, byi:byi + window_xy, bxi:bxi + window_xy]
                if subA.max() < energy_threshold or subB.max() < energy_threshold:
                    continue

                disp_vox, peak_ratio, on_edge = _cross_correlate_window(subA, subB)
                applied_shift = np.array([bzi - zi, byi - yi, bxi - xi], dtype=np.float64)
                total_disp = disp_vox + applied_shift

                grid[zk][yk][xk] = {
                    "disp_vox": total_disp, "peak_ratio": peak_ratio, "on_edge": on_edge,
                    "zi": zi, "yi": yi, "xi": xi,
                }

    return grid, z_starts, y_starts, x_starts, window_z


def _make_offset_lookup(coarse_grid: list, coarse_window_z: int, coarse_window_xy: int):
    """Builds a nearest-neighbour predictor from a coarse pass's results, for
    use as the fine pass's offset_lookup. Only coarse windows that passed
    validation (not on-edge, adequate peak ratio) are used as predictors --
    an invalid coarse displacement is worse than no prediction at all."""
    centers = []
    disps = []
    for plane in coarse_grid:
        for row in plane:
            for cell in row:
                if cell is None or cell["on_edge"] or cell["peak_ratio"] < PEAK_RATIO_MIN:
                    continue
                centers.append((
                    cell["zi"] + coarse_window_z / 2.0,
                    cell["yi"] + coarse_window_xy / 2.0,
                    cell["xi"] + coarse_window_xy / 2.0,
                ))
                disps.append(cell["disp_vox"])

    if not centers:
        return None
    centers_arr = np.array(centers)
    disps_arr = np.array(disps)

    def lookup(cz: float, cy: float, cx: float) -> np.ndarray:
        d2 = (
            (centers_arr[:, 0] - cz) ** 2
            + (centers_arr[:, 1] - cy) ** 2
            + (centers_arr[:, 2] - cx) ** 2
        )
        return disps_arr[int(np.argmin(d2))]

    return lookup


def _universal_outlier_detection(sample_grid: list, threshold: float = UOD_THRESHOLD) -> int:
    """Westerweel & Scarano (2005) normalized median test, applied in-place
    over a 3D grid of VectorSample-or-None (a 26-neighbour box around each
    vector). Each valid vector's deviation from its neighbours' median is
    normalized by the neighbours' own median absolute deviation (a robust,
    local estimate of expected scatter) -- this is what makes a single
    threshold work regardless of how noisy or clean a given region is,
    unlike a fixed absolute cutoff. Vectors with too few valid neighbours to
    judge are left alone rather than guessed at. Mutates sample.valid in
    place; returns how many were newly flagged as outliers."""
    nz, ny, nx = len(sample_grid), len(sample_grid[0]), len(sample_grid[0][0])
    to_flag = []

    for zi in range(nz):
        for yi in range(ny):
            for xi in range(nx):
                center = sample_grid[zi][yi][xi]
                if center is None or not center.valid:
                    continue
                neighbours = []
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            zz, yy, xx = zi + dz, yi + dy, xi + dx
                            if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                                n = sample_grid[zz][yy][xx]
                                if n is not None and n.valid:
                                    neighbours.append(n)
                if len(neighbours) < UOD_MIN_NEIGHBOURS:
                    continue

                is_outlier = False
                for attr in ("dx_mm", "dy_mm", "dz_mm"):
                    center_val = getattr(center, attr)
                    nvals = np.array([getattr(n, attr) for n in neighbours])
                    med = float(np.median(nvals))
                    spread = float(np.median(np.abs(nvals - med)))
                    normalized = abs(center_val - med) / (spread + UOD_EPSILON_VOX)
                    if normalized > threshold:
                        is_outlier = True
                        break
                if is_outlier:
                    to_flag.append(center)

    for s in to_flag:
        s.valid = False
    return len(to_flag)


def compute_average_velocity(
    samples: list["VectorSample"], dt_s: float, exclude_upward: bool = True,
) -> tuple[float, float, float, int]:
    """Mean (u, v, w) in m/s over valid vectors. exclude_upward drops vectors
    with a positive Y displacement -- in this rig, seeded droplets are not
    expected to genuinely move upward, so a +Y vector is almost always a
    spurious correlation result rather than real flow, and would bias the
    average if included. Returns (u, v, w, n_samples_used)."""
    used = [s for s in samples if s.valid and (not exclude_upward or s.dy_mm <= 0)]
    if not used:
        return 0.0, 0.0, 0.0, 0
    u = float(np.mean([s.dx_mm / 1000.0 / dt_s for s in used]))
    v = float(np.mean([s.dy_mm / 1000.0 / dt_s for s in used]))
    w = float(np.mean([s.dz_mm / 1000.0 / dt_s for s in used]))
    return u, v, w, len(used)


def _resolve_volume_paths(run_name: str, layout, mart_iterations: int | None) -> tuple[Path, Path, str, str]:
    """Returns (path_A, path_B, pair_label, file_stem) for whichever pairing
    mode applies. Both branches auto-build whatever's missing (build_*_if_missing
    checks first, only reconstructs what's not already on disk) so this never
    fails on a missing file -- it just takes longer the first time.

    file_stem identifies the pair in output filenames (velocity_<file_stem>.csv/.vtk)
    so correlating a different pair never silently overwrites a previous result:

    HAS_T2=1 (double-pulse): the t1/t2 pair for the FIRST value in
    BURST_FRAME_INDEX_FOR_VOLUME (project_layout.py -- that setting can be a
    single int, a list, or a range for building multiple volumes via
    reconstruct_mart.py directly, but correlation always needs exactly one
    instant, so this always picks the first) -- named volume_b{NNN}_t1 /
    volume_b{NNN}_t2; file_stem = "b{NNN}_t1_t2".

    HAS_T2=0 (single-pulse): CROSS_CORR_BURST_INDEX_X/Y (project_layout.py)
    -- two arbitrary burst instants, named volume_b{X} / volume_b{Y};
    file_stem = "{X}to{Y}"."""
    # Local import: avoids pulling in CuPy/tifffile/etc for callers who only
    # ever hit volumes that are already built.
    from pipeline.reconstruct_mart import N_ITERATIONS, build_double_pulse_volumes_if_missing, build_volume_if_missing

    n_iter = mart_iterations if mart_iterations is not None else N_ITERATIONS

    if HAS_T2:
        burst_index = volume_burst_indices()[0]
        log.info(
            "HAS_T2=1 -- correlating the t1/t2 pulse pair for burst instant %d "
            "(first value of BURST_FRAME_INDEX_FOR_VOLUME in project_layout.py); building whichever "
            "isn't already reconstructed (iterations=%d if needed).",
            burst_index, n_iter,
        )
        path_a, path_b = build_double_pulse_volumes_if_missing(run_name, burst_index, n_iter)
        return path_a, path_b, f"t1/t2 pulse pair, burst instant {burst_index}", f"b{burst_index:03d}_t1_t2"

    log.info(
        "HAS_T2=0 -- correlating burst instants X=%d vs Y=%d (CROSS_CORR_BURST_INDEX_X/Y in "
        "project_layout.py); building whichever isn't already reconstructed (iterations=%d if needed).",
        CROSS_CORR_BURST_INDEX_X, CROSS_CORR_BURST_INDEX_Y, n_iter,
    )
    path_a = build_volume_if_missing(run_name, CROSS_CORR_BURST_INDEX_X, n_iter)
    path_b = build_volume_if_missing(run_name, CROSS_CORR_BURST_INDEX_Y, n_iter)
    pair_label = f"burst instants {CROSS_CORR_BURST_INDEX_X} vs {CROSS_CORR_BURST_INDEX_Y}"
    return path_a, path_b, pair_label, f"{CROSS_CORR_BURST_INDEX_X}to{CROSS_CORR_BURST_INDEX_Y}"


def run_cross_correlation(
    run_name: str = DEFAULT_RUN_NAME,
    window_xy: int = WINDOW_XY,
    window_z: int = WINDOW_Z,
    overlap_frac: float = OVERLAP_FRAC,
    dt_us: float = ASSUMED_DT_US,
    mart_iterations: int | None = None,
    window_xy_fine: int = WINDOW_XY_FINE,
) -> tuple[list[VectorSample], float, str]:
    layout = get_run_layout(run_name)
    layout.ensure_dirs()
    volumes_dir = layout.volumes_dir

    path_a, path_b, pair_label, file_stem = _resolve_volume_paths(run_name, layout, mart_iterations)
    log.info("Correlating %s (%s) against %s (%s)", path_a.name, pair_label, path_b.name, pair_label)

    grid_info = json.loads((volumes_dir / "volume_grid.json").read_text())
    pitch_mm = grid_info["pitch_mm"]
    x0, y0, z0 = grid_info["x0_mm"], grid_info["y0_mm"], grid_info["z0_mm"]

    log.info(
        "dt = %.1f us (see cross_correlate.py header for how this was derived -- override with --dt-us)",
        dt_us,
    )
    dt_s = dt_us * 1e-6

    volA = np.load(path_a)
    volB = np.load(path_b)
    if volA.shape != volB.shape:
        raise ValueError(f"volume shape mismatch: A={volA.shape} B={volB.shape}")

    energy_threshold = ENERGY_MIN_FRAC * max(float(volA.max()), float(volB.max()), 1e-12)

    # Pass 1 (coarse): big window, no predictor -- exists purely to capture
    # the full bulk displacement safely (this is exactly the old single-pass
    # behaviour, using the project's measured/tuned CROSS_CORR_WINDOW_XY).
    coarse_grid, coarse_z, coarse_y, coarse_x, coarse_window_z = _correlate_pass(
        volA, volB, window_xy, window_z, overlap_frac, energy_threshold, offset_lookup=None,
    )
    n_coarse = len(coarse_z) * len(coarse_y) * len(coarse_x)
    log.info(
        "Coarse pass: %d x %d x %d = %d windows (window xy=%d z=%d voxels)",
        len(coarse_x), len(coarse_y), len(coarse_z), n_coarse, window_xy, coarse_window_z,
    )

    # Pass 2 (fine): smaller window, pre-shifted per-window by the nearest
    # valid coarse displacement -- see _correlate_pass / _make_offset_lookup
    # docstrings for why (Scarano & Riethmuller 2000 window-offset method).
    offset_lookup = _make_offset_lookup(coarse_grid, coarse_window_z, window_xy)
    fine_window_xy = min(window_xy_fine, window_xy)
    fine_grid, z_starts, y_starts, x_starts, window_z = _correlate_pass(
        volA, volB, fine_window_xy, window_z, overlap_frac, energy_threshold, offset_lookup=offset_lookup,
    )
    n_fine = len(z_starts) * len(y_starts) * len(x_starts)
    log.info(
        "Fine pass: %d x %d x %d = %d windows (window xy=%d z=%d voxels; "
        "offset-corrected using the coarse pass as a predictor)",
        len(x_starts), len(y_starts), len(z_starts), n_fine, fine_window_xy, window_z,
    )

    sample_grid: list = [[[None] * len(x_starts) for _ in range(len(y_starts))] for _ in range(len(z_starts))]
    n_skipped_empty = 0
    n_skipped_edge = 0
    n_skipped_lowratio = 0

    for zk, zi in enumerate(z_starts):
        for yk, yi in enumerate(y_starts):
            for xk, xi in enumerate(x_starts):
                cell = fine_grid[zk][yk][xk]
                if cell is None:
                    n_skipped_empty += 1
                    continue

                valid = True
                if cell["on_edge"]:
                    valid = False
                    n_skipped_edge += 1
                elif cell["peak_ratio"] < PEAK_RATIO_MIN:
                    valid = False
                    n_skipped_lowratio += 1

                cx = x0 + (xi + fine_window_xy / 2.0) * pitch_mm
                cy = y0 + (yi + fine_window_xy / 2.0) * pitch_mm
                cz = z0 + (zi + window_z / 2.0) * pitch_mm
                dz_mm, dy_mm, dx_mm = cell["disp_vox"] * pitch_mm  # disp_vox is (dz,dy,dx)

                sample_grid[zk][yk][xk] = VectorSample(
                    x_mm=cx, y_mm=cy, z_mm=cz,
                    dx_mm=dx_mm, dy_mm=dy_mm, dz_mm=dz_mm,
                    peak_ratio=cell["peak_ratio"], valid=valid,
                )

    n_uod_flagged = _universal_outlier_detection(sample_grid, threshold=UOD_THRESHOLD)

    samples: list[VectorSample] = [
        s for plane in sample_grid for row in plane for s in row if s is not None
    ]

    n_valid = sum(1 for s in samples if s.valid)
    log.info(
        "Correlated %d windows (skipped %d empty). UOD flagged %d additional outliers "
        "(threshold=%.1f). Valid vectors: %d, invalid: %d (on-edge: %d, low peak-ratio: %d, UOD: %d)",
        len(samples), n_skipped_empty, n_uod_flagged, UOD_THRESHOLD,
        n_valid, len(samples) - n_valid, n_skipped_edge, n_skipped_lowratio, n_uod_flagged,
    )
    return samples, dt_s, file_stem


def _save_csv(samples: list[VectorSample], dt_s: float, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x_mm", "y_mm", "z_mm", "dx_mm", "dy_mm", "dz_mm",
            "u_mps", "v_mps", "w_mps", "speed_mps", "peak_ratio", "valid",
        ])
        for s in samples:
            u, v, w = s.dx_mm / 1000.0 / dt_s, s.dy_mm / 1000.0 / dt_s, s.dz_mm / 1000.0 / dt_s
            speed = float(np.sqrt(u * u + v * v + w * w))
            writer.writerow([s.x_mm, s.y_mm, s.z_mm, s.dx_mm, s.dy_mm, s.dz_mm,
                              u, v, w, speed, s.peak_ratio, int(s.valid)])


def _save_vtk(samples: list[VectorSample], dt_s: float, path: Path) -> None:
    """Legacy ASCII VTK PolyData: opens directly in ParaView. Use the Glyph
    filter (Arrow) on the 'velocity' vectors to visualize direction/magnitude."""
    valid = [s for s in samples if s.valid]
    n = len(valid)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Tomo-PIV velocity vectors (Volume A -> Volume B cross-correlation)\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for s in valid:
            f.write(f"{s.x_mm} {s.y_mm} {s.z_mm}\n")
        # One vertex cell per point -- without this, the file has POINTS but
        # zero cell topology, and filters that operate on cells (e.g.
        # ParaView's Threshold) silently produce empty output no matter how
        # correct the scalar range/settings are.
        f.write(f"VERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        f.write("VECTORS velocity_mps float\n")
        for s in valid:
            u, v, w = s.dx_mm / 1000.0 / dt_s, s.dy_mm / 1000.0 / dt_s, s.dz_mm / 1000.0 / dt_s
            f.write(f"{u} {v} {w}\n")
        f.write("SCALARS speed_mps float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in valid:
            u, v, w = s.dx_mm / 1000.0 / dt_s, s.dy_mm / 1000.0 / dt_s, s.dz_mm / 1000.0 / dt_s
            f.write(f"{float(np.sqrt(u*u+v*v+w*w))}\n")
        f.write("SCALARS peak_ratio float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in valid:
            f.write(f"{s.peak_ratio}\n")


def run_step6(run_name: str = DEFAULT_RUN_NAME, window_xy: int = WINDOW_XY, window_z: int = WINDOW_Z,
              overlap_frac: float = OVERLAP_FRAC, dt_us: float = ASSUMED_DT_US,
              mart_iterations: int | None = None, window_xy_fine: int = WINDOW_XY_FINE) -> None:
    layout = get_run_layout(run_name)
    samples, dt_s, file_stem = run_cross_correlation(
        run_name, window_xy, window_z, overlap_frac, dt_us, mart_iterations, window_xy_fine,
    )

    # Named after the specific pair correlated (e.g. "vector_1to2.vtk" for
    # burst instants 1 vs 2, or "vector_b004_t1_t2.vtk" for a double-pulse
    # instant) so correlating a different pair never overwrites a previous
    # result -- see _resolve_volume_paths for how file_stem is built.
    csv_path = layout.volumes_dir / f"vector_{file_stem}.csv"
    vtk_path = layout.volumes_dir / f"vector_{file_stem}.vtk"
    summary_path = layout.volumes_dir / f"vector_{file_stem}_summary.json"
    _save_csv(samples, dt_s, csv_path)
    _save_vtk(samples, dt_s, vtk_path)

    u, v, w, n_used = compute_average_velocity(samples, dt_s, exclude_upward=True)
    speed = float(np.sqrt(u * u + v * v + w * w))
    n_valid_total = sum(1 for s in samples if s.valid)
    log.info(
        "Average velocity over %d valid vectors (excluding %d upward/+Y vectors): "
        "u=%.4f v=%.4f w=%.4f m/s, speed=%.4f m/s",
        n_used, n_valid_total - n_used, u, v, w, speed,
    )
    summary_path.write_text(json.dumps({
        "n_samples": len(samples),
        "n_valid": n_valid_total,
        "n_used_for_average": n_used,
        "average_u_mps": u,
        "average_v_mps": v,
        "average_w_mps": w,
        "average_speed_mps": speed,
        "excluded_upward_plus_y": True,
    }, indent=2))
    log.info("Saved %s, %s and %s", csv_path.name, vtk_path.name, summary_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: 3D cross-correlation velocity field (Volume A -> Volume B)")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    parser.add_argument("--window-xy", type=int, default=WINDOW_XY,
                         help="coarse-pass in-plane window size in voxels (must exceed ~2x the real bulk displacement)")
    parser.add_argument("--window-xy-fine", type=int, default=WINDOW_XY_FINE,
                         help="fine-pass in-plane window size in voxels (offset-corrected using the coarse pass; "
                              "smaller = finer spatial resolution, but must still exceed the local residual "
                              "displacement between neighbouring coarse windows)")
    parser.add_argument("--window-z", type=int, default=WINDOW_Z,
                         help="depth interrogation window size in voxels (capped at the volume depth)")
    parser.add_argument("--overlap", type=float, default=OVERLAP_FRAC, help="fractional overlap between windows")
    parser.add_argument("--dt-us", type=float, default=ASSUMED_DT_US,
                         help="inter-pulse time in microseconds (see module header for how the default was derived)")
    parser.add_argument("--iterations", type=int, default=None,
                         help="MART iterations, only used if HAS_T2=0 and a volume needs building "
                              "(defaults to reconstruct_mart.N_ITERATIONS)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_step6(args.run, args.window_xy, args.window_z, args.overlap, args.dt_us, args.iterations, args.window_xy_fine)


if __name__ == "__main__":
    main()
