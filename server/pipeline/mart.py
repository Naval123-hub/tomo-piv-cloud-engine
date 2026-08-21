"""Multiplicative Algebraic Reconstruction Technique (MART) for Tomo-PIV.

Standard formulation (Elsinga et al. 2006 style), run per camera per outer
iteration:
  1. Forward-project (scatter/splat, bilinear weights) the whole current volume
     estimate into that camera's image plane -> estimate_img.
  2. ratio_img = measured_img / estimate_img (clipped for stability).
  3. Back-project (gather, bilinear sample) ratio_img onto every voxel and apply
     the multiplicative correction: E(X) *= ratio(proj(X)) ** relaxation.

Starting MART from a *uniform* initial guess (every voxel equal) is what
causes the classic "ghost ray" artifact: early on, every voxel along a bright
pixel's line of sight gets boosted by roughly the same amount, since there is
nothing yet to distinguish "the true particle" from "just somewhere along the
ray" -- these get carved away only gradually, over many iterations, and
incompletely with few cameras. The standard fix in the Tomo-PIV literature
(Elsinga et al. 2006; "Fast MART", Discetti & Astarita 2012; Atkinson &
Soria's MLOS-SMART) is MLOS (Multiplicative Line-of-Sight) initialization:
seed each voxel with the *product* of all cameras' image intensity at that
voxel's projection, instead of a flat value. A voxel only starts bright if
every single camera agrees there's something there; since MART's update is
multiplicative, a voxel that starts at (near) zero can never be revived later,
so most ghost-ray content is suppressed before iteration even begins. This
lets MART reach a clean result in far fewer iterations than a uniform start
(the literature above reports ~8 iterations as a good quality/speed
trade-off with MLOS init, vs. needing many more from a flat start).

Works on numpy or CuPy arrays interchangeably via the `xp` module parameter.
"""
from __future__ import annotations

import logging

import numpy as np

from pipeline.calibration import PinholeCamera
from pipeline.voxel_grid import VoxelGrid

log = logging.getLogger(__name__)

RATIO_CLIP = (0.2, 5.0)


def project_slice(cam: PinholeCamera, Xg, Yg, z, xp):
    """Project a (ny,nx) world (X,Y) slice at fixed Z to pixel (u,v), same shape."""
    R = cam.R
    T = cam.tvec
    Xc = R[0, 0] * Xg + R[0, 1] * Yg + R[0, 2] * z + T[0]
    Yc = R[1, 0] * Xg + R[1, 1] * Yg + R[1, 2] * z + T[1]
    Zc = R[2, 0] * Xg + R[2, 1] * Yg + R[2, 2] * z + T[2]
    Zc = xp.where(xp.abs(Zc) < 1e-6, 1e-6, Zc)

    x = Xc / Zc
    y = Yc / Zc
    r2 = x * x + y * y
    k1, k2, p1, p2 = (float(c) for c in cam.dist_coeffs)
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    fx, fy = float(cam.camera_matrix[0, 0]), float(cam.camera_matrix[1, 1])
    cx, cy = float(cam.camera_matrix[0, 2]), float(cam.camera_matrix[1, 2])
    u = fx * x_d + cx
    v = fy * y_d + cy
    return u, v


def _bilinear_corners(u, v, H, W, xp):
    u0 = xp.floor(u).astype(xp.int32)
    v0 = xp.floor(v).astype(xp.int32)
    valid = (u0 >= 0) & (u0 < W - 1) & (v0 >= 0) & (v0 < H - 1)
    u0c = xp.clip(u0, 0, W - 2)
    v0c = xp.clip(v0, 0, H - 2)
    du = u - u0c.astype(xp.float32)
    dv = v - v0c.astype(xp.float32)
    wa = (1 - du) * (1 - dv)
    wb = du * (1 - dv)
    wc = (1 - du) * dv
    wd = du * dv
    return u0c, v0c, wa, wb, wc, wd, valid


def bilinear_gather(img, u, v, xp, fill_value: float = 1.0):
    """fill_value is what a voxel gets when it projects OUTSIDE this camera's
    sensor. For the MART ratio correction the neutral value is 1.0 (multiply
    by 1 = leave the voxel alone, since this camera has no opinion about it).
    For MLOS initialization it must be 0.0 instead: a voxel no camera can see
    must not be seeded bright, or it appears as a bright ghost slab at the
    edge of the volume where a camera's field of view runs out."""
    H, W = img.shape
    u0c, v0c, wa, wb, wc, wd, valid = _bilinear_corners(u, v, H, W, xp)
    Ia = img[v0c, u0c]
    Ib = img[v0c, u0c + 1]
    Ic = img[v0c + 1, u0c]
    Id = img[v0c + 1, u0c + 1]
    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return xp.where(valid, out, fill_value)


def bilinear_scatter_add(estimate_img, weight_img, u, v, vals, xp, using_gpu):
    H, W = estimate_img.shape
    u0c, v0c, wa, wb, wc, wd, valid = _bilinear_corners(u, v, H, W, xp)

    wa = xp.where(valid, wa, 0.0).reshape(-1)
    wb = xp.where(valid, wb, 0.0).reshape(-1)
    wc = xp.where(valid, wc, 0.0).reshape(-1)
    wd = xp.where(valid, wd, 0.0).reshape(-1)
    vals_flat = vals.reshape(-1)

    idx_a = (v0c * W + u0c).reshape(-1)
    idx_b = (v0c * W + (u0c + 1)).reshape(-1)
    idx_c = ((v0c + 1) * W + u0c).reshape(-1)
    idx_d = ((v0c + 1) * W + (u0c + 1)).reshape(-1)

    flat_est = estimate_img.reshape(-1)
    flat_w = weight_img.reshape(-1)

    if using_gpu:
        import cupyx

        cupyx.scatter_add(flat_est, idx_a, vals_flat * wa)
        cupyx.scatter_add(flat_est, idx_b, vals_flat * wb)
        cupyx.scatter_add(flat_est, idx_c, vals_flat * wc)
        cupyx.scatter_add(flat_est, idx_d, vals_flat * wd)
        cupyx.scatter_add(flat_w, idx_a, wa)
        cupyx.scatter_add(flat_w, idx_b, wb)
        cupyx.scatter_add(flat_w, idx_c, wc)
        cupyx.scatter_add(flat_w, idx_d, wd)
    else:
        np.add.at(flat_est, idx_a, vals_flat * wa)
        np.add.at(flat_est, idx_b, vals_flat * wb)
        np.add.at(flat_est, idx_c, vals_flat * wc)
        np.add.at(flat_est, idx_d, vals_flat * wd)
        np.add.at(flat_w, idx_a, wa)
        np.add.at(flat_w, idx_b, wb)
        np.add.at(flat_w, idx_c, wc)
        np.add.at(flat_w, idx_d, wd)


MLOS_FLOOR = 1e-4  # tiny nonzero epsilon so exact 0.0**exponent edge cases can't occur

# Tolerance, in image pixels, applied ONLY when building the MLOS initial
# guess. MLOS demands that all cameras see something at the same voxel, which
# with a sparse particle field is brittle: a residual calibration error of a
# fraction of a pixel, or a particle whose image is 1px in one camera and 2px
# in another, is enough for the strict intersection to come out completely
# empty (measured on this data: 0 four-camera coincidences at 0px tolerance,
# but a healthy depth-peaked 300 at 2px). Too much tolerance is equally bad --
# at 3px the count stops peaking with depth, meaning it is matching everything
# with everything and seeding ghosts. 2px keeps a clear depth peak.
# Applies to the seed only; the MART correction itself uses the raw images.
MLOS_TOLERANCE_PX = 2


def _dilate_2d(img, radius: int, xp):
    """Grey dilation by a square structuring element, done with shifted
    maximums so it works identically on numpy and CuPy (no SciPy/OpenCV
    dependency on the GPU path)."""
    out = img
    for axis in (0, 1):
        acc = out
        for s in range(1, radius + 1):
            acc = xp.maximum(acc, xp.roll(out, s, axis=axis))
            acc = xp.maximum(acc, xp.roll(out, -s, axis=axis))
        out = acc
    return out


def compute_mlos_initial_volume(cameras, images, grid: VoxelGrid, xp, Xg, Yg, z_centers, using_gpu=False):
    """MLOS initial guess: a voxel is seeded bright only if EVERY camera sees
    something at that voxel's projection.

    Uses the geometric mean of the per-camera intensities rather than their
    raw product. Both keep the essential "all cameras must agree" property
    (any camera seeing zero -> voxel zero), but the raw product collapses the
    scale -- four intensities of ~0.3 multiply to 0.008, so the volume starts
    ~100x too dim, its forward projection lands far below the measured image,
    and MART's correction ratio saturates at the clip limit for every
    iteration instead of converging. The geometric mean keeps the initial
    volume in the same 0..1 intensity range as the images themselves.

    A final global scale then matches the volume's forward projection to the
    measured intensity, so MART starts near the right magnitude and only has
    to refine the spatial distribution rather than climb several decades."""
    n_cams = len(cameras)
    volume = xp.empty(grid.shape, dtype=xp.float32)
    cam_items = list(cameras.items())

    # Dilated copies used for the seed only (see MLOS_TOLERANCE_PX).
    seed_images = images
    if MLOS_TOLERANCE_PX > 0:
        seed_images = {cid: _dilate_2d(img, MLOS_TOLERANCE_PX, xp) for cid, img in images.items()}

    for zi in range(grid.nz):
        slice_val = xp.ones((grid.ny, grid.nx), dtype=xp.float32)
        for cam_id, cam in cam_items:
            u, v = project_slice(cam, Xg, Yg, z_centers[zi], xp)
            slice_val *= bilinear_gather(seed_images[cam_id], u, v, xp, fill_value=0.0)
        volume[zi] = xp.clip(slice_val, 0.0, None) ** (1.0 / n_cams)

    # Global scale so the forward projection sits at the measured intensity
    # level (computed on one reference camera; MART handles the rest).
    ref_id, ref_cam = cam_items[0]
    ref_img = images[ref_id]
    H, W = ref_img.shape
    est = xp.zeros((H, W), dtype=xp.float32)
    wgt = xp.zeros((H, W), dtype=xp.float32)
    for zi in range(grid.nz):
        u, v = project_slice(ref_cam, Xg, Yg, z_centers[zi], xp)
        bilinear_scatter_add(est, wgt, u, v, volume[zi], xp, using_gpu)
    est_norm = est / xp.clip(wgt, 1.0, None)
    signal = (wgt > 0) & (ref_img > 0)
    if bool(signal.any()):
        num = float(ref_img[signal].mean())
        den = float(est_norm[signal].mean())
        if den > 1e-12:
            volume *= num / den
    return xp.clip(volume, MLOS_FLOOR, None)


def run_mart(
    cameras: dict[int, PinholeCamera],
    images: dict[int, "np.ndarray"],
    grid: VoxelGrid,
    xp,
    using_gpu: bool,
    n_iterations: int = 12,
    relaxation: float = 1.0,
    init_value: float = 0.05,
    use_mlos_init: bool = True,
):
    x_centers = grid.x_centers(xp)
    y_centers = grid.y_centers(xp)
    z_centers = grid.z_centers(xp)
    Xg, Yg = xp.meshgrid(x_centers, y_centers)  # each (ny, nx)

    if use_mlos_init:
        volume = compute_mlos_initial_volume(cameras, images, grid, xp, Xg, Yg, z_centers, using_gpu)
    else:
        volume = xp.full(grid.shape, init_value, dtype=xp.float32)

    cam_items = list(cameras.items())

    for iteration in range(n_iterations):
        residuals = []
        for cam_id, cam in cam_items:
            img = images[cam_id]
            H, W = img.shape
            estimate_img = xp.zeros((H, W), dtype=xp.float32)
            weight_img = xp.zeros((H, W), dtype=xp.float32)

            for zi in range(grid.nz):
                u, v = project_slice(cam, Xg, Yg, z_centers[zi], xp)
                bilinear_scatter_add(estimate_img, weight_img, u, v, volume[zi], xp, using_gpu)

            estimate_norm = estimate_img / xp.clip(weight_img, 1.0, None)
            ratio_img = xp.clip(img, 0, None) / xp.clip(estimate_norm, 1e-3, None)
            ratio_img = xp.clip(ratio_img, RATIO_CLIP[0], RATIO_CLIP[1])
            ratio_img = xp.where(weight_img > 0, ratio_img, 1.0)

            for zi in range(grid.nz):
                u, v = project_slice(cam, Xg, Yg, z_centers[zi], xp)
                r = bilinear_gather(ratio_img, u, v, xp)
                volume[zi] *= r ** relaxation

            # Residual is measured ONLY where the camera actually recorded
            # signal. Averaging over every pixel instead makes this metric
            # useless: ~99% of an enhanced PIV image is pure black background,
            # whose ratio sits pinned at the low clip forever, so the mean
            # barely moves between iterations and hides whether the particle
            # intensities are actually converging.
            signal_px = (weight_img > 0) & (img > 0)
            if bool(signal_px.any()):
                res = xp.abs(ratio_img[signal_px] - 1.0).mean()
            else:
                res = xp.asarray(0.0)
            residuals.append(float(res.get()) if using_gpu else float(res))

        log.info("MART iter %d/%d: mean|ratio-1| per camera = %s", iteration + 1, n_iterations,
                  ", ".join(f"{r:.4f}" for r in residuals))

    return volume
