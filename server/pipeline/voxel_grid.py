"""Voxel grid definition and auto-crop ROI estimation for the MART reconstruction.

Tomo-PIV always restricts reconstruction to a defined volume-of-interest rather
than the full camera field of view (both for correctness and because a full-FOV
native-pixel-pitch grid is far too large to fit in GPU memory). The ROI here is
estimated from Step 3's droplet detections: for each camera, the pixel bounding
box of its detections is back-projected (via the pinhole model) onto several
world Z-planes spanning the calibrated depth range; the union of all resulting
world (X,Y) corners, across all cameras/timesteps/z-planes, gives a conservative
envelope guaranteed to contain the true (sparser, single-camera-ambiguous) 3D
particle cloud.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.calibration import PinholeCamera, ray_plane_z_intersect
from pipeline.project_layout import VOLUME_Z_MAX_MM, VOLUME_Z_MIN_MM

log = logging.getLogger(__name__)

# Kept as Z_MIN_MM/Z_MAX_MM here for backward compatibility with existing
# imports; the actual values now live in project_layout.py (single source of
# truth for "which part of physical space to reconstruct").
Z_MIN_MM = VOLUME_Z_MIN_MM
Z_MAX_MM = VOLUME_Z_MAX_MM
Z_PLANES_FOR_ROI = 5
ROI_MARGIN_FRAC = 0.08
BBOX_MARGIN_PX = 20
BBOX_PERCENTILE_LO = 5.0  # trim outlier detections (e.g. residual rig-structure
BBOX_PERCENTILE_HI = 95.0  # false positives near frame edges) before taking the bbox
MAX_VOXELS = 150_000_000


@dataclass
class VoxelGrid:
    nx: int
    ny: int
    nz: int
    x0: float
    y0: float
    z0: float
    pitch: float  # mm, isotropic voxel size

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    def x_centers(self, xp=np):
        return self.x0 + (xp.arange(self.nx, dtype=xp.float32) + 0.5) * self.pitch

    def y_centers(self, xp=np):
        return self.y0 + (xp.arange(self.ny, dtype=xp.float32) + 0.5) * self.pitch

    def z_centers(self, xp=np):
        return self.z0 + (xp.arange(self.nz, dtype=xp.float32) + 0.5) * self.pitch

    def voxel_volume_mm3(self) -> float:
        return self.pitch ** 3


def _load_detection_pixels(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        return np.zeros((0, 2))
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            rows.append((float(row[0]), float(row[1])))
    return np.array(rows) if rows else np.zeros((0, 2))


def estimate_world_xy_bounds(
    cameras: dict[int, PinholeCamera],
    detection_csvs: list[Path],
    z_min: float = Z_MIN_MM,
    z_max: float = Z_MAX_MM,
) -> tuple[float, float, float, float]:
    """Returns (x_min, x_max, y_min, y_max) in mm.

    A real particle must be visible to every camera to be reconstructed at all,
    so the correct measurement volume is the INTERSECTION of each camera's
    back-projected footprint, not their union -- a union stays needlessly huge
    since 4 cameras at very different angles each see a differently-shaped
    (and mostly non-overlapping-at-the-edges) swath of world space.
    """
    z_planes = np.linspace(z_min, z_max, Z_PLANES_FOR_ROI)

    per_camera_bounds = []
    for camera_id, cam in cameras.items():
        pixels_for_cam = []
        for csv_path in detection_csvs:
            if csv_path.name.startswith(f"Camera{camera_id}_"):
                pixels_for_cam.append(_load_detection_pixels(csv_path))
        if not pixels_for_cam:
            continue
        pts = np.concatenate(pixels_for_cam, axis=0)
        if pts.shape[0] == 0:
            continue

        # Percentile rather than strict min/max: robust against a residual of
        # spurious detections (rig structure near frame edges the Step 3 detector
        # doesn't fully suppress) that would otherwise blow up the bbox.
        row_min, col_min = np.percentile(pts, BBOX_PERCENTILE_LO, axis=0) - BBOX_MARGIN_PX
        row_max, col_max = np.percentile(pts, BBOX_PERCENTILE_HI, axis=0) + BBOX_MARGIN_PX
        corner_rows = np.array([row_min, row_min, row_max, row_max])
        corner_cols = np.array([col_min, col_max, col_min, col_max])

        cam_x, cam_y = [], []
        for z in z_planes:
            origins, dirs = cam.pixel_ray(corner_cols, corner_rows)
            world_pts = ray_plane_z_intersect(origins, dirs, float(z))
            cam_x.extend(world_pts[:, 0].tolist())
            cam_y.extend(world_pts[:, 1].tolist())
        per_camera_bounds.append((min(cam_x), max(cam_x), min(cam_y), max(cam_y)))

    if not per_camera_bounds:
        raise RuntimeError("no droplet detections found in any camera; cannot estimate ROI. Run Step 3 first.")

    x_min = max(b[0] for b in per_camera_bounds)
    x_max = min(b[1] for b in per_camera_bounds)
    y_min = max(b[2] for b in per_camera_bounds)
    y_max = min(b[3] for b in per_camera_bounds)
    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError(
            f"camera view intersection is empty/degenerate (x:[{x_min:.1f},{x_max:.1f}] "
            f"y:[{y_min:.1f},{y_max:.1f}]) -- check calibration/detections"
        )
    x_pad = (x_max - x_min) * ROI_MARGIN_FRAC
    y_pad = (y_max - y_min) * ROI_MARGIN_FRAC
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def build_voxel_grid(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    ideal_pitch_mm: float,
    max_voxels: int = MAX_VOXELS,
) -> VoxelGrid:
    x_extent = x_bounds[1] - x_bounds[0]
    y_extent = y_bounds[1] - y_bounds[0]
    z_extent = z_bounds[1] - z_bounds[0]

    pitch = ideal_pitch_mm
    while (x_extent / pitch) * (y_extent / pitch) * (z_extent / pitch) > max_voxels:
        pitch *= 1.05

    nx = max(1, int(np.ceil(x_extent / pitch)))
    ny = max(1, int(np.ceil(y_extent / pitch)))
    nz = max(1, int(np.ceil(z_extent / pitch)))

    log.info(
        "Voxel grid: %dx%dx%d = %.1fM voxels, pitch=%.4fmm (ideal=%.4fmm), extent=%.1fx%.1fx%.1fmm",
        nx, ny, nz, nx * ny * nz / 1e6, pitch, ideal_pitch_mm, x_extent, y_extent, z_extent,
    )

    return VoxelGrid(nx=nx, ny=ny, nz=nz, x0=x_bounds[0], y0=y_bounds[0], z0=z_bounds[0], pitch=pitch)
