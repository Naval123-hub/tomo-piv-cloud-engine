"""Parse the existing DaVis Calibration.xml (PinholeOpenCV model) and provide
projection / back-projection helpers used to build the MART voxel geometry.

RESOLUTION HANDLING -- the intrinsics MUST be rescaled to whatever image size
is actually being processed, and different projects in this workspace record at
different sizes:
  * Calibration.xml is fit at <OriginalImageSize> (1216x1024 here) with
    <Transformations BinningX/Y> (2x2), i.e. the calibration reference pixel is
    2x2 native sensor pixels: ref_pitch = SensorPixelSizeMm * Binning.
  * Project_Imaging_260709_174128 records UNBINNED 2448x2048 -> intrinsics must
    be scaled UP by ~2.
  * LED_PIV/Project_Imaging_260717_210627 records BINNED 1216x1024, the same
    sampling the calibration was fit at -> scale factor 1, NO rescale.
Getting this wrong is silent and total: projecting into a 1216-wide image with
intrinsics built for 2448 puts every point at twice its true pixel position, so
no two cameras' rays ever intersect, MLOS/MART find zero agreement, and the
reconstruction comes out empty. Always pass the real image size.

Distortion coefficients and extrinsics (R, T) are in normalized/physical units
and are resolution-invariant, so they are used as-is.

`PrincipalPoint` (camera-specific) is used as (cx, cy) -- NOT `OriginPixelPosition`,
which is identical across all 4 cameras in the XML and is a dewarping/rectification
bookkeeping constant, not part of the raw pinhole model.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.project_layout import CALIBRATION_XML


@dataclass
class PinholeCamera:
    camera_id: int
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (4,) k1 k2 p1 p2
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,) mm
    R: np.ndarray  # 3x3, world -> camera
    center_world: np.ndarray  # (3,) mm, camera center in world coords

    def project(self, world_xyz: np.ndarray) -> np.ndarray:
        """world_xyz: (N,3) mm -> pixel (N,2) in full-resolution image coords."""
        pts = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        return uv.reshape(-1, 2)

    def pixel_ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Back-project pixel coords to world-space rays (origin=camera center,
        unit direction), for the auto-crop ROI estimate."""
        pts = np.stack([np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)], axis=-1)
        pts = pts.reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs)  # (N,1,2), camera-frame x,y at z=1
        norm = norm.reshape(-1, 2)
        dirs_cam = np.concatenate([norm, np.ones((norm.shape[0], 1))], axis=1)  # (N,3)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
        dirs_world = dirs_cam @ self.R  # R^T @ d, since R is world->camera (row-vector form)
        origins = np.tile(self.center_world, (dirs_world.shape[0], 1))
        return origins, dirs_world


def ray_plane_z_intersect(origin: np.ndarray, direction: np.ndarray, z_world: float) -> np.ndarray:
    """Intersect rays (origin, direction) with the world plane Z=z_world -> (N,3) points."""
    t = (z_world - origin[:, 2]) / direction[:, 2]
    return origin + t[:, None] * direction


def _f(elem: ET.Element, tag: str, attr: str) -> float:
    return float(elem.find(tag).attrib[attr])


def parse_pixels_per_mm(xml_path: Path = CALIBRATION_XML, image_width: float | None = None) -> float:
    """PixelPerMmFactor rescaled to the image resolution actually being
    processed, for sizing the MART voxel pitch."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mapper = next(root.iter("CoordinateMapper"))
    common = mapper.find("PinholeParameters").find("CommonParameters")
    px_per_mm_calibrated = _f(common, "PixelPerMmFactor", "Value")
    orig_w = _f(common, "OriginalImageSize", "Width")
    return px_per_mm_calibrated * ((image_width or orig_w) / orig_w)


def parse_calibration(
    xml_path: Path = CALIBRATION_XML,
    image_width: float | None = None,
    image_height: float | None = None,
) -> dict[int, PinholeCamera]:
    """image_width/height are the dimensions of the images this calibration will
    be used against (read them from the frame header). Defaults to the
    calibration's own <OriginalImageSize>, i.e. no rescale. See module docstring:
    passing the wrong size silently breaks every camera-ray intersection."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cameras: dict[int, PinholeCamera] = {}

    for mapper in root.iter("CoordinateMapper"):
        camera_id = int(mapper.attrib["CameraIdentifier"])
        pinhole = mapper.find("PinholeParameters")
        common = pinhole.find("CommonParameters")
        internal = pinhole.find("InternalCameraParameters")
        external = pinhole.find("ExternalCameraParameters")
        transforms = pinhole.find("Transformations")

        orig_w = _f(common, "OriginalImageSize", "Width")
        orig_h = _f(common, "OriginalImageSize", "Height")
        # Scale from the calibration's reference resolution to the resolution of
        # the images we will actually project into.
        sx = (image_width or orig_w) / orig_w
        sy = (image_height or orig_h) / orig_h

        # Despite the name, <FocalLengthPixel> is the physical focal length in mm,
        # not a pixel count (confirmed: per-camera focal length / working distance
        # (TranslationMm Tz) is constant across all 4 cameras, the signature of a
        # real focal length covering a fixed field of view from different
        # distances). SensorPixelSizeMm is the *native* (unbinned) pixel pitch, so
        # the calibration-reference pixel pitch includes the binning factor.
        sensor_pixel_size_mm = _f(internal, "SensorPixelSizeMm", "Value")
        binning_x = float(transforms.attrib.get("BinningX", 1)) if transforms is not None else 1.0
        binning_y = float(transforms.attrib.get("BinningY", 1)) if transforms is not None else 1.0
        ref_pitch_x = sensor_pixel_size_mm * binning_x
        ref_pitch_y = sensor_pixel_size_mm * binning_y
        fx = _f(internal, "FocalLengthPixel", "x") / ref_pitch_x * sx
        fy = _f(internal, "FocalLengthPixel", "y") / ref_pitch_y * sy
        cx = _f(internal, "PrincipalPoint", "x") * sx
        cy = _f(internal, "PrincipalPoint", "y") * sy

        k1 = _f(internal, "RadialDistortion", "radialDistortionCoefficient1")
        k2 = _f(internal, "RadialDistortion", "radialDistortionCoefficient2")
        p1 = _f(internal, "TangentialDistortion", "tangentialDistortionCoefficient1")
        p2 = _f(internal, "TangentialDistortion", "tangentialDistortionCoefficient2")

        tvec = np.array([
            _f(external, "TranslationMm", "Tx"),
            _f(external, "TranslationMm", "Ty"),
            _f(external, "TranslationMm", "Tz"),
        ])
        rvec = np.array([
            _f(external, "RotationAngles", "Rx"),
            _f(external, "RotationAngles", "Ry"),
            _f(external, "RotationAngles", "Rz"),
        ])

        R, _ = cv2.Rodrigues(rvec)
        center_world = -R.T @ tvec

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)

        cameras[camera_id] = PinholeCamera(
            camera_id=camera_id,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            R=R,
            center_world=center_world,
        )

    return cameras
