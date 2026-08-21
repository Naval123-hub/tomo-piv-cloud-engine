"""Step 2: enhance raw frames so faint/small seeding droplets become clearly visible.

Pipeline:
  1. Per-column baseline correction. The sensor has a strong (~300-count)
     column-to-column fixed-pattern offset (adjacent columns read very
     differently even in pure dark background -- confirmed empirically), which
     a small-kernel top-hat would otherwise preserve and misread as texture/signal
     everywhere. Each column's robust median is subtracted and replaced with the
     image-wide median, flattening this offset while leaving sparse droplet
     pixels (a small minority of any column) essentially unaffected.
  2. Static-structure column mask. The test rig has near-perfectly-vertical metal
     rails in frame that catch direct/specular light and saturate. Unlike
     droplets -- small, sparse, spatially random -- a rail lights up the same
     sensor column continuously over nearly the full image height, so columns
     bright over an anomalously large fraction of rows are masked out as rig
     structure, not seeding.
  3. White top-hat: removes remaining large-scale background glow, keeps small
     bright blobs.
  4. Percentile-based noise floor: with a sparse particle field the vast
     majority of pixels are background, so a fixed high percentile (tuned to
     roughly match expected droplet coverage) is clipped to black *before*
     stretching contrast -- this is what keeps the background black instead of
     a stretched noise fog.
  5. Gamma boost of the remaining particle signal (brings up dim/small
     droplets) -> scale to full 12-bit range.
Output is saved both as 8-bit JPEG and as a repacked 12-bit .ims (+ decoder.xml).
"""
from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np
from PIL import Image

from pipeline import ims_io
from pipeline.project_layout import (
    CROP_BOTTOM_PX, CROP_LEFT_PX, CROP_RIGHT_PX, CROP_TOP_PX, HAS_T1, HAS_T2, DEFAULT_RUN_NAME, MAX_PARTICLE_PX,
    NOISE_FLOOR_PERCENTILE, burst_indices, get_run_layout, image_stem,
)

log = logging.getLogger(__name__)

TOPHAT_KERNEL_PX = 35
GAMMA = 0.55
# NOISE_FLOOR_PERCENTILE now lives in project_layout.py -- it's the single
# most sensitive, most project-specific knob in this pipeline (whether the 3D
# reconstruction gets any signal at all depends on it), so it's set alongside
# the other per-project values instead of buried here. See the comment there
# for the measured 4-camera-coincidence numbers behind the current default.
SIGNAL_REF_PERCENTILE = 99.9  # this percentile of the remaining (post-clip) signal maps to full scale

STRUCTURAL_BRIGHT_THRESHOLD = 2000  # raw 12-bit counts
STRUCTURAL_PERSISTENCE_FRAC = 0.08  # column flagged if bright over >8% of image height
STRUCTURAL_MASK_MARGIN_PX = 25  # widen flagged columns to cover the dimmer blooming shoulder around a rail


def correct_column_fpn(img12: np.ndarray) -> np.ndarray:
    """Flattens the sensor's column-to-column fixed-pattern offset. Uses each
    column's median as the offset estimate: droplets are a small minority of
    pixels in any given column, so the median robustly reflects that column's
    true background level regardless of how many droplets pass through it."""
    col_baseline = np.median(img12, axis=0)
    global_baseline = np.median(col_baseline)
    corrected = img12.astype(np.float32) - col_baseline[None, :] + global_baseline
    return np.clip(corrected, 0, None)


def structural_column_mask(img12: np.ndarray) -> np.ndarray:
    """Flags columns that are bright over an anomalously large fraction of the
    image height -- real droplets are small and spatially sparse, so no
    droplet-only column should be lit more than a few percent of the time.
    Persistently-lit columns are rig structure (rails) catching direct light."""
    bright = img12 > STRUCTURAL_BRIGHT_THRESHOLD
    col_frac = bright.mean(axis=0)
    flagged = col_frac > STRUCTURAL_PERSISTENCE_FRAC
    if STRUCTURAL_MASK_MARGIN_PX > 0:
        kernel = np.ones(2 * STRUCTURAL_MASK_MARGIN_PX + 1, dtype=np.uint8)
        flagged = np.convolve(flagged.astype(np.uint8), kernel, mode="same") > 0
    return flagged


def apply_crop_mask(img: np.ndarray) -> np.ndarray:
    """Zeros out the configured edge margins (CROP_*_PX in project_layout.py)
    so that region can never register as a droplet -- use this to exclude
    rig rails, nozzle hardware, etc. near the frame edges.

    MUST be applied at the very END of enhancement, never before the
    morphological top-hat: zeroing a region creates a sharp intensity step at
    the crop boundary, and a top-hat run afterwards responds strongly to that
    step, painting a bright artifact ring along the crop edge (worst at the
    corners). Those artifacts are large blobs, and large 2D blobs become
    full-depth ghost rays in the 3D reconstruction -- see remove_oversized_blobs
    below for why blob size matters so much."""
    if not (CROP_TOP_PX or CROP_BOTTOM_PX or CROP_LEFT_PX or CROP_RIGHT_PX):
        return img
    img = img.copy()
    h, w = img.shape
    if CROP_TOP_PX > 0:
        img[:CROP_TOP_PX, :] = 0
    if CROP_BOTTOM_PX > 0:
        img[h - CROP_BOTTOM_PX:, :] = 0
    if CROP_LEFT_PX > 0:
        img[:, :CROP_LEFT_PX] = 0
    if CROP_RIGHT_PX > 0:
        img[:, w - CROP_RIGHT_PX:] = 0
    return img


def remove_oversized_blobs(img12: np.ndarray) -> np.ndarray:
    """Zeroes connected bright regions wider than MAX_PARTICLE_PX.

    Why this matters far more than it looks: with a small camera aperture, a
    reconstructed particle is stretched along the depth axis by roughly
    (in-plane size / tan(half-aperture)). Measured on this data, the z-extent
    of a reconstructed blob tracks its 2D image size with correlation 0.93 --
    so a 2px droplet stays compact (~0.2mm deep) while a 14px blob smears
    across the ENTIRE volume depth as a "ray". Real tracer droplets here are
    1-3px; anything much larger is merged/saturated structure or rig hardware,
    not a trackable particle, so removing it costs no real signal and removes
    the dominant source of ghost rays.
    Set MAX_PARTICLE_PX = 0 in project_layout.py to disable."""
    if MAX_PARTICLE_PX <= 0:
        return img12
    mask = img12 > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return img12
    # stats columns: x, y, width, height, area
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    too_big = extent > MAX_PARTICLE_PX
    too_big[0] = False  # label 0 is the background
    if not too_big.any():
        return img12
    out = img12.copy()
    out[too_big[labels]] = 0
    return out


def enhance_image(img12: np.ndarray) -> np.ndarray:
    """Return an enhanced image, still in the 0-4095 (12-bit) range."""
    struct_cols = structural_column_mask(img12)
    corrected = correct_column_fpn(img12)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_PX, TOPHAT_KERNEL_PX))
    tophat = cv2.morphologyEx(corrected, cv2.MORPH_TOPHAT, kernel)
    tophat[:, struct_cols] = 0

    noise_floor = float(np.percentile(tophat, NOISE_FLOOR_PERCENTILE))
    signal = np.clip(tophat - noise_floor, 0, None)

    positive = signal[signal > 0]
    ref = float(np.percentile(positive, SIGNAL_REF_PERCENTILE)) if positive.size else 1.0
    ref = max(ref, 1.0)

    normalized = np.clip(signal / ref, 0, 1)
    boosted = np.power(normalized, GAMMA)
    final = (boosted * ims_io.MAX_12BIT).astype(np.uint16)

    # Both of these run LAST, after all morphology, so neither can create a
    # sharp intensity step that a later filter would turn into a bright edge.
    final = apply_crop_mask(final)
    final = remove_oversized_blobs(final)
    return final


def run_step2(run_name: str = DEFAULT_RUN_NAME) -> None:
    layout = get_run_layout(run_name)
    layout.ensure_dirs()

    for ref in ims_io.frame_refs_for_run(HAS_T1, HAS_T2):
        header_path = layout.run_dir / f"Frame{ref.frame_index}-0.ims"
        decoder_path = layout.run_dir / f"Frame{ref.frame_index}-decoder.xml"
        image_path = layout.run_dir / f"Frame{ref.frame_index}-1.ims"
        width, height = ims_io.read_frame_header(header_path)
        decoder_id = ims_io.read_decoder_id(decoder_path)

        for burst_index in burst_indices():
            img12 = ims_io.load_ims_image(image_path, width, height, decoder_id=decoder_id, burst_index=burst_index)
            enhanced12 = enhance_image(img12)

            stem = image_stem(ref.camera, ref.timestep, burst_index)

            jpg_path = layout.filtered_dir / f"{stem}.jpg"
            img8 = np.round(enhanced12.astype(np.float32) * (255.0 / ims_io.MAX_12BIT)).clip(0, 255).astype(np.uint8)
            Image.fromarray(img8).save(jpg_path, quality=95)

            # Filtered output keeps the same decoder format as the source, so it
            # stays self-describing and (in principle) re-importable into DaVis.
            ims_path = layout.filtered_dir / f"{stem}.ims"
            ims_io.save_ims_image(enhanced12, ims_path, decoder_id=decoder_id)
            ims_io.write_decoder_xml(layout.filtered_dir / f"{stem}-decoder.xml", decoder_id=decoder_id)

            log.info(
                "Step2: %s [burst %d] -> %s (raw max=%d, enhanced non-zero frac=%.3f)",
                image_path.name, burst_index, jpg_path.name, int(img12.max()), float((enhanced12 > 50).mean()),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: enhance frames for droplet visibility")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_step2(args.run)


if __name__ == "__main__":
    main()
