"""Step 3: detect seeding droplets (even small/faint ones) on the enhanced images.

Approach: Difference-of-Gaussians band-pass (isolates particle-scale blobs from
background) + robust median/MAD threshold + local-maxima peak finding. Thresholds
are deliberately permissive (favor recall) since even small/faint droplets should
be detected. Detections are circled on the image (drop/*.jpg) and their pixel
centroids saved to CSV for reuse when auto-cropping the MART reconstruction ROI.
"""
from __future__ import annotations

import argparse
import csv
import logging

import cv2
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians

from pipeline import ims_io
from pipeline.project_layout import HAS_T1, HAS_T2, DEFAULT_RUN_NAME, burst_indices, get_run_layout, image_stem

log = logging.getLogger(__name__)

DOG_LOW_SIGMA = 1.0
DOG_HIGH_SIGMA = 6.0
MIN_PEAK_DISTANCE_PX = 3
THRESHOLD_MAD_K = 3.0  # lower = more sensitive/higher recall
MARK_RADIUS_PX = 6


def detect_centroids(enhanced12: np.ndarray) -> np.ndarray:
    """Return an (N, 2) array of (row, col) droplet centroids."""
    img = enhanced12.astype(np.float32)
    dog = difference_of_gaussians(img, DOG_LOW_SIGMA, DOG_HIGH_SIGMA)
    dog = np.clip(dog, 0, None)

    median = float(np.median(dog))
    mad = float(np.median(np.abs(dog - median))) + 1e-6
    threshold = median + THRESHOLD_MAD_K * 1.4826 * mad

    coords = peak_local_max(
        dog,
        min_distance=MIN_PEAK_DISTANCE_PX,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return coords


def mark_image(enhanced12: np.ndarray, coords: np.ndarray) -> np.ndarray:
    img8 = np.round(enhanced12.astype(np.float32) * (255.0 / ims_io.MAX_12BIT)).clip(0, 255).astype(np.uint8)
    marked = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    for row, col in coords:
        cv2.circle(marked, (int(col), int(row)), MARK_RADIUS_PX, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    return marked


def run_step3(run_name: str = DEFAULT_RUN_NAME) -> None:
    layout = get_run_layout(run_name)
    layout.ensure_dirs()

    for ref in ims_io.frame_refs_for_run(HAS_T1, HAS_T2):
        header_path = layout.run_dir / f"Frame{ref.frame_index}-0.ims"
        width, height = ims_io.read_frame_header(header_path)

        for burst_index in burst_indices():
            stem = image_stem(ref.camera, ref.timestep, burst_index)

            filtered_ims_path = layout.filtered_dir / f"{stem}.ims"
            if not filtered_ims_path.exists():
                raise FileNotFoundError(f"missing Step 2 output {filtered_ims_path}; run enhance.py first")
            decoder_id = ims_io.read_decoder_id(layout.filtered_dir / f"{stem}-decoder.xml")
            # burst_index=0 here is correct regardless of the loop variable above:
            # each filtered .ims Step 2 writes always holds exactly one frame.
            enhanced12 = ims_io.load_ims_image(filtered_ims_path, width, height, decoder_id=decoder_id, burst_index=0)

            coords = detect_centroids(enhanced12)
            marked = mark_image(enhanced12, coords)

            jpg_path = layout.drop_dir / f"{stem}.jpg"
            Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)).save(jpg_path, quality=95)

            csv_path = layout.drop_dir / f"{stem}_detections.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["row_px", "col_px"])
                writer.writerows(coords.tolist())

            log.info("Step3: %s -> %d droplets detected -> %s", stem, len(coords), jpg_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: detect and mark droplets")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_step3(args.run)


if __name__ == "__main__":
    main()
