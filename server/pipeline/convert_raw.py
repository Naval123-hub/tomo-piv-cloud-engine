"""Step 1: decode raw .ims camera frames and save straight (linear) 8-bit JPEGs."""
from __future__ import annotations

import argparse
import logging

import numpy as np
from PIL import Image

from pipeline import ims_io
from pipeline.project_layout import HAS_T1, HAS_T2, DEFAULT_RUN_NAME, burst_indices, get_run_layout, image_stem

log = logging.getLogger(__name__)


def to_8bit_linear(img12: np.ndarray) -> np.ndarray:
    return np.round(img12.astype(np.float32) * (255.0 / ims_io.MAX_12BIT)).clip(0, 255).astype(np.uint8)


def run_step1(run_name: str = DEFAULT_RUN_NAME) -> None:
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
            img8 = to_8bit_linear(img12)

            stem = image_stem(ref.camera, ref.timestep, burst_index)
            out_path = layout.converted_jpg_dir / f"{stem}.jpg"
            Image.fromarray(img8).save(out_path, quality=95)
            log.info("Step1: %s [burst %d] -> %s (%dx%d, max=%d)",
                      image_path.name, burst_index, out_path.name, width, height, int(img12.max()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: convert raw .ims frames to JPEG")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_step1(args.run)


if __name__ == "__main__":
    main()
