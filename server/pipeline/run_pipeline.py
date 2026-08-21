"""End-to-end Tomo-PIV pipeline CLI.

    python -m pipeline.run_pipeline --run "LI_16072026_7.2mm_09"

Runs, in order:
  1. convert_raw   - raw .ims -> straight 8-bit JPEG
  2. enhance       - contrast/visibility enhancement -> filtered JPEG + .ims
  3. detect_drops  - droplet detection & marking -> drop JPEG + centroid CSV
  4-5. reconstruct_mart - MART 3D reconstruction -> Volume A (t1) / Volume B (t2)

Each step is also independently runnable as its own module, and re-running any
step only touches its own output folder -- the original Frame*.ims files are
never read for writing, only for input.
"""
from __future__ import annotations

import argparse
import logging
import time

from pipeline.convert_raw import run_step1
from pipeline.detect_drops import run_step3
from pipeline.enhance import run_step2
from pipeline.project_layout import DEFAULT_RUN_NAME, get_run_layout
from pipeline.reconstruct_mart import N_ITERATIONS, run_steps_4_5

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Tomo-PIV processing pipeline")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS, help="MART iterations")
    parser.add_argument("--skip-reconstruction", action="store_true", help="only run Steps 1-3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    layout = get_run_layout(args.run)
    log.info("Run folder: %s", layout.run_dir)
    log.info("Output folder: %s", layout.output_dir)

    t0 = time.time()
    log.info("=== Step 1: raw -> jpg ===")
    run_step1(args.run)

    log.info("=== Step 2: enhance/filter ===")
    run_step2(args.run)

    log.info("=== Step 3: detect & mark droplets ===")
    run_step3(args.run)

    if not args.skip_reconstruction:
        log.info("=== Steps 4-5: MART 3D reconstruction ===")
        run_steps_4_5(args.run, args.iterations)

    log.info("Pipeline finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
