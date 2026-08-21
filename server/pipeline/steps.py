"""One definition of "what the pipeline's steps are" and how to invoke them.

Every caller -- the remote GPU worker (``app_modal.py``) and the
``python -m pipeline.run_pipeline`` CLI alike -- drives the pipeline through
this module, so adding or reordering a step is a single edit rather than
several that can drift apart.

This module is a thin dispatcher only: it calls the existing ``run_stepN``
entry points unchanged and holds no numerics of its own. Every heavy import
(OpenCV, scikit-image, tifffile, CuPy) is deferred into the branch that needs
it, so listing the steps costs nothing and a CPU-only step never pays to
import the GPU stack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepSpec:
    key: str
    label: str
    #: Whether this step benefits from CUDA. Steps 1-3 are 2D OpenCV/
    #: scikit-image work and do not; Steps 4-6 (MART voxel reconstruction and
    #: 3D FFT cross-correlation) are what pipeline/gpu_utils.py reaches for
    #: CuPy for, and do.
    needs_gpu: bool
    description: str
    #: Steps whose output must already exist on disk.
    requires: tuple[str, ...] = ()


STEPS: dict[str, StepSpec] = {
    "step1": StepSpec(
        "step1", "Step 1 - Decode raw frames", False,
        "Decodes .ims camera frames to linear 8-bit JPEG previews.",
    ),
    "step2": StepSpec(
        "step2", "Step 2 - Enhance", False,
        "Column-FPN correction, rail masking, top-hat, noise floor, gamma boost. "
        "Writes filtered .ims + JPEG.",
        ("step1",),
    ),
    "step3": StepSpec(
        "step3", "Step 3 - Detect droplets", False,
        "DoG band-pass + robust MAD threshold peak finding. Writes marked JPEG + centroid CSV.",
        ("step2",),
    ),
    "steps45": StepSpec(
        "steps45", "Steps 4-5 - MART reconstruction", True,
        "Builds the voxel grid from Step 3 detections and runs MART. Writes .npy, a .tif "
        "stack and a max-intensity-projection JPEG per volume.",
        ("step3",),
    ),
    "step6": StepSpec(
        "step6", "Step 6 - Cross-correlation", True,
        "Two-pass (coarse + window-offset fine) 3D FFT cross-correlation. Writes the "
        "velocity vector field as .vtk and .csv.",
        ("steps45",),
    ),
    "full": StepSpec(
        "full", "Full pipeline", True,
        "Steps 1 through 6 in order, sharing one process (and one voxel grid).",
    ),
}

#: Parameters that are *arguments to the step functions* rather than pipeline
#: constants. Anything else in a request payload is routed to
#: project_layout.apply_overrides().
RUNTIME_PARAM_DEFAULTS: dict[str, Any] = {
    "mart_iterations": 15,        # reconstruct_mart.N_ITERATIONS
    "overlap_frac": 0.5,          # cross_correlate.OVERLAP_FRAC
    "include_step6": True,        # consulted only by step="full"
}


def split_params(params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Splits one flat request payload into (pipeline overrides, step arguments).

    Unknown keys are dropped with a warning rather than raising, so a worker
    running slightly older code still accepts a payload from a newer UI.
    """
    from pipeline import project_layout as pl

    overrides: dict[str, Any] = {}
    runtime: dict[str, Any] = dict(RUNTIME_PARAM_DEFAULTS)
    for key, value in (params or {}).items():
        if key in RUNTIME_PARAM_DEFAULTS:
            runtime[key] = value
        elif key in pl.TUNABLES_BY_NAME or key in ("PROJECT_ROOT", "OUTPUT_ROOT", "CALIBRATION_XML"):
            overrides[key] = value
        else:
            log.warning("Ignoring unrecognised parameter %r", key)
    return overrides, runtime


def run_step(step: str, run_name: str, runtime: dict[str, Any] | None = None) -> None:
    """Runs one step (or, for ``step="full"``, all of them in order).

    Assumes ``project_layout.apply_overrides()`` has already been called --
    the step functions read their hyperparameters from module state, so the
    configuration must be in place before this is invoked.
    """
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}; expected one of {', '.join(STEPS)}")
    runtime = {**RUNTIME_PARAM_DEFAULTS, **(runtime or {})}

    if step in ("step1", "full"):
        from pipeline.convert_raw import run_step1

        log.info("=== Step 1: raw -> jpg ===")
        run_step1(run_name)

    if step in ("step2", "full"):
        from pipeline.enhance import run_step2

        log.info("=== Step 2: enhance/filter ===")
        run_step2(run_name)

    if step in ("step3", "full"):
        from pipeline.detect_drops import run_step3

        log.info("=== Step 3: detect & mark droplets ===")
        run_step3(run_name)

    if step in ("steps45", "full"):
        from pipeline.reconstruct_mart import run_steps_4_5

        log.info("=== Steps 4-5: MART 3D reconstruction ===")
        run_steps_4_5(run_name, int(runtime["mart_iterations"]))

    if step == "step6" or (step == "full" and runtime.get("include_step6", True)):
        from pipeline import project_layout as pl
        from pipeline.cross_correlate import run_step6

        log.info("=== Step 6: 3D cross-correlation ===")
        # Window sizes and dt are passed explicitly from the (possibly
        # overridden) constants rather than relying on the callee's defaults,
        # so the values used are unambiguous in the job log.
        run_step6(
            run_name,
            window_xy=pl.CROSS_CORR_WINDOW_XY,
            window_z=pl.CROSS_CORR_WINDOW_Z,
            overlap_frac=float(runtime["overlap_frac"]),
            dt_us=pl.CROSS_CORR_DT_US,
            mart_iterations=int(runtime["mart_iterations"]),
            window_xy_fine=pl.CROSS_CORR_WINDOW_XY_FINE,
        )
