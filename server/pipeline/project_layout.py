"""Shared paths, tunable hyperparameters and small conventions for the
Tomo-PIV processing pipeline.

Two things live here:

1. **Paths.** Everything is anchored to ``NAVAL_ROOT``, derived from this
   file's own location (see ``_derive_repo_root``) so the pipeline runs
   unchanged on a laptop, in a Modal container, or in CI. Every root can still
   be redirected with an environment variable (see ``_env_path``) -- that is
   how the remote GPU worker points the pipeline at its scratch volume instead
   of a developer's Desktop.

2. **Hyperparameters.** The per-project knobs (noise floor, particle size cap,
   Z depth, window sizes, dt, pulse mode ...). They stay plain module-level
   constants so the algorithm modules keep importing them exactly as before,
   but ``apply_overrides()`` can now rewrite them at runtime -- that is what
   lets the web UI / API drive a run without editing this file. See the
   docstring of ``apply_overrides`` for why a plain ``setattr`` here is not
   enough on its own.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger(__name__)


def _env_path(var: str, default: Path) -> Path:
    """Environment override for a root path, else the derived default."""
    raw = os.getenv(var)
    return Path(raw).expanduser().resolve() if raw else default


def _derive_repo_root() -> Path:
    """The directory that holds the acquisition data folders.

    In the checkout this package lives at
    ``<repo>/3d_piv/server/pipeline/project_layout.py`` while the DaVis project
    folders (``Project_Imaging_*``, ``LED_PIV``) and ``output/`` sit at
    ``<repo>``, four levels up.

    That depth is not universal, though: the Modal image copies the package to
    ``/root/pipeline``, where four levels up does not exist. So the layout is
    checked by name rather than assumed, and anything unrecognised falls back
    to the package's parent -- which is what the container wants, and where
    ``app_modal.py`` overrides the roots explicitly anyway.
    """
    parents = Path(__file__).resolve().parents
    if len(parents) > 3 and parents[1].name == "server" and parents[2].name == "3d_piv":
        return parents[3]
    return parents[1]


# Derived, never hardcoded -- the same checkout has to work on Windows, on a
# Modal Linux container and inside a Docker build without edits. Override with
# TOMOPIV_ROOT if the data lives somewhere else entirely.
NAVAL_ROOT = _env_path("TOMOPIV_ROOT", _derive_repo_root())
PROJECT_ROOT = _env_path("TOMOPIV_PROJECT_ROOT", NAVAL_ROOT / "Project_Imaging_260709_174128")
CALIBRATION_XML = _env_path("TOMOPIV_CALIBRATION_XML", PROJECT_ROOT / "Properties" / "Calibration" / "Calibration.xml")
LIVE_IMAGES_DIR = PROJECT_ROOT / "Live images"
OUTPUT_ROOT = _env_path("TOMOPIV_OUTPUT_ROOT", NAVAL_ROOT / "output")

DEFAULT_RUN_NAME = os.getenv("TOMOPIV_DEFAULT_RUN", "LI_16072026_7.2mm_09")

# Acquisition mode: does this run have one exposure per camera (single-pulse)
# or two (double-pulse, t1+t2)? Confirmed by directly counting each run's raw
# files in this project -- the two modes are NOT just "t2 missing", they use a
# different frame-index layout entirely:
#   double-pulse (8 raw Frame*-1.ims files): Frame0,1->Camera1(t1,t2);
#     Frame2,3->Camera2; Frame4,5->Camera3; Frame6,7->Camera4
#   single-pulse (4 raw Frame*-1.ims files): Frame0,1,2,3 -> Camera1,2,3,4
#     (t1 only -- there is no t2 file at all, not an empty one)
# Runs checked in THIS project: _06 and _08 are single-pulse (4 files);
# _11, _12, _13, _14 are double-pulse (8 files); _10 has no raw data (skip).
#
#   HAS_T1=1, HAS_T2=1  ->  double-pulse: reconstruct Volume A (t1) AND
#                           Volume B (t2), Step 6 cross-correlation runs.
#   HAS_T1=1, HAS_T2=0  ->  single-pulse: reconstruct Volume A (t1) ONLY --
#                           reconstruct_mart.py skips Volume B entirely, and
#                           cross_correlate.py refuses to run (there is no
#                           Volume B to correlate against) rather than relying
#                           on remembering not to call it.
HAS_T1 = 1
HAS_T2 = 1

# When HAS_T2=0 (single-pulse: no true t1/t2 pulse pair), cross_correlate.py
# cannot use the usual "Volume A (t1) vs Volume B (t2)" comparison -- instead
# it correlates two arbitrary BURST INSTANTS against each other (both must be
# in BURST_FRAME_INDICES so Steps 1-3 have processed them). Ignored when
# HAS_T2=1 (that case already has a real t1/t2 pair to use).
# cross_correlate.py checks whether each volume already exists in volumes/
# (as volume_bNNN.npy) before building it -- so the first run reconstructs
# whichever of these two is missing, and later runs (e.g. after changing just
# one of the two, to compare a new instant against an already-built one)
# reuse whatever's already there instead of rebuilding it.
CROSS_CORR_BURST_INDEX_X = 1
CROSS_CORR_BURST_INDEX_Y = 2

# For burst-mode acquisitions (e.g. LED_PIV, where each FrameN-1.ims holds 50
# concatenated frames instead of 1): which frame(s) *within* the burst to run
# Steps 1-3 (convert_raw.py, enhance.py, detect_drops.py) on. Accepts a
# single int, a list, or a range -- e.g.:
#   BURST_FRAME_INDICES = [0]                  # just the first instant (default)
#   BURST_FRAME_INDICES = range(0, 10)         # first 10 instants
#   BURST_FRAME_INDICES = [0, 5, 10, 15, 20]   # specific instants
# Each processed instant gets its own output files, suffixed "_bNNN" (e.g.
# Camera1_t1_b005.jpg) -- except index 0, which keeps the plain "Camera1_t1.jpg"
# naming so existing single-instant output/workflows are unaffected.
# Ignored (harmless) for single-frame-per-file projects like the original
# LI_16072026_7.2mm_09 dataset, since index 0 is the only frame present there.
BURST_FRAME_INDICES = [0]   # LI_16072026_7.2mm_09 (old project): 1 frame per file, index 0 is the only one present

# Which burst frame(s) to reconstruct into volume(s) (Steps 4-5). Accepts a
# single int, a list, or a range -- same convention as BURST_FRAME_INDICES
# above -- every value used here must also be in BURST_FRAME_INDICES (that
# instant has to have been filtered first, by Step 2):
#   BURST_FRAME_INDEX_FOR_VOLUME = 4                     # just instant 4
#   BURST_FRAME_INDEX_FOR_VOLUME = [0, 5, 10]            # specific instants
#   BURST_FRAME_INDEX_FOR_VOLUME = range(0, 10)          # first 10 instants
# Each instant gets its own volume file(s) (volume_b{NNN}[_t1/_t2].npy -- see
# reconstruct_mart.py:volume_stem), so building several here is safe, nothing
# gets overwritten. `python -m pipeline.reconstruct_mart` builds all of them
# in one call, sharing one voxel grid (see _prepare_calibration_and_grid).
# cross_correlate.py's double-pulse (HAS_T2=1) path uses the FIRST value here
# as the instant to correlate -- set a single int if that's your only use.
BURST_FRAME_INDEX_FOR_VOLUME = 0   # LI_16072026_7.2mm_09 (old project): only instant that exists


def burst_indices() -> list[int]:
    """Normalizes BURST_FRAME_INDICES (int, list, or range) to a plain list."""
    if isinstance(BURST_FRAME_INDICES, int):
        return [BURST_FRAME_INDICES]
    return list(BURST_FRAME_INDICES)


def volume_burst_indices() -> list[int]:
    """Normalizes BURST_FRAME_INDEX_FOR_VOLUME (int, list, or range) to a plain list."""
    if isinstance(BURST_FRAME_INDEX_FOR_VOLUME, int):
        return [BURST_FRAME_INDEX_FOR_VOLUME]
    return list(BURST_FRAME_INDEX_FOR_VOLUME)


# Current project's native frame size: WIDTH=1216 px, HEIGHT=1024 px
# (LED_PIV / LI_16072026_7.2mm_14 -- 2448x2048 sensor, 2x2 binned).
# Use these as the ceiling when picking crop values below:
#
# (If you switch DEFAULT_RUN_NAME/PROJECT_ROOT to a different project, check
# its actual size -- e.g. the original LI_16072026_7.2mm_09 project was
# 2448x2048, unbinned. Run pipeline.ims_io.read_frame_header() on that
# project's Frame0-0.ims if unsure.)
#
# Image-space crop margins, in pixels -- how much to cut off each edge of the
# frame BEFORE filtering/detection (Steps 2-3). Anything inside these margins
# is zeroed out so it can never be enhanced or detected as a droplet -- use
# this to exclude rig rails, nozzle/hardware, or any other known non-droplet
# structure near the frame edges.
# 0 = don't crop that edge (default -- process the full frame, unchanged).
#
#   CROP_LEFT_PX + CROP_RIGHT_PX must stay well under 1216 (width)
#   CROP_TOP_PX + CROP_BOTTOM_PX must stay well under 1024 (height)

CROP_LEFT_PX = 0
CROP_RIGHT_PX = 0
CROP_TOP_PX = 0
CROP_BOTTOM_PX = 50

# Maximum allowed droplet image size, in pixels (width or height of a
# connected bright region). Anything larger is deleted before reconstruction.
# This is the single most effective control over the "continuous ray from
# front to back" artifact in the 3D volume: because the cameras here span
# only a ~12.6 deg aperture, a reconstructed particle is stretched along the
# depth axis by roughly (in-plane size / tan(half-aperture)) ~= 9x its 2D
# size. Measured on this data, reconstructed z-extent tracks 2D blob size
# with correlation 0.93 -- a 2px droplet stays ~0.2mm deep, but a 14px blob
# smears through the whole 4mm depth as a ray. Real tracer droplets here are
# 1-3px across (median 1-2px), so blobs above ~8px are merged clumps,
# saturated reflections or rig hardware -- not trackable particles.
# 0 = disable this filter entirely.
MAX_PARTICLE_PX = 0

# Physical depth (Z) range of the reconstruction volume, in mm, centered on
# the calibration's z=0 plane. This is the main "which part of physical
# space" knob for Steps 4-5 -- X/Y bounds are auto-detected from where
# droplets actually show up (see voxel_grid.py), but Z has no such signal
# (all cameras look through the same depth range), so it's set directly
# here. Default matches the calibrated planes (z0, zminus=-5mm, zplus=+5mm)
# with a small margin.
#
# LI_16072026_7.2mm_09 (old project): calibration planes at z=-5,0,+5mm -> +-6mm.
# (widen back to +-10mm if switching to LED_PIV, where the flow isn't as
# tightly bounded by the calibrated planes.)
VOLUME_Z_MIN_MM = -6.0
VOLUME_Z_MAX_MM = 6.0

# Percentile of the enhanced (top-hat) image clipped to black in Step 2 --
# the single most sensitive knob for whether reconstruction works at all,
# since MART can only place a particle where ALL FOUR cameras see it, so
# anything this threshold discards in even one camera's exposure is lost
# from the volume entirely. Higher = stricter/cleaner but risks deleting real
# (dim) signal; lower = more permissive but risks admitting noise.
#   LI_16072026_7.2mm_09 (old project, bright/saturated in both exposures): 99.3
#   LI_16072026_7.2mm_14 (LED_PIV, t2 exposure ~4x dimmer than t1): 97.0
#     (measured: at 99.3 the t2 volume came out completely EMPTY -- 0
#     four-camera coincidences; at 97.0, 885)
# If switching projects, re-check with the 4-camera-coincidence test rather
# than assuming -- what's "signal" vs "noise" depends on how bright/saturated
# that project's raw images are.
NOISE_FLOOR_PERCENTILE = 99.3

# Cross-correlation (Step 6) interrogation window size, in voxels, and the
# assumed inter-pulse time (dt) in microseconds. ALL THREE must be re-derived
# per project -- they depend on the real droplet displacement and recording
# rate, which differ a lot between a laser double-pulse project and an LED
# burst project:
#   1. Measure real displacement: 2D-correlate a camera's t1 image against
#      its t2 image directly (not the reconstructed volume) and read off the
#      peak shift in pixels -- this is ground truth, don't guess.
#   2. WINDOW_XY/WINDOW_Z must comfortably exceed that displacement (in
#      voxels = displacement_px * image_pixel_size_mm / voxel_pitch_mm) or
#      the correlation peak gets clipped and returns spurious near-zero
#      vectors. WINDOW_Z is capped at the volume depth automatically, so it
#      can be much smaller than WINDOW_XY for a thin volume.
#   3. CROSS_CORR_DT_US: read the acquisition Timing XML's raw <dt> tick value
#      and the camera's frameRate <period>; both are undocumented custom
#      units, but reading them as picoseconds gives physically sane numbers
#      on both projects checked so far (matches the exposure time order of
#      magnitude, and gives a realistic droplet speed rather than an
#      impossible one -- sanity-check the resulting speed against what's
#      physically plausible for your flow).
#   LI_16072026_7.2mm_09 (old project): displacement small enough for a
#     32-voxel isotropic window; dt = 50 us.
#   LI_16072026_7.2mm_14 (LED_PIV): measured displacement ~144 voxels
#     in-plane, volume only 24-47 voxels deep; dt = 1476.4 us.
CROSS_CORR_WINDOW_XY = 32   # or whatever isotropic window fits that project's measured displacement
CROSS_CORR_WINDOW_Z = 32
CROSS_CORR_DT_US = 50.0

# Step 6 runs TWO correlation passes (standard iterative/multigrid PIV
# window-offset technique -- Scarano & Riethmuller 2000): a coarse pass at
# CROSS_CORR_WINDOW_XY (big enough to safely capture the full bulk
# displacement, as measured above), then a second, finer pass at this size
# that pre-shifts its window by the coarse pass's local displacement before
# correlating -- since the bulk motion is already compensated, this pass only
# has to resolve a small residual, which both increases spatial resolution
# (more, smaller windows fit across the same volume) and improves SNR/
# reduces noise (standard result in the PIV literature). Must stay small
# enough to meaningfully subdivide CROSS_CORR_WINDOW_XY, but big enough to
# still contain several droplets and comfortably exceed the expected residual
# displacement (local velocity variation between neighbouring coarse windows,
# not the full bulk motion). Start at half the coarse window and tune from
# there if vectors still look too sparse/noisy.
CROSS_CORR_WINDOW_XY_FINE = 128


@dataclass(frozen=True)
class RunLayout:
    run_name: str
    run_dir: Path
    output_dir: Path
    converted_jpg_dir: Path
    filtered_dir: Path
    drop_dir: Path
    volumes_dir: Path

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.converted_jpg_dir, self.filtered_dir, self.drop_dir, self.volumes_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_run_layout(run_name: str = DEFAULT_RUN_NAME, project_root: Path = PROJECT_ROOT) -> RunLayout:
    run_dir = project_root / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run folder not found: {run_dir}")
    output_dir = OUTPUT_ROOT / run_name
    return RunLayout(
        run_name=run_name,
        run_dir=run_dir,
        output_dir=output_dir,
        converted_jpg_dir=output_dir / "converted_jpg",
        filtered_dir=output_dir / "filtered",
        drop_dir=output_dir / "drop",
        volumes_dir=output_dir / "volumes",
    )


def image_stem(camera: int, timestep: int, burst_index: int = 0) -> str:
    if burst_index == 0:
        return f"Camera{camera}_t{timestep}"
    return f"Camera{camera}_t{timestep}_b{burst_index:03d}"


# ---------------------------------------------------------------------------
# Runtime hyperparameter overrides (web UI / API / remote GPU worker)
# ---------------------------------------------------------------------------
#
# The algorithm modules were written for a single-user CLI and bind their
# knobs at import time:
#
#     from pipeline.project_layout import NOISE_FLOOR_PERCENTILE
#
# That is a value copy, not a reference -- rebinding the name *here* after
# enhance.py has been imported would leave enhance.NOISE_FLOOR_PERCENTILE
# pointing at the old float. Rather than rewrite those modules (they hold the
# validated maths and are deliberately left untouched), apply_overrides()
# rebinds the constant in this module AND in every already-imported
# `pipeline.*` module that holds a copy of it.
#
# Two further wrinkles are handled below:
#   * aliases -- voxel_grid re-exports VOLUME_Z_MIN_MM as Z_MIN_MM, and
#     cross_correlate re-exports CROSS_CORR_WINDOW_XY as WINDOW_XY, etc.
#   * default arguments -- `def get_run_layout(..., project_root=PROJECT_ROOT)`
#     froze its default when the `def` executed, so the function object's
#     __defaults__ tuple has to be rebuilt too.


@dataclass(frozen=True)
class Tunable:
    """UI/validation metadata for one hyperparameter. The web sidebar is built
    straight from this table, so a knob added here shows up in the UI, in the
    API payload and in the run provenance record without further edits."""
    name: str
    kind: str            # "int" | "float" | "bool" | "int_list" | "path" | "str"
    group: str
    label: str
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None


TUNABLES: tuple[Tunable, ...] = (
    Tunable("HAS_T2", "bool", "Acquisition", "Double-pulse (t1 + t2)",
            help="On: 8 raw frames, a t1/t2 pulse pair per camera, Step 6 correlates t1 vs t2. "
                 "Off: 4 raw frames, single exposure; Step 6 correlates two burst instants instead."),
    Tunable("BURST_FRAME_INDICES", "int_list", "Acquisition", "Burst instants for Steps 1-3",
            help="Which frame(s) inside each FrameN-1.ims to process. Comma-separated, e.g. 0,5,10."),
    Tunable("BURST_FRAME_INDEX_FOR_VOLUME", "int_list", "Acquisition", "Burst instants to reconstruct",
            help="Which processed instant(s) Steps 4-5 build volumes for. Must be a subset of the row above."),
    Tunable("CROSS_CORR_BURST_INDEX_X", "int", "Acquisition", "Single-pulse: instant X",
            help="Only used when double-pulse is off: first of the two instants to correlate.",
            min=0, max=999, step=1),
    Tunable("CROSS_CORR_BURST_INDEX_Y", "int", "Acquisition", "Single-pulse: instant Y",
            help="Only used when double-pulse is off: second of the two instants to correlate.",
            min=0, max=999, step=1),

    Tunable("NOISE_FLOOR_PERCENTILE", "float", "Step 2 - Enhance", "Noise floor percentile",
            help="Percentile of the top-hat image clipped to black. The most sensitive knob in the pipeline: "
                 "MART only places a particle where all 4 cameras see it, so signal cut here is lost entirely. "
                 "99.3 for the bright laser project, ~97.0 for the dimmer LED burst project.",
            min=80.0, max=99.99, step=0.05),
    Tunable("MAX_PARTICLE_PX", "int", "Step 2 - Enhance", "Max particle size (px)",
            help="Connected bright regions wider/taller than this are deleted before reconstruction -- the main "
                 "control over full-depth ghost rays. 0 disables the filter.",
            min=0, max=64, step=1),
    Tunable("CROP_TOP_PX", "int", "Step 2 - Enhance", "Crop top (px)",
            help="Edge margin zeroed before filtering/detection, to exclude rig hardware.",
            min=0, max=1024, step=1),
    Tunable("CROP_BOTTOM_PX", "int", "Step 2 - Enhance", "Crop bottom (px)", min=0, max=1024, step=1),
    Tunable("CROP_LEFT_PX", "int", "Step 2 - Enhance", "Crop left (px)", min=0, max=1216, step=1),
    Tunable("CROP_RIGHT_PX", "int", "Step 2 - Enhance", "Crop right (px)", min=0, max=1216, step=1),

    Tunable("VOLUME_Z_MIN_MM", "float", "Steps 4-5 - MART", "Z min (mm)",
            help="Depth range of the reconstructed volume, centred on the calibration z=0 plane. X/Y bounds are "
                 "auto-detected from Step 3 detections; Z has no such signal so it is set here.",
            min=-50.0, max=0.0, step=0.5),
    Tunable("VOLUME_Z_MAX_MM", "float", "Steps 4-5 - MART", "Z max (mm)", min=0.0, max=50.0, step=0.5),

    Tunable("CROSS_CORR_WINDOW_XY", "int", "Step 6 - Correlate", "Coarse window XY (voxels)",
            help="Coarse-pass interrogation window. Must comfortably exceed the real bulk displacement in voxels "
                 "or the correlation peak is clipped and returns spurious near-zero vectors.",
            min=8, max=512, step=8),
    Tunable("CROSS_CORR_WINDOW_XY_FINE", "int", "Step 6 - Correlate", "Fine window XY (voxels)",
            help="Second (window-offset) pass. Start at half the coarse window.",
            min=8, max=512, step=8),
    Tunable("CROSS_CORR_WINDOW_Z", "int", "Step 6 - Correlate", "Window Z (voxels)",
            help="Capped at the volume depth automatically, so it can be much smaller than the XY window.",
            min=4, max=256, step=4),
    Tunable("CROSS_CORR_DT_US", "float", "Step 6 - Correlate", "Laser dt (us)",
            help="Inter-pulse time. Displacement and direction are independent of this; only absolute speed "
                 "(m/s) scales with it. 50 us for the laser project, 1476.4 us for the LED burst project.",
            min=0.01, max=100000.0, step=0.1),
)

TUNABLES_BY_NAME: dict[str, Tunable] = {t.name: t for t in TUNABLES}

#: Names re-exported under a different identifier by a downstream module.
_ALIASES: dict[str, tuple[str, ...]] = {
    "VOLUME_Z_MIN_MM": ("Z_MIN_MM",),           # pipeline.voxel_grid, pipeline.reconstruct_mart
    "VOLUME_Z_MAX_MM": ("Z_MAX_MM",),
    "CROSS_CORR_WINDOW_XY": ("WINDOW_XY",),     # pipeline.cross_correlate
    "CROSS_CORR_WINDOW_XY_FINE": ("WINDOW_XY_FINE",),
    "CROSS_CORR_WINDOW_Z": ("WINDOW_Z",),
    "CROSS_CORR_DT_US": ("ASSUMED_DT_US",),
}

#: (module, function, {parameter: constant-name}) -- functions whose default
#: arguments captured a constant at definition time and must be rebuilt.
_DEFAULT_REBINDS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("pipeline.project_layout", "get_run_layout",
     {"run_name": "DEFAULT_RUN_NAME", "project_root": "PROJECT_ROOT"}),
    ("pipeline.voxel_grid", "estimate_world_xy_bounds",
     {"z_min": "VOLUME_Z_MIN_MM", "z_max": "VOLUME_Z_MAX_MM"}),
    ("pipeline.cross_correlate", "run_cross_correlation",
     {"run_name": "DEFAULT_RUN_NAME", "window_xy": "CROSS_CORR_WINDOW_XY", "window_z": "CROSS_CORR_WINDOW_Z",
      "dt_us": "CROSS_CORR_DT_US", "window_xy_fine": "CROSS_CORR_WINDOW_XY_FINE"}),
    ("pipeline.cross_correlate", "run_step6",
     {"run_name": "DEFAULT_RUN_NAME", "window_xy": "CROSS_CORR_WINDOW_XY", "window_z": "CROSS_CORR_WINDOW_Z",
      "dt_us": "CROSS_CORR_DT_US", "window_xy_fine": "CROSS_CORR_WINDOW_XY_FINE"}),
    ("pipeline.convert_raw", "run_step1", {"run_name": "DEFAULT_RUN_NAME"}),
    ("pipeline.enhance", "run_step2", {"run_name": "DEFAULT_RUN_NAME"}),
    ("pipeline.detect_drops", "run_step3", {"run_name": "DEFAULT_RUN_NAME"}),
    ("pipeline.reconstruct_mart", "run_steps_4_5", {"run_name": "DEFAULT_RUN_NAME"}),
)

#: Path constants recomputed from PROJECT_ROOT unless overridden explicitly.
_PROJECT_DERIVED = {
    "CALIBRATION_XML": lambda root: root / "Properties" / "Calibration" / "Calibration.xml",
    "LIVE_IMAGES_DIR": lambda root: root / "Live images",
}

_PATH_CONSTANTS = ("NAVAL_ROOT", "PROJECT_ROOT", "OUTPUT_ROOT", "CALIBRATION_XML", "LIVE_IMAGES_DIR")


def _coerce(name: str, value: Any) -> Any:
    """Parse a value coming from a web form / JSON payload into what the
    pipeline expects. Raises ValueError naming the offending parameter so the
    API can answer 422 instead of failing deep inside MART."""
    if name in _PATH_CONSTANTS:
        return Path(value).expanduser()

    spec = TUNABLES_BY_NAME.get(name)
    kind = spec.kind if spec else None
    if kind == "bool":
        if isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        return int(bool(value))          # HAS_T1/HAS_T2 are ints by convention
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "int_list":
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
            return [int(p) for p in parts]
        return [int(v) for v in value]
    return value


def coerce_param(name: str, value: Any) -> Any:
    """Public wrapper over the payload parser, so callers (the web UI's
    pre-flight validation, an API layer) can normalise a value the same way
    apply_overrides() will, without applying it."""
    return _coerce(name, value)


def _validate(name: str, value: Any) -> None:
    spec = TUNABLES_BY_NAME.get(name)
    if spec is None or spec.kind not in ("int", "float"):
        return
    if spec.min is not None and value < spec.min:
        raise ValueError(f"{name}={value} is below the allowed minimum {spec.min}")
    if spec.max is not None and value > spec.max:
        raise ValueError(f"{name}={value} is above the allowed maximum {spec.max}")


def _pipeline_modules() -> Iterator[Any]:
    """Every already-imported `pipeline.*` module, this one included."""
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name == "pipeline" or mod_name.startswith("pipeline."):
            yield mod


def _propagate(name: str, value: Any) -> list[str]:
    """Rebind `name` (and its aliases) in every pipeline module holding a copy."""
    touched: list[str] = []
    targets = (name,) + _ALIASES.get(name, ())
    for mod in _pipeline_modules():
        for target in targets:
            if hasattr(mod, target):
                setattr(mod, target, value)
                touched.append(f"{mod.__name__}.{target}")
    return touched


def _rebind_defaults() -> None:
    """Rebuild the __defaults__ of functions that captured a constant when
    their `def` executed. Modules not yet imported are skipped -- a later
    import picks up the current value anyway."""
    current = globals()
    for mod_name, func_name, param_map in _DEFAULT_REBINDS:
        mod = sys.modules.get(mod_name)
        func = getattr(mod, func_name, None) if mod is not None else None
        if func is None or not getattr(func, "__defaults__", None):
            continue
        params = [
            p for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is not p.empty
        ]
        if len(params) != len(func.__defaults__):
            continue  # signature changed underneath us; leave it alone
        func.__defaults__ = tuple(
            current[param_map[p.name]] if p.name in param_map and param_map[p.name] in current else old
            for p, old in zip(params, func.__defaults__)
        )


def apply_overrides(overrides: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Rewrite pipeline hyperparameters at runtime; returns what was applied.

    Accepts a dict, keyword arguments, or both::

        apply_overrides({"NOISE_FLOOR_PERCENTILE": 97.0}, HAS_T2=0)

    Unknown names are ignored with a warning rather than raising, so an older
    worker still accepts a payload from a newer UI instead of hard-failing the
    job. Out-of-range values for a known Tunable *do* raise ValueError.
    """
    merged: dict[str, Any] = dict(overrides or {})
    merged.update(kwargs)

    applied: dict[str, Any] = {}
    current = globals()
    for name, raw in merged.items():
        if raw is None:
            continue
        if name not in TUNABLES_BY_NAME and name not in current:
            log.warning("apply_overrides: ignoring unknown parameter %r", name)
            continue
        value = _coerce(name, raw)
        _validate(name, value)
        current[name] = value
        _propagate(name, value)
        applied[name] = value

    # PROJECT_ROOT moved -> recompute anything derived from it that the caller
    # did not set explicitly. Otherwise switching datasets silently keeps
    # pointing the reconstruction at the previous project's calibration.
    if "PROJECT_ROOT" in applied:
        root = current["PROJECT_ROOT"]
        for derived, fn in _PROJECT_DERIVED.items():
            if derived not in applied:
                current[derived] = fn(root)
                _propagate(derived, current[derived])
                applied[derived] = current[derived]

    _rebind_defaults()
    if applied:
        log.info("Applied %d pipeline override(s): %s", len(applied),
                 ", ".join(f"{k}={v}" for k, v in applied.items()))
    return applied


def config_snapshot() -> dict[str, Any]:
    """Every tunable's current value, JSON-safe. Written alongside each job's
    output so a result can always be traced back to the settings that made it."""
    current = globals()
    snap: dict[str, Any] = {t.name: current[t.name] for t in TUNABLES}
    snap.update({
        "HAS_T1": HAS_T1,
        "DEFAULT_RUN_NAME": current["DEFAULT_RUN_NAME"],
        "PROJECT_ROOT": str(current["PROJECT_ROOT"]),
        "OUTPUT_ROOT": str(current["OUTPUT_ROOT"]),
        "CALIBRATION_XML": str(current["CALIBRATION_XML"]),
    })
    for key in ("BURST_FRAME_INDICES", "BURST_FRAME_INDEX_FOR_VOLUME"):
        value = snap[key]
        snap[key] = [value] if isinstance(value, int) else list(value)
    return snap


def overrides_from_env(prefix: str = "TOMOPIV_PARAM_") -> dict[str, Any]:
    """Reads overrides out of the environment, e.g. TOMOPIV_PARAM_HAS_T2=0.
    Used by the remote worker, where threading a dict through the container
    boundary is not always convenient."""
    return {k[len(prefix):]: v for k, v in os.environ.items() if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Discovery helpers (used by the web UI and the API to populate their pickers)
# ---------------------------------------------------------------------------

def discover_project_roots(search_root: Path | None = None) -> list[Path]:
    """DaVis project folders under the repo -- any directory carrying a
    Properties/Calibration folder. That matches how the pipeline identifies a
    project (calibration is the one thing every run in a project shares).
    Searches one and two levels deep, so both `Project_Imaging_*/` and
    `LED_PIV/Project_Imaging_*/` are found."""
    base = Path(search_root) if search_root else NAVAL_ROOT
    found: list[Path] = []
    for pattern in ("*/Properties/Calibration", "*/*/Properties/Calibration"):
        for cal_dir in sorted(base.glob(pattern)):
            root = cal_dir.parent.parent
            if root not in found:
                found.append(root)
    return found


def list_runs(project_root: Path | None = None) -> list[str]:
    """Run folder names inside a project (LI_16072026_7.2mm_09, ...)."""
    root = Path(project_root) if project_root else globals()["PROJECT_ROOT"]
    if not root.is_dir():
        return []
    skip = {"Properties", "Live images"}
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name not in skip and not p.name.startswith("Recording_Date")
    )


#: Output artifact categories, in the order the UI presents them:
#: (label, RunLayout attribute holding the directory, glob patterns).
ARTIFACT_KINDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Vector fields (VTK)", "volumes_dir", ("vector_*.vtk",)),
    ("Volume stacks (TIFF)", "volumes_dir", ("volume_*.tif",)),
    ("Vector tables (CSV)", "volumes_dir", ("vector_*.csv",)),
    ("Detections (CSV)", "drop_dir", ("*_detections.csv",)),
    ("Volume previews (JPG)", "volumes_dir", ("*_MIP.jpg",)),
    ("Marked droplets (JPG)", "drop_dir", ("*.jpg",)),
    ("Filtered frames (JPG)", "filtered_dir", ("*.jpg",)),
    ("Converted frames (JPG)", "converted_jpg_dir", ("*.jpg",)),
    ("Run metadata (JSON)", "volumes_dir", ("*.json",)),
)


def iter_run_artifacts(layout: RunLayout, kinds: Iterable[str] | None = None) -> Iterator[tuple[str, Path]]:
    """Yields (kind_label, path) for every downloadable output of a run.
    One definition of "an output" keeps the local download buttons and the
    cloud uploader from ever drifting apart."""
    wanted = set(kinds) if kinds else None
    for label, attr, patterns in ARTIFACT_KINDS:
        if wanted is not None and label not in wanted:
            continue
        directory: Path = getattr(layout, attr)
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.is_file():
                    yield label, path
