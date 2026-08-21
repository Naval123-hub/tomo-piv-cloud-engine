"""Remote GPU worker + REST API for the Tomo-PIV pipeline.

    modal serve app_modal.py     # hot-reloading dev server
    modal deploy app_modal.py    # persistent deployment

Architecture
------------
::

    React client --> Express gateway --HTTPS--> FastAPI (Modal container, CPU)
                                  |  .spawn()          ^  status/logs
                                  v                    |
                          Worker container   <---->  Redis / Upstash
                       (T4 GPU for Steps 4-6)          (job records)
                                  |
                                  v
                       Cloudflare R2  (raw .ims in, .vtk/.tif/.csv/.jpg out)

The API returns immediately with a ``job_id``; the worker runs detached and
publishes status, streamed logs and artifact links into the job store. Nothing
about the numerical pipeline changed to make this work -- ``project_layout.
apply_overrides()`` rewrites the hyperparameters in-process before the step
functions are called, and ``JobLogHandler`` taps the logging the pipeline
modules already emit.

Compute placement is deliberate: Steps 1-3 are OpenCV/scikit-image work on 2D
frames and get **no** GPU (a T4 sitting idle behind single-threaded morphology
is pure cost), while Steps 4-6 -- MART voxel reconstruction and 3D FFT
cross-correlation, which is what ``pipeline/gpu_utils.py`` reaches for CuPy for
-- run on a **T4**.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import modal

from pipeline.steps import RUNTIME_PARAM_DEFAULTS, STEPS, run_step, split_params

from cloud_io import (
    Artifact, JobLogHandler, JobRecord, JobStore, Manifest, R2Storage,
    STATUS_CANCELLED, STATUS_FAILED, STATUS_QUEUED, STATUS_RUNNING, STATUS_SUCCEEDED,
    calibration_prefix, input_prefix, manifest_key, new_job_id, output_prefix,
)

log = logging.getLogger(__name__)

APP_NAME = os.getenv("MODAL_APP_NAME", "tomo-piv")
GPU_TYPE = os.getenv("MODAL_GPU", "T4")

#: Paths inside the container. The Modal Volume is the pipeline's working
#: filesystem: raw frames land here from R2 once and are reused by every later
#: job for the same run, which is what keeps a Step-6-only rerun cheap.
#:
#: Kept as plain POSIX strings, NOT Path objects: this module is also imported
#: on the developer's machine at `modal deploy` time, and on Windows
#: `str(Path("/workspace"))` is "\workspace" -- which would be sent as the
#: volume mount point and the container's output root. Container paths must
#: never be round-tripped through the host's path flavour.
WORKSPACE = "/workspace"
PROJECTS_DIR = "/workspace/projects"
OUTPUT_DIR = "/workspace/output"

app = modal.App(APP_NAME)

# Raw acquisitions and reconstructed volumes are far too large to ship in the
# image; they live on a persistent Volume mounted by every function.
volume = modal.Volume.from_name("tomo-piv-workspace", create_if_missing=True)

# R2 credentials + Redis URL. Create with:
#   modal secret create tomo-piv-secrets \
#       CLOUDFLARE_R2_ENDPOINT=... CLOUDFLARE_R2_ACCESS_KEY=... \
#       CLOUDFLARE_R2_SECRET_KEY=... CLOUDFLARE_R2_BUCKET=... REDIS_URL=...
secret = modal.Secret.from_name("tomo-piv-secrets")

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    # OpenCV's shared-library dependencies. Even the headless wheel needs
    # libglib; the rest keep scikit-image/tifffile codecs happy.
    .apt_install("libglib2.0-0", "libgomp1")
    .pip_install(
        "numpy>=1.26,<3",
        # headless, not plain opencv-python: identical cv2 API, no GUI/GL
        # dependency chain, which is what you want in a container.
        "opencv-python-headless>=4.9",
        "cupy-cuda12x>=13.0",       # CUDA 12 build; matches NVIDIA 12.2 devel base
        "scikit-image>=0.22",
        "pillow>=10.0",
        "tifffile>=2024.1.30",
        "boto3>=1.34",
        "fastapi>=0.110",
        "pydantic>=2.6",
        "redis>=5.0",
    )
    .env({
        # project_layout derives its roots from __file__ but honours these,
        # so the pipeline writes to the Volume rather than the container's
        # ephemeral filesystem.
        "TOMOPIV_OUTPUT_ROOT": OUTPUT_DIR,
        "PYTHONUNBUFFERED": "1",
    })
    # Local source last: these layers change most often, so everything above
    # them stays cached between deploys.
    .add_local_python_source("cloud_io")
    .add_local_python_source("pipeline")
)


# ---------------------------------------------------------------------------
# Worker internals (run inside the container)
# ---------------------------------------------------------------------------

def _sync_inputs(storage: R2Storage, project_key: str, run_name: str) -> Path:
    """Mirrors this run's raw frames and the project calibration from R2 onto
    the Volume, then checks that everything the pipeline will open is actually
    present.

    Both counts are reported as "N new, M already cached": files already on the
    Volume at a matching size are skipped, so a second job on the same run is
    free -- but a bare "synced 0" reads like "nothing exists", which is exactly
    the wrong conclusion when the truth is "all cached".

    The calibration check matters more than it looks. Without it a missing
    Calibration.xml surfaces as a bare FileNotFoundError from ElementTree deep
    inside Step 4, after the container has started and Steps 1-3 have run --
    minutes of billed GPU time to learn that an 8 KB file was never uploaded.
    """
    project_dir = Path(PROJECTS_DIR) / project_key
    run_dir = project_dir / run_name
    properties_dir = project_dir / "Properties"
    calibration_xml = properties_dir / "Calibration" / "Calibration.xml"

    def _count(directory: Path) -> int:
        return sum(1 for p in directory.rglob("*") if p.is_file()) if directory.is_dir() else 0

    if storage.configured:
        log.info("Syncing calibration from R2: %s", calibration_prefix(project_key))
        n_cal = storage.download_prefix(calibration_prefix(project_key), properties_dir)
        log.info("Syncing raw frames from R2: %s", input_prefix(project_key, run_name))
        n_raw = storage.download_prefix(input_prefix(project_key, run_name), run_dir)
        log.info(
            "Calibration: %d new, %d already cached. Raw frames: %d new, %d already cached.",
            n_cal, max(0, _count(properties_dir) - n_cal), n_raw, max(0, _count(run_dir) - n_raw),
        )
        volume.commit()
    else:
        log.warning("R2 not configured -- expecting inputs to already be on the Modal Volume.")

    bucket = storage.settings.bucket or "<bucket>"

    if not run_dir.is_dir() or not any(run_dir.iterdir()):
        raise FileNotFoundError(
            f"No input data for run '{run_name}' (project '{project_key}'). Upload the raw "
            f"Frame*-0.ims / Frame*-1.ims / Frame*-decoder.xml files to "
            f"s3://{bucket}/{input_prefix(project_key, run_name)}/ first."
        )

    if not calibration_xml.is_file():
        # Name the exact key -- the usual mistake is dropping the file at
        # Properties/Calibration.xml, one level above where the pipeline looks.
        found = sorted(p.name for p in properties_dir.rglob("*.xml")) if properties_dir.is_dir() else []
        raise FileNotFoundError(
            f"Calibration.xml is missing for project '{project_key}'. The pipeline needs it at "
            f"s3://{bucket}/{calibration_prefix(project_key)}/Calibration/Calibration.xml "
            f"(note the 'Calibration/' subfolder). "
            + (f"Found under Properties/ instead: {', '.join(found)}. " if found
               else "Nothing was found under Properties/ at all. ")
            + "It is one ~8 KB file, shared by every run in the project."
        )

    return project_dir


def _write_provenance(layout, job_id: str, step: str, params: dict[str, Any]) -> None:
    """Records exactly which settings produced these outputs, next to them."""
    from pipeline import project_layout as pl

    record = {
        "job_id": job_id,
        "step": step,
        "run_name": layout.run_name,
        "requested_params": params,
        "effective_config": pl.config_snapshot(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    layout.volumes_dir.mkdir(parents=True, exist_ok=True)
    (layout.volumes_dir / f"job_{job_id}_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _upload_outputs(storage: R2Storage, layout, project_key: str, run_name: str, step: str,
                    job_id: str, params: dict[str, Any], duration_s: float) -> list[Artifact]:
    """Pushes every downloadable output to R2, then writes the manifest that
    makes the result cacheable.

    Uses project_layout.iter_run_artifacts so the UI's local download list and
    the cloud one are generated from the same definition.

    The write order is the load-bearing part. Outputs live at a fixed
    ``outputs/{project}/{run}/{step}/`` prefix so the gateway can find them
    without a job id, which means a re-run *replaces* what is there. So:

        1. delete the old manifest   -- the folder is now officially incomplete
        2. delete the old artifacts  -- no orphans from a longer previous run
        3. upload the new artifacts
        4. write the new manifest    -- the folder is complete again

    Anything that interrupts steps 1-3 leaves no manifest, so the next submit
    misses the cache and recomputes. The old result is only destroyed after the
    new one has been computed successfully, since this runs after ``run_step``.
    """
    from pipeline import project_layout as pl

    prefix = output_prefix(project_key, run_name, step)
    marker = manifest_key(project_key, run_name, step)

    if storage.configured:
        storage.delete_keys([marker])
        replaced = storage.delete_prefix(f"{prefix}/")
        if replaced:
            log.info("Replacing %d file(s) from a previous run of this step", replaced)

    artifacts: list[Artifact] = []
    for kind, path in pl.iter_run_artifacts(layout):
        artifact = Artifact(
            key=f"{prefix}/{path.parent.name}/{path.name}",
            name=path.name,
            kind=kind,
            size=path.stat().st_size,
        )
        if storage.configured:
            storage.upload_file(path, artifact.key)
        else:
            artifact.key = str(path)     # Volume path; still addressable from a later job
        artifacts.append(artifact)

    log.info("Collected %d artifact(s)%s", len(artifacts),
             " and uploaded them to R2" if storage.configured else " (R2 disabled, left on the Volume)")

    if storage.configured:
        manifest = Manifest(
            job_id=job_id,
            project_key=project_key,
            run_name=run_name,
            step=step,
            params=params or {},
            artifacts=[a.__dict__ for a in artifacts],
            total_bytes=sum(a.size for a in artifacts),
            duration_s=round(duration_s, 1),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            effective_config=pl.config_snapshot(),
        )
        storage.upload_bytes(manifest.to_json(), marker, "application/json")
        log.info("Wrote %s -- this result is now reusable for identical parameters", marker)

    return artifacts


def _execute(job_id: str, run_name: str, step: str, params: dict[str, Any], project_key: str) -> dict[str, Any]:
    """The body every worker function shares. Runs one step (or the whole
    pipeline) and keeps the job record current throughout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)

    store = JobStore()
    storage = R2Storage()
    handler = JobLogHandler(store, job_id)
    logging.getLogger().addHandler(handler)

    store.set_status(job_id, STATUS_RUNNING, message=STEPS[step].label)
    started = time.time()
    try:
        from pipeline import project_layout as pl

        project_dir = _sync_inputs(storage, project_key, run_name)

        overrides, runtime = split_params(params)
        overrides.setdefault("PROJECT_ROOT", str(project_dir))
        overrides.setdefault("OUTPUT_ROOT", OUTPUT_DIR)
        pl.apply_overrides(overrides)

        layout = pl.get_run_layout(run_name)
        layout.ensure_dirs()
        log.info("Run folder: %s", layout.run_dir)
        log.info("Output folder: %s", layout.output_dir)

        run_step(step, run_name, runtime)
        _write_provenance(layout, job_id, step, params)

        volume.commit()
        artifacts = _upload_outputs(storage, layout, project_key, run_name, step,
                                    job_id, params, time.time() - started)

        store.set_artifacts(job_id, artifacts)
        store.set_status(job_id, STATUS_SUCCEEDED,
                         message=f"{STEPS[step].label} finished in {time.time() - started:.1f}s")
        return {"job_id": job_id, "status": STATUS_SUCCEEDED, "artifacts": [a.__dict__ for a in artifacts]}

    except Exception as exc:                          # noqa: BLE001 - recorded, then re-raised
        detail = traceback.format_exc(limit=12)
        log.error("Job %s failed: %s", job_id, exc)
        store.append_logs(job_id, detail.splitlines())
        store.set_status(job_id, STATUS_FAILED, message=str(exc), error=detail)
        raise
    finally:
        handler.flush()
        logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _assert_gpu_usable() -> None:
    """Fail loudly if a GPU job cannot actually use the GPU.

    gpu_utils.get_xp() is deliberately forgiving: it catches any CuPy problem
    and returns NumPy, which is the right behaviour on a laptop. On a billed T4
    it is the worst possible outcome -- MART runs on the CPU, an order of
    magnitude slower, while the GPU sits idle and charged. A 130M-voxel grid
    turns a few GPU-minutes into hours.

    So the GPU entry point checks first and refuses, with the fix in the
    message, rather than quietly degrading.
    """
    try:
        import cupy as cp

        probe = cp.asarray([1.0, 2.0]) * 2.0
        float(probe.sum())
        # scatter_add is what mart.py actually calls, and it JIT-compiles --
        # a plain array op can succeed while the JIT path still fails.
        import cupyx

        target = cp.zeros(4, dtype=cp.float32)
        cupyx.scatter_add(target, cp.asarray([0, 1]), cp.asarray([1.0, 1.0], dtype=cp.float32))
        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        free, total = cp.cuda.runtime.memGetInfo()
        log.info("GPU ready: %s, %.1f GB free of %.1f GB", device, free / 1e9, total / 1e9)
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        raise RuntimeError(
            f"This is a GPU job but CuPy is unusable ({type(exc).__name__}: {exc}). "
            "The pipeline would silently fall back to NumPy and run MART on the CPU "
            "while the T4 sits idle and billed. Most likely the image lacks the CUDA "
            "toolkit headers that cupyx.scatter_add needs to JIT-compile: install "
            "'cupy-cuda12x[ctk]' (the [ctk] extra), then `modal deploy app_modal.py`."
        ) from exc


@app.function(
    image=image,
    volumes={WORKSPACE: volume},
    secrets=[secret],
    cpu=4.0,
    memory=8192,
    timeout=3600,
    retries=0,      # a failed run must surface, not silently re-bill
)
def run_cpu_job(job_id: str, run_name: str, step: str, params: dict, project_key: str) -> dict:
    """Steps 1-3: 2D image work (OpenCV morphology, scikit-image DoG). CPU-bound
    and single-frame -- a GPU would idle here."""
    return _execute(job_id, run_name, step, params, project_key)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={WORKSPACE: volume},
    secrets=[secret],
    cpu=8.0,
    memory=32768,
    timeout=7200,   # MART over a large voxel grid is the long pole
    retries=0,
)
def run_gpu_job(job_id: str, run_name: str, step: str, params: dict, project_key: str) -> dict:
    """Steps 4-6 and the full pipeline: MART voxel reconstruction and 3D FFT
    cross-correlation on an NVIDIA T4."""
    _assert_gpu_usable()
    return _execute(job_id, run_name, step, params, project_key)


@app.function(image=image, volumes={WORKSPACE: volume}, secrets=[secret], timeout=600)
def inspect_workspace() -> dict:
    """What is currently staged on the Volume -- lets the UI show which runs
    are ready without re-listing R2."""
    projects_dir, output_dir = Path(PROJECTS_DIR), Path(OUTPUT_DIR)
    projects: dict[str, list[str]] = {}
    if projects_dir.is_dir():
        for project in sorted(p for p in projects_dir.iterdir() if p.is_dir()):
            projects[project.name] = sorted(
                r.name for r in project.iterdir() if r.is_dir() and r.name != "Properties"
            )
    outputs = sorted(p.name for p in output_dir.iterdir()) if output_dir.is_dir() else []
    return {"projects": projects, "outputs": outputs}


def _submit(run_name: str, step: str, params: dict, project_key: str, store: JobStore) -> JobRecord:
    """Creates the job record and spawns it on the right hardware."""
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}; expected one of {', '.join(STEPS)}")

    record = JobRecord(
        job_id=new_job_id(), run_name=run_name, step=step, status=STATUS_QUEUED,
        backend="modal", project_key=project_key, params=params or {},
        message=f"queued: {STEPS[step].label}",
    )
    store.create(record)

    fn = run_gpu_job if STEPS[step].needs_gpu else run_cpu_job
    handle = fn.spawn(record.job_id, run_name, step, params or {}, project_key)
    store.update(record.job_id, modal_call_id=handle.object_id)
    record.modal_call_id = handle.object_id
    log.info("Spawned %s for job %s (%s)", fn.info.function_name, record.job_id,
             "T4 GPU" if STEPS[step].needs_gpu else "CPU")
    return record


# ---------------------------------------------------------------------------
# FastAPI surface
# ---------------------------------------------------------------------------

def _build_api():
    """Constructed lazily so importing this module without FastAPI installed
    (e.g. from a thin CLI) still works."""
    from fastapi import Depends, FastAPI, HTTPException, Header, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    api = FastAPI(
        title="Tomo-PIV Pipeline API",
        version="1.0.0",
        description="Asynchronous 3D Tomographic PIV processing on NVIDIA T4 GPUs.",
    )
    # The React client / Express gateway are served from a different origin.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("API_ALLOWED_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        """Bearer-token gate. Disabled when API_AUTH_TOKEN is unset so a local
        `modal serve` needs no setup; set the secret before exposing the URL."""
        expected = os.getenv("API_AUTH_TOKEN")
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    class JobRequest(BaseModel):
        run_name: str = Field(..., description="Acquisition run folder, e.g. LI_16072026_7.2mm_09")
        step: str = Field(..., description=f"One of: {', '.join(STEPS)}")
        project_key: str = Field("default", description="Project namespace inside the bucket")
        params: dict[str, Any] = Field(default_factory=dict,
                                       description="Hyperparameter overrides + runtime arguments")

    class PresignRequest(BaseModel):
        project_key: str = "default"
        run_name: str
        filenames: list[str]
        target: str = Field("run", description="'run' for raw frames, 'calibration' for Properties/")

    # `from __future__ import annotations` (top of this file) turns every
    # endpoint annotation into a string, and FastAPI resolves those strings
    # against the endpoint function's *module* globals. These models are
    # defined inside this factory -- so the module still imports on a machine
    # without FastAPI -- which means they must be published to module scope
    # here. Without this, FastAPI cannot resolve `request: JobRequest`, quietly
    # downgrades it to a query parameter, and every POST fails with a 422.
    globals().update(JobRequest=JobRequest, PresignRequest=PresignRequest)

    def store() -> JobStore:
        return JobStore()

    @api.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "tomo-piv", "docs": "/docs", "health": "/health"}

    @api.get("/health")
    def health() -> dict:
        r2_ok, r2_detail = R2Storage().health()
        js = JobStore()
        return {
            "status": "ok",
            "gpu": GPU_TYPE,
            "r2": {"ok": r2_ok, "detail": r2_detail},
            "job_store": js.backend,
            "steps": list(STEPS),
        }

    @api.get("/config/tunables", dependencies=[Depends(require_auth)])
    def tunables() -> dict:
        """Serves the hyperparameter schema and the worker's current defaults,
        so a client can render the whole control panel without hardcoding it."""
        from pipeline import project_layout as pl

        return {
            "tunables": [t.__dict__ for t in pl.TUNABLES],
            "runtime_params": RUNTIME_PARAM_DEFAULTS,
            "defaults": pl.config_snapshot(),
            "steps": [s.__dict__ for s in STEPS.values()],
        }

    @api.post("/jobs", status_code=202, dependencies=[Depends(require_auth)])
    def create_job(request: JobRequest) -> dict:
        try:
            record = _submit(request.run_name, request.step, request.params, request.project_key, store())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"job_id": record.job_id, "status": record.status, "step": record.step,
                "modal_call_id": record.modal_call_id}

    @api.get("/jobs", dependencies=[Depends(require_auth)])
    def list_jobs(limit: int = Query(25, ge=1, le=200)) -> dict:
        return {"jobs": [r.to_dict() for r in store().list_jobs(limit)]}

    @api.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def get_job(job_id: str, include_logs: bool = True) -> dict:
        record = store().get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        data = record.to_dict()
        if not include_logs:
            data["logs"] = []
        data["duration_s"] = round(record.duration_s, 1)
        return data

    @api.get("/jobs/{job_id}/logs", dependencies=[Depends(require_auth)])
    def get_logs(job_id: str, since: int = Query(0, ge=0)) -> dict:
        """Incremental log tail -- the UI polls with the count it already has."""
        record = store().get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return {"job_id": job_id, "status": record.status,
                "next_offset": len(record.logs), "lines": record.logs[since:]}

    @api.get("/jobs/{job_id}/artifacts", dependencies=[Depends(require_auth)])
    def get_artifacts(job_id: str, expires_in: int = Query(3600, ge=60, le=604800)) -> dict:
        record = store().get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        storage = R2Storage()
        artifacts = []
        for raw in record.artifacts:
            item = dict(raw)
            if storage.configured and not item["key"].startswith("/"):
                item["url"] = storage.presigned_get_url(item["key"], expires_in, download_as=item["name"])
            artifacts.append(item)
        return {"job_id": job_id, "status": record.status, "artifacts": artifacts}

    @api.delete("/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def cancel_job(job_id: str) -> dict:
        js = store()
        record = js.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        if record.done:
            return {"job_id": job_id, "status": record.status, "detail": "already finished"}
        if record.modal_call_id:
            try:
                modal.FunctionCall.from_id(record.modal_call_id).cancel()
            except Exception as exc:                  # noqa: BLE001
                raise HTTPException(status_code=502, detail=f"could not cancel: {exc}") from exc
        js.set_status(job_id, STATUS_CANCELLED, message="cancelled by user")
        return {"job_id": job_id, "status": STATUS_CANCELLED}

    @api.post("/uploads/presign", dependencies=[Depends(require_auth)])
    def presign_uploads(request: PresignRequest) -> dict:
        """Presigned PUT links so the browser streams raw frames straight into
        R2. Keeping multi-GB uploads out of the API container is the whole
        point -- it stays a small, fast, stateless dispatcher."""
        storage = R2Storage()
        if not storage.configured:
            raise HTTPException(status_code=503, detail="R2 is not configured on the server")
        base = (calibration_prefix(request.project_key) if request.target == "calibration"
                else input_prefix(request.project_key, request.run_name))
        return {
            "bucket": storage.settings.bucket,
            "uploads": [
                {"filename": name, "key": f"{base}/{name}", "url": storage.presigned_put_url(f"{base}/{name}")}
                for name in request.filenames
            ],
        }

    @api.get("/runs", dependencies=[Depends(require_auth)])
    def list_available_runs(project_key: str = "default") -> dict:
        """Runs discoverable in R2 (uploaded) -- what the UI's run picker offers."""
        storage = R2Storage()
        if not storage.configured:
            return {"project_key": project_key, "runs": [], "detail": "R2 not configured"}
        prefix = f"inputs/{project_key}/"
        runs: dict[str, int] = {}
        for obj in storage.list_objects(prefix):
            rel = obj["Key"][len(prefix):]
            head = rel.split("/", 1)[0]
            if head and head != "Properties":
                runs[head] = runs.get(head, 0) + 1
        return {"project_key": project_key,
                "runs": [{"run_name": k, "file_count": v} for k, v in sorted(runs.items())]}

    return api


@app.function(image=image, secrets=[secret], min_containers=0, timeout=600)
@modal.asgi_app()
def fastapi_app():
    """Public REST surface. Deployed URL looks like
    https://<workspace>--tomo-piv-fastapi-app.modal.run"""
    return _build_api()


# ---------------------------------------------------------------------------
# CLI helpers:  modal run app_modal.py --run-name ... --step ...
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(run_name: str = "LI_16072026_7.2mm_09", step: str = "full",
         project_key: str = "default", params_json: str = "{}", wait: bool = True) -> None:
    """Fire one job from the terminal, e.g.::

        modal run app_modal.py --run-name LI_16072026_7.2mm_09 --step steps45 \
            --params-json '{"NOISE_FLOOR_PERCENTILE": 97.0, "mart_iterations": 20}'
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = json.loads(params_json)
    if step not in STEPS:
        raise SystemExit(f"unknown step {step!r}; expected one of {', '.join(STEPS)}")

    fn = run_gpu_job if STEPS[step].needs_gpu else run_cpu_job
    job_id = new_job_id()
    print(f"job_id={job_id}  step={step}  gpu={'T4' if STEPS[step].needs_gpu else 'none'}")

    if wait:
        result = fn.remote(job_id, run_name, step, params, project_key)
        print(json.dumps(result, indent=2, default=str))
    else:
        handle = fn.spawn(job_id, run_name, step, params, project_key)
        print(f"spawned; modal_call_id={handle.object_id}")
