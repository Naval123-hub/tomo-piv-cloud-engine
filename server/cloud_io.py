"""Cloud plumbing for the GPU worker (`app_modal.py`): Cloudflare R2 object
storage, job-status persistence and the job record schema.

It lives outside `pipeline/` on purpose -- nothing here knows anything about
Tomo-PIV, and nothing in `pipeline/` imports it, so the numerical pipeline
stays runnable with zero cloud dependencies installed.

Every third-party import (boto3, redis, modal) is deferred to the point of
use. Importing this module on a machine with none of them installed is fine
and gives you the local-only code paths; `R2Settings.configured` and
`JobStore.backend` tell you which paths are actually live.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger(__name__)

#: How long a finished job record is kept before expiring out of Redis.
JOB_TTL_SECONDS = int(os.getenv("TOMOPIV_JOB_TTL_SECONDS", str(7 * 24 * 3600)))
#: Presigned download links are short-lived by default; the UI re-mints them
#: on demand rather than caching a long-lived URL anywhere.
DEFAULT_URL_TTL_SECONDS = int(os.getenv("TOMOPIV_URL_TTL_SECONDS", "3600"))
#: Cap on how many log lines a job record retains (Redis values stay small).
MAX_LOG_LINES = 2000

#: Content types that matter for "click the link and get a usable file".
#: .vtk and .npy are deliberately octet-stream so browsers download rather
#: than render them; ParaView does not care about the MIME type.
_CONTENT_TYPES = {
    ".vtk": "application/octet-stream",
    ".npy": "application/octet-stream",
    ".ims": "application/octet-stream",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".csv": "text/csv",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pvd": "application/xml",
}


def content_type_for(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _CONTENT_TYPES:
        return _CONTENT_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


# ---------------------------------------------------------------------------
# Cloudflare R2 (S3-compatible object storage)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class R2Settings:
    """Credentials for the R2 bucket. Read from the environment so the same
    code runs locally (via a .env file) and inside a Modal container (via a
    Modal Secret) with no branching."""
    endpoint: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    bucket: str | None = None
    #: Optional public base URL (an r2.dev subdomain or a custom domain) for a
    #: bucket exposed publicly. When set, artifact URLs are permanent public
    #: links instead of expiring presigned ones.
    public_base_url: str | None = None

    @classmethod
    def from_env(cls) -> "R2Settings":
        return cls(
            endpoint=os.getenv("CLOUDFLARE_R2_ENDPOINT"),
            access_key=os.getenv("CLOUDFLARE_R2_ACCESS_KEY"),
            secret_key=os.getenv("CLOUDFLARE_R2_SECRET_KEY"),
            bucket=os.getenv("CLOUDFLARE_R2_BUCKET"),
            public_base_url=os.getenv("CLOUDFLARE_R2_PUBLIC_URL"),
        )

    @property
    def configured(self) -> bool:
        return all((self.endpoint, self.access_key, self.secret_key, self.bucket))

    def missing(self) -> list[str]:
        names = {
            "CLOUDFLARE_R2_ENDPOINT": self.endpoint,
            "CLOUDFLARE_R2_ACCESS_KEY": self.access_key,
            "CLOUDFLARE_R2_SECRET_KEY": self.secret_key,
            "CLOUDFLARE_R2_BUCKET": self.bucket,
        }
        return [k for k, v in names.items() if not v]


class R2Storage:
    """Thin boto3 wrapper for the handful of S3 operations this app needs.

    R2 quirks that this class encodes so callers never have to think about
    them: the region is always the literal string ``auto``, signature v4 is
    required for presigned URLs to validate, and the addressing style must be
    path-style because R2 does not serve virtual-host-style bucket subdomains
    on the S3 API endpoint.
    """

    def __init__(self, settings: R2Settings | None = None) -> None:
        self.settings = settings or R2Settings.from_env()
        self._client = None
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.settings.configured

    def _require(self) -> None:
        if not self.configured:
            raise RuntimeError(
                "Cloudflare R2 is not configured; missing environment variable(s): "
                + ", ".join(self.settings.missing())
            )

    @property
    def client(self):
        self._require()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import boto3
                    from botocore.config import Config

                    self._client = boto3.client(
                        "s3",
                        endpoint_url=self.settings.endpoint,
                        aws_access_key_id=self.settings.access_key,
                        aws_secret_access_key=self.settings.secret_key,
                        region_name="auto",
                        config=Config(
                            signature_version="s3v4",
                            s3={"addressing_style": "path"},
                            retries={"max_attempts": 5, "mode": "standard"},
                        ),
                    )
        return self._client

    def health(self) -> tuple[bool, str]:
        """(ok, detail) -- used by /health and by the UI's connection badge."""
        if not self.configured:
            return False, "not configured: " + ", ".join(self.settings.missing())
        try:
            self.client.head_bucket(Bucket=self.settings.bucket)
            return True, f"bucket '{self.settings.bucket}' reachable"
        except Exception as exc:                      # noqa: BLE001 - surfaced verbatim to the UI
            return False, f"{type(exc).__name__}: {exc}"

    # -- upload -------------------------------------------------------------
    def upload_file(self, local_path: str | Path, key: str) -> str:
        local_path = Path(local_path)
        self.client.upload_file(
            str(local_path), self.settings.bucket, key,
            ExtraArgs={"ContentType": content_type_for(local_path)},
        )
        log.info("R2 upload: %s -> s3://%s/%s (%d bytes)",
                 local_path.name, self.settings.bucket, key, local_path.stat().st_size)
        return key

    def upload_bytes(self, data: bytes, key: str, content_type: str | None = None) -> str:
        self.client.put_object(
            Bucket=self.settings.bucket, Key=key, Body=data,
            ContentType=content_type or content_type_for(key),
        )
        return key

    def upload_dir(self, local_dir: str | Path, key_prefix: str,
                   patterns: Iterable[str] = ("*",)) -> list[str]:
        """Uploads matching files under `local_dir`, preserving relative paths."""
        local_dir = Path(local_dir)
        keys: list[str] = []
        for pattern in patterns:
            for path in sorted(local_dir.rglob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(local_dir).as_posix()
                keys.append(self.upload_file(path, f"{key_prefix.rstrip('/')}/{rel}"))
        return keys

    # -- download -----------------------------------------------------------
    def download_file(self, key: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.settings.bucket, key, str(local_path))
        return local_path

    def download_prefix(self, prefix: str, dest_dir: str | Path, skip_existing: bool = True) -> int:
        """Mirrors everything under `prefix` into `dest_dir`. Returns the file
        count. `skip_existing` makes re-running a job on a warm container cheap:
        raw .ims frames are large and never change once uploaded."""
        dest_dir = Path(dest_dir)
        count = 0
        for obj in self.list_objects(prefix):
            rel = obj["Key"][len(prefix):].lstrip("/")
            if not rel:
                continue
            target = dest_dir / rel
            if skip_existing and target.exists() and target.stat().st_size == obj["Size"]:
                continue
            self.download_file(obj["Key"], target)
            count += 1
        return count

    def object_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.settings.bucket, Key=key)["Body"].read()

    # -- delete -------------------------------------------------------------
    def delete_keys(self, keys: Iterable[str]) -> int:
        """Removes objects in batches of 1000 (the S3 DeleteObjects limit).
        Returns how many were actually deleted."""
        keys = [k for k in keys if k]
        deleted = 0
        for start in range(0, len(keys), 1000):
            batch = keys[start:start + 1000]
            response = self.client.delete_objects(
                Bucket=self.settings.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
            for err in response.get("Errors", []):
                log.warning("R2 delete failed for %s: %s", err.get("Key"), err.get("Message"))
            deleted += len(batch) - len(response.get("Errors", []))
        return deleted

    def delete_prefix(self, prefix: str) -> int:
        """Removes everything under `prefix`. Refuses an empty prefix, because
        that is a request to empty the bucket by typo."""
        if not prefix or not prefix.strip("/"):
            raise ValueError("refusing to delete an empty prefix")
        return self.delete_keys(obj["Key"] for obj in self.list_objects(prefix))

    # -- listing / URLs -----------------------------------------------------
    def list_objects(self, prefix: str) -> Iterator[dict[str, Any]]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.settings.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj

    def presigned_get_url(self, key: str, expires_in: int = DEFAULT_URL_TTL_SECONDS,
                          download_as: str | None = None) -> str:
        """A time-limited download link. If the bucket has a public base URL
        configured we hand back the permanent public link instead -- cheaper,
        cacheable, and it survives being pasted into a lab notebook."""
        if self.settings.public_base_url:
            return f"{self.settings.public_base_url.rstrip('/')}/{key.lstrip('/')}"
        params: dict[str, Any] = {"Bucket": self.settings.bucket, "Key": key}
        if download_as:
            params["ResponseContentDisposition"] = f'attachment; filename="{download_as}"'
        return self.client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_in)

    def presigned_put_url(self, key: str, expires_in: int = DEFAULT_URL_TTL_SECONDS) -> str:
        """Upload link handed to the browser so raw frames stream straight to
        R2 instead of through the API container."""
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.settings.bucket, "Key": key, "ContentType": content_type_for(key)},
            ExpiresIn=expires_in,
        )


# ---------------------------------------------------------------------------
# Key conventions -- one place, so worker and UI always agree
# ---------------------------------------------------------------------------

def input_prefix(project_key: str, run_name: str) -> str:
    """Raw acquisition frames for one run."""
    return f"inputs/{project_key}/{run_name}"


def calibration_prefix(project_key: str) -> str:
    """Per-project calibration (shared by every run in that project)."""
    return f"inputs/{project_key}/Properties"


#: Name of the completion marker written at the end of every successful run.
#: Its presence is what makes a result cacheable; see :class:`Manifest`.
MANIFEST_NAME = "manifest.json"


def output_prefix(project_key: str, run_name: str, step: str) -> str:
    """Results of one step of one run.

    Deliberately *not* keyed by job id. A given (project, run, step,
    hyperparameters) tuple has exactly one correct answer, so the gateway can
    look here before dispatching and serve an earlier result instead of buying
    another GPU minute. The trade-off is real and chosen: re-running a step
    replaces the previous outputs rather than accumulating a new folder per
    job. The manifest records which parameters produced what is currently
    there, so "replaced" never means "silently mismatched".
    """
    return f"outputs/{project_key}/{run_name}/{step}"


def manifest_key(project_key: str, run_name: str, step: str) -> str:
    return f"{output_prefix(project_key, run_name, step)}/{MANIFEST_NAME}"


# ---------------------------------------------------------------------------
# Job records
# ---------------------------------------------------------------------------

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED)


@dataclass
class Artifact:
    key: str            # object key in R2, or an absolute path for local jobs
    name: str           # file name shown in the UI
    kind: str           # category label from project_layout.ARTIFACT_KINDS
    size: int = 0
    url: str | None = None   # filled in on demand; presigned links expire


@dataclass
class Manifest:
    """The completion marker for one (project, run, step) output folder.

    Two rules make it trustworthy as a cache key, and both are about ordering
    rather than content:

    * it is written **last**, after every artifact has been uploaded, so its
      presence means the folder is complete;
    * it is deleted **first**, before a re-run replaces those artifacts, so a
      run that dies halfway leaves no manifest and simply misses the cache.

    Anything else -- a crashed worker, a cancelled job, a half-finished
    upload -- therefore reads as "not cached" rather than as a stale hit.

    ``params`` holds the resolved hyperparameter dictionary exactly as the
    worker received it. The gateway compares its own freshly built dictionary
    against this one structurally; there is no hash, because a hash would have
    to be computed identically in Python and JavaScript and float formatting
    does not survive that round trip (``json.dumps(1.0)`` is ``"1.0"``,
    ``JSON.stringify(1.0)`` is ``"1"``).
    """
    schema: int = 1
    job_id: str = ""
    project_key: str = ""
    run_name: str = ""
    step: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0
    duration_s: float = 0.0
    completed_at: str = ""
    effective_config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), indent=2, default=str).encode("utf-8")


@dataclass
class JobRecord:
    job_id: str
    run_name: str
    step: str
    status: str = STATUS_QUEUED
    backend: str = "modal"           # "modal" | "local"
    project_key: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    message: str = ""
    modal_call_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def duration_s(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - (self.started_at or self.created_at))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        known = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def new_job_id() -> str:
    """Sortable-ish, human-legible job id: 20260820-143355-ab12cd34."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


class JobStore:
    """Job status/log persistence with three interchangeable backends, tried
    in order:

    1. **Redis / Upstash** (``REDIS_URL`` or ``UPSTASH_REDIS_URL``) -- the
       production choice. Shared by every API container and every worker, and
       expires finished jobs automatically via TTL.
    2. **modal.Dict** -- a distributed dict Modal provides natively. Used when
       Redis is not configured but we are running on Modal, so the app is
       still correct across containers without provisioning anything.
    3. **In-process dict** -- local development, e.g. running the FastAPI app
       directly without Redis. Correct for a single process only, which is
       exactly that case.

    All three are wrapped so callers never branch on which one is live; read
    :attr:`backend` if you want to display it.
    """

    def __init__(self, namespace: str = "tomopiv:jobs", redis_url: str | None = None) -> None:
        self.namespace = namespace
        self._redis = None
        self._modal_dict = None
        self._memory: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.backend = "memory"

        url = redis_url or os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")
        if url:
            try:
                import redis

                # decode_responses so values come back as str, not bytes.
                # Upstash requires TLS -- use a rediss:// URL there.
                client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
                client.ping()
                self._redis = client
                self.backend = "redis"
                log.info("JobStore backend: Redis (%s)", url.split("@")[-1])
                return
            except Exception as exc:                  # noqa: BLE001 - degrade, never crash the API
                log.warning("Redis unavailable (%s); falling back.", exc)

        if os.getenv("MODAL_TASK_ID") or os.getenv("MODAL_IS_REMOTE"):
            try:
                import modal

                self._modal_dict = modal.Dict.from_name("tomo-piv-jobs", create_if_missing=True)
                self.backend = "modal.Dict"
                log.info("JobStore backend: modal.Dict")
                return
            except Exception as exc:                  # noqa: BLE001
                log.warning("modal.Dict unavailable (%s); falling back to in-process store.", exc)

        log.info("JobStore backend: in-process dict (single process only)")

    # -- primitives ---------------------------------------------------------
    def _key(self, job_id: str) -> str:
        return f"{self.namespace}:{job_id}"

    def _index_key(self) -> str:
        return f"{self.namespace}:index"

    def _read(self, job_id: str) -> dict[str, Any] | None:
        if self._redis is not None:
            raw = self._redis.get(self._key(job_id))
            return json.loads(raw) if raw else None
        if self._modal_dict is not None:
            return self._modal_dict.get(job_id, None)
        with self._lock:
            return self._memory.get(job_id)

    def _write(self, record: JobRecord) -> None:
        record.updated_at = time.time()
        data = record.to_dict()
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.set(self._key(record.job_id), json.dumps(data), ex=JOB_TTL_SECONDS)
            pipe.zadd(self._index_key(), {record.job_id: record.created_at})
            # Trim the index so it cannot grow without bound as jobs expire.
            pipe.zremrangebyrank(self._index_key(), 0, -501)
            pipe.execute()
            return
        if self._modal_dict is not None:
            self._modal_dict[record.job_id] = data
            return
        with self._lock:
            self._memory[record.job_id] = data

    # -- public API ---------------------------------------------------------
    def create(self, record: JobRecord) -> JobRecord:
        self._write(record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        data = self._read(job_id)
        return JobRecord.from_dict(data) if data else None

    def update(self, job_id: str, **changes: Any) -> JobRecord | None:
        record = self.get(job_id)
        if record is None:
            return None
        for key, value in changes.items():
            setattr(record, key, value)
        self._write(record)
        return record

    def set_status(self, job_id: str, status: str, message: str = "", error: str | None = None) -> None:
        record = self.get(job_id)
        if record is None:
            return
        record.status = status
        if message:
            record.message = message
        if error is not None:
            record.error = error
        if status == STATUS_RUNNING and record.started_at is None:
            record.started_at = time.time()
        if status in TERMINAL_STATUSES:
            record.finished_at = time.time()
        self._write(record)

    def append_logs(self, job_id: str, lines: Iterable[str]) -> None:
        lines = [str(line) for line in lines]
        if not lines:
            return
        record = self.get(job_id)
        if record is None:
            return
        record.logs.extend(lines)
        if len(record.logs) > MAX_LOG_LINES:
            dropped = len(record.logs) - MAX_LOG_LINES
            record.logs = [f"... {dropped} earlier line(s) trimmed ..."] + record.logs[-MAX_LOG_LINES:]
        self._write(record)

    def set_artifacts(self, job_id: str, artifacts: Iterable[Artifact]) -> None:
        self.update(job_id, artifacts=[asdict(a) for a in artifacts])

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        if self._redis is not None:
            ids = self._redis.zrevrange(self._index_key(), 0, max(0, limit - 1))
            return [r for r in (self.get(i) for i in ids) if r is not None]
        if self._modal_dict is not None:
            records = [JobRecord.from_dict(v) for _, v in self._modal_dict.items()]
        else:
            with self._lock:
                records = [JobRecord.from_dict(v) for v in self._memory.values()]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]


class JobLogHandler(logging.Handler):
    """Pipes the pipeline's own `logging` output into a job record.

    The pipeline modules already log every meaningful step (`Step2: ... ->
    ...`, `MART reconstruction took ...s`), so attaching this handler gives a
    live progress feed with no changes to the algorithm code at all.

    Records are batched -- one store write per `flush_interval` seconds or per
    `batch_size` lines -- because a MART run emits far too many lines to write
    each one to Redis individually.
    """

    def __init__(self, store: JobStore, job_id: str, batch_size: int = 20, flush_interval: float = 2.0) -> None:
        super().__init__(level=logging.INFO)
        self.store = store
        self.job_id = job_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:                            # noqa: BLE001 - a broken log line must not kill a job
            return
        with self._lock:
            self._buffer.append(line)
            due = (len(self._buffer) >= self.batch_size
                   or time.monotonic() - self._last_flush >= self.flush_interval)
            if not due:
                return
            pending, self._buffer = self._buffer, []
            self._last_flush = time.monotonic()
        self._safe_append(pending)

    def flush(self) -> None:
        with self._lock:
            pending, self._buffer = self._buffer, []
            self._last_flush = time.monotonic()
        self._safe_append(pending)

    def _safe_append(self, lines: list[str]) -> None:
        if not lines:
            return
        try:
            self.store.append_logs(self.job_id, lines)
        except Exception as exc:                     # noqa: BLE001
            # Losing log lines is acceptable; failing the compute job is not.
            log.debug("Could not persist job logs: %s", exc)


def load_dotenv_if_present(path: str | Path = ".env") -> dict[str, str]:
    """Minimal .env loader so local development needs no extra dependency.
    Existing environment variables always win, matching python-dotenv."""
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
