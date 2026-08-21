# Tomo-PIV SaaS — running and deploying

Everything in this document is run from `3d_piv/server/`, which holds both the
Python GPU worker and the Node gateway. The numerical pipeline (`pipeline/`) is
unchanged; these layers were added around it:

| Layer | File | Role |
|---|---|---|
| Core numerics | `pipeline/` | Steps 1-6: decode, enhance, detect, MART, cross-correlate |
| Step registry | `pipeline/steps.py` | One definition of the six steps, shared by every front end |
| GPU worker + API | `app_modal.py` | Modal T4 containers + FastAPI; async job dispatch |
| Cloud plumbing | `cloud_io.py` | Cloudflare R2 storage, Redis/Upstash job store, log capture |
| API gateway (Node) | `src/` | Express: presigned R2 URLs, Redis job reads, Modal dispatch |
| Web UI (JS) | `../client/` | React + Vite console — see [../README.md](../README.md) |

Repository layout:

```
Naval/
├── 3d_piv/
│   ├── client/          React + Vite console
│   └── server/          <- you are here
│       ├── pipeline/    core numerical code
│       ├── app_modal.py Modal GPU worker + FastAPI
│       ├── cloud_io.py  R2 + job store
│       └── src/         Node Express gateway
├── Project_Imaging_*/   acquisition data (not in git)
└── output/              results
```

The React console talks to the Express gateway, which dispatches to this
`app_modal.py` API and reads job status back out of Redis.

Configuration flows through `project_layout.apply_overrides()`, which rewrites the pipeline's
module-level constants at runtime — so the UI drives a run without anyone editing a `.py` file.

---

## 1. Local development

### 1.1 Install

```bash
cd "path/to/Naval/3d_piv/server"
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

GPU (optional, Steps 4–6). Only on a machine with CUDA 12 + an NVIDIA card:

```bash
pip install cupy-cuda12x
```

Without CuPy the pipeline transparently falls back to NumPy — `pipeline/gpu_utils.py`
probes the device at import and logs which backend it picked.

### 1.2 Run the web stack

From the repository root (`3d_piv/`):

```bash
npm run setup     # Node deps for both apps + this requirements.txt
npm run dev       # Express gateway :5000 + React console :5173
```

The console needs `MODAL_API_URL` and the R2 keys in `server/.env` before it can
actually dispatch a job — see sections 2 to 4.

### 1.3 Command line

The pipeline runs standalone, no web tier and no cloud account, against data on
this machine:

```bash
python -m pipeline.run_pipeline --run LI_16072026_7.2mm_09
python -m pipeline.reconstruct_mart --run LI_16072026_7.2mm_09 --iterations 15
python -m pipeline.cross_correlate --run LI_16072026_7.2mm_09 --window-xy 32 --dt-us 50
```

---

## 2. Cloudflare R2

1. Cloudflare dashboard → **R2** → *Create bucket* (e.g. `tomo-piv`).
2. **Manage R2 API Tokens** → *Create API token* → Object Read & Write. Copy the
   Access Key ID, Secret Access Key, and the **S3 API endpoint**
   (`https://<account_id>.r2.cloudflarestorage.com` — not the public bucket URL).
3. `cp .env.example .env` and fill in the four `CLOUDFLARE_R2_*` values.

Bucket layout the app expects:

```
inputs/<project_key>/Properties/Calibration/Calibration.xml    # per project
inputs/<project_key>/<run_name>/Frame0-0.ims                   # raw frames
                                Frame0-1.ims
                                Frame0-decoder.xml  …
outputs/<project_key>/<run_name>/<step>/manifest.json           # cache marker
                                        Volumes/vector_b000_t1_t2.vtk
                                        Volumes/volume_b000_t1.tif
                                        Drop/Camera1_t1_detections.csv …
```

Results are keyed by **step, not by job id**, which is what makes them
reusable: before dispatching, the gateway reads `manifest.json` from that
folder and — if the hyperparameters recorded in it match the ones being
submitted, and every file it lists is still in the bucket — returns signed
download links without starting a GPU container at all.

The trade-off is that re-running a step *replaces* its previous outputs rather
than accumulating a folder per job. The manifest is written last and deleted
first, so a run that fails partway leaves no manifest and simply misses the
cache; it never leaves a manifest describing files that are half old and half
new. Pass `force: true` to `POST /api/jobs/submit` to recompute regardless.

Upload a run (one-off, using any S3 client):

```bash
aws s3 cp "../../Project_Imaging_260709_174128/LI_16072026_7.2mm_09" \
  "s3://tomo-piv/inputs/default/LI_16072026_7.2mm_09" --recursive \
  --endpoint-url "$CLOUDFLARE_R2_ENDPOINT"

aws s3 cp "../../Project_Imaging_260709_174128/Properties" \
  "s3://tomo-piv/inputs/default/Properties" --recursive \
  --endpoint-url "$CLOUDFLARE_R2_ENDPOINT"
```

Or from the app: `POST /uploads/presign` returns presigned PUT URLs so a browser
streams frames straight into R2 without passing through the API container.

---

## 3. Redis / Upstash (job status)

Create a database at <https://upstash.com> → copy the **TLS** connection string
(`rediss://…`) into `REDIS_URL`.

This is optional. `JobStore` degrades in a defined order:

1. **Redis** — shared across all API and worker containers, with TTL expiry. Use this in production.
2. **`modal.Dict`** — used automatically on Modal when `REDIS_URL` is unset. Correct across
   containers, no provisioning needed.
3. **In-process dict** — local development, e.g. running the FastAPI app directly
   without Redis.

`GET /health` reports which one is live.

---

## 4. Deploy the GPU worker + API to Modal

```bash
pip install modal
modal setup                     # one-time browser auth
```

Create the secret (names must match `.env`):

```bash
modal secret create tomo-piv-secrets \
  CLOUDFLARE_R2_ENDPOINT="https://<account_id>.r2.cloudflarestorage.com" \
  CLOUDFLARE_R2_ACCESS_KEY="…" \
  CLOUDFLARE_R2_SECRET_KEY="…" \
  CLOUDFLARE_R2_BUCKET="tomo-piv" \
  REDIS_URL="rediss://…" \
  API_AUTH_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
```

Develop against a live, hot-reloading deployment:

```bash
modal serve app_modal.py
```

Deploy for real:

```bash
modal deploy app_modal.py
```

Modal prints the API URL, e.g.
`https://<workspace>--tomo-piv-fastapi-app.modal.run`. Check it:

```bash
curl https://<workspace>--tomo-piv-fastapi-app.modal.run/health
```

Interactive API docs are at `/docs`.

Fire a job from the terminal without the UI:

```bash
modal run app_modal.py --run-name LI_16072026_7.2mm_09 --step steps45 \
  --params-json '{"NOISE_FLOOR_PERCENTILE": 97.0, "mart_iterations": 20}'
```

### Compute placement

| Step | Hardware | Why |
|---|---|---|
| 1–3 (decode, enhance, detect) | CPU, 4 cores | OpenCV morphology and scikit-image DoG on 2D frames. A T4 would idle. |
| 4–5 (MART), 6 (correlation), full | **T4 GPU**, 8 cores / 32 GB | Voxel reconstruction and 3D FFT correlation — what `gpu_utils.py` reaches for CuPy for. |

Raw frames are cached on a Modal Volume (`tomo-piv-workspace`) and skipped on re-download when
size matches, so a Step-6-only rerun does not re-fetch gigabytes.

---

## 5. Point the front end at the cloud

Set `MODAL_API_URL` (the URL `modal deploy` printed) and `API_AUTH_TOKEN` in
`server/.env`, then restart the gateway. `GET /api/health` reports the whole
chain — gateway, R2, Redis and Modal — and the console shows it as three badges.

### Deploying the client and gateway

The React client is a static bundle; the gateway is a small stateless Node
service. Any host works — the only requirements are that the gateway holds the
R2 and Modal secrets, and that the client knows where the gateway is.

```bash
cd 3d_piv/client && npm run build      # -> client/dist, deploy as static files
```

Set `VITE_API_URL` at build time to the deployed gateway URL, and `CORS_ORIGIN`
on the gateway to the client's origin. Then set `API_ALLOWED_ORIGINS` in the
Modal secret to the gateway's origin and `modal deploy app_modal.py` again.

Keep the GPU work on Modal: the gateway never does heavy compute, so it can run
on the smallest instance available.

---

## 6. API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | API, GPU type, R2 reachability, job-store backend |
| `GET` | `/config/tunables` | Hyperparameter schema + worker defaults (drives the UI) |
| `POST` | `/jobs` | Submit `{run_name, step, project_key, params}` → `202 {job_id}` |
| `GET` | `/jobs` | Recent jobs |
| `GET` | `/jobs/{id}` | Status, params, artifacts, logs |
| `GET` | `/jobs/{id}/logs?since=N` | Incremental log tail |
| `GET` | `/jobs/{id}/artifacts` | Presigned R2 download URLs |
| `DELETE` | `/jobs/{id}` | Cancel a running job |
| `POST` | `/uploads/presign` | Presigned PUT URLs for raw frame upload |
| `GET` | `/runs` | Runs already uploaded to R2 |

`step` is one of `step1`, `step2`, `step3`, `steps45`, `step6`, `full`.

Example:

```bash
curl -X POST https://<workspace>--tomo-piv-fastapi-app.modal.run/jobs \
  -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"run_name":"LI_16072026_7.2mm_09","step":"full",
       "params":{"NOISE_FLOOR_PERCENTILE":99.3,"CROSS_CORR_DT_US":50.0,
                 "mart_iterations":15}}'
```

---

## 7. Hyperparameters

The sidebar is generated from `project_layout.TUNABLES`, so adding a knob there makes it appear
in the UI, in the API payload and in each job's provenance record with no further edits.
Every job writes `volumes/job_<job_id>_config.json` recording the full effective configuration
next to the outputs it produced.

Pre-flight validation splits into two kinds:

* **Errors** (run buttons disabled) — configurations the pipeline rejects outright: a burst
  instant queued for reconstruction that Steps 1–3 never processed, or `Z min ≥ Z max`.
* **Advisories** (informational) — settings that will run but are probably not intended, e.g.
  a fine correlation window larger than the coarse one.

> **Note on the shipped defaults:** `CROSS_CORR_WINDOW_XY = 32` with
> `CROSS_CORR_WINDOW_XY_FINE = 128` are carried over unchanged from the original
> `project_layout.py`, and they belong to *different* project presets — 32/50 µs is the laser
> `_09` preset, while the 128 fine window pairs with the commented-out 256 coarse window for
> LED_PIV. The fine pass is meant to subdivide the coarse one (about half). The UI flags this
> as an advisory rather than silently changing a validated number.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `R2 not configured` in `/health` | The four `CLOUDFLARE_R2_*` variables are not all present in the Modal secret. |
| `No input data for run …` | Raw frames were never uploaded to `inputs/<project_key>/<run_name>/`. |
| Reconstruction produces an empty volume | `NOISE_FLOOR_PERCENTILE` too high — MART needs a particle in **all four** cameras. Drop toward 97.0 for dim exposures. |
| Step 6 returns near-zero vectors | `CROSS_CORR_WINDOW_XY` is smaller than the real displacement, so the peak is clipped. Measure the true 2D shift first. |
| `burst_index=N was not processed by Steps 1-3` | Add N to *Burst instants for Steps 1-3* and re-run Step 2. |
| `modal.exception.NotFoundError: secret` | Run the `modal secret create tomo-piv-secrets …` command in §4. |
| Full-depth "ray" artifacts in the volume | Set `MAX_PARTICLE_PX` to ~8; large 2D blobs smear across the whole depth at this aperture. |

---

## 9. Where things run from

| Task | Run from | Command |
|---|---|---|
| Pipeline CLI | `3d_piv/server/` | `python -m pipeline.run_pipeline --run <name>` |
| Deploy GPU worker | `3d_piv/server/` | `modal deploy app_modal.py` |
| Node gateway | `3d_piv/server/` | `npm run dev` |
| React console | `3d_piv/client/` | `npm run dev` |

`modal deploy` must be run from `3d_piv/server/` — the image is built with
`add_local_python_source("pipeline")` and `("cloud_io")`, which resolves those
modules from the working directory.

The pipeline finds the acquisition data by walking up from
`pipeline/project_layout.py` to the repository root (see `_derive_repo_root`),
so `Project_Imaging_*/` and `output/` stay where they are at the top level.
Override with `TOMOPIV_ROOT` if your data lives elsewhere.
