# Tomo-PIV — 3D Tomographic Particle Image Velocimetry

Reconstructs a 3D particle volume from four calibrated camera views and measures
the velocity field inside it, with a web console and GPU-accelerated cloud
execution.

The numerical core is a MART (Multiplicative Algebraic Reconstruction Technique)
tomographic solver followed by two-pass 3D FFT cross-correlation, following
standard Tomo-PIV practice (Elsinga et al. 2006). Around it sit a React control
console, a Node API gateway, and a Modal worker that runs the heavy steps on an
NVIDIA T4.

```
 client/                      server/
 ┌────────────┐    /api    ┌──────────────┐        ┌────────────────────┐
 │   React    │──────────▶ │   Express    │──────▶ │   app_modal.py     │
 │   :5173    │            │    :5000     │        │   FastAPI + T4     │
 └────────────┘            └──────────────┘        │   runs pipeline/   │
       ▲                      │       ▲            └────────────────────┘
       │                      │       └── Redis (job records) ◀──┤
       └─── presigned URL ────┴─────▶ Cloudflare R2 ◀─────────────┘
```

## Pipeline

| Step | Does | Hardware |
|---|---|---|
| 1 | Decode raw `.ims` camera frames → 8-bit previews | CPU |
| 2 | Column-FPN correction, rail masking, top-hat, noise floor, gamma | CPU |
| 3 | Difference-of-Gaussians detection of droplet centroids | CPU |
| 4–5 | Build the voxel grid and reconstruct the volume with MART | **T4 GPU** |
| 6 | Two-pass (coarse + window-offset fine) 3D cross-correlation | **T4 GPU** |

Outputs are `.vtk` vector fields (open in ParaView), `.tif` volume stacks,
`.csv` vector tables and `.jpg` previews.

## Quick start

Prerequisites: **Node ≥ 18** and **Python ≥ 3.10**.

```bash
git clone <your-repo-url> && cd 3d_piv
npm run setup        # installs Node deps for both apps + Python requirements
npm run dev          # gateway on :5000 and console on :5173, one Ctrl-C stops both
```

Open <http://localhost:5173>. The three badges top-left show whether Modal, R2
and Redis are reachable — hover for detail.

Running the pipeline for real needs cloud credentials:

```bash
cp server/.env.example server/.env     # then fill in R2 keys + MODAL_API_URL
cd server && modal deploy app_modal.py
```

Full setup for Cloudflare R2, Upstash Redis and Modal is in
**[server/DEPLOYMENT.md](server/DEPLOYMENT.md)**.

### Other entry points

```bash
npm run dev:server                    # gateway only
npm run dev:client                    # console only
npm run build                         # production client bundle -> client/dist
npm run deploy:worker                 # modal deploy
cd server && npm test                 # gateway tests (cache, storage, routes)
cd client && npm test                 # viewer stage/frame model

cd server && python -m pipeline.run_pipeline --run <run_name>   # no web tier
```

## Layout

```
3d_piv/
├── client/                  React + Vite console  (:5173)
│   ├── src/
│   │   ├── App.jsx          the whole console
│   │   ├── imageViewer.js   image viewer: folder scope, wrap, zoom, URLs
│   │   ├── main.jsx
│   │   └── index.css        Tailwind entry
│   ├── test/                node:test suite for the viewer model
│   ├── index.html
│   └── vite.config.js       React + Tailwind plugins, /api dev proxy
│
├── server/                  backend — Node gateway + Python worker
│   ├── src/                 Express API gateway  (:5000)
│   │   ├── index.js         entry point
│   │   ├── app.js           express assembly, error middleware
│   │   ├── config.js        env, constants
│   │   ├── r2.js            Cloudflare R2 client + URL signing
│   │   ├── jobs.js          job store reader (Redis → Modal) + progress
│   │   ├── manifest.js      R2 result cache: lookup, param diff, signing
│   │   ├── storage.js       bucket browse · usage/quota · deletion
│   │   ├── params.js        friendly names → pipeline constants + validation
│   │   └── routes/          health · uploads · jobs · storage
│   │
│   ├── test/                node:test suite (no framework, no network)
│   │
│   ├── pipeline/            core numerical code (Steps 1-6)
│   │   ├── project_layout.py   paths + tunable hyperparameters
│   │   ├── steps.py            step registry, shared by every front end
│   │   ├── mart.py             MART tomographic reconstruction
│   │   ├── cross_correlate.py  3D FFT cross-correlation
│   │   ├── calibration.py      DaVis pinhole model
│   │   ├── voxel_grid.py       measurement-volume estimation
│   │   ├── enhance.py          Step 2 image enhancement
│   │   ├── detect_drops.py     Step 3 droplet detection
│   │   └── ims_io.py           raw .ims decode/encode
│   │
│   ├── app_modal.py         Modal T4 GPU worker + FastAPI
│   ├── cloud_io.py          R2 storage + Redis job store
│   ├── requirements.txt     Python dependencies
│   └── DEPLOYMENT.md        Modal / R2 / Redis setup
│
├── scripts/dev.mjs          zero-dependency two-process dev launcher
└── package.json             root scripts (setup, dev, build)
```

## Where the data lives

Acquisition data and results are **not in this repository** — a single raw run is
several GB. They are expected one level above the repo:

```
<parent>/
├── 3d_piv/                        this repository
├── Project_Imaging_.../           DaVis project: Properties/Calibration + run folders
└── output/                        generated volumes, vectors, previews
```

`server/pipeline/project_layout.py` walks up from its own location to find that
parent (see `_derive_repo_root`). Point it somewhere else with `TOMOPIV_ROOT`,
or set `TOMOPIV_PROJECT_ROOT` / `TOMOPIV_OUTPUT_ROOT` individually. For cloud
runs the worker pulls inputs from Cloudflare R2 instead, so no local data is
needed at all.

## Configuration

Hyperparameters are defined once, in `server/pipeline/project_layout.py`
(`TUNABLES`), and flow outward from there — `GET /config/tunables` serves that
table so the client renders its controls without hardcoding them.

`project_layout.apply_overrides()` rewrites those module-level constants at
runtime, which is what lets the web UI drive a run without editing any `.py`
file. It rebinds each constant across every already-imported `pipeline.*` module
(and rebuilds frozen function defaults), because the algorithm modules bind
their values at import time.

The main knobs:

| Parameter | Meaning |
|---|---|
| `has_t2` | Double-pulse (8 raw frames, t1/t2 pair) vs single-pulse (4 frames) |
| `noise_floor_pct` | Percentile clipped to black in Step 2 — the most sensitive setting; MART only places a particle all four cameras see |
| `z_range` | Depth of the reconstruction volume in mm (X/Y are auto-detected) |
| `mart_iterations` | MART passes; sharper particles, cost grows ~linearly |
| `window_xy` | Coarse interrogation window; must exceed the real displacement |
| `dt_us` | Inter-pulse time; only absolute speed scales with it |
| `crop_*_px` | Edge margins zeroed before Steps 2-3, to exclude rig hardware |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Gateway, R2, Redis and Modal status |
| `POST` | `/api/upload-url` | Presigned PUT URLs for raw frames |
| `POST` | `/api/jobs/submit` | Start a step; echoes `resolved_params` |
| `GET` | `/api/jobs` | Recent jobs |
| `GET` | `/api/jobs/:id/status` | Status + derived progress + log tail |
| `GET` | `/api/jobs/:id/logs?since=N` | Incremental log tail |
| `GET` | `/api/jobs/:id/download` | Presigned download URLs, grouped by kind |
| `DELETE` | `/api/jobs/:id` | Cancel a running job |
| `GET` | `/api/runs` | Runs already uploaded to R2 |
| `GET` | `/api/storage/list` | Bucket contents, grouped by folder and by run |
| `GET` | `/api/storage/usage` | Bytes stored against the 10 GB quota |
| `GET` | `/api/storage/download` | Signed URL for one stored object (`&redirect=1` / `&inline=1`) |
| `DELETE` | `/api/storage/delete` | Remove a file, a folder, or a whole run |

`step` is one of `step1`, `step2`, `step3`, `steps45`, `step6`, `full`.

```bash
curl -X POST http://localhost:5000/api/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{"step":"full","run_name":"LI_16072026_7.2mm_09","has_t2":true,
       "noise_floor_pct":99.3,"mart_iterations":15,"z_range":[-6,6],
       "dt_us":50,"window_xy":32}'
```

## Design notes

**Job polling reads Redis directly.** The Python worker writes every record to
`tomopiv:jobs:<job_id>`, so watching a running job costs no GPU container time.
Modal is contacted only to start or cancel work. Without `REDIS_URL` it falls
back to proxying Modal, so it works either way.

**Results are cached in R2, not in Redis.** Every successful run writes a
`manifest.json` beside its outputs at `outputs/{project}/{run}/{step}/`,
recording the exact hyperparameters that produced them. A submit reads that one
small object first; if the parameters match and every file it lists is still in
the bucket, the gateway signs those files and returns `from_cache: true` without
starting a container. R2 is the right home for this because job records expire
after a week while results do not — a Redis-backed cache would forget answers
that are still sitting in the bucket. Send `force: true` to recompute anyway.

The manifest is written last and deleted first, so an interrupted re-run leaves
no manifest and simply misses; it never leaves one describing files that are
half old and half new. Two things it deliberately does not do: it does not
reason about upstream steps (the console sends the full parameter set with every
submit, so any change misses every step — conservative, occasionally wasteful),
and it does not detect re-uploaded input frames under an existing run name.

**Storage is a first-class surface, because the free tier is 10 GB and one raw
run is ~700 MB.** The console's Cloud storage tab lists the bucket, and the
gateway refuses dataset uploads past 9 GB with a 507 rather than letting a
multi-GB PUT fail unexplained. Calibration uploads stay allowed — one ~8 KB file
that every run depends on. Sizes are decimal GB throughout, matching how
Cloudflare meters storage; binary units would disagree with the invoice by 7%.

**Images are viewed where the files are.** Every file in the Cloud storage tab
carries Open · Download · Delete, and Open — offered for images only — is a
folder-scoped lightbox: scroll or arrow through the images beside it, ctrl-scroll
or −/+ to zoom to 8x, drag to pan. A folder in this bucket is already a
meaningful set (`converted_jpg/` is every raw frame of a run, `drop/` every
detection overlay), so no cross-folder model is needed to make "the next image"
mean something.

Its `<img src>` points at `/api/storage/download?key=…&inline=1`, which 302s to
a freshly signed R2 URL. Signing is offline, so this costs no R2 round trip —
and unlike a batch of links held in state, a viewer left open past the one-hour
expiry keeps working rather than filling with broken images.

**Large files never touch the gateway.** Uploads and downloads are presigned
URLs; the browser talks to R2 directly. The gateway stays small and stateless.

**Progress is derived, not reported.** The pipeline emits no percentage, so the
gateway reads the `=== Step N` banners and `MART iter n/N` lines out of the job
log — during reconstruction the bar advances per MART iteration. The UI labels
it as an estimate rather than implying precision it does not have.

**`window_xy_fine` is derived from `window_xy`.** The second correlation pass
subdivides the first, so it must be smaller. `project_layout.py` ships a
coarse/fine pair (32 / 128) inherited from two different project presets, which
is backwards — so sending only `window_xy` yields a fine window at half that
rather than letting the stale default apply. An explicit `window_xy_fine` wins,
and fine > coarse is rejected with a 400.

**Two security checks worth keeping.** `?key=` on the download route is
validated against that job's own artifact list, or the endpoint would presign any
object in the bucket for anyone who can guess a job id. Upload filenames reject
`..`, leading `/` and backslashes, since a signed PUT is write access.

## Further reading

* **[CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md)** — architecture, execution
  flow, and a deep dive on each algorithm
* **[server/DEPLOYMENT.md](server/DEPLOYMENT.md)** — Modal, Cloudflare R2 and
  Upstash Redis deployment

## References

* Elsinga, Scarano, Wieneke & van Oudheusden (2006), *Tomographic particle image
  velocimetry*, Experiments in Fluids 41:933–947
* Keane & Adrian (1992), *Theory of cross-correlation analysis of PIV images*
* Scarano & Riethmuller (2000), iterative multigrid window-offset correlation
* Westerweel & Scarano (2005), universal outlier detection
