/**
 * Tomo-PIV development console.
 *
 * Talks only to the Express gateway (../../server), which holds the R2
 * credentials and the Modal token. No secret ever reaches the browser.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ZOOM_STEPS,
  downloadUrl,
  extOf,
  imagesIn,
  inlineUrl,
  isImage,
  stepZoom,
  wrapIndex,
} from "./imageViewer.js";

// Empty in dev: vite.config.js proxies /api to the gateway, so the browser
// stays on one origin and there is no preflight. Set VITE_API_URL for a
// production build where the client and API are served from different hosts.
const API = import.meta.env?.VITE_API_URL || "";
const POLL_MS = 2000;

const STEPS = [
  { key: "step1", label: "Step 1", sub: "Decode raw frames", hint: ".ims -> linear 8-bit JPEG previews" },
  { key: "step2", label: "Step 2", sub: "Enhance", hint: "Column-FPN, rail mask, top-hat, noise floor, gamma" },
  { key: "step3", label: "Step 3", sub: "Detect droplets", hint: "DoG band-pass + MAD threshold peak finding" },
  { key: "steps45", label: "Steps 4-5", sub: "MART reconstruction", hint: "Voxel grid + MART on the T4", gpu: true },
  { key: "step6", label: "Step 6", sub: "Cross-correlation", hint: "Coarse + window-offset fine 3D FFT", gpu: true },
];

const STATUS_STYLE = {
  queued: { dot: "bg-amber-400", text: "text-amber-300", label: "Queued" },
  running: { dot: "bg-sky-400 animate-pulse", text: "text-sky-300", label: "Running" },
  succeeded: { dot: "bg-emerald-400", text: "text-emerald-300", label: "Succeeded" },
  failed: { dot: "bg-rose-500", text: "text-rose-300", label: "Failed" },
  cancelled: { dot: "bg-neutral-500", text: "text-neutral-400", label: "Cancelled" },
};

/* Extension -> badge colour, so the download grid is scannable at a glance. */
const FILE_BADGE = {
  vtk: "bg-violet-500/15 text-violet-300 ring-violet-500/30",
  tif: "bg-sky-500/15 text-sky-300 ring-sky-500/30",
  tiff: "bg-sky-500/15 text-sky-300 ring-sky-500/30",
  csv: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
  jpg: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  jpeg: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  ims: "bg-fuchsia-500/15 text-fuchsia-300 ring-fuchsia-500/30",
  xml: "bg-teal-500/15 text-teal-300 ring-teal-500/30",
  json: "bg-neutral-500/15 text-neutral-300 ring-neutral-500/30",
};


/**
 * Decimal units throughout, not binary.
 *
 * Cloudflare meters and bills R2 in decimal GB, so a console that showed 9.31
 * GiB next to a 10 GB quota would disagree with the invoice by 7% -- exactly
 * the margin that matters when you are deciding whether one more run fits.
 */
const formatBytes = (bytes = 0) => {
  if (bytes < 1000) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ${units[unit]}`;
};

async function callApi(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

/* ------------------------------------------------------------------ *
 * Small presentational pieces
 * ------------------------------------------------------------------ */

function Field({ label, hint, children, value }) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-medium text-neutral-200">{label}</span>
        {value !== undefined && (
          <span className="font-mono text-xs tabular-nums text-sky-300">{value}</span>
        )}
      </div>
      {children}
      {hint && <p className="mt-1 text-[11px] leading-snug text-neutral-500">{hint}</p>}
    </label>
  );
}

function Slider({ value, onChange, min, max, step = 1 }) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-neutral-700
                 accent-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40"
    />
  );
}

function NumberInput({ value, onChange, min, max, suffix }) {
  return (
    <div className="relative">
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          // Clamp here as well as on the server: the spinner and keyboard entry
          // both let a value out of range, and a 400 half a second after a GPU
          // submit is a worse way to learn that than the field refusing it.
          const next = Number(e.target.value);
          if (e.target.value === "") return onChange(0);
          if (!Number.isFinite(next)) return undefined;
          return onChange(Math.min(max, Math.max(min, Math.round(next))));
        }}
        className="w-full rounded-lg border border-neutral-700 bg-neutral-950 py-1.5 pl-2.5 pr-7
                   font-mono text-xs tabular-nums text-neutral-100
                   focus:border-sky-500 focus:outline-none"
      />
      {suffix && (
        <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-neutral-600">
          {suffix}
        </span>
      )}
    </div>
  );
}

function StatusPill({ status }) {
  const style = STATUS_STYLE[status] || STATUS_STYLE.queued;
  return (
    <span className={`inline-flex items-center gap-2 text-sm font-medium ${style.text}`}>
      <span className={`h-2 w-2 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  );
}

function HealthBadge({ ok, name, detail }) {
  return (
    <span
      title={detail}
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ring-1 ${
        ok
          ? "bg-emerald-500/10 text-emerald-300 ring-emerald-500/25"
          : "bg-rose-500/10 text-rose-300 ring-rose-500/25"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-rose-400"}`} />
      {name}
    </span>
  );
}

function FileBadge({ name }) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ring-1 ${
        FILE_BADGE[extOf(name)] || FILE_BADGE.json
      }`}
    >
      {extOf(name)}
    </span>
  );
}

/* ------------------------------------------------------------------ *
 * Storage quota
 * ------------------------------------------------------------------ */

/** Which of the three states the bucket is in. The colour is a status colour,
 *  never the only signal -- every state also carries text. */
function quotaLevel(usage) {
  if (!usage) return "ok";
  if (usage.bytes >= usage.limit_bytes) return "critical";
  return usage.over_threshold ? "warning" : "ok";
}

const METER_FILL = { ok: "bg-sky-500", warning: "bg-amber-400", critical: "bg-rose-500" };

function UsageMeter({ usage }) {
  if (!usage) return null;
  const level = quotaLevel(usage);

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-medium text-neutral-200">Storage</span>
        <span className="font-mono text-xs tabular-nums text-neutral-500">
          <span className="text-neutral-100">{usage.gb.toFixed(2)} GB</span> / {usage.limit_gb} GB
        </span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-neutral-800">
        <div
          className={`h-full rounded-full transition-[width] duration-500 ease-out ${METER_FILL[level]}`}
          style={{ width: `${Math.max(1.5, Math.min(100, usage.percent ?? 0))}%` }}
        />
      </div>
      <p className="mt-1 text-[11px] text-neutral-500">
        {usage.object_count.toLocaleString()} objects · uploads pause at {usage.warn_gb} GB
      </p>
    </div>
  );
}

/**
 * The banner is the part that has to be impossible to miss: past the
 * threshold, every dataset upload is refused by the gateway, and without this
 * the only symptom would be a 507 in the network tab.
 */
function QuotaBanner({ usage, onManage }) {
  if (!usage?.over_threshold) return null;
  const critical = quotaLevel(usage) === "critical";

  return (
    <div
      role="alert"
      className={`flex flex-wrap items-center gap-3 rounded-xl border px-4 py-3 ${
        critical
          ? "border-rose-500/40 bg-rose-500/10"
          : "border-amber-500/40 bg-amber-500/10"
      }`}
    >
      <span aria-hidden="true" className="text-base leading-none">⚠️</span>
      <p className={`min-w-0 flex-1 text-sm ${critical ? "text-rose-100" : "text-amber-100"}`}>
        {usage.message}
      </p>
      <button
        onClick={onManage}
        className={`shrink-0 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
          critical
            ? "bg-rose-500/20 text-rose-100 hover:bg-rose-500/30"
            : "bg-amber-500/20 text-amber-100 hover:bg-amber-500/30"
        }`}
      >
        Manage storage
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Image lightbox
 * ------------------------------------------------------------------ */

/**
 * Full-screen viewer over the images of one folder.
 *
 * Scoped to the folder it was opened from, because a folder here is already a
 * meaningful set: `converted_jpg/` is every raw frame of a run, `drop/` is
 * every detection overlay. Wheel steps through them; ctrl/cmd-wheel zooms,
 * which is the split every image viewer uses and the only one that leaves
 * plain scroll free for its obvious job.
 */
function ImageLightbox({ folder, images, index, onIndex, onClose }) {
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [failed, setFailed] = useState(false);
  const drag = useRef(null);
  const wheelAt = useRef(0);

  const file = images[index];

  // A different picture invalidates the pan/zoom that was framing the old one.
  useEffect(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setFailed(false);
  }, [file?.key]);

  const go = useCallback(
    (delta) => onIndex(wrapIndex(index, delta, images.length)),
    [index, images.length, onIndex]
  );

  // Fetch the neighbours so stepping does not flash an empty frame.
  useEffect(() => {
    for (const delta of [-1, 1]) {
      const neighbour = images[wrapIndex(index, delta, images.length)];
      if (neighbour && neighbour.key !== file?.key) new Image().src = inlineUrl(neighbour.key);
    }
  }, [images, index, file?.key]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "+" || e.key === "=") setZoom((z) => stepZoom(z, 1));
      else if (e.key === "-") setZoom((z) => stepZoom(z, -1));
      else if (e.key === "0") setZoom(1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    // The page behind must not scroll while the overlay is up.
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [go, onClose]);

  const onWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      setZoom((z) => {
        const next = Math.min(ZOOM_STEPS.at(-1), Math.max(1, z * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
        if (next === 1) setOffset({ x: 0, y: 0 });
        return next;
      });
      return;
    }
    // A trackpad emits a burst of small deltas per flick; without a floor and a
    // cooldown one gesture would skip half the folder.
    const now = Date.now();
    if (Math.abs(e.deltaY) < 4 || now - wheelAt.current < 140) return;
    wheelAt.current = now;
    go(e.deltaY > 0 ? 1 : -1);
  };

  const startDrag = (e) => {
    if (zoom === 1) return;
    drag.current = { x: e.clientX - offset.x, y: e.clientY - offset.y };
  };
  const onDrag = (e) => {
    if (!drag.current) return;
    setOffset({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y });
  };
  const endDrag = () => {
    drag.current = null;
  };

  if (!file) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${file.name} — image ${index + 1} of ${images.length}`}
      className="fixed inset-0 z-50 flex flex-col bg-neutral-950/95 backdrop-blur-sm"
      onWheel={onWheel}
    >
      {/* -- tag bar -- */}
      <header className="flex flex-wrap items-center gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <FileBadge name={file.name} />
            <span className="truncate font-mono text-sm text-neutral-100">{file.name}</span>
            <span className="shrink-0 rounded-full bg-neutral-800 px-2 py-0.5 font-mono text-[11px] tabular-nums text-neutral-300">
              {index + 1} / {images.length}
            </span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[11px] text-neutral-500">
            <span className="truncate font-mono">{folder.prefix}/</span>
            <span>{formatBytes(file.size)}</span>
            {file.last_modified && <span>{file.last_modified.slice(0, 19).replace("T", " ")} UTC</span>}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <button
            onClick={() => setZoom((z) => stepZoom(z, -1))}
            disabled={zoom <= 1}
            aria-label="Zoom out"
            className="h-8 w-8 rounded-lg border border-neutral-800 text-neutral-300 transition
                       hover:border-sky-500/50 hover:text-sky-300 disabled:opacity-30"
          >
            &minus;
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setOffset({ x: 0, y: 0 });
            }}
            className="h-8 min-w-14 rounded-lg border border-neutral-800 px-2 font-mono text-[11px]
                       tabular-nums text-neutral-300 transition hover:border-sky-500/50 hover:text-sky-300"
          >
            {zoom.toFixed(1)}&times;
          </button>
          <button
            onClick={() => setZoom((z) => stepZoom(z, 1))}
            disabled={zoom >= ZOOM_STEPS.at(-1)}
            aria-label="Zoom in"
            className="h-8 w-8 rounded-lg border border-neutral-800 text-neutral-300 transition
                       hover:border-sky-500/50 hover:text-sky-300 disabled:opacity-30"
          >
            +
          </button>

          <a
            href={downloadUrl(file.key)}
            className="ml-2 rounded-lg border border-neutral-800 px-3 py-1.5 text-xs text-neutral-300
                       transition hover:border-sky-500/50 hover:text-sky-300"
          >
            Download
          </a>
          <button
            onClick={onClose}
            aria-label="Close viewer"
            className="ml-1 h-8 w-8 rounded-lg border border-neutral-800 text-neutral-300 transition
                       hover:border-rose-500/50 hover:text-rose-300"
          >
            &times;
          </button>
        </div>
      </header>

      {/* -- image -- */}
      <div
        className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden"
        onMouseDown={startDrag}
        onMouseMove={onDrag}
        onMouseUp={endDrag}
        onMouseLeave={endDrag}
        style={{ cursor: zoom > 1 ? (drag.current ? "grabbing" : "grab") : "default" }}
      >
        {failed ? (
          <p className="px-6 text-center text-sm text-amber-300">
            This image could not be loaded from R2.
          </p>
        ) : (
          <img
            key={file.key}
            src={inlineUrl(file.key)}
            alt={file.name}
            draggable={false}
            onError={() => setFailed(true)}
            className="max-h-full max-w-full select-none object-contain"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              transformOrigin: "center",
              transition: drag.current ? "none" : "transform 120ms ease-out",
            }}
          />
        )}

        {images.length > 1 && (
          <>
            <button
              onClick={() => go(-1)}
              aria-label="Previous image"
              className="absolute left-3 top-1/2 -translate-y-1/2 rounded-full bg-neutral-900/80 px-3 py-4
                         text-neutral-300 ring-1 ring-neutral-800 transition hover:text-sky-300"
            >
              &lsaquo;
            </button>
            <button
              onClick={() => go(1)}
              aria-label="Next image"
              className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-neutral-900/80 px-3 py-4
                         text-neutral-300 ring-1 ring-neutral-800 transition hover:text-sky-300"
            >
              &rsaquo;
            </button>
          </>
        )}
      </div>

      {/* -- filmstrip + hint -- */}
      <footer className="border-t border-neutral-800 px-4 py-3">
        {images.length > 1 && (
          <div className="mb-2 flex gap-1.5 overflow-x-auto pb-1">
            {images.map((each, i) => (
              <button
                key={each.key}
                onClick={() => onIndex(i)}
                title={each.name}
                className={`shrink-0 rounded px-2 py-1 font-mono text-[10px] transition ${
                  i === index
                    ? "bg-sky-500/15 text-sky-200 ring-1 ring-sky-500/40"
                    : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                {each.name.replace(/\.[^.]+$/, "")}
              </button>
            ))}
          </div>
        )}
        <p className="text-center text-[11px] text-neutral-600">
          Scroll or &larr; &rarr; for the next image in this folder · ctrl/&#8984;+scroll or &minus;/+ to
          zoom · drag to pan when zoomed · Esc to close
        </p>
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Cloud storage manager
 * ------------------------------------------------------------------ */

function FolderCard({ folder, expanded, onToggle, onDelete, onOpen, filter, busy }) {
  const files = filter
    ? folder.files.filter((file) => file.name.toLowerCase().includes(filter))
    : folder.files;
  if (filter && files.length === 0) return null;

  const open = expanded || Boolean(filter);
  // Indexed against the *unfiltered* folder: the viewer shows everything next
  // to the image you opened, not just what happened to match the search box.
  const viewable = imagesIn(folder);
  const positionOf = new Map(viewable.map((file, i) => [file.key, i]));

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-950">
      <div className="flex items-center gap-3 px-3 py-2.5">
        <button
          onClick={onToggle}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
          aria-expanded={open}
        >
          <span className="w-3 shrink-0 text-neutral-600">{open ? "▾" : "▸"}</span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13px] font-medium text-neutral-200">
              {folder.label || folder.prefix}
            </span>
            <span className="block truncate font-mono text-[10px] text-neutral-600">
              {folder.prefix}/
            </span>
          </span>
        </button>

        <span className="shrink-0 text-right">
          <span className="block font-mono text-xs tabular-nums text-neutral-300">
            {formatBytes(folder.bytes)}
          </span>
          <span className="block text-[10px] text-neutral-600">
            {folder.object_count} file{folder.object_count === 1 ? "" : "s"}
          </span>
        </span>

        {viewable.length > 0 && (
          <button
            onClick={() => onOpen(folder, 0)}
            className="shrink-0 rounded-lg border border-neutral-800 px-2.5 py-1 text-[11px] text-neutral-300
                       transition hover:border-sky-500/50 hover:text-sky-300"
          >
            View {viewable.length} image{viewable.length === 1 ? "" : "s"}
          </button>
        )}

        <button
          onClick={() => onDelete({ prefix: folder.prefix }, `folder ${folder.prefix}/`, folder.bytes)}
          disabled={busy}
          className="shrink-0 rounded-lg border border-neutral-800 px-2.5 py-1 text-[11px] text-neutral-400
                     transition hover:border-rose-500/50 hover:text-rose-300 disabled:opacity-40"
        >
          Delete folder
        </button>
      </div>

      {open && (
        <ul className="border-t border-neutral-800/70">
          {files.map((file) => (
            <li
              key={file.key}
              className="flex items-center gap-2.5 border-b border-neutral-900 px-3 py-1.5 last:border-b-0"
            >
              <FileBadge name={file.name} />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-mono text-[11px] text-neutral-300" title={file.key}>
                  {file.name}
                </span>
                {file.manifest && (
                  <span className="text-[10px] text-emerald-400/80">
                    cache manifest — deleting it forces a recompute
                  </span>
                )}
              </span>
              <span className="shrink-0 font-mono text-[11px] tabular-nums text-neutral-500">
                {formatBytes(file.size)}
              </span>

              <span className="flex shrink-0 items-center gap-2.5 text-[11px]">
                {isImage(file.name) ? (
                  <button
                    onClick={() => onOpen(folder, positionOf.get(file.key) ?? 0)}
                    className="text-neutral-400 transition hover:text-sky-300"
                  >
                    Open
                  </button>
                ) : (
                  <span className="text-neutral-800" aria-hidden="true">
                    Open
                  </span>
                )}
                <a
                  href={downloadUrl(file.key)}
                  className="text-neutral-400 transition hover:text-sky-300"
                >
                  Download
                </a>
                <button
                  onClick={() => onDelete({ key: file.key }, file.name, file.size)}
                  disabled={busy}
                  className="text-neutral-600 transition hover:text-rose-300 disabled:opacity-40"
                >
                  Delete
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function StorageManager({
  data,
  busy,
  filter,
  onFilter,
  onRefresh,
  onDelete,
  onOpen,
  expanded,
  onToggle,
}) {
  if (!data) {
    return (
      <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
        <p className="text-xs text-neutral-500">
          {busy ? "Listing the bucket…" : "Nothing listed yet."}
        </p>
      </section>
    );
  }

  const needle = filter.trim().toLowerCase();

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-neutral-200">Runs</h2>
            <p className="mt-0.5 text-xs text-neutral-500">
              Deleting a run removes its raw frames <em>and</em> every stored result, so the next
              submit is a real recompute rather than a cache hit on orphaned outputs.
            </p>
          </div>
          <span className="font-mono text-[11px] text-neutral-500">
            {data.object_count.toLocaleString()} objects · {formatBytes(data.total_bytes)}
          </span>
        </div>

        {data.runs.length === 0 ? (
          <p className="mt-3 text-xs text-neutral-500">No runs uploaded yet.</p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-130 text-left">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-neutral-600">
                  <th className="pb-2 font-semibold">Run</th>
                  <th className="pb-2 text-right font-semibold">Inputs</th>
                  <th className="pb-2 text-right font-semibold">Outputs</th>
                  <th className="pb-2 text-right font-semibold">Total</th>
                  <th className="pb-2" />
                </tr>
              </thead>
              <tbody>
                {data.runs.map((run) => (
                  <tr key={`${run.project_key}/${run.run_name}`} className="border-t border-neutral-800/70">
                    <td className="py-2 pr-3">
                      <span className="block truncate font-mono text-[11px] text-neutral-200">
                        {run.run_name}
                      </span>
                      <span className="text-[10px] text-neutral-600">{run.project_key}</span>
                    </td>
                    <td className="py-2 text-right font-mono text-[11px] tabular-nums text-neutral-400">
                      {formatBytes(run.input_bytes)}
                    </td>
                    <td className="py-2 text-right font-mono text-[11px] tabular-nums text-neutral-400">
                      {formatBytes(run.output_bytes)}
                    </td>
                    <td className="py-2 text-right font-mono text-[11px] tabular-nums text-neutral-200">
                      {formatBytes(run.bytes)}
                    </td>
                    <td className="py-2 pl-3 text-right">
                      <button
                        onClick={() =>
                          onDelete(
                            { run_name: run.run_name, project_key: run.project_key },
                            `run ${run.run_name} (inputs + outputs)`,
                            run.bytes
                          )
                        }
                        disabled={busy}
                        className="rounded-lg border border-neutral-800 px-2.5 py-1 text-[11px] text-neutral-400
                                   transition hover:border-rose-500/50 hover:text-rose-300 disabled:opacity-40"
                      >
                        Delete run
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {data.legacy?.object_count > 0 && (
        <section className="rounded-2xl border border-amber-500/25 bg-amber-500/5 p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-amber-100">Legacy outputs</h2>
              <p className="mt-0.5 max-w-2xl text-xs text-amber-200/70">
                {data.legacy.object_count.toLocaleString()} file(s) across{" "}
                {data.legacy.prefixes.length} job folder(s), written under the old{" "}
                <span className="font-mono">outputs/&#123;run&#125;/&#123;job_id&#125;/</span> layout.
                Results are now keyed by step so they can be reused, which means these can never
                produce a cache hit — they only occupy quota.
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-3">
              <span className="font-mono text-sm tabular-nums text-amber-100">
                {formatBytes(data.legacy.bytes)}
              </span>
              <button
                onClick={() =>
                  onDelete(
                    { prefixes: data.legacy.prefixes.map((entry) => entry.prefix) },
                    `all ${data.legacy.object_count} legacy output file(s)`,
                    data.legacy.bytes
                  )
                }
                disabled={busy}
                className="rounded-lg bg-amber-500/20 px-3 py-1.5 text-xs font-semibold text-amber-100
                           transition hover:bg-amber-500/30 disabled:opacity-40"
              >
                Delete all
              </button>
            </div>
          </div>
        </section>
      )}

      <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-neutral-200">Files</h2>
          <div className="flex items-center gap-2">
            <input
              type="search"
              value={filter}
              onChange={(e) => onFilter(e.target.value)}
              placeholder="Filter by file name…"
              className="w-52 rounded-lg border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-[11px]
                         text-neutral-200 placeholder:text-neutral-600 focus:border-sky-500 focus:outline-none"
            />
            <button
              onClick={onRefresh}
              disabled={busy}
              className="rounded-lg border border-neutral-800 px-2.5 py-1.5 text-[11px] text-neutral-300
                         transition hover:border-sky-500/50 hover:text-sky-300 disabled:opacity-40"
            >
              {busy ? "Working…" : "Refresh"}
            </button>
          </div>
        </div>

        {data.truncated && (
          <p className="mt-3 text-[11px] text-amber-300">
            Listing was capped — some objects are not shown, and the totals above are lower bounds.
          </p>
        )}

        <div className="mt-4 space-y-6">
          {data.sections.map((section) => (
            <div key={section.id}>
              <div className="flex items-baseline justify-between">
                <h3 className="font-mono text-[11px] font-semibold uppercase tracking-wide text-neutral-500">
                  {section.prefix}
                </h3>
                <span className="font-mono text-[11px] text-neutral-600">
                  {section.object_count} files · {formatBytes(section.bytes)}
                </span>
              </div>

              {section.folders.length === 0 ? (
                <p className="mt-2 text-xs text-neutral-600">empty</p>
              ) : (
                <div className="mt-2 space-y-1.5">
                  {section.folders.map((folder) => (
                    <FolderCard
                      key={folder.prefix}
                      folder={folder}
                      expanded={Boolean(expanded[folder.prefix])}
                      onToggle={() => onToggle(folder.prefix)}
                      onDelete={onDelete}
                      onOpen={onOpen}
                      filter={needle}
                      busy={busy}
                    />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Main component
 * ------------------------------------------------------------------ */

export default function App() {
  const [params, setParams] = useState({
    run_name: "LI_16072026_7.2mm_09",
    project_key: "default",
    has_t2: true,
    noise_floor_pct: 99.3,
    mart_iterations: 15,
    z_min: -6,
    z_max: 6,
    dt_us: 50,
    window_xy: 32,
    crop_top_px: 0,
    crop_bottom_px: 0,
    crop_left_px: 0,
    crop_right_px: 0,
  });

  const [tab, setTab] = useState("pipeline");
  const [health, setHealth] = useState(null);
  const [job, setJob] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [submitting, setSubmitting] = useState(null);
  const [uploading, setUploading] = useState(null);

  const [usage, setUsage] = useState(null);
  const [listing, setListing] = useState(null);
  const [storageBusy, setStorageBusy] = useState(false);
  const [storageFilter, setStorageFilter] = useState("");
  const [expanded, setExpanded] = useState({});
  const [viewing, setViewing] = useState(null); // { folder, index }

  const logRef = useRef(null);
  // Records that the storage tab has already tried to load itself. Without it,
  // a failing listing loops: the effect fires, refreshListing() sets busy then
  // clears it on error, listing is still null, and the dependency change fires
  // the effect again forever.
  const listingRequested = useRef(false);

  const set = (key) => (value) => setParams((prev) => ({ ...prev, [key]: value }));

  /* -- health ------------------------------------------------------- */
  useEffect(() => {
    callApi("/api/health").then(setHealth).catch(() => setHealth({ status: "unreachable" }));
  }, []);

  /* -- storage ------------------------------------------------------ */
  const refreshUsage = useCallback(async (fresh = false) => {
    try {
      setUsage(await callApi(`/api/storage/usage${fresh ? "?fresh=1" : ""}`));
    } catch {
      // R2 unconfigured or unreachable -- the health badge already says so, and
      // a missing gauge is better than a scary banner about an unknown number.
      setUsage(null);
    }
  }, []);

  const refreshListing = useCallback(async () => {
    setStorageBusy(true);
    try {
      const data = await callApi("/api/storage/list");
      setListing(data);
      setUsage(data.usage);
    } catch (err) {
      setError(err.message);
    } finally {
      setStorageBusy(false);
    }
  }, []);

  useEffect(() => {
    refreshUsage();
  }, [refreshUsage]);

  useEffect(() => {
    if (tab === "storage" && !listing && !listingRequested.current) {
      listingRequested.current = true;
      refreshListing();
    }
  }, [tab, listing, refreshListing]);

  const deleteFromStorage = useCallback(
    async (body, label, bytes) => {
      const freed = bytes ? ` This frees ${formatBytes(bytes)}.` : "";
      // eslint-disable-next-line no-alert
      if (!window.confirm(`Delete ${label}?${freed}\n\nThis cannot be undone.`)) return;

      setStorageBusy(true);
      setError(null);
      try {
        const result = await callApi("/api/storage/delete", {
          method: "DELETE",
          body: JSON.stringify(body),
        });
        if (result.usage) setUsage(result.usage);
        setNotice(
          result.deleted === 0
            ? result.detail || "Nothing matched — it may already be gone."
            : `Deleted ${result.deleted} object(s), freed ${formatBytes(result.freed_bytes)}.`
        );
      } catch (err) {
        setError(err.message);
      } finally {
        setStorageBusy(false);
      }
      setViewing(null);
      await refreshListing();
    },
    [refreshListing]
  );

  const toggleFolder = useCallback(
    (prefix) => setExpanded((prev) => ({ ...prev, [prefix]: !prev[prefix] })),
    []
  );

  const openViewer = useCallback((folder, index) => setViewing({ folder, index }), []);

  /* -- poll while a job is in flight -------------------------------- */
  useEffect(() => {
    if (!job?.job_id || job.done) return undefined;
    let cancelled = false;

    const tick = async () => {
      try {
        const next = await callApi(`/api/jobs/${job.job_id}/status`);
        if (cancelled) return;
        setJob(next);
        if (next.done && next.status === "succeeded") {
          const payload = await callApi(`/api/jobs/${next.job_id}/download`);
          if (!cancelled) setArtifacts(payload);
          // A finished job just wrote its outputs to the bucket.
          refreshUsage(true);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    const id = setInterval(tick, POLL_MS);
    tick();
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [job?.job_id, job?.done, refreshUsage]);

  /* Keep the log pane pinned to the newest line. */
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log_count]);

  /* -- actions ------------------------------------------------------ */

  /** The request body for one step. Factored out because refreshing a cached
   *  result's expired links means asking the cache the same question again. */
  const payloadFor = useCallback(
    (step, force = false) => ({
      step,
      run_name: params.run_name,
      project_key: params.project_key,
      has_t2: params.has_t2,
      noise_floor_pct: params.noise_floor_pct,
      mart_iterations: params.mart_iterations,
      z_range: [params.z_min, params.z_max],
      dt_us: params.dt_us,
      window_xy: params.window_xy,
      crop_top_px: params.crop_top_px,
      crop_bottom_px: params.crop_bottom_px,
      crop_left_px: params.crop_left_px,
      crop_right_px: params.crop_right_px,
      force,
    }),
    [params]
  );

  const submit = useCallback(
    async (step, { force = false } = {}) => {
      setError(null);
      setNotice(null);
      setArtifacts(null);
      setSubmitting(step);
      try {
        const payload = payloadFor(step, force);
        const response = await callApi("/api/jobs/submit", {
          method: "POST",
          body: JSON.stringify(payload),
        });

        // A cache hit is already finished: the gateway found an identical
        // completed run in R2 and signed its files instead of starting a
        // container. It arrives shaped like a download payload, so the results
        // pane needs no special case -- only the progress pane does, to say
        // where they came from.
        if (response.from_cache) {
          setJob({ ...response, logs: [], log_count: 0, duration_s: 0 });
          setArtifacts(response);
          return;
        }
        setJob({ ...response, progress: 0, logs: [], done: false });
      } catch (err) {
        setError(err.message);
      } finally {
        setSubmitting(null);
      }
    },
    [payloadFor]
  );

  const cancel = useCallback(async () => {
    if (!job?.job_id) return;
    try {
      await callApi(`/api/jobs/${job.job_id}`, { method: "DELETE" });
    } catch (err) {
      setError(err.message);
    }
  }, [job?.job_id]);

  /**
   * Raw frames go browser -> R2 directly on a presigned PUT; they never pass
   * through the gateway. The Content-Type must match what was signed.
   */
  const upload = useCallback(
    /**
     * kind "run"         -> the Frame*-0.ims / Frame*-1.ims / Frame*-decoder.xml
     *                       files of one experiment (24 for a double-pulse run).
     * kind "calibration" -> Calibration.xml, once per project. It has to land at
     *                       Properties/Calibration/Calibration.xml, so the
     *                       subfolder is prefixed here: a file input only ever
     *                       hands us the bare basename, and dropping the file
     *                       one level too high is the failure the worker cannot
     *                       recover from.
     */
    async (fileList, kind = "run") => {
      const files = Array.from(fileList || []);
      if (!files.length) return;
      setError(null);
      setUploading({ done: 0, total: files.length, kind });
      try {
        const keyFor = (file) => (kind === "calibration" ? `Calibration/${file.name}` : file.name);

        const { uploads } = await callApi("/api/upload-url", {
          method: "POST",
          body: JSON.stringify({
            run_name: params.run_name,
            project_key: params.project_key,
            target: kind,
            filenames: files.map(keyFor),
          }),
        });

        const byName = new Map(uploads.map((u) => [u.filename, u]));
        let done = 0;
        for (const file of files) {
          const target = byName.get(keyFor(file));
          if (!target) continue;
          const put = await fetch(target.url, {
            method: "PUT",
            headers: { "Content-Type": target.contentType },
            body: file,
          });
          if (!put.ok) throw new Error(`upload failed for ${file.name} (${put.status})`);
          done += 1;
          setUploading({ done, total: files.length, kind });
        }
        setUploading({ done, total: files.length, kind, finished: true });
      } catch (err) {
        setError(err.message);
        setUploading(null);
      } finally {
        // The bytes landed in R2 without passing through the gateway, so the
        // usage total is only correct after an explicit recount.
        refreshUsage(true);
        setListing(null);
        listingRequested.current = false;   // new objects: re-list on next visit
      }
    },
    [params.run_name, params.project_key, refreshUsage]
  );

  const busy = Boolean(job && !job.done);
  const uploadsBlocked = Boolean(usage?.uploads_blocked);

  const groupedArtifacts = useMemo(() => {
    if (!artifacts?.by_kind) return [];
    return Object.entries(artifacts.by_kind).sort((a, b) => a[0].localeCompare(b[0]));
  }, [artifacts]);

  /* -- render ------------------------------------------------------- */
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-6 p-6 lg:flex-row">
        {/* ============ Sidebar ============ */}
        <aside className="w-full shrink-0 lg:w-80">
          <div className="sticky top-6 space-y-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
            <div>
              <h1 className="text-lg font-semibold tracking-tight">Tomo-PIV</h1>
              <p className="text-xs text-neutral-500">3D tomographic reconstruction &amp; velocimetry</p>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {health ? (
                <>
                  <HealthBadge ok={health.modal?.ok} name="Modal" detail={health.modal?.detail} />
                  <HealthBadge ok={health.r2?.ok} name="R2" detail={health.r2?.detail} />
                  <HealthBadge ok={health.redis?.ok} name="Redis" detail={health.redis?.detail} />
                </>
              ) : (
                <span className="text-[11px] text-neutral-500">checking services…</span>
              )}
            </div>

            <UsageMeter usage={usage} />

            <div className="h-px bg-neutral-800" />

            <Field label="Run name" hint="Acquisition folder uploaded to R2.">
              <input
                type="text"
                value={params.run_name}
                onChange={(e) => set("run_name")(e.target.value)}
                className="mt-1.5 w-full rounded-lg border border-neutral-700 bg-neutral-950 px-3 py-2
                           font-mono text-xs text-neutral-100 focus:border-sky-500 focus:outline-none"
              />
            </Field>

            {/* Pulse mode changes the raw frame layout entirely, not just
                whether t2 exists -- 8 files vs 4. */}
            <Field label="Acquisition mode" hint="Double-pulse gives Step 6 a true t1/t2 pair to correlate.">
              <div className="mt-2 space-y-1.5">
                {[
                  { value: true, label: "Double-pulse (t1 + t2)", note: "8 raw frames" },
                  { value: false, label: "Single-pulse (t1 only)", note: "4 raw frames" },
                ].map((option) => (
                  <label
                    key={String(option.value)}
                    className={`flex cursor-pointer items-center gap-2.5 rounded-lg border px-3 py-2 text-xs transition ${
                      params.has_t2 === option.value
                        ? "border-sky-500/50 bg-sky-500/10 text-sky-200"
                        : "border-neutral-800 text-neutral-400 hover:border-neutral-700"
                    }`}
                  >
                    <input
                      type="radio"
                      name="has_t2"
                      checked={params.has_t2 === option.value}
                      onChange={() => set("has_t2")(option.value)}
                      className="accent-sky-500"
                    />
                    <span className="flex-1">{option.label}</span>
                    <span className="text-[10px] text-neutral-600">{option.note}</span>
                  </label>
                ))}
              </div>
            </Field>

            <Field
              label="Noise floor percentile"
              value={params.noise_floor_pct.toFixed(2)}
              hint="The most sensitive knob: MART only places a particle all 4 cameras see, so signal cut here is lost. ~99.3 bright, ~97.0 dim."
            >
              <Slider value={params.noise_floor_pct} onChange={set("noise_floor_pct")} min={90} max={99.9} step={0.05} />
            </Field>

            <Field
              label="MART iterations"
              value={params.mart_iterations}
              hint="More iterations sharpen particles; cost grows roughly linearly."
            >
              <Slider value={params.mart_iterations} onChange={set("mart_iterations")} min={1} max={50} />
            </Field>

            <Field
              label="Z range (mm)"
              value={`${params.z_min} … ${params.z_max}`}
              hint="Volume depth about the calibration z=0 plane. X/Y bounds are auto-detected from Step 3."
            >
              <Slider value={params.z_min} onChange={set("z_min")} min={-20} max={0} step={0.5} />
              <Slider value={params.z_max} onChange={set("z_max")} min={0} max={20} step={0.5} />
            </Field>

            <Field
              label="Laser dt (µs)"
              value={params.dt_us}
              hint="Inter-pulse time. Direction and displacement are independent of this; only absolute speed scales with it."
            >
              <Slider value={params.dt_us} onChange={set("dt_us")} min={1} max={2000} step={0.5} />
            </Field>

            <Field
              label="Coarse window XY (voxels)"
              value={params.window_xy}
              hint="Must exceed the real bulk displacement or the correlation peak clips. The fine pass is derived at half this."
            >
              <Slider value={params.window_xy} onChange={set("window_xy")} min={8} max={256} step={8} />
            </Field>

            {/* Edge margins are zeroed before Step 2 filtering and Step 3
                detection, so anything inside them can never be enhanced or
                detected -- the way to exclude rig rails and nozzle hardware. */}
            <Field
              label="Edge crop (px)"
              value={
                params.crop_top_px + params.crop_bottom_px + params.crop_left_px + params.crop_right_px ===
                0
                  ? "off"
                  : `${params.crop_top_px}/${params.crop_bottom_px}/${params.crop_left_px}/${params.crop_right_px}`
              }
              hint="Zeroed before Steps 2-3 so rig hardware near the frame edges is never detected as a droplet."
            >
              <div className="mt-2 grid grid-cols-2 gap-2">
                {[
                  ["crop_top_px", "Top"],
                  ["crop_bottom_px", "Bottom"],
                  ["crop_left_px", "Left"],
                  ["crop_right_px", "Right"],
                ].map(([key, label]) => (
                  <label key={key} className="block">
                    <span className="mb-1 block text-[10px] uppercase tracking-wide text-neutral-500">
                      {label}
                    </span>
                    <NumberInput
                      value={params[key]}
                      onChange={set(key)}
                      min={0}
                      max={500}
                      suffix="px"
                    />
                  </label>
                ))}
              </div>
              {params.crop_bottom_px === 0 && (
                <p className="mt-2 text-[11px] leading-snug text-amber-300/90">
                  Note: the pipeline itself ships <span className="font-mono">CROP_BOTTOM_PX = 50</span>{" "}
                  to mask the rig rail. Sending 0 overrides that, so the rail is processed and can be
                  detected as droplets.
                </p>
              )}
            </Field>

            <div className="h-px bg-neutral-800" />

            {/* Two separate controls, because the two inputs have different
                lifetimes: frames are per experiment, calibration is once per
                project and is shared by every run in it. */}
            <div className="space-y-3">
              <div>
                <span className="text-[13px] font-medium text-neutral-200">1 · Raw frames</span>
                <span className="ml-1.5 text-[10px] text-neutral-500">per experiment</span>
                <input
                  type="file"
                  multiple
                  disabled={uploadsBlocked}
                  onChange={(e) => upload(e.target.files, "run")}
                  className="mt-2 block w-full text-[11px] text-neutral-400
                             file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-neutral-800
                             file:px-3 file:py-1.5 file:text-[11px] file:font-medium file:text-neutral-200
                             hover:file:bg-neutral-700
                             disabled:cursor-not-allowed disabled:opacity-40 disabled:file:cursor-not-allowed"
                />
                {uploadsBlocked ? (
                  <p className="mt-1 text-[11px] text-amber-300">
                    Paused: the bucket is at {usage.gb.toFixed(2)} GB of {usage.limit_gb} GB. Free space
                    in Cloud storage first — the gateway refuses dataset uploads past {usage.warn_gb} GB.
                  </p>
                ) : (
                  <p className="mt-1 text-[11px] text-neutral-500">
                    Every Frame*-0.ims, Frame*-1.ims and Frame*-decoder.xml from one run folder — 24
                    files for a double-pulse run. Skip FrameScales / MetaData / Settings.
                  </p>
                )}
              </div>

              <div>
                <span className="text-[13px] font-medium text-neutral-200">2 · Calibration.xml</span>
                <span className="ml-1.5 text-[10px] text-neutral-500">once per project</span>
                <input
                  type="file"
                  accept=".xml"
                  onChange={(e) => upload(e.target.files, "calibration")}
                  className="mt-2 block w-full text-[11px] text-neutral-400
                             file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-neutral-800
                             file:px-3 file:py-1.5 file:text-[11px] file:font-medium file:text-neutral-200
                             hover:file:bg-neutral-700"
                />
                <p className="mt-1 text-[11px] text-neutral-500">
                  From Properties/Calibration/. Steps 4-6 cannot run without it.
                </p>
              </div>

              {uploading && (
                <p className="text-[11px] text-sky-300">
                  {uploading.finished
                    ? `Uploaded ${uploading.done} ${
                        uploading.kind === "calibration" ? "calibration" : "frame"
                      } file(s) to R2.`
                    : `Uploading ${uploading.done}/${uploading.total}…`}
                </p>
              )}
            </div>
          </div>
        </aside>

        {/* ============ Main ============ */}
        <main className="min-w-0 flex-1 space-y-6">
          <nav className="flex gap-1 rounded-xl border border-neutral-800 bg-neutral-900/60 p-1">
            {[
              { id: "pipeline", label: "Pipeline" },
              { id: "storage", label: "Cloud storage" },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                aria-current={tab === item.id ? "page" : undefined}
                className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition ${
                  tab === item.id
                    ? "bg-neutral-800 text-neutral-100"
                    : "text-neutral-400 hover:text-neutral-200"
                }`}
              >
                {item.label}
                {item.id === "storage" && usage && (
                  <span
                    className={`ml-2 font-mono text-[11px] tabular-nums ${
                      usage.over_threshold ? "text-amber-300" : "text-neutral-600"
                    }`}
                  >
                    {usage.gb.toFixed(1)} GB
                  </span>
                )}
              </button>
            ))}
          </nav>

          <QuotaBanner usage={usage} onManage={() => setTab("storage")} />

          {error && (
            <div className="flex items-start justify-between gap-4 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3">
              <p className="text-sm text-rose-200">{error}</p>
              <button onClick={() => setError(null)} className="text-xs text-rose-300 hover:text-rose-100">
                dismiss
              </button>
            </div>
          )}

          {notice && (
            <div className="flex items-start justify-between gap-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3">
              <p className="text-sm text-emerald-200">{notice}</p>
              <button
                onClick={() => setNotice(null)}
                className="text-xs text-emerald-300 hover:text-emerald-100"
              >
                dismiss
              </button>
            </div>
          )}

          {tab === "storage" ? (
            <StorageManager
              data={listing}
              busy={storageBusy}
              filter={storageFilter}
              onFilter={setStorageFilter}
              onRefresh={refreshListing}
              onDelete={deleteFromStorage}
              onOpen={openViewer}
              expanded={expanded}
              onToggle={toggleFolder}
            />
          ) : (
            <>
              {/* -- step execution -- */}
              <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
                <h2 className="text-sm font-semibold text-neutral-200">Run pipeline</h2>
                <p className="mt-0.5 text-xs text-neutral-500">
                  Each step reads the previous step's output, so run them in order the first time.
                  A step already computed with these exact settings is served from R2 without
                  starting a GPU.
                </p>

                <div className="mt-4 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
                  {STEPS.map((step) => (
                    <button
                      key={step.key}
                      onClick={() => submit(step.key)}
                      disabled={busy || submitting !== null}
                      title={step.hint}
                      className="group relative rounded-xl border border-neutral-800 bg-neutral-950 p-3 text-left
                                 transition hover:border-sky-500/50 hover:bg-sky-500/5
                                 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-neutral-800
                                 disabled:hover:bg-neutral-950"
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-[13px] font-semibold text-neutral-100">{step.label}</span>
                        {step.gpu && (
                          <span className="rounded bg-violet-500/15 px-1 py-px text-[9px] font-semibold text-violet-300">
                            T4
                          </span>
                        )}
                      </div>
                      <div className="mt-0.5 text-[11px] leading-tight text-neutral-500">{step.sub}</div>
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => submit("full")}
                  disabled={busy || submitting !== null}
                  className="mt-3 w-full rounded-xl bg-sky-600 px-4 py-3 text-sm font-semibold text-white
                             transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-neutral-800
                             disabled:text-neutral-500"
                >
                  {submitting === "full" ? "Submitting…" : "▶  Run full pipeline (Steps 1 → 6)"}
                </button>
              </section>

              {/* -- progress -- */}
              {job && (
                <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-3">
                        <h2 className="text-sm font-semibold text-neutral-200">{job.step}</h2>
                        <StatusPill status={job.status} />
                        {job.from_cache && (
                          <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-300 ring-1 ring-emerald-500/25">
                            from cache · no GPU used
                          </span>
                        )}
                      </div>
                      <p className="mt-0.5 font-mono text-[11px] text-neutral-500">
                        {job.job_id || "—"}
                      </p>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="font-mono text-lg tabular-nums text-neutral-100">
                          {job.progress ?? 0}%
                        </div>
                        <div className="text-[11px] text-neutral-500">
                          {(job.duration_s ?? 0).toFixed(0)}s elapsed
                        </div>
                      </div>
                      {!job.done && (
                        <button
                          onClick={cancel}
                          className="rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300
                                     hover:border-rose-500/50 hover:text-rose-300"
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 h-2 overflow-hidden rounded-full bg-neutral-800">
                    <div
                      className={`h-full rounded-full transition-[width] duration-500 ease-out ${
                        job.status === "failed" ? "bg-rose-500" : job.done ? "bg-emerald-500" : "bg-sky-500"
                      }`}
                      style={{ width: `${job.progress ?? 0}%` }}
                    />
                  </div>

                  {job.from_cache ? (
                    <div className="mt-3 space-y-2 text-xs text-neutral-400">
                      <p>
                        These files were already in R2 under{" "}
                        <span className="font-mono text-[11px] text-neutral-300">{job.prefix}/</span>,
                        produced with identical parameters
                        {job.cached_at ? ` on ${job.cached_at.replace("T", " ").replace("Z", " UTC")}` : ""}
                        {job.original_duration_s ? ` in ${job.original_duration_s}s of compute` : ""}.
                        Nothing was recomputed.
                      </p>
                      <button
                        onClick={() => submit(job.step, { force: true })}
                        disabled={submitting !== null}
                        className="rounded-lg border border-neutral-700 px-3 py-1.5 text-[11px] text-neutral-300
                                   transition hover:border-sky-500/50 hover:text-sky-300 disabled:opacity-40"
                      >
                        Recompute anyway
                      </button>
                    </div>
                  ) : (
                    /* Honest about what the number is: the pipeline emits step
                       banners and MART iteration counts, not a true percentage. */
                    <p className="mt-1.5 text-[11px] text-neutral-600">
                      Estimated from step markers and MART iteration count in the job log.
                    </p>
                  )}

                  {job.message && <p className="mt-3 text-xs text-neutral-400">{job.message}</p>}

                  {job.error && (
                    <pre className="mt-3 max-h-40 overflow-auto rounded-lg border border-rose-500/25 bg-rose-500/5 p-3 text-[11px] text-rose-200">
                      {job.error}
                    </pre>
                  )}

                  {job.logs?.length > 0 && (
                    <div
                      ref={logRef}
                      className="mt-3 max-h-64 overflow-auto rounded-lg border border-neutral-800 bg-neutral-950 p-3"
                    >
                      <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-neutral-400">
                        {job.logs.join("\n")}
                      </pre>
                    </div>
                  )}
                </section>
              )}

              {/* -- downloads -- */}
              <section className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
                <div className="flex items-baseline justify-between">
                  <h2 className="text-sm font-semibold text-neutral-200">Download centre</h2>
                  {artifacts?.artifacts?.length > 0 && (
                    <span className="text-[11px] text-neutral-500">
                      {artifacts.artifacts.length} files · {formatBytes(artifacts.total_bytes)} · links
                      expire in {Math.round((artifacts.expires_in ?? 3600) / 60)} min
                    </span>
                  )}
                </div>

                {!artifacts?.artifacts?.length ? (
                  <p className="mt-3 text-xs text-neutral-500">
                    {busy ? "Results appear here when the job finishes." : "No results yet — run a step above."}
                  </p>
                ) : (
                  <div className="mt-4 space-y-5">
                    {groupedArtifacts.map(([kind, files]) => (
                      <div key={kind}>
                        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-500">
                          {kind}
                        </h3>
                        <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                          {files.map((file) => (
                            <a
                              key={file.key}
                              href={file.url}
                              download={file.name}
                              className="flex items-center gap-2.5 rounded-lg border border-neutral-800 bg-neutral-950
                                         px-3 py-2.5 transition hover:border-sky-500/50 hover:bg-sky-500/5"
                            >
                              <FileBadge name={file.name} />
                              <span className="min-w-0 flex-1">
                                <span className="block truncate font-mono text-[11px] text-neutral-200">
                                  {file.name}
                                </span>
                                <span className="text-[10px] text-neutral-500">{formatBytes(file.size)}</span>
                              </span>
                              <span className="text-neutral-600">↓</span>
                            </a>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}
        </main>
      </div>

      {viewing && (
        <ImageLightbox
          folder={viewing.folder}
          images={imagesIn(viewing.folder)}
          index={viewing.index}
          onIndex={(index) => setViewing((prev) => ({ ...prev, index }))}
          onClose={() => setViewing(null)}
        />
      )}
    </div>
  );
}
