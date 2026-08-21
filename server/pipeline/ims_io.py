"""Read/write LaVision DaVis raw ``.ims`` frame files. Two decoder formats are
supported, auto-detected per-frame from that frame's ``FrameN-decoder.xml``:

  * ``mono-12pmsb`` (confirmed empirically against ``LI_16072026_7.2mm_09``):
    a headerless buffer of ``width*height`` 12-bit pixels, packed 2 pixels per
    3 bytes, MSB-first: P0 = (B0<<4)|(B1&0x0F), P1 = (B2<<4)|(B1>>4).
  * ``raw-8-bit`` (confirmed against the ``LED_PIV`` project): plain 1
    byte/pixel. Values are rescaled 0-255 -> 0-MAX_12BIT on load (and back on
    save) purely so every other module in this pipeline -- tuned around a
    0-4095 dynamic range -- keeps working unmodified regardless of source
    bit depth.

Some projects (e.g. ``LED_PIV``, a burst-mode acquisition) store multiple
frames concatenated in one ``FrameN-1.ims`` file rather than a single image;
``burst_index`` selects which one to decode (default 0). ``FrameN-0.ims`` is
always a small per-frame header, not image data -- bytes [12:16] and [16:20]
are little-endian uint32 width/height.

Frame index -> (camera, timestep) mapping for an 8-frame (4-camera) acquisition,
confirmed via ``Acq.Camera.Label`` in ``MetaDataN.attr`` (holds across both
projects checked so far):
    Frame0,1 -> Camera1 (t1,t2); Frame2,3 -> Camera2; Frame4,5 -> Camera3; Frame6,7 -> Camera4
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_WIDTH = 2448
DEFAULT_HEIGHT = 2048
MAX_12BIT = 4095
DEFAULT_DECODER_ID = "mono-12pmsb"


@dataclass(frozen=True)
class FrameRef:
    frame_index: int
    camera: int  # 1-based
    timestep: int  # 1 or 2


def frame_refs_for_run(has_t1: bool = True, has_t2: bool = True) -> list[FrameRef]:
    """Frame-index -> (camera, timestep) mapping. Confirmed empirically (see
    project_layout.py HAS_T1/HAS_T2) that double- and single-pulse
    acquisitions use genuinely different layouts, not just a missing file:
      double-pulse (has_t1 and has_t2): 8 raw files, FrameN/N+1 pairs -->
        Frame0,1->Camera1(t1,t2); Frame2,3->Camera2; Frame4,5->Camera3; Frame6,7->Camera4
      single-pulse (has_t1 only): 4 raw files, one per camera, t1 only -->
        Frame0,1,2,3 -> Camera1,2,3,4
    """
    if not has_t1:
        raise ValueError("has_t1=False is not a supported acquisition mode (every project has at least t1)")

    refs = []
    if has_t2:
        for n in range(8):
            refs.append(FrameRef(frame_index=n, camera=n // 2 + 1, timestep=n % 2 + 1))
    else:
        for n in range(4):
            refs.append(FrameRef(frame_index=n, camera=n + 1, timestep=1))
    return refs


def read_frame_header(header_path: Path) -> tuple[int, int]:
    """Parse width/height from the small FrameN-0.ims header. Falls back to defaults."""
    try:
        data = Path(header_path).read_bytes()
        width = struct.unpack_from("<I", data, 12)[0]
        height = struct.unpack_from("<I", data, 16)[0]
        if 0 < width <= 16384 and 0 < height <= 16384:
            return width, height
    except (FileNotFoundError, struct.error):
        pass
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def decode_mono12p_msb(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unpack a Mono12p MSB-first byte buffer into a (height, width) uint16 image."""
    expected = width * height * 3 // 2
    if raw.size != expected:
        raise ValueError(f"raw buffer has {raw.size} bytes, expected {expected} for {width}x{height}")
    triplets = raw.reshape(-1, 3)
    b0 = triplets[:, 0].astype(np.uint16)
    b1 = triplets[:, 1].astype(np.uint16)
    b2 = triplets[:, 2].astype(np.uint16)
    p0 = (b0 << 4) | (b1 & 0x0F)
    p1 = (b2 << 4) | (b1 >> 4)
    out = np.empty(triplets.shape[0] * 2, dtype=np.uint16)
    out[0::2] = p0
    out[1::2] = p1
    return out.reshape(height, width)


def encode_mono12p_msb(img: np.ndarray) -> np.ndarray:
    """Pack a (height, width) array of 12-bit values (0-4095) back into Mono12p MSB bytes."""
    img = np.clip(img, 0, MAX_12BIT).astype(np.uint16)
    flat = img.reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError("pixel count must be even to pack into byte triplets")
    p0 = flat[0::2]
    p1 = flat[1::2]
    b0 = (p0 >> 4).astype(np.uint8)
    b1 = (((p0 & 0x0F) | ((p1 & 0x0F) << 4))).astype(np.uint8)
    b2 = (p1 >> 4).astype(np.uint8)
    out = np.empty(p0.size * 3, dtype=np.uint8)
    out[0::3] = b0
    out[1::3] = b1
    out[2::3] = b2
    return out


def decode_raw8(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unpack a plain 1-byte/pixel buffer, rescaled 0-255 -> 0-MAX_12BIT."""
    expected = width * height
    if raw.size != expected:
        raise ValueError(f"raw buffer has {raw.size} bytes, expected {expected} for {width}x{height} (raw-8-bit)")
    img8 = raw.reshape(height, width).astype(np.uint32)
    return (img8 * MAX_12BIT // 255).astype(np.uint16)


def encode_raw8(img: np.ndarray) -> np.ndarray:
    """Inverse of decode_raw8: 0-MAX_12BIT -> 0-255, 1 byte/pixel."""
    img = np.clip(img, 0, MAX_12BIT).astype(np.uint32)
    return (img * 255 // MAX_12BIT).astype(np.uint8).reshape(-1)


def bytes_per_pixel_for_decoder(decoder_id: str) -> float:
    return 1.0 if decoder_id == "raw-8-bit" else 1.5


def read_decoder_id(decoder_xml_path: Path) -> str:
    """Reads <id> from a FrameN-decoder.xml sidecar; falls back to the
    Mono12p-MSB format (this pipeline's original/default project) if the
    file is missing or unrecognized, so old callers keep working."""
    try:
        text = Path(decoder_xml_path).read_text(encoding="utf-8")
        m = re.search(r"<id>([^<]+)</id>", text)
        if m:
            return m.group(1).strip()
    except FileNotFoundError:
        pass
    return DEFAULT_DECODER_ID


def load_ims_image(
    path: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    decoder_id: str = DEFAULT_DECODER_ID,
    burst_index: int = 0,
) -> np.ndarray:
    """Decodes one frame from ``path``. If the file holds multiple concatenated
    frames (burst-mode acquisition), ``burst_index`` selects which one --
    default 0 works unchanged for single-frame files too."""
    bytes_per_pixel = bytes_per_pixel_for_decoder(decoder_id)
    frame_bytes = int(round(width * height * bytes_per_pixel))
    with open(str(path), "rb") as f:
        f.seek(burst_index * frame_bytes)
        raw = np.frombuffer(f.read(frame_bytes), dtype=np.uint8)
    if decoder_id == "raw-8-bit":
        return decode_raw8(raw, width, height)
    return decode_mono12p_msb(raw, width, height)


def save_ims_image(img: np.ndarray, path: Path, decoder_id: str = DEFAULT_DECODER_ID) -> None:
    packed = encode_raw8(img) if decoder_id == "raw-8-bit" else encode_mono12p_msb(img)
    packed.tofile(str(path))


DECODER_XML_TEMPLATES = {
    "mono-12pmsb": """<?xml version="1.0" encoding="UTF-8"?>
<FrameDecoder>
    <id>mono-12pmsb</id>
    <label>Mono12pmsb to 16 bit</label>
    <minimalDaVisVersion>10.2.0</minimalDaVisVersion>
</FrameDecoder>
""",
    "raw-8-bit": """<?xml version="1.0" encoding="UTF-8"?>
<FrameDecoder>
    <id>raw-8-bit</id>
    <label>8 to 16 bit</label>
    <minimalDaVisVersion>10.2.0</minimalDaVisVersion>
</FrameDecoder>
""",
}


def write_decoder_xml(path: Path, decoder_id: str = DEFAULT_DECODER_ID) -> None:
    template = DECODER_XML_TEMPLATES.get(decoder_id, DECODER_XML_TEMPLATES[DEFAULT_DECODER_ID])
    Path(path).write_text(template, encoding="utf-8")
