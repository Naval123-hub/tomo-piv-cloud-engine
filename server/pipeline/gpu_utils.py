"""GPU/CPU array-module selection. Uses CuPy if it is installed and a CUDA
device actually works; otherwise falls back transparently to numpy."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_cupy_module = None
_checked = False


def get_xp(prefer_gpu: bool = True):
    """Return (xp_module, using_gpu: bool)."""
    global _cupy_module, _checked
    if not prefer_gpu:
        return np, False

    if not _checked:
        _checked = True
        try:
            import cupy as cp

            probe = cp.asarray([1.0, 2.0]) * 2.0
            float(probe.sum())  # forces kernel compilation/execution
            _cupy_module = cp
            log.info("GPU acceleration available: using CuPy (%s)", cp.cuda.runtime.getDeviceProperties(0)["name"])
        except Exception as exc:  # ImportError, RuntimeError from CUDA/toolkit issues, etc.
            log.warning("CuPy not usable (%s); falling back to CPU numpy.", exc)
            _cupy_module = None

    if _cupy_module is not None:
        return _cupy_module, True
    return np, False


def to_numpy(arr):
    xp, using_gpu = get_xp()
    if using_gpu and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)
