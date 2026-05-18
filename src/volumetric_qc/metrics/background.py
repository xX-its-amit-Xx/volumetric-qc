"""Background uniformity and autofluorescence estimation.

Background is estimated by morphological opening (rolling-ball-like) followed by
sampling the low-percentile patches. The coefficient of variation of these
patches across the volume tells us whether background is uniform (good clearing,
uniform illumination) or patchy (vignetting, residual lipid, dust).

Autofluorescence ratio: mode of low-intensity (background) pixel histogram
divided by mean of high-intensity (signal) pixels. High ratio means signal is
not well separated from background — i.e. high autofluorescence relative to
target stain.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from volumetric_qc.io.chunked import iter_z_chunks, block_mean_downsample


def _patch_means(arr2d: np.ndarray, patch: int = 32) -> np.ndarray:
    """Mean-pool 2D array into patches of given size; return flat list of patch means."""
    y, x = arr2d.shape
    ny, nx = y // patch, x // patch
    if ny == 0 or nx == 0:
        return np.array([float(arr2d.mean())])
    cropped = arr2d[: ny * patch, : nx * patch]
    return cropped.reshape(ny, patch, nx, patch).mean(axis=(1, 3)).ravel()


def background_uniformity(
    zyx: da.Array,
    *,
    z_stride: int = 1,
    xy_downsample: int = 1,
    chunk_z: int = 16,
    background_percentile: float = 20.0,
    signal_percentile: float = 90.0,
) -> dict[str, Any]:
    """Estimate per-slice background level, background uniformity, and autofluorescence ratio.

    Parameters
    ----------
    zyx
        Dask array of shape ``(Z, Y, X)``.
    z_stride
        Stride for sampling z-slices.
    xy_downsample
        Block-mean downsample in y / x.
    chunk_z
        Slab size for materialization.
    background_percentile
        Percentile defining background pixels per slice (defaults to 20th).
    signal_percentile
        Percentile defining signal pixels per slice (defaults to 90th).

    Returns
    -------
    dict
        Keys: ``z``, ``background`` (per-slice background level),
        ``signal`` (per-slice signal level), ``background_cv`` (CV of the
        per-patch background levels across the volume — uniformity measure),
        ``autofluor_ratio`` (overall background / signal ratio).
    """
    zs: list[int] = []
    bg_levels: list[float] = []
    sig_levels: list[float] = []
    all_patch_bg: list[float] = []

    for z_start, block in iter_z_chunks(zyx, z_stride=z_stride, chunk_z=chunk_z):
        block_d = block_mean_downsample(block, xy_downsample) if xy_downsample > 1 else block
        for i in range(block_d.shape[0]):
            sl = block_d[i].astype(np.float32, copy=False)
            zs.append(z_start + i * z_stride)
            bg = float(np.percentile(sl, background_percentile))
            sig = float(np.percentile(sl, signal_percentile))
            bg_levels.append(bg)
            sig_levels.append(sig)
            patches = _patch_means(sl, patch=max(16, sl.shape[0] // 16))
            # Only keep low-percentile patches as "background patches".
            low_thresh = np.percentile(patches, background_percentile)
            all_patch_bg.extend(patches[patches <= low_thresh].tolist())

    bg_arr = np.asarray(all_patch_bg, dtype=np.float64)
    overall_bg = float(bg_arr.mean()) if bg_arr.size else float(np.mean(bg_levels) if bg_levels else 0.0)
    overall_sig = float(np.mean(sig_levels)) if sig_levels else 0.0
    bg_cv = float(bg_arr.std() / max(overall_bg, 1e-9)) if bg_arr.size else 0.0
    autofluor_ratio = float(overall_bg / max(overall_sig, 1e-9)) if overall_sig > 0 else 0.0

    return {
        "z": zs,
        "background": bg_levels,
        "signal": sig_levels,
        "background_cv": bg_cv,
        "autofluor_ratio": autofluor_ratio,
        "overall_background": overall_bg,
        "overall_signal": overall_sig,
    }
