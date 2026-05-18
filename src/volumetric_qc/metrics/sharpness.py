"""Focus / sharpness metrics via Laplacian variance.

The variance of the Laplacian is a classic no-reference focus measure: a sharp
image has many strong second-derivative responses, an out-of-focus image is
smooth. Per-slice values are normalized by the maximum across the volume so the
output is comparable across runs. Outliers (z-slices with <= peak * fraction)
are flagged.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from volumetric_qc.io.chunked import iter_z_chunks, block_mean_downsample


_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _laplacian_variance(arr: np.ndarray) -> float:
    """Compute variance of the Laplacian of a 2D array."""
    lap = ndi.convolve(arr.astype(np.float32), _LAPLACIAN, mode="reflect")
    return float(lap.var())


def sharpness_profile(
    zyx: da.Array,
    *,
    z_stride: int = 1,
    xy_downsample: int = 1,
    chunk_z: int = 16,
    outlier_fraction: float = 0.25,
) -> dict[str, Any]:
    """Per-slice Laplacian variance, with outlier z-slice flagging.

    Parameters
    ----------
    zyx
        Dask array of shape ``(Z, Y, X)``.
    z_stride
        Stride for sampling z-slices.
    xy_downsample
        Block-mean downsample in y / x before Laplacian.
    chunk_z
        Slab size for materialization.
    outlier_fraction
        A z-slice is flagged if its sharpness is below ``peak * outlier_fraction``.

    Returns
    -------
    dict
        Keys: ``z``, ``laplacian_var`` (per-slice), ``relative`` (per-slice /
        peak), ``min_relative``, ``outlier_z`` (list of z indices below the
        outlier fraction), ``peak_z`` (index of the sharpest slice).
    """
    zs: list[int] = []
    lvars: list[float] = []
    for z_start, block in iter_z_chunks(zyx, z_stride=z_stride, chunk_z=chunk_z):
        block_d = block_mean_downsample(block, xy_downsample) if xy_downsample > 1 else block
        for i in range(block_d.shape[0]):
            zs.append(z_start + i * z_stride)
            lvars.append(_laplacian_variance(block_d[i]))

    arr = np.asarray(lvars, dtype=np.float64)
    peak = float(arr.max()) if arr.size else 0.0
    rel = (arr / peak).tolist() if peak > 0 else [0.0] * len(arr)
    outlier_z = [zs[i] for i, r in enumerate(rel) if r < outlier_fraction]
    peak_z = int(zs[int(np.argmax(arr))]) if arr.size else -1

    return {
        "z": zs,
        "laplacian_var": lvars,
        "relative": rel,
        "min_relative": float(min(rel)) if rel else 0.0,
        "outlier_z": outlier_z,
        "peak_z": peak_z,
    }
