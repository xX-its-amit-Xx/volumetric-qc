"""Intensity statistics and drift detection.

This module produces per-z-slice mean, std, and percentile statistics, and from
those derives a normalized drift slope (linear regression of slice mean vs z,
normalized by the overall mean). A large positive or negative drift indicates
imaging-depth-dependent intensity changes — common with iDISCO bleaching, with
SHIELD when refractive-index matching is suboptimal, or with light-sheet engines
that drift over a long acquisition.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np

from volumetric_qc.io.chunked import iter_z_chunks, block_mean_downsample


def intensity_profile(
    zyx: da.Array,
    *,
    z_stride: int = 1,
    xy_downsample: int = 1,
    percentiles: tuple[float, ...] = (1.0, 50.0, 99.0),
    chunk_z: int = 16,
) -> dict[str, Any]:
    """Compute per-z-slice intensity statistics and drift metrics.

    Parameters
    ----------
    zyx
        Dask array of shape ``(Z, Y, X)``.
    z_stride
        Sample every ``z_stride``-th slice.
    xy_downsample
        Downsample factor for y and x (block-mean) before computing stats.
    percentiles
        Percentiles to record per slice.
    chunk_z
        Number of slices to load per dask materialization.

    Returns
    -------
    dict
        Keys:

        * ``z`` — list of z indices sampled.
        * ``mean``, ``std`` — per-slice mean / std.
        * ``p{P}`` — per-slice percentile P for each requested P.
        * ``drift_slope`` — slope of per-slice mean over z, normalized by the
          overall mean (dimensionless). Large absolute value indicates drift.
        * ``cv`` — coefficient of variation of per-slice mean (std / mean).
        * ``global_mean``, ``global_std`` — aggregate stats.
    """
    zs: list[int] = []
    means: list[float] = []
    stds: list[float] = []
    pcts: dict[str, list[float]] = {f"p{int(p)}": [] for p in percentiles}

    for z_start, block in iter_z_chunks(zyx, z_stride=z_stride, chunk_z=chunk_z):
        block_d = block_mean_downsample(block, xy_downsample) if xy_downsample > 1 else block
        for i in range(block_d.shape[0]):
            sl = block_d[i].astype(np.float64, copy=False)
            zs.append(z_start + i * z_stride)
            means.append(float(sl.mean()))
            stds.append(float(sl.std()))
            for p in percentiles:
                pcts[f"p{int(p)}"].append(float(np.percentile(sl, p)))

    arr_means = np.asarray(means, dtype=np.float64)
    arr_z = np.asarray(zs, dtype=np.float64)
    global_mean = float(arr_means.mean()) if arr_means.size else 0.0
    global_std = float(np.asarray(stds, dtype=np.float64).mean()) if stds else 0.0

    drift_slope = 0.0
    if arr_z.size >= 2 and global_mean > 0:
        # Normalize x to [0, 1] so slope is per-volume not per-slice.
        x = (arr_z - arr_z.min()) / max(arr_z.max() - arr_z.min(), 1.0)
        slope, _ = np.polyfit(x, arr_means, 1)
        drift_slope = float(slope) / max(global_mean, 1e-9)

    cv = float(arr_means.std() / max(global_mean, 1e-9)) if arr_means.size else 0.0

    return {
        "z": zs,
        "mean": means,
        "std": stds,
        **pcts,
        "drift_slope": drift_slope,
        "cv": cv,
        "global_mean": global_mean,
        "global_std": global_std,
    }
