"""Chunked iteration helpers for large volumes.

Most QC metrics process the volume z-slab by z-slab so peak memory stays bounded
by chunk size, not volume size. These helpers provide a uniform iteration
interface over dask arrays.
"""

from __future__ import annotations

from typing import Iterator

import dask.array as da
import numpy as np


def iter_z_chunks(
    arr: da.Array,
    *,
    z_stride: int = 1,
    chunk_z: int = 16,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(z_start, block)`` pairs for a ``(Z, Y, X)`` array.

    Each yielded block is a numpy array materialized via ``compute()``.
    The block spans up to ``chunk_z`` z-slices.

    Parameters
    ----------
    arr
        Dask array with shape ``(Z, Y, X)``.
    z_stride
        Stride for iterating z; only every ``z_stride``-th slice is loaded.
    chunk_z
        Number of stride-selected slices to bundle into one materialized block.

    Yields
    ------
    z_start : int
        Index of the first z-slice in the block.
    block : np.ndarray
        Array of shape ``(<= chunk_z, Y, X)``.
    """
    if arr.ndim != 3:
        raise ValueError(f"iter_z_chunks expects 3D (Z, Y, X), got ndim={arr.ndim}")
    nz = arr.shape[0]
    sampled = list(range(0, nz, z_stride))
    for i in range(0, len(sampled), chunk_z):
        zs = sampled[i : i + chunk_z]
        z0, z1 = zs[0], zs[-1] + 1
        # Pull a contiguous slab then slice the stride pattern.
        slab = np.asarray(arr[z0:z1].compute())
        keep = [z - z0 for z in zs]
        yield zs[0], slab[keep]


def block_mean_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D / 3D array by block-mean averaging.

    Parameters
    ----------
    arr
        Array of shape ``(Y, X)`` or ``(Z, Y, X)``.
    factor
        Integer downsample factor applied to the trailing two axes.

    Returns
    -------
    np.ndarray
        Downsampled array. Trailing axes are floor-divided by ``factor``.
    """
    if factor <= 1:
        return arr
    if arr.ndim == 2:
        y, x = arr.shape
        ynew, xnew = y // factor, x // factor
        if ynew == 0 or xnew == 0:
            return arr
        cropped = arr[: ynew * factor, : xnew * factor]
        return cropped.reshape(ynew, factor, xnew, factor).mean(axis=(1, 3))
    if arr.ndim == 3:
        z, y, x = arr.shape
        ynew, xnew = y // factor, x // factor
        if ynew == 0 or xnew == 0:
            return arr
        cropped = arr[:, : ynew * factor, : xnew * factor]
        return cropped.reshape(z, ynew, factor, xnew, factor).mean(axis=(2, 4))
    raise ValueError(f"block_mean_downsample expects 2D or 3D input, got ndim={arr.ndim}")


def stratified_z_indices(nz: int, n_samples: int, *, seed: int = 0) -> list[int]:
    """Pick ``n_samples`` z-indices stratified across the depth of the volume.

    The stratified sample is deterministic for a given ``seed`` and chooses one
    index from each of ``n_samples`` equal-width strata. Useful for metrics that
    are expensive per-slice (blob detection, FFT) and don't need every slice.
    """
    if n_samples <= 0:
        return []
    if n_samples >= nz:
        return list(range(nz))
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, nz, n_samples + 1, dtype=int)
    picks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        picks.append(int(rng.integers(lo, hi)))
    return picks
