"""Artifact detection: stripes (FFT), bubbles (blob detection), folding (gradients).

Stripes
-------
Light-sheet stripe artifacts appear as horizontal/vertical bands due to absorbing
features in the illumination path. In Fourier space they concentrate energy
along a thin band orthogonal to the stripe direction. We detect them by
computing the 2D power spectrum of each sampled slice and measuring the fraction
of energy concentrated in a narrow wedge along the (kx=0) or (ky=0) axes.

Bubbles
-------
Round dark/light inclusions in the cleared tissue (air bubbles, debris). We use
the Laplacian-of-Gaussian blob detector at multiple scales on sampled slices and
count blobs above a contrast threshold.

Folding / tears
---------------
Sharp discontinuities in the gradient field that don't look like normal cellular
structure. Estimated as the fraction of pixels where the gradient magnitude is
more than a robust z-score above the median.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log

from volumetric_qc.io.chunked import block_mean_downsample, iter_z_chunks, stratified_z_indices


def _power_spectrum(arr2d: np.ndarray) -> np.ndarray:
    """2D normalized power spectrum (DC-centered)."""
    a = arr2d.astype(np.float32, copy=False)
    a = a - a.mean()
    f = np.fft.fftshift(np.fft.fft2(a))
    p = np.abs(f) ** 2
    s = p.sum()
    return p / s if s > 0 else p


def _stripe_energy_ratio(arr2d: np.ndarray, wedge_fraction: float = 0.02) -> float:
    """Fraction of FFT energy concentrated in horizontal or vertical wedges.

    A clean image has roughly isotropic spectrum. A striped image has bright
    bands along kx=0 (horizontal stripes -> vertical band in FFT) or ky=0.
    """
    p = _power_spectrum(arr2d)
    h, w = p.shape
    cy, cx = h // 2, w // 2
    band_y = max(1, int(h * wedge_fraction))
    band_x = max(1, int(w * wedge_fraction))
    # Exclude the DC by skipping a small central neighborhood.
    dc_skip = 2
    mask = np.zeros_like(p, dtype=bool)
    mask[cy - band_y : cy + band_y + 1, :] = True
    mask[:, cx - band_x : cx + band_x + 1] = True
    mask[cy - dc_skip : cy + dc_skip + 1, cx - dc_skip : cx + dc_skip + 1] = False
    return float(p[mask].sum())


def stripe_energy(
    zyx: da.Array,
    *,
    z_stride: int = 1,
    chunk_z: int = 8,
    tile_size: int = 512,
    n_samples: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate per-slice stripe energy ratio (FFT-based).

    Returns
    -------
    dict
        Keys: ``z_sampled``, ``energy_ratio`` (per sampled slice),
        ``mean_ratio`` summary, ``max_ratio``.
    """
    nz = int(zyx.shape[0])
    z_idx = stratified_z_indices(nz, n_samples, seed=seed)
    ratios: list[float] = []
    for z in z_idx:
        sl = np.asarray(zyx[z].compute()).astype(np.float32, copy=False)
        # Crop to a square power-of-two-ish tile from the center for stable FFT.
        y, x = sl.shape
        ts = min(tile_size, y, x)
        y0 = (y - ts) // 2
        x0 = (x - ts) // 2
        tile = sl[y0 : y0 + ts, x0 : x0 + ts]
        ratios.append(_stripe_energy_ratio(tile))

    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "z_sampled": z_idx,
        "energy_ratio": ratios,
        "mean_ratio": float(arr.mean()) if arr.size else 0.0,
        "max_ratio": float(arr.max()) if arr.size else 0.0,
    }


def bubble_count(
    zyx: da.Array,
    *,
    n_samples: int = 12,
    seed: int = 42,
    min_sigma: float = 6.0,
    max_sigma: float = 20.0,
    num_sigma: int = 5,
    threshold: float = 0.12,
) -> dict[str, Any]:
    """Detect bubble-like blobs per z-slice using Laplacian-of-Gaussian.

    Returns
    -------
    dict
        Keys: ``z_sampled``, ``counts`` per slice, ``max_per_slice``,
        ``mean_per_slice``, ``blobs`` (list of (z, y, x, sigma) tuples).
    """
    nz = int(zyx.shape[0])
    z_idx = stratified_z_indices(nz, n_samples, seed=seed)
    counts: list[int] = []
    blobs_all: list[tuple[int, float, float, float]] = []

    for z in z_idx:
        sl = np.asarray(zyx[z].compute()).astype(np.float32, copy=False)
        # Normalize to [0, 1] for stable blob threshold.
        lo, hi = float(sl.min()), float(sl.max())
        if hi > lo:
            sl_n = (sl - lo) / (hi - lo)
        else:
            sl_n = np.zeros_like(sl)
        try:
            blobs = blob_log(sl_n, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        except Exception:
            blobs = np.empty((0, 3))
        counts.append(int(blobs.shape[0]))
        for b in blobs:
            blobs_all.append((int(z), float(b[0]), float(b[1]), float(b[2])))

    arr = np.asarray(counts, dtype=np.float64)
    return {
        "z_sampled": z_idx,
        "counts": counts,
        "max_per_slice": int(arr.max()) if arr.size else 0,
        "mean_per_slice": float(arr.mean()) if arr.size else 0.0,
        "blobs": blobs_all,
    }


def folding_score(
    zyx: da.Array,
    *,
    n_samples: int = 12,
    seed: int = 42,
    z_score: float = 4.0,
) -> dict[str, Any]:
    """Fraction of high-gradient outlier pixels per sampled slice.

    Tissue folding and tears produce sharp, locally coherent gradient
    discontinuities that look qualitatively different from neuronal structure.
    We estimate this by computing the gradient magnitude, then the fraction of
    pixels where the magnitude exceeds ``median + z_score * MAD``.
    """
    nz = int(zyx.shape[0])
    z_idx = stratified_z_indices(nz, n_samples, seed=seed)
    fractions: list[float] = []
    for z in z_idx:
        sl = np.asarray(zyx[z].compute()).astype(np.float32, copy=False)
        gy = ndi.sobel(sl, axis=0)
        gx = ndi.sobel(sl, axis=1)
        mag = np.hypot(gy, gx)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-9
        thresh = med + z_score * 1.4826 * mad
        frac = float((mag > thresh).mean())
        fractions.append(frac)

    arr = np.asarray(fractions, dtype=np.float64)
    return {
        "z_sampled": z_idx,
        "outlier_fraction_per_slice": fractions,
        "outlier_fraction": float(arr.mean()) if arr.size else 0.0,
        "max_outlier_fraction": float(arr.max()) if arr.size else 0.0,
    }
