"""Clearing-specific artifact detection.

In cleared-tissue imaging, residual lipid pockets and refractive-index (RI)
mismatch between sample and immersion medium produce characteristic
high-frequency speckle that's distinct from genuine staining signal. We
estimate it as the energy fraction in a high-frequency band of the FFT,
optionally restricted to the upper portion of the volume's depth (closer to
the surface, where uncleared tissue is more common).
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np

from volumetric_qc.io.chunked import stratified_z_indices


def _hf_energy_fraction(arr2d: np.ndarray, low_cutoff: float = 0.4) -> float:
    """Fraction of FFT energy above a normalized spatial frequency cutoff."""
    a = arr2d.astype(np.float32, copy=False)
    a = a - a.mean()
    f = np.fft.fftshift(np.fft.fft2(a))
    p = np.abs(f) ** 2
    total = p.sum()
    if total <= 0:
        return 0.0

    h, w = p.shape
    cy, cx = h // 2, w // 2
    ky = (np.arange(h) - cy) / (h / 2)
    kx = (np.arange(w) - cx) / (w / 2)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    r = np.sqrt(KX**2 + KY**2)
    hf_mask = r >= low_cutoff
    return float(p[hf_mask].sum() / total)


def clearing_residue(
    zyx: da.Array,
    *,
    n_samples: int = 12,
    seed: int = 42,
    low_cutoff: float = 0.4,
    tile_size: int = 512,
) -> dict[str, Any]:
    """Estimate residual lipid / RI-mismatch speckle as high-frequency energy fraction.

    Parameters
    ----------
    zyx
        Dask array ``(Z, Y, X)``.
    n_samples, seed
        How many z-slices to sample (stratified).
    low_cutoff
        Lower bound of the high-frequency band (normalized to Nyquist).
    tile_size
        Center-cropped tile size for FFT.

    Returns
    -------
    dict
        Keys: ``z_sampled``, ``hf_energy_per_slice``, ``speckle_energy``
        (mean across sampled slices), ``max_hf_energy``.
    """
    nz = int(zyx.shape[0])
    z_idx = stratified_z_indices(nz, n_samples, seed=seed)
    hfs: list[float] = []
    for z in z_idx:
        sl = np.asarray(zyx[z].compute()).astype(np.float32, copy=False)
        y, x = sl.shape
        ts = min(tile_size, y, x)
        y0 = (y - ts) // 2
        x0 = (x - ts) // 2
        tile = sl[y0 : y0 + ts, x0 : x0 + ts]
        hfs.append(_hf_energy_fraction(tile, low_cutoff=low_cutoff))

    arr = np.asarray(hfs, dtype=np.float64)
    return {
        "z_sampled": z_idx,
        "hf_energy_per_slice": hfs,
        "speckle_energy": float(arr.mean()) if arr.size else 0.0,
        "max_hf_energy": float(arr.max()) if arr.size else 0.0,
    }
