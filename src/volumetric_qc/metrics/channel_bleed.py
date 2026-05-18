"""Cross-channel bleed-through detection.

Spectral bleed-through happens when fluorophore A's emission spectrum overlaps
with fluorophore B's detection channel. The signal in channel B then contains a
scaled copy of channel A. We detect this by:

1. Sampling z-slices across the volume.
2. Restricting attention to "signal" pixels in channel A (above its 90th
   percentile) to avoid background-driven spurious correlation.
3. Computing the Pearson correlation between channel A and channel B intensity
   in those pixels.

A high positive correlation in signal pixels is the bleed-through signature.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from volumetric_qc.io.chunked import block_mean_downsample, stratified_z_indices
from volumetric_qc.io.readers import LazyVolume


def bleed_through(
    vol: LazyVolume,
    *,
    z_stride: int = 1,
    xy_downsample: int = 1,
    n_samples: int = 16,
    signal_percentile: float = 90.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate pairwise channel bleed-through correlation.

    Parameters
    ----------
    vol
        :class:`LazyVolume` with at least 2 channels.
    z_stride, xy_downsample, n_samples, seed
        Sampling controls.
    signal_percentile
        Percentile in the *source* channel above which a pixel is considered
        "signal" for the correlation computation.

    Returns
    -------
    dict
        Keys: ``pairwise_corr`` mapping ``(label_a, label_b)`` tuples to Pearson
        correlation in signal pixels; ``n_samples`` used; ``z_sampled``.
    """
    nc = vol.nchannels
    if nc < 2:
        return {"pairwise_corr": {}, "n_samples": 0, "z_sampled": []}

    nz = vol.nz
    z_idx = stratified_z_indices(nz, n_samples, seed=seed)
    labels = vol.channel_names or [f"ch{i}" for i in range(nc)]

    # Pre-load sampled slices once per channel.
    sampled: dict[int, np.ndarray] = {}
    for c in range(nc):
        ch = vol.channel(c)
        slabs = [np.asarray(ch[z].compute()) for z in z_idx]
        arr = np.stack(slabs, axis=0) if slabs else np.zeros((0,))
        if xy_downsample > 1 and arr.ndim == 3:
            arr = block_mean_downsample(arr, xy_downsample)
        sampled[c] = arr.astype(np.float32, copy=False)

    corr: dict[tuple[str, str], float] = {}
    for i in range(nc):
        for j in range(nc):
            if i == j:
                continue
            a = sampled[i].ravel()
            b = sampled[j].ravel()
            if a.size == 0 or b.size == 0:
                corr[(labels[i], labels[j])] = 0.0
                continue
            # Mask: signal pixels in source channel.
            thresh = np.percentile(a, signal_percentile)
            mask = a >= thresh
            if mask.sum() < 50:
                corr[(labels[i], labels[j])] = 0.0
                continue
            am = a[mask] - a[mask].mean()
            bm = b[mask] - b[mask].mean()
            denom = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
            corr[(labels[i], labels[j])] = float((am * bm).sum() / denom) if denom > 0 else 0.0

    # Stringify tuple keys for JSON safety.
    return {
        "pairwise_corr": {f"{a}->{b}": v for (a, b), v in corr.items()},
        "n_samples": len(z_idx),
        "z_sampled": z_idx,
        "channels": labels,
    }
