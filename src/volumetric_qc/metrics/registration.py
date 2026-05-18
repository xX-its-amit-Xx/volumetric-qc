"""Cross-channel registration error via phase correlation.

When multiple wavelengths are imaged through different optical paths, there is
typically a per-axis pixel shift between channels. We measure this with
sub-pixel phase correlation on stratified z-slices and report the median shift
per axis along with the spread.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.registration import phase_cross_correlation

from volumetric_qc.io.chunked import block_mean_downsample, stratified_z_indices
from volumetric_qc.io.readers import LazyVolume


def _phase_shift(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Sub-pixel shift estimation between two 2D arrays via phase correlation."""
    try:
        shift, _err, _diff = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None)
    except TypeError:
        # Older skimage signatures.
        shift = phase_cross_correlation(ref, mov, upsample_factor=10)[0]
    return np.asarray(shift, dtype=np.float64)


def cross_channel_shifts(
    vol: LazyVolume,
    *,
    z_stride: int = 1,
    xy_downsample: int = 1,
    n_samples: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate per-channel-pair registration shifts.

    Parameters
    ----------
    vol
        :class:`LazyVolume` with >= 2 channels.
    z_stride, xy_downsample, n_samples, seed
        Sampling controls. We sample ``n_samples`` z-slices (stratified) and
        estimate the (y, x) shift per slice, then report the median shift.

    Returns
    -------
    dict
        Keys: ``pairwise_shifts`` mapping ``"chA->chB"`` to a numpy-list of
        ``[dy, dx]`` median shifts in voxels, ``pairwise_shifts_per_slice``
        with raw per-slice shifts, ``n_samples``, ``z_sampled``, ``channels``.
    """
    nc = vol.nchannels
    if nc < 2:
        return {"pairwise_shifts": {}, "n_samples": 0, "z_sampled": []}

    nz = vol.nz
    z_idx = stratified_z_indices(nz, n_samples, seed=seed)
    labels = vol.channel_names or [f"ch{i}" for i in range(nc)]

    sampled: dict[int, np.ndarray] = {}
    for c in range(nc):
        ch = vol.channel(c)
        slabs = [np.asarray(ch[z].compute()).astype(np.float32) for z in z_idx]
        arr = np.stack(slabs, axis=0) if slabs else np.zeros((0,))
        if xy_downsample > 1 and arr.ndim == 3:
            arr = block_mean_downsample(arr, xy_downsample)
        sampled[c] = arr

    pairwise_shifts: dict[str, list[float]] = {}
    per_slice: dict[str, list[list[float]]] = {}

    for i in range(nc):
        for j in range(nc):
            if i >= j:
                continue
            ref_arr = sampled[i]
            mov_arr = sampled[j]
            shifts: list[np.ndarray] = []
            for z in range(ref_arr.shape[0]):
                ref = ref_arr[z]
                mov = mov_arr[z]
                if ref.std() < 1e-6 or mov.std() < 1e-6:
                    continue
                shifts.append(_phase_shift(ref, mov))
            key = f"{labels[i]}->{labels[j]}"
            if shifts:
                arr_s = np.stack(shifts, axis=0)
                # Scale back to original-pixel space if we downsampled.
                if xy_downsample > 1:
                    arr_s = arr_s * xy_downsample
                med = np.median(arr_s, axis=0).tolist()
                pairwise_shifts[key] = med
                per_slice[key] = arr_s.tolist()
            else:
                pairwise_shifts[key] = [0.0, 0.0]
                per_slice[key] = []

    return {
        "pairwise_shifts": pairwise_shifts,
        "pairwise_shifts_per_slice": per_slice,
        "n_samples": len(z_idx),
        "z_sampled": z_idx,
        "channels": labels,
    }
