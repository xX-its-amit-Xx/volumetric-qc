"""Synthetic volume generators and artifact injectors.

The :func:`clean_volume` function generates a structured but artifact-free
multi-channel z-stack — neuron-like blob clouds on a low-intensity background.
The ``inject_*`` functions corrupt that volume with a specific artifact so we
can test metric sensitivity.

All injectors are deterministic given a numpy ``Generator`` seed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def clean_volume(
    shape: tuple[int, int, int, int] = (2, 48, 256, 256),
    *,
    n_cells: int = 800,
    background: float = 80.0,
    signal: float = 1200.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a clean multi-channel z-stack with neuron-like blob structure.

    Parameters
    ----------
    shape
        Output ``(C, Z, Y, X)``.
    n_cells
        Approximate number of distinct cell-like blobs.
    background
        Background intensity offset.
    signal
        Peak per-blob intensity above background.
    seed
        Random seed.

    Returns
    -------
    np.ndarray
        Float32 volume of given shape.
    """
    c, z, y, x = shape
    rng = np.random.default_rng(seed)
    vol = np.full(shape, background, dtype=np.float32)
    # Add tiny Gaussian noise everywhere.
    vol += rng.normal(0, 5, size=shape).astype(np.float32)

    # Place blobs as Gaussians sprinkled in 3D, somewhat correlated across channels
    # to mimic shared anatomy.
    centers = np.stack([
        rng.integers(0, z, size=n_cells),
        rng.integers(0, y, size=n_cells),
        rng.integers(0, x, size=n_cells),
    ], axis=1).astype(np.float32)
    radii = rng.uniform(2.0, 5.0, size=n_cells).astype(np.float32)
    amps = rng.uniform(0.4, 1.0, size=n_cells).astype(np.float32)
    # Different channels see different subsets / amplitudes.
    chan_amp = rng.uniform(0.3, 1.0, size=(c, n_cells)).astype(np.float32)

    z_idx, y_idx, x_idx = np.indices((z, y, x), dtype=np.float32)
    for cc in range(c):
        for k in range(n_cells):
            zc, yc, xc = centers[k]
            r = radii[k]
            # Only render within a tight bounding box per blob.
            zlo, zhi = max(0, int(zc - 3 * r)), min(z, int(zc + 3 * r) + 1)
            ylo, yhi = max(0, int(yc - 3 * r)), min(y, int(yc + 3 * r) + 1)
            xlo, xhi = max(0, int(xc - 3 * r)), min(x, int(xc + 3 * r) + 1)
            if zhi <= zlo or yhi <= ylo or xhi <= xlo:
                continue
            dz = z_idx[zlo:zhi, ylo:yhi, xlo:xhi] - zc
            dy = y_idx[zlo:zhi, ylo:yhi, xlo:xhi] - yc
            dx = x_idx[zlo:zhi, ylo:yhi, xlo:xhi] - xc
            g = np.exp(-(dz * dz + dy * dy + dx * dx) / (2 * r * r))
            vol[cc, zlo:zhi, ylo:yhi, xlo:xhi] += signal * amps[k] * chan_amp[cc, k] * g
    return vol


def inject_intensity_drift(vol: np.ndarray, slope: float = 0.5, *, channel: int | None = None) -> np.ndarray:
    """Multiply each z-slice by a linear factor that ranges from 1-slope/2 at z=0 to 1+slope/2 at z=Z.

    Parameters
    ----------
    vol
        ``(C, Z, Y, X)`` array.
    slope
        Total fractional change end-to-end.
    channel
        If given, only affect that channel; otherwise all channels.
    """
    out = vol.copy()
    z = out.shape[1]
    factors = np.linspace(1 - slope / 2, 1 + slope / 2, z, dtype=np.float32)
    if channel is None:
        for c in range(out.shape[0]):
            out[c] = out[c] * factors[:, None, None]
    else:
        out[channel] = out[channel] * factors[:, None, None]
    return out


def inject_stripes(vol: np.ndarray, amplitude: float = 0.3, period: int = 8, *, channel: int | None = None) -> np.ndarray:
    """Add horizontal stripe pattern by multiplying with a sinusoidal y-modulation."""
    out = vol.copy()
    _, _, y, _ = out.shape
    pattern = 1 + amplitude * np.sin(2 * np.pi * np.arange(y) / period).astype(np.float32)
    pattern = pattern[None, None, :, None]
    channels = [channel] if channel is not None else range(out.shape[0])
    for c in channels:
        out[c] = out[c] * pattern[0]
    return out


def inject_bubbles(
    vol: np.ndarray,
    n_bubbles: int = 30,
    radius_range: tuple[float, float] = (6.0, 14.0),
    *,
    channel: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Punch dark circular regions into random z-slices to simulate bubbles."""
    out = vol.copy()
    rng = np.random.default_rng(seed)
    c, z, y, x = out.shape
    y_idx, x_idx = np.indices((y, x), dtype=np.float32)
    channels = [channel] if channel is not None else range(c)
    for _ in range(n_bubbles):
        zc = int(rng.integers(0, z))
        yc = float(rng.integers(0, y))
        xc = float(rng.integers(0, x))
        r = float(rng.uniform(*radius_range))
        mask = ((y_idx - yc) ** 2 + (x_idx - xc) ** 2) < r * r
        for cc in channels:
            sl = out[cc, zc]
            sl[mask] *= 0.2  # darken the bubble area
    return out


def inject_bleed_through(vol: np.ndarray, source: int = 0, target: int = 1, factor: float = 0.4) -> np.ndarray:
    """Add ``factor * vol[source]`` to ``vol[target]``."""
    out = vol.copy()
    if max(source, target) >= out.shape[0]:
        raise ValueError("source/target channel index out of range")
    out[target] = out[target] + factor * out[source]
    return out


def inject_registration_shift(vol: np.ndarray, channel: int = 1, shift: tuple[int, int] = (3, 4)) -> np.ndarray:
    """Shift one channel by ``(dy, dx)`` voxels using zero-fill at the edges."""
    out = vol.copy()
    dy, dx = shift
    out[channel] = np.roll(out[channel], shift=(dy, dx), axis=(-2, -1))
    return out


def inject_clearing_residue(vol: np.ndarray, amplitude: float = 0.6, scale: float = 1.5, *, channel: int | None = None, seed: int = 0) -> np.ndarray:
    """Add high-frequency speckle noise to mimic residual lipid / RI-mismatch."""
    rng = np.random.default_rng(seed)
    out = vol.copy()
    speckle = rng.normal(0, 1, size=out.shape[1:]).astype(np.float32)
    speckle = ndi.gaussian_filter(speckle, sigma=(0, scale, scale))
    speckle = speckle - speckle.mean()
    speckle = speckle / max(speckle.std(), 1e-6)
    channels = [channel] if channel is not None else range(out.shape[0])
    for c in channels:
        out[c] = out[c] + amplitude * out[c].mean() * speckle
    return out


def inject_focus_blur(vol: np.ndarray, z_indices: list[int], sigma: float = 4.0, *, channel: int | None = None) -> np.ndarray:
    """Blur specific z-slices to simulate localized loss of focus."""
    out = vol.copy()
    channels = [channel] if channel is not None else range(out.shape[0])
    for c in channels:
        for z in z_indices:
            if 0 <= z < out.shape[1]:
                out[c, z] = ndi.gaussian_filter(out[c, z], sigma=sigma)
    return out


def inject_folding(vol: np.ndarray, z_index: int, *, channel: int | None = None, fold_width: int = 12) -> np.ndarray:
    """Add a sharp gradient discontinuity (fold/tear) at one z-slice."""
    out = vol.copy()
    _, _, y, x = out.shape
    yc = y // 2
    band = np.zeros((y, x), dtype=np.float32)
    band[yc - fold_width : yc, :] = out[0, z_index].mean() * 2.0
    band[yc : yc + fold_width, :] = -out[0, z_index].mean() * 0.5
    channels = [channel] if channel is not None else range(out.shape[0])
    for c in channels:
        out[c, z_index] = out[c, z_index] + band
    return out
