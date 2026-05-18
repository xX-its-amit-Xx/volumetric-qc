"""Lazy chunked readers for OME-Zarr, OME-TIFF, and NIfTI volumes.

All readers return a :class:`LazyVolume` wrapper around a ``dask.array.Array``.
The wrapper exposes a unified ``(C, Z, Y, X)`` indexing contract regardless of
the source format. Single-channel volumes are reshaped to a leading axis of 1
so downstream code never has to special-case channel count.

The point of this layer is that downstream metrics receive a dask array they
can ``.compute()`` chunk-wise — the full TB volume is never materialized in
RAM. Reads happen lazily as metrics request specific z-slices or tiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import dask.array as da
import numpy as np


@dataclass
class LazyVolume:
    """A lazy 4D volume in ``(C, Z, Y, X)`` order.

    Attributes
    ----------
    data
        Dask array of shape ``(C, Z, Y, X)``. Reads happen on ``.compute()``.
    voxel_size_um
        ``(z, y, x)`` voxel size in micrometers, or ``None`` if unknown.
    channel_names
        Optional per-channel labels.
    source
        Path or URI the volume was loaded from.
    fmt
        Format string: ``"ome-zarr"``, ``"ome-tiff"``, ``"nifti"``, or ``"array"``.
    """

    data: da.Array
    voxel_size_um: tuple[float, float, float] | None
    channel_names: list[str] | None
    source: str
    fmt: str

    @property
    def shape(self) -> tuple[int, int, int, int]:
        s = tuple(int(x) for x in self.data.shape)
        if len(s) != 4:
            raise ValueError(f"LazyVolume must be 4D (C,Z,Y,X), got shape {s}.")
        return s  # type: ignore[return-value]

    @property
    def nchannels(self) -> int:
        return self.shape[0]

    @property
    def nz(self) -> int:
        return self.shape[1]

    def channel(self, c: int) -> da.Array:
        """Return the ``(Z, Y, X)`` dask array for channel ``c``."""
        if not 0 <= c < self.nchannels:
            raise IndexError(f"channel {c} out of range [0, {self.nchannels})")
        return self.data[c]

    def z_slice(self, c: int, z: int) -> np.ndarray:
        """Materialize one ``(Y, X)`` slice into a numpy array."""
        return np.asarray(self.data[c, z].compute())

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "format": self.fmt,
            "shape_czyx": list(self.shape),
            "dtype": str(self.data.dtype),
            "voxel_size_um": list(self.voxel_size_um) if self.voxel_size_um else None,
            "channel_names": self.channel_names,
        }


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _open_ome_zarr(path: str | Path) -> LazyVolume:
    """Open an OME-Zarr store. Reads the highest-resolution multiscale level by default."""
    import zarr

    store_path = str(path)
    root = zarr.open(store_path, mode="r")

    # OME-NGFF multiscales metadata lives in the group's attributes.
    attrs = dict(root.attrs) if hasattr(root, "attrs") else {}
    multiscales = attrs.get("multiscales") or []
    voxel_size: tuple[float, float, float] | None = None
    channel_names: list[str] | None = None

    if multiscales:
        # Pick the first multiscale and the highest-resolution dataset.
        ms = multiscales[0]
        datasets = ms.get("datasets") or [{"path": "0"}]
        arr_path = datasets[0]["path"]
        try:
            arr = root[arr_path]
        except KeyError:
            arr = root["0"]

        # Coordinate transformations: first global, then per-dataset.
        scale: list[float] = []
        for ct in (ms.get("coordinateTransformations") or []) + (datasets[0].get("coordinateTransformations") or []):
            if ct.get("type") == "scale":
                scale = list(ct.get("scale") or [])
        if scale:
            axes = ms.get("axes") or []
            # Map axes -> name; we want (z, y, x).
            axis_names = [a.get("name", "").lower() if isinstance(a, dict) else str(a).lower() for a in axes]
            zyx = {}
            for nm, sc in zip(axis_names, scale):
                if nm in ("z", "y", "x"):
                    zyx[nm] = float(sc)
            if {"z", "y", "x"} <= set(zyx):
                voxel_size = (zyx["z"], zyx["y"], zyx["x"])

        omero = attrs.get("omero", {})
        chans = omero.get("channels") if isinstance(omero, dict) else None
        if chans:
            channel_names = [c.get("label") or c.get("name") or f"ch{i}" for i, c in enumerate(chans)]
    else:
        # No multiscales metadata — try common conventions.
        try:
            arr = root["0"]
        except Exception:
            # Maybe the root itself is the array.
            arr = root  # type: ignore[assignment]

    darr = da.from_zarr(arr)
    darr = _to_czyx(darr)
    return LazyVolume(
        data=darr,
        voxel_size_um=voxel_size,
        channel_names=channel_names,
        source=str(path),
        fmt="ome-zarr",
    )


def _open_ome_tiff(path: str | Path) -> LazyVolume:
    """Open an OME-TIFF using tifffile. Whole-array is exposed as a dask array via memmap."""
    import tifffile

    p = str(path)
    with tifffile.TiffFile(p) as tf:
        # Prefer the OME-XML axes order if present.
        series = tf.series[0]
        axes = series.axes  # e.g. "TCZYX", "CZYX", "ZYX", "ZCYX"
        arr = series.asarray(out="memmap") if series.size * series.dtype.itemsize > 5 * 10**8 else series.asarray()
        voxel_size: tuple[float, float, float] | None = None
        # tifffile exposes resolution and OME metadata when present.
        try:
            ome_meta = tf.ome_metadata
        except Exception:
            ome_meta = None
        if ome_meta:
            # Very lightweight parse — full OME parsing is out of scope.
            import re

            phys_x = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', ome_meta)
            phys_y = re.search(r'PhysicalSizeY="([\d.eE+-]+)"', ome_meta)
            phys_z = re.search(r'PhysicalSizeZ="([\d.eE+-]+)"', ome_meta)
            if phys_x and phys_y and phys_z:
                voxel_size = (float(phys_z.group(1)), float(phys_y.group(1)), float(phys_x.group(1)))

    # Normalize axes to CZYX.
    arr = _reorder_to_czyx(np.asarray(arr), axes)
    darr = da.from_array(arr, chunks=_default_chunks(arr.shape))
    return LazyVolume(
        data=darr,
        voxel_size_um=voxel_size,
        channel_names=None,
        source=str(path),
        fmt="ome-tiff",
    )


def _open_nifti(path: str | Path) -> LazyVolume:
    """Open a NIfTI (.nii/.nii.gz). NIfTI is single-channel; we prepend a C=1 axis."""
    import nibabel as nib

    img = nib.load(str(path))
    # Use dataobj for lazy memory-mapped access.
    dataobj = img.dataobj
    # NIfTI axes are conventionally (x, y, z, t). We want (z, y, x); ignore time.
    arr = np.asanyarray(dataobj)
    if arr.ndim == 4:
        # Treat the 4th dim as channels.
        arr = np.moveaxis(arr, -1, 0)
        # Now (C, X, Y, Z) -> (C, Z, Y, X)
        arr = arr.transpose(0, 3, 2, 1)
    elif arr.ndim == 3:
        # (X, Y, Z) -> (Z, Y, X) and add a channel axis.
        arr = arr.transpose(2, 1, 0)[None, ...]
    else:
        raise ValueError(f"Unsupported NIfTI ndim={arr.ndim}")

    voxel_size: tuple[float, float, float] | None = None
    zooms = img.header.get_zooms()
    if len(zooms) >= 3:
        # zooms are (x, y, z[, t]) -> (z, y, x)
        voxel_size = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

    darr = da.from_array(arr, chunks=_default_chunks(arr.shape))
    return LazyVolume(
        data=darr,
        voxel_size_um=voxel_size,
        channel_names=None,
        source=str(path),
        fmt="nifti",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def open_volume(source: str | Path | np.ndarray | da.Array, *, fmt: str | None = None) -> LazyVolume:
    """Open a 3D/4D volume from a path or in-memory array.

    Parameters
    ----------
    source
        Either a filesystem path / URI to an OME-Zarr, OME-TIFF, or NIfTI file,
        or an in-memory numpy / dask array. Arrays are accepted in ``(Z, Y, X)``,
        ``(C, Z, Y, X)``, or ``(Z, C, Y, X)`` order and reshaped to ``(C, Z, Y, X)``.
    fmt
        Force a specific reader (``"ome-zarr"``, ``"ome-tiff"``, ``"nifti"``).
        If omitted, the format is inferred from the path suffix.

    Returns
    -------
    LazyVolume
        Lazy 4D wrapper exposing a dask array of shape ``(C, Z, Y, X)``.

    Notes
    -----
    For OME-Zarr stores with multiscale pyramids, this function reads the
    highest-resolution level. Use the underlying ``zarr`` API directly if you
    need a lower-resolution level.
    """
    if isinstance(source, (np.ndarray, da.Array)):
        arr = source if isinstance(source, da.Array) else da.from_array(source, chunks=_default_chunks(source.shape))
        arr = _to_czyx(arr)
        return LazyVolume(data=arr, voxel_size_um=None, channel_names=None, source="<array>", fmt="array")

    path = Path(source)
    if fmt is None:
        s = str(path).lower()
        if s.endswith(".zarr") or os.path.isdir(path) and (path / ".zarray").exists() or (path / ".zgroup").exists() or (path / "zarr.json").exists():
            fmt = "ome-zarr"
        elif s.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
            fmt = "ome-tiff"
        elif s.endswith((".nii", ".nii.gz")):
            fmt = "nifti"
        else:
            raise ValueError(f"Could not infer format from path {path!s}. Pass fmt= explicitly.")

    if fmt == "ome-zarr":
        return _open_ome_zarr(path)
    if fmt == "ome-tiff":
        return _open_ome_tiff(path)
    if fmt == "nifti":
        return _open_nifti(path)
    raise ValueError(f"Unknown format {fmt!r}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Pick reasonable default chunks. One slab per chunk along z, full y/x."""
    if len(shape) == 3:
        z, y, x = shape
        return (max(1, min(z, 16)), y, x)
    if len(shape) == 4:
        c, z, y, x = shape
        return (1, max(1, min(z, 16)), y, x)
    return tuple(min(s, 64) for s in shape)


def _reorder_to_czyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reorder a numpy array from a tifffile axis string to ``(C, Z, Y, X)``.

    Strips any leading time axis (uses ``T=0``).
    """
    ax = axes.upper()
    # Drop T axis by indexing into 0.
    if "T" in ax:
        idx = [slice(None)] * arr.ndim
        idx[ax.index("T")] = 0  # type: ignore[index]
        arr = arr[tuple(idx)]
        ax = ax.replace("T", "")
    if "S" in ax:  # sample (RGB) axis — collapse with channels
        ax = ax.replace("S", "C")

    # Ensure C, Z, Y, X all present.
    for needed in "ZYX":
        if needed not in ax:
            arr = arr[None, ...]
            ax = needed + ax
    if "C" not in ax:
        arr = arr[None, ...]
        ax = "C" + ax

    target = "CZYX"
    # Build permutation.
    perm = [ax.index(c) for c in target]
    return np.transpose(arr, perm)


def _to_czyx(arr: da.Array) -> da.Array:
    """Coerce a dask array to ``(C, Z, Y, X)`` from 3D / 4D / 5D inputs.

    Heuristic for 4D: if the smallest axis is also leading-or-second, assume that
    is C; otherwise treat as (Z, C, Y, X). For 5D, assume (T, C, Z, Y, X) and take T=0.
    """
    if arr.ndim == 3:
        return arr[None, ...]
    if arr.ndim == 4:
        # OME-Zarr convention is (T, C, Z, Y, X) but multiscale arrays often drop T.
        c, z, y, x = arr.shape
        # If first axis is large and second is small, assume (Z, C, Y, X) and swap.
        if c > z and z <= 8:
            return da.moveaxis(arr, 1, 0)
        return arr
    if arr.ndim == 5:
        # Assume (T, C, Z, Y, X)
        return arr[0]
    if arr.ndim == 2:
        return arr[None, None, ...]
    raise ValueError(f"Cannot coerce array with ndim={arr.ndim} to (C, Z, Y, X)")


def iter_channels(vol: LazyVolume) -> Iterator[tuple[int, da.Array]]:
    """Yield ``(channel_index, dask_array_zyx)`` for each channel."""
    for c in range(vol.nchannels):
        yield c, vol.channel(c)
