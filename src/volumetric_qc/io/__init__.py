"""Lazy chunked I/O for OME-Zarr, OME-TIFF, NIfTI."""

from volumetric_qc.io.readers import LazyVolume, open_volume
from volumetric_qc.io.chunked import iter_z_chunks, block_mean_downsample

__all__ = ["LazyVolume", "open_volume", "iter_z_chunks", "block_mean_downsample"]
