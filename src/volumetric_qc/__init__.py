"""volumetric-qc — QC and artifact detection for large 3D fluorescence microscopy volumes.

The package is organized into a few cooperating layers:

* :mod:`volumetric_qc.io` — lazy chunked readers for OME-Zarr, OME-TIFF, NIfTI.
* :mod:`volumetric_qc.metrics` — per-volume / per-channel QC metrics.
* :mod:`volumetric_qc.pipeline` — orchestrates metrics with dask and pydantic config.
* :mod:`volumetric_qc.reports` — JSON + standalone HTML dashboards.
* :mod:`volumetric_qc.synthetic` — synthetic artifact injection for unit-testing metric sensitivity.

The top-level :func:`run_qc` convenience function provides a one-call entry point
that mirrors the CLI.
"""

from __future__ import annotations

__version__ = "0.1.0"

from volumetric_qc.io.readers import open_volume
from volumetric_qc.pipeline.config import QCConfig, load_preset
from volumetric_qc.pipeline.runner import run_qc

__all__ = [
    "__version__",
    "open_volume",
    "QCConfig",
    "load_preset",
    "run_qc",
]
