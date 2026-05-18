"""QC metrics. Each submodule exposes functions that take a dask array or
LazyVolume and return a JSON-serializable dict with per-slice and summary stats.
"""

from volumetric_qc.metrics import (
    intensity,
    sharpness,
    background,
    channel_bleed,
    registration,
    artifacts,
    clearing,
)

__all__ = [
    "intensity",
    "sharpness",
    "background",
    "channel_bleed",
    "registration",
    "artifacts",
    "clearing",
]
