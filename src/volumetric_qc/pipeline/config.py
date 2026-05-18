"""Pydantic config schemas and per-modality presets.

Thresholds determine which metric values are considered "pass" vs "fail" vs "warn".
Presets bundle defaults appropriate for a tissue-clearing modality. Users can override
any field via a YAML config; missing fields fall back to the preset defaults.

Example
-------
>>> from volumetric_qc.pipeline.config import load_preset
>>> cfg = load_preset("shield")
>>> cfg.thresholds.intensity_drift_max
0.25
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


Preset = Literal["shield", "idisco", "clarity", "generic"]


class MetricThresholds(BaseModel):
    """Per-metric thresholds. Each metric has an *acceptable* range / max.

    All values are dimensionless ratios unless noted otherwise. Values outside
    these ranges trigger a "warn" or "fail" flag in the QC report.
    """

    # Intensity drift across z (relative slope of mean intensity vs depth).
    intensity_drift_max: float = Field(0.30, ge=0.0, description="Max acceptable normalized intensity slope across z.")
    intensity_cv_max: float = Field(0.50, ge=0.0, description="Max coeff of variation of per-slice mean intensity.")

    # Sharpness — per-slice Laplacian variance must exceed this floor (relative to volume max).
    sharpness_min_relative: float = Field(0.05, ge=0.0, le=1.0, description="Min Laplacian variance relative to in-focus slice.")
    sharpness_outlier_zscore: float = Field(3.0, ge=0.0, description="Z-score above which a z-slice is flagged as out of focus.")

    # Background uniformity.
    background_cv_max: float = Field(0.40, ge=0.0, description="Max coefficient of variation of background patches.")
    autofluorescence_ratio_max: float = Field(0.30, ge=0.0, description="Max ratio of background-mode to signal-mean intensity.")

    # Channel bleed-through. Acceptable normalized cross-correlation between channels.
    channel_bleed_corr_max: float = Field(0.40, ge=0.0, le=1.0, description="Max acceptable cross-channel correlation in signal regions.")

    # Cross-channel registration error (in voxels).
    registration_shift_max_voxels: float = Field(2.0, ge=0.0, description="Max acceptable per-axis registration shift in voxels.")

    # Stripe artifacts (FFT energy ratio along a directional band).
    stripe_energy_ratio_max: float = Field(0.20, ge=0.0, description="Max FFT energy in stripe band / total energy.")

    # Bubbles per slice (blob count).
    bubbles_per_slice_max: int = Field(5, ge=0, description="Max bubbles permitted per z-slice.")

    # Folding / tear discontinuities (gradient outlier fraction).
    folding_outlier_fraction_max: float = Field(0.02, ge=0.0, le=1.0, description="Max fraction of gradient outlier pixels.")

    # Clearing residue (high-frequency lipid speckle energy).
    clearing_residue_max: float = Field(0.15, ge=0.0, description="Max high-frequency speckle energy fraction.")


class SamplingConfig(BaseModel):
    """How to subsample the volume for metric estimation.

    A TB-scale volume cannot be processed in full for every metric; some metrics
    (e.g. blob detection) are run on a stratified sample of z-slices. Other metrics
    (e.g. intensity drift) need every z-slice but only a downsampled (y, x) view.
    """

    z_stride: int = Field(1, ge=1, description="Take every Nth z-slice for per-slice metrics.")
    xy_downsample: int = Field(1, ge=1, description="Downsample factor in x and y (block-mean).")
    blob_z_sample: int = Field(20, ge=1, description="Number of z-slices to sample for blob/bubble detection.")
    fft_tile_size: int = Field(512, ge=64, description="Tile size for FFT-based stripe detection.")
    random_seed: int = Field(42, description="Seed for any random sampling.")


class MetricToggles(BaseModel):
    """Enable/disable individual metric families."""

    intensity: bool = True
    sharpness: bool = True
    background: bool = True
    channel_bleed: bool = True
    registration: bool = True
    stripes: bool = True
    bubbles: bool = True
    folding: bool = True
    clearing_residue: bool = True


class QCConfig(BaseModel):
    """Top-level config bundling thresholds, sampling, and toggles.

    Parameters
    ----------
    preset
        Modality name. Determines default threshold values.
    thresholds
        :class:`MetricThresholds` instance. Overrides preset defaults.
    sampling
        :class:`SamplingConfig` instance controlling subsampling.
    metrics
        :class:`MetricToggles` enabling/disabling metrics.
    channels
        Optional human-readable channel labels (length must match volume channels).
    voxel_size_um
        Voxel size in micrometers as (z, y, x). Used to convert pixel-space metrics
        (e.g. registration shifts) to physical units in the report.
    """

    preset: Preset = "generic"
    thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    metrics: MetricToggles = Field(default_factory=MetricToggles)
    channels: list[str] | None = None
    voxel_size_um: tuple[float, float, float] | None = None

    @field_validator("voxel_size_um", mode="before")
    @classmethod
    def _coerce_voxel_size(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(float(x) for x in v)
        raise ValueError("voxel_size_um must be a length-3 sequence of floats.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QCConfig":
        """Load a config from a YAML file. Unknown keys are rejected."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this config to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Modality presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    # SHIELD (Park et al. 2018) preserves protein and lipid; tends to have more
    # autofluorescence residue and mild refractive index speckle.
    "shield": {
        "preset": "shield",
        "thresholds": {
            "intensity_drift_max": 0.25,
            "autofluorescence_ratio_max": 0.35,
            "clearing_residue_max": 0.20,
            "background_cv_max": 0.45,
        },
    },
    # iDISCO+ (Renier et al. 2016) — solvent-based, strong clearing, but can
    # introduce tissue shrinkage and folding artifacts.
    "idisco": {
        "preset": "idisco",
        "thresholds": {
            "intensity_drift_max": 0.35,
            "folding_outlier_fraction_max": 0.03,
            "clearing_residue_max": 0.10,
            "registration_shift_max_voxels": 3.0,
        },
    },
    # CLARITY (Chung et al. 2013) — hydrogel embedding, can have channel
    # bleed-through from residual lipid autofluorescence.
    "clarity": {
        "preset": "clarity",
        "thresholds": {
            "intensity_drift_max": 0.30,
            "channel_bleed_corr_max": 0.45,
            "clearing_residue_max": 0.18,
            "bubbles_per_slice_max": 8,
        },
    },
    "generic": {"preset": "generic"},
}


def load_preset(name: Preset | str) -> QCConfig:
    """Return a :class:`QCConfig` populated with the named preset.

    Parameters
    ----------
    name
        One of ``shield``, ``idisco``, ``clarity``, ``generic``.

    Returns
    -------
    QCConfig
        Config with modality-specific threshold overrides applied.
    """
    if name not in _PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose from {sorted(_PRESETS)}.")
    return QCConfig.model_validate(_PRESETS[name])
