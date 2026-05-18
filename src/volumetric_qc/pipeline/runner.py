"""Pipeline orchestration — runs configured metrics over a :class:`LazyVolume`.

The runner is intentionally a thin layer: it discovers enabled metrics from the
config, dispatches them with the appropriate sampling strategy, collects per-
channel results, evaluates threshold flags, and packages everything into a
:class:`QCResult` that the report layer can serialize.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from volumetric_qc.io.readers import LazyVolume, open_volume
from volumetric_qc.pipeline.config import QCConfig


@dataclass
class FlagStatus:
    """Outcome of a single threshold check.

    Attributes
    ----------
    name
        Short identifier for the check (e.g. ``intensity_drift``).
    value
        Numeric value being checked.
    threshold
        Threshold value used for the comparison.
    passed
        True if the metric is within acceptable bounds.
    severity
        One of ``"pass"``, ``"warn"``, ``"fail"``.
    message
        Human-readable description of the flag.
    """

    name: str
    value: float
    threshold: float
    passed: bool
    severity: str
    message: str


@dataclass
class QCResult:
    """Bundle of all per-channel metric outputs plus pass/fail flags.

    Attributes
    ----------
    volume_info
        Dict with shape, dtype, voxel size, source path.
    metrics
        Nested dict: ``metrics[metric_name][channel_label]`` -> per-metric output.
        Each metric output is itself a dict with keys like ``per_z``, ``summary``.
    flags
        List of :class:`FlagStatus` entries summarizing pass/fail per check.
    config
        The :class:`QCConfig` used for this run (serialized).
    elapsed_seconds
        Wall-clock time for the run.
    """

    volume_info: dict[str, Any]
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    flags: list[FlagStatus] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def overall_pass(self) -> bool:
        return all(f.severity != "fail" for f in self.flags)

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.flags if f.severity == "warn")

    @property
    def n_fail(self) -> int:
        return sum(1 for f in self.flags if f.severity == "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_info": self.volume_info,
            "metrics": _jsonify(self.metrics),
            "flags": [vars(f) for f in self.flags],
            "config": self.config,
            "elapsed_seconds": self.elapsed_seconds,
            "overall_pass": self.overall_pass,
            "n_warn": self.n_warn,
            "n_fail": self.n_fail,
        }


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy arrays / scalars to plain Python types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _channel_label(idx: int, names: list[str] | None) -> str:
    if names and idx < len(names):
        return names[idx]
    return f"channel_{idx}"


def run_qc(
    source: Any,
    config: QCConfig | None = None,
    *,
    progress: bool = False,
) -> QCResult:
    """Run all enabled QC metrics on a volume.

    Parameters
    ----------
    source
        Anything :func:`open_volume` accepts (path, ndarray, dask array, or an
        existing :class:`LazyVolume`).
    config
        :class:`QCConfig` controlling thresholds, sampling, and toggles.
        Defaults to ``QCConfig()`` (generic preset).
    progress
        If True, print one line per metric as it completes.

    Returns
    -------
    QCResult
        Bundle of metric outputs and pass/fail flags. Pass to the reports
        module to render JSON / HTML.
    """
    # Lazy imports so the package __init__ does not pull in scikit-image.
    from volumetric_qc.metrics import (
        intensity as m_intensity,
        sharpness as m_sharpness,
        background as m_background,
        channel_bleed as m_channel_bleed,
        registration as m_registration,
        artifacts as m_artifacts,
        clearing as m_clearing,
    )

    t0 = time.time()
    config = config or QCConfig()
    vol = source if isinstance(source, LazyVolume) else open_volume(source)
    if config.voxel_size_um is None and vol.voxel_size_um is not None:
        config = config.model_copy(update={"voxel_size_um": vol.voxel_size_um})
    if config.channels is None and vol.channel_names is not None:
        config = config.model_copy(update={"channels": vol.channel_names})

    channel_labels = [_channel_label(i, config.channels) for i in range(vol.nchannels)]
    # Propagate human-readable channel names to the volume so cross-channel
    # metrics surface them in their result dicts.
    if config.channels:
        vol.channel_names = list(config.channels[: vol.nchannels])
    result = QCResult(
        volume_info=vol.describe(),
        config=config.model_dump(mode="json"),
    )

    def _log(msg: str) -> None:
        if progress:
            print(f"[volumetric-qc] {msg}")

    # ---- per-channel metrics ------------------------------------------------
    sampling = config.sampling
    tog = config.metrics
    thr = config.thresholds

    for c in range(vol.nchannels):
        label = channel_labels[c]
        zyx = vol.channel(c)

        if tog.intensity:
            _log(f"intensity drift / cv for {label}")
            res = m_intensity.intensity_profile(zyx, z_stride=sampling.z_stride, xy_downsample=sampling.xy_downsample)
            result.metrics.setdefault("intensity", {})[label] = res
            result.flags.append(_flag_threshold(
                f"intensity_drift::{label}", abs(res["drift_slope"]),
                thr.intensity_drift_max, "Normalized intensity slope across z",
            ))
            result.flags.append(_flag_threshold(
                f"intensity_cv::{label}", res["cv"], thr.intensity_cv_max,
                "Coefficient of variation of per-slice mean intensity",
            ))

        if tog.sharpness:
            _log(f"sharpness for {label}")
            res = m_sharpness.sharpness_profile(zyx, z_stride=sampling.z_stride, xy_downsample=sampling.xy_downsample)
            result.metrics.setdefault("sharpness", {})[label] = res
            result.flags.append(_flag_threshold(
                f"sharpness_min::{label}",
                res["min_relative"],
                thr.sharpness_min_relative,
                "Minimum Laplacian variance relative to peak slice",
                direction="ge",
            ))
            result.flags.append(_flag_threshold(
                f"sharpness_outliers::{label}",
                float(len(res["outlier_z"])),
                0,  # any outlier triggers warn
                "Count of z-slices flagged as out-of-focus",
                severity_if_fail="warn",
            ))

        if tog.background:
            _log(f"background for {label}")
            res = m_background.background_uniformity(zyx, z_stride=sampling.z_stride, xy_downsample=sampling.xy_downsample)
            result.metrics.setdefault("background", {})[label] = res
            result.flags.append(_flag_threshold(
                f"background_cv::{label}", res["background_cv"], thr.background_cv_max,
                "Background patch coefficient of variation",
            ))
            result.flags.append(_flag_threshold(
                f"autofluorescence::{label}", res["autofluor_ratio"], thr.autofluorescence_ratio_max,
                "Background-mode / signal-mean ratio (autofluorescence proxy)",
            ))

        if tog.stripes:
            _log(f"stripe artifacts for {label}")
            res = m_artifacts.stripe_energy(zyx, z_stride=sampling.z_stride, tile_size=sampling.fft_tile_size, n_samples=sampling.blob_z_sample, seed=sampling.random_seed)
            result.metrics.setdefault("stripes", {})[label] = res
            result.flags.append(_flag_threshold(
                f"stripe_energy::{label}", res["mean_ratio"], thr.stripe_energy_ratio_max,
                "FFT energy concentrated in stripe direction",
            ))

        if tog.bubbles:
            _log(f"bubbles for {label}")
            res = m_artifacts.bubble_count(zyx, n_samples=sampling.blob_z_sample, seed=sampling.random_seed)
            result.metrics.setdefault("bubbles", {})[label] = res
            result.flags.append(_flag_threshold(
                f"bubbles::{label}", res["max_per_slice"], thr.bubbles_per_slice_max,
                "Maximum bubble count on any sampled slice",
            ))

        if tog.folding:
            _log(f"folding for {label}")
            res = m_artifacts.folding_score(zyx, n_samples=sampling.blob_z_sample, seed=sampling.random_seed)
            result.metrics.setdefault("folding", {})[label] = res
            result.flags.append(_flag_threshold(
                f"folding::{label}", res["outlier_fraction"], thr.folding_outlier_fraction_max,
                "Fraction of gradient-discontinuity pixels (folding/tear proxy)",
            ))

        if tog.clearing_residue:
            _log(f"clearing residue for {label}")
            res = m_clearing.clearing_residue(zyx, n_samples=sampling.blob_z_sample, seed=sampling.random_seed)
            result.metrics.setdefault("clearing_residue", {})[label] = res
            result.flags.append(_flag_threshold(
                f"clearing_residue::{label}", res["speckle_energy"], thr.clearing_residue_max,
                "High-frequency speckle energy (residual lipid / RI mismatch proxy)",
            ))

    # ---- cross-channel metrics ---------------------------------------------
    if vol.nchannels >= 2:
        if tog.channel_bleed:
            _log("channel bleed-through")
            res = m_channel_bleed.bleed_through(vol, z_stride=sampling.z_stride, xy_downsample=sampling.xy_downsample)
            result.metrics["channel_bleed"] = res
            for pair_key, corr in res["pairwise_corr"].items():
                result.flags.append(_flag_threshold(
                    f"bleed::{pair_key}", abs(corr), thr.channel_bleed_corr_max,
                    f"Cross-channel correlation in signal pixels ({pair_key})",
                ))

        if tog.registration:
            _log("cross-channel registration")
            res = m_registration.cross_channel_shifts(vol, z_stride=sampling.z_stride, xy_downsample=sampling.xy_downsample)
            result.metrics["registration"] = res
            for pair, shift in res["pairwise_shifts"].items():
                max_shift = float(np.max(np.abs(shift)))
                result.flags.append(_flag_threshold(
                    f"reg_shift::{pair}", max_shift, thr.registration_shift_max_voxels,
                    f"Max per-axis registration shift between {pair} (voxels)",
                ))

    result.elapsed_seconds = time.time() - t0
    return result


def _flag_threshold(
    name: str,
    value: float,
    threshold: float,
    message: str,
    *,
    direction: str = "le",
    severity_if_fail: str = "fail",
) -> FlagStatus:
    """Build a FlagStatus from a value/threshold comparison.

    ``direction="le"`` means value must be <= threshold to pass.
    ``direction="ge"`` means value must be >= threshold to pass.
    """
    if direction == "le":
        passed = bool(value <= threshold)
    elif direction == "ge":
        passed = bool(value >= threshold)
    else:
        raise ValueError(f"direction must be 'le' or 'ge', got {direction!r}")
    if passed:
        severity = "pass"
    else:
        # Warn if within 1.5x threshold; fail beyond.
        ratio = (value / threshold) if (direction == "le" and threshold > 0) else 0
        severity = "warn" if (direction == "le" and threshold > 0 and ratio < 1.5) else severity_if_fail
        if severity_if_fail == "warn":
            severity = "warn"
    return FlagStatus(
        name=name,
        value=float(value),
        threshold=float(threshold),
        passed=passed,
        severity=severity,
        message=message,
    )
