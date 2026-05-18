"""End-to-end pipeline + config + report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from volumetric_qc import QCConfig, load_preset, open_volume, run_qc
from volumetric_qc.pipeline.batch import detect_outliers, feature_vector
from volumetric_qc.pipeline.runner import QCResult
from volumetric_qc.reports import write_html_report, write_json_summary


# ---------------------------------------------------------------------------
# Config / presets
# ---------------------------------------------------------------------------


def test_load_preset_shield_overrides_defaults():
    cfg = load_preset("shield")
    assert cfg.preset == "shield"
    # SHIELD preset tightens intensity drift.
    assert cfg.thresholds.intensity_drift_max == 0.25


def test_load_preset_unknown_raises():
    with pytest.raises(ValueError):
        load_preset("not-a-real-preset")


def test_config_roundtrip_yaml(tmp_path: Path):
    cfg = load_preset("idisco")
    cfg.channels = ["DAPI", "GFP"]
    cfg.voxel_size_um = (4.0, 1.6, 1.6)
    cfg.to_yaml(tmp_path / "qc.yaml")

    cfg2 = QCConfig.from_yaml(tmp_path / "qc.yaml")
    assert cfg2.preset == "idisco"
    assert cfg2.channels == ["DAPI", "GFP"]
    assert cfg2.voxel_size_um == (4.0, 1.6, 1.6)


def test_config_voxel_size_validation():
    cfg = QCConfig.model_validate({"voxel_size_um": [4.0, 1.6, 1.6]})
    assert cfg.voxel_size_um == (4.0, 1.6, 1.6)
    with pytest.raises(Exception):
        QCConfig.model_validate({"voxel_size_um": [4.0, 1.6]})


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def test_run_qc_returns_qcresult(tiny_clean_volume):
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 32
    res = run_qc(open_volume(tiny_clean_volume), cfg)
    assert isinstance(res, QCResult)
    assert len(res.flags) > 0
    assert "intensity" in res.metrics
    assert isinstance(res.to_dict(), dict)


def test_run_qc_passes_clean_volume_intensity(tiny_clean_volume):
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 32
    res = run_qc(open_volume(tiny_clean_volume), cfg)
    drift_flag = next(f for f in res.flags if f.name.startswith("intensity_drift::"))
    assert drift_flag.severity in ("pass", "warn", "fail")  # well-formed


def test_run_qc_disabled_metric_absent(tiny_clean_volume):
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 32
    cfg.metrics.stripes = False
    res = run_qc(open_volume(tiny_clean_volume), cfg)
    assert "stripes" not in res.metrics


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def test_write_json_summary(tmp_path: Path, tiny_clean_volume):
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 32
    res = run_qc(open_volume(tiny_clean_volume), cfg)
    p = write_json_summary(res, tmp_path / "summary.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "metrics" in data and "flags" in data and "status" in data


def test_write_html_report(tmp_path: Path, tiny_clean_volume):
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 32
    res = run_qc(open_volume(tiny_clean_volume), cfg)
    p = write_html_report(res, tmp_path / "dash.html", title="test")
    html = p.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    # Should embed plotly.
    assert "plotly" in html.lower()


# ---------------------------------------------------------------------------
# Batch outlier detection
# ---------------------------------------------------------------------------


def test_feature_vector_fixed_length(tiny_clean_volume):
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 64
    res = run_qc(open_volume(tiny_clean_volume), cfg)
    fv = feature_vector(res)
    assert fv.ndim == 1
    assert fv.shape[0] >= 8  # at least 8 features defined


def test_detect_outliers_runs_on_three_samples(tiny_clean_volume):
    import numpy as np
    cfg = QCConfig()
    cfg.sampling.blob_z_sample = 3
    cfg.sampling.fft_tile_size = 32

    # Three slightly different volumes — one with extreme drift.
    from volumetric_qc.synthetic import inject_intensity_drift
    rs = []
    for slope in (0.05, 0.05, 0.7):
        vol = inject_intensity_drift(tiny_clean_volume, slope=slope)
        rs.append(run_qc(open_volume(vol), cfg))

    bor = detect_outliers(rs, sample_names=["a", "b", "c"])
    assert len(bor.outlier_flags) == 3
    assert bor.feature_matrix.shape == (3, len(bor.feature_names))
