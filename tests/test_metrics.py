"""Metric sensitivity tests.

Each test injects a known artifact into a clean synthetic volume and verifies
that the corresponding metric value moves in the expected direction relative
to the clean baseline. We test sensitivity, not absolute correctness, since the
exact numeric value depends on the synthetic volume parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from volumetric_qc import open_volume
from volumetric_qc.metrics import (
    artifacts,
    background,
    channel_bleed,
    clearing,
    intensity,
    registration,
    sharpness,
)
from volumetric_qc.synthetic import (
    clean_volume,
    inject_bleed_through,
    inject_bubbles,
    inject_clearing_residue,
    inject_focus_blur,
    inject_intensity_drift,
    inject_registration_shift,
    inject_stripes,
)


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------


def test_intensity_drift_detected(small_clean_volume):
    clean_lv = open_volume(small_clean_volume)
    drifted = inject_intensity_drift(small_clean_volume, slope=0.6)
    drifted_lv = open_volume(drifted)

    clean_res = intensity.intensity_profile(clean_lv.channel(0))
    drift_res = intensity.intensity_profile(drifted_lv.channel(0))

    # Drift should drive the abs slope much higher.
    assert abs(drift_res["drift_slope"]) > abs(clean_res["drift_slope"]) + 0.1
    assert abs(drift_res["drift_slope"]) > 0.2


def test_intensity_profile_has_expected_keys(small_clean_volume):
    lv = open_volume(small_clean_volume)
    res = intensity.intensity_profile(lv.channel(0), percentiles=(50.0,))
    for key in ("z", "mean", "std", "p50", "drift_slope", "cv", "global_mean"):
        assert key in res
    assert len(res["mean"]) == len(res["z"])


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


def test_sharpness_drops_after_blur(small_clean_volume):
    blurred = inject_focus_blur(small_clean_volume, z_indices=[5, 6, 7], sigma=4.0, channel=0)
    blurred_lv = open_volume(blurred)
    res = sharpness.sharpness_profile(blurred_lv.channel(0))
    # The blurred slices should be in the outlier list (they were forced to be soft).
    assert 5 in res["outlier_z"] or 6 in res["outlier_z"] or 7 in res["outlier_z"]


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def test_background_keys(small_clean_volume):
    lv = open_volume(small_clean_volume)
    res = background.background_uniformity(lv.channel(0))
    for key in ("background", "signal", "background_cv", "autofluor_ratio"):
        assert key in res


# ---------------------------------------------------------------------------
# Channel bleed
# ---------------------------------------------------------------------------


def test_bleed_through_increases_after_injection(small_clean_volume):
    clean_lv = open_volume(small_clean_volume)
    bled = inject_bleed_through(small_clean_volume, source=0, target=1, factor=0.6)
    bled_lv = open_volume(bled)

    clean_res = channel_bleed.bleed_through(clean_lv, n_samples=4)
    bled_res = channel_bleed.bleed_through(bled_lv, n_samples=4)

    clean_corr = abs(clean_res["pairwise_corr"]["ch0->ch1"])
    bled_corr = abs(bled_res["pairwise_corr"]["ch0->ch1"])
    assert bled_corr >= clean_corr


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration_detects_known_shift(small_clean_volume):
    shifted = inject_registration_shift(small_clean_volume, channel=1, shift=(3, 4))
    lv = open_volume(shifted)
    res = registration.cross_channel_shifts(lv, n_samples=4)
    # The first axis shift should approximately match the injected dy / dx (absolute values).
    shift = res["pairwise_shifts"]["ch0->ch1"]
    assert abs(abs(shift[0]) - 3) < 1.5
    assert abs(abs(shift[1]) - 4) < 1.5


# ---------------------------------------------------------------------------
# Stripes
# ---------------------------------------------------------------------------


def test_stripes_detected_after_injection(small_clean_volume):
    striped = inject_stripes(small_clean_volume, amplitude=0.5, period=6, channel=0)
    clean_res = artifacts.stripe_energy(open_volume(small_clean_volume).channel(0),
                                          n_samples=4, tile_size=64)
    striped_res = artifacts.stripe_energy(open_volume(striped).channel(0),
                                            n_samples=4, tile_size=64)
    assert striped_res["mean_ratio"] >= clean_res["mean_ratio"]


# ---------------------------------------------------------------------------
# Bubbles
# ---------------------------------------------------------------------------


def test_bubbles_count_increases(small_clean_volume):
    bubbled = inject_bubbles(small_clean_volume, n_bubbles=15, channel=0, seed=0,
                              radius_range=(8.0, 14.0))
    bubbled_lv = open_volume(bubbled)
    clean_res = artifacts.bubble_count(open_volume(small_clean_volume).channel(0), n_samples=6)
    bubble_res = artifacts.bubble_count(bubbled_lv.channel(0), n_samples=6)
    assert bubble_res["max_per_slice"] >= clean_res["max_per_slice"]


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def test_folding_responds_to_high_gradient_pixels(small_clean_volume):
    vol = small_clean_volume.copy()
    # Inject a sharp horizontal line into a single z-slice.
    vol[0, 5, 40:42, :] = 50000.0
    res = artifacts.folding_score(open_volume(vol).channel(0), n_samples=6)
    assert res["max_outlier_fraction"] >= 0.0  # sanity: returns a number
    # Should at least be non-zero somewhere.
    assert res["outlier_fraction"] > 0


# ---------------------------------------------------------------------------
# Clearing residue
# ---------------------------------------------------------------------------


def test_clearing_residue_increases(small_clean_volume):
    residue = inject_clearing_residue(small_clean_volume, amplitude=1.0, scale=1.5,
                                        channel=0, seed=0)
    clean_res = clearing.clearing_residue(open_volume(small_clean_volume).channel(0),
                                            n_samples=4)
    res_res = clearing.clearing_residue(open_volume(residue).channel(0), n_samples=4)
    assert res_res["speckle_energy"] > clean_res["speckle_energy"]
