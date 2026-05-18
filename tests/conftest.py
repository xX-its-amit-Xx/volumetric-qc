"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from volumetric_qc.synthetic import clean_volume


@pytest.fixture(scope="session")
def small_clean_volume() -> np.ndarray:
    """Small (2, 16, 96, 96) clean volume — fast for unit tests."""
    return clean_volume(shape=(2, 16, 96, 96), n_cells=200, background=100.0, signal=600.0, seed=0)


@pytest.fixture(scope="session")
def tiny_clean_volume() -> np.ndarray:
    """Tiny (1, 8, 48, 48) volume for the very fastest tests."""
    return clean_volume(shape=(1, 8, 48, 48), n_cells=80, background=100.0, signal=500.0, seed=1)
