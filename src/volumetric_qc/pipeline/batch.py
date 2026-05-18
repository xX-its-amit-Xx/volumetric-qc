"""Batch-level outlier detection across multiple QC runs.

Given a list of :class:`QCResult` instances (one per sample), build a fixed-
length feature vector per sample (summary metric scalars only), then flag
outlier samples using:

* Per-feature z-score against the batch median (robust MAD-based).
* IsolationForest on the joint feature space.
* Optional 2D embedding via PCA + UMAP for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from volumetric_qc.pipeline.runner import QCResult


# Fixed ordering of summary features. Channel-bearing features are aggregated
# across channels (max of |value|) so the vector length is sample-independent.
_FEATURE_KEYS = [
    "intensity_drift_max_abs",
    "intensity_cv_max",
    "sharpness_min_relative",
    "background_cv_max",
    "autofluor_ratio_max",
    "stripe_energy_max",
    "bubbles_max",
    "folding_max",
    "clearing_residue_max",
    "bleed_corr_max",
    "registration_shift_max",
]


def feature_vector(result: QCResult) -> np.ndarray:
    """Build a fixed-length summary feature vector from a single QCResult."""

    def _agg(metric: str, key: str, reducer=max) -> float:
        m = result.metrics.get(metric) or {}
        if not isinstance(m, dict):
            return 0.0
        vals = []
        for v in m.values():
            if isinstance(v, dict) and key in v:
                vals.append(abs(float(v[key])))
        return float(reducer(vals)) if vals else 0.0

    bleed = result.metrics.get("channel_bleed") or {}
    bleed_corrs = list(bleed.get("pairwise_corr", {}).values())
    bleed_max = max((abs(v) for v in bleed_corrs), default=0.0)

    reg = result.metrics.get("registration") or {}
    reg_shifts = reg.get("pairwise_shifts", {})
    reg_max = 0.0
    for sh in reg_shifts.values():
        reg_max = max(reg_max, float(np.max(np.abs(sh))))

    feats = {
        "intensity_drift_max_abs": _agg("intensity", "drift_slope"),
        "intensity_cv_max": _agg("intensity", "cv"),
        "sharpness_min_relative": _agg("sharpness", "min_relative", reducer=min),
        "background_cv_max": _agg("background", "background_cv"),
        "autofluor_ratio_max": _agg("background", "autofluor_ratio"),
        "stripe_energy_max": _agg("stripes", "mean_ratio"),
        "bubbles_max": _agg("bubbles", "max_per_slice"),
        "folding_max": _agg("folding", "outlier_fraction"),
        "clearing_residue_max": _agg("clearing_residue", "speckle_energy"),
        "bleed_corr_max": bleed_max,
        "registration_shift_max": reg_max,
    }
    return np.array([feats[k] for k in _FEATURE_KEYS], dtype=np.float64)


@dataclass
class BatchOutlierResult:
    sample_names: list[str]
    feature_matrix: np.ndarray  # (n_samples, n_features)
    feature_names: list[str]
    robust_z: np.ndarray  # (n_samples, n_features) — MAD-based z-score
    sample_max_robust_z: np.ndarray  # (n_samples,)
    outlier_flags: list[bool]
    isolation_score: np.ndarray | None
    pca_2d: np.ndarray | None
    umap_2d: np.ndarray | None


def detect_outliers(
    results: list[QCResult],
    sample_names: list[str] | None = None,
    *,
    robust_z_threshold: float = 4.0,
    use_isolation_forest: bool = True,
    embed: bool = True,
    random_state: int = 42,
) -> BatchOutlierResult:
    """Identify outlier samples in a batch of QC runs.

    Parameters
    ----------
    results
        List of :class:`QCResult` (one per sample).
    sample_names
        Optional human-readable labels.
    robust_z_threshold
        A sample is flagged if any feature exceeds this MAD-based z-score.
    use_isolation_forest
        If True, also fit an IsolationForest and include its anomaly score.
    embed
        If True, compute 2D PCA and (if ``umap-learn`` installed) UMAP embeddings.

    Returns
    -------
    BatchOutlierResult
        Bundle of feature matrix, robust z-scores, outlier flags, and embeddings.
    """
    if not results:
        raise ValueError("results list is empty")
    names = sample_names if sample_names else [f"sample_{i}" for i in range(len(results))]
    X = np.stack([feature_vector(r) for r in results], axis=0)

    # Robust z-score per feature: (x - median) / (1.4826 * MAD).
    med = np.median(X, axis=0, keepdims=True)
    mad = np.median(np.abs(X - med), axis=0, keepdims=True)
    denom = 1.4826 * mad
    denom[denom == 0] = 1.0
    rz = (X - med) / denom
    sample_max_rz = np.max(np.abs(rz), axis=1)
    outlier_flags = (sample_max_rz > robust_z_threshold).tolist()

    iso_scores: np.ndarray | None = None
    if use_isolation_forest and X.shape[0] >= 3:
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(random_state=random_state, contamination="auto")
            iso.fit(X)
            iso_scores = -iso.score_samples(X)  # higher = more anomalous
        except Exception:
            iso_scores = None

    pca_2d: np.ndarray | None = None
    umap_2d: np.ndarray | None = None
    if embed and X.shape[0] >= 3:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, X.shape[1]))
            pca_2d = pca.fit_transform(X)
        except Exception:
            pca_2d = None
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(5, X.shape[0] - 1))
            umap_2d = reducer.fit_transform(X)
        except Exception:
            umap_2d = None

    return BatchOutlierResult(
        sample_names=names,
        feature_matrix=X,
        feature_names=list(_FEATURE_KEYS),
        robust_z=rz,
        sample_max_robust_z=sample_max_rz,
        outlier_flags=outlier_flags,
        isolation_score=iso_scores,
        pca_2d=pca_2d,
        umap_2d=umap_2d,
    )
