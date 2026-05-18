"""Synthetic volume generation and artifact injection.

These helpers are used in unit tests to verify metric sensitivity (e.g. that
the stripe detector flags a striped volume but not a clean one) and in the
synthetic_artifacts example notebook.
"""

from volumetric_qc.synthetic.generators import (
    clean_volume,
    inject_intensity_drift,
    inject_stripes,
    inject_bubbles,
    inject_bleed_through,
    inject_registration_shift,
    inject_clearing_residue,
    inject_focus_blur,
    inject_folding,
)

__all__ = [
    "clean_volume",
    "inject_intensity_drift",
    "inject_stripes",
    "inject_bubbles",
    "inject_bleed_through",
    "inject_registration_shift",
    "inject_clearing_residue",
    "inject_focus_blur",
    "inject_folding",
]
