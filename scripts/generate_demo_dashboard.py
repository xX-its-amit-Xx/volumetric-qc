"""Generate the README demo dashboard from a synthetic artifact-laden volume.

This script is checked in so the dashboard screenshot in the README is
reproducible: ``python scripts/generate_demo_dashboard.py`` regenerates
the volume, runs the full pipeline, writes the HTML dashboard, and saves a
PNG screenshot to assets/.

The volume is synthetic but its artifacts are realistic in character — see
the synthetic_artifacts notebook for an end-to-end walk-through.
"""

from __future__ import annotations

from pathlib import Path

from volumetric_qc import open_volume, run_qc
from volumetric_qc.pipeline.config import load_preset
from volumetric_qc.reports import write_html_report, write_json_summary
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


def build_demo_volume() -> "open_volume":
    """Build a 3-channel SHIELD-style volume with seven injected artifact families."""
    vol = clean_volume(shape=(3, 64, 320, 320), n_cells=1500, background=120.0, signal=900.0, seed=7)
    vol = inject_intensity_drift(vol, slope=0.45)
    vol = inject_stripes(vol, amplitude=0.18, period=12, channel=1)
    vol = inject_bubbles(vol, n_bubbles=18, channel=0, seed=3)
    vol = inject_bleed_through(vol, source=1, target=2, factor=0.35)
    vol = inject_registration_shift(vol, channel=2, shift=(3, 4))
    vol = inject_clearing_residue(vol, amplitude=0.4, channel=0, seed=5)
    vol = inject_focus_blur(vol, z_indices=[5, 6, 7, 58, 59, 60], sigma=3.5)
    return open_volume(vol)


def main(output_dir: Path = Path("demo_output"), assets_dir: Path = Path("assets")) -> None:
    lv = build_demo_volume()
    cfg = load_preset("shield")
    cfg.channels = ["DAPI", "GFP", "RFP"]
    cfg.voxel_size_um = (4.0, 1.6, 1.6)
    cfg.sampling.z_stride = 1
    cfg.sampling.xy_downsample = 2
    cfg.sampling.blob_z_sample = 12
    cfg.sampling.fft_tile_size = 256

    result = run_qc(lv, cfg, progress=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = write_html_report(
        result,
        output_dir / "qc_dashboard.html",
        title="Volumetric QC – Synthetic SHIELD demo",
    )
    json_path = write_json_summary(result, output_dir / "qc_summary.json")

    print()
    print(f"overall_pass : {result.overall_pass}")
    print(f"n_fail       : {result.n_fail}")
    print(f"n_warn       : {result.n_warn}")
    print(f"elapsed      : {result.elapsed_seconds:.2f}s")
    print(f"dashboard    : {html_path}")
    print(f"summary      : {json_path}")


if __name__ == "__main__":
    main()
