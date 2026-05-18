"""Command-line interface — `volumetric-qc run`, `batch`, `dashboard`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from volumetric_qc.io.readers import open_volume
from volumetric_qc.pipeline.config import QCConfig, load_preset
from volumetric_qc.pipeline.runner import run_qc
from volumetric_qc.reports.html import write_html_report
from volumetric_qc.reports.summary import write_json_summary

app = typer.Typer(
    name="volumetric-qc",
    help="QC and artifact detection for large 3D fluorescence microscopy volumes.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_config(preset: str | None, config_path: Path | None) -> QCConfig:
    if config_path is not None:
        cfg = QCConfig.from_yaml(config_path)
    elif preset:
        cfg = load_preset(preset)
    else:
        cfg = QCConfig()
    return cfg


@app.command()
def run(
    source: str = typer.Argument(..., help="Path to OME-Zarr, OME-TIFF, or NIfTI volume."),
    output: Path = typer.Option(Path("qc_output"), "--output", "-o", help="Output directory."),
    preset: Optional[str] = typer.Option(None, "--preset", help="Modality preset (shield, idisco, clarity, generic)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config to override defaults."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Print per-metric progress lines."),
    html: bool = typer.Option(True, "--html/--no-html", help="Emit HTML dashboard."),
    json_summary: bool = typer.Option(True, "--json/--no-json", help="Emit JSON summary."),
    title: str = typer.Option("Volumetric QC Report", "--title", help="Title for the HTML dashboard."),
) -> None:
    """Run QC on a single volume."""
    cfg = _load_config(preset, config)
    typer.echo(f"[volumetric-qc] preset={cfg.preset} source={source}")
    vol = open_volume(source)
    typer.echo(f"[volumetric-qc] loaded shape={vol.shape} dtype={vol.data.dtype}")
    result = run_qc(vol, cfg, progress=progress)
    output.mkdir(parents=True, exist_ok=True)
    if json_summary:
        p = write_json_summary(result, output / "qc_summary.json")
        typer.echo(f"[volumetric-qc] wrote {p}")
    if html:
        p = write_html_report(result, output / "qc_dashboard.html", title=title)
        typer.echo(f"[volumetric-qc] wrote {p}")
    typer.echo(
        f"[volumetric-qc] overall_pass={result.overall_pass} "
        f"n_warn={result.n_warn} n_fail={result.n_fail} elapsed={result.elapsed_seconds:.1f}s"
    )
    if not result.overall_pass:
        raise typer.Exit(code=1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Directory containing one volume per subfolder (or *.tif / *.zarr)."),
    output: Path = typer.Option(Path("qc_batch"), "--output", "-o", help="Output directory."),
    preset: Optional[str] = typer.Option(None, "--preset", help="Modality preset."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config."),
    pattern: str = typer.Option("*", "--pattern", help="Glob pattern for volumes inside input_dir."),
    progress: bool = typer.Option(False, "--progress/--no-progress"),
) -> None:
    """Run QC on every volume in a directory and write per-sample + batch outputs."""
    from volumetric_qc.pipeline.batch import detect_outliers

    cfg = _load_config(preset, config)
    output.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in input_dir.glob(pattern) if p.suffix.lower() in {".tif", ".tiff", ".nii", ".gz"} or p.is_dir())
    if not sources:
        typer.echo(f"No matching volumes in {input_dir}")
        raise typer.Exit(code=2)

    results = []
    names = []
    for src in sources:
        typer.echo(f"[volumetric-qc] processing {src.name}")
        sample_dir = output / src.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        vol = open_volume(src)
        res = run_qc(vol, cfg, progress=progress)
        write_json_summary(res, sample_dir / "qc_summary.json")
        write_html_report(res, sample_dir / "qc_dashboard.html", title=f"QC – {src.name}")
        results.append(res)
        names.append(src.name)

    if len(results) >= 2:
        bor = detect_outliers(results, sample_names=names)
        (output / "batch_outliers.json").write_text(
            json.dumps({
                "sample_names": bor.sample_names,
                "feature_names": bor.feature_names,
                "feature_matrix": bor.feature_matrix.tolist(),
                "robust_z": bor.robust_z.tolist(),
                "sample_max_robust_z": bor.sample_max_robust_z.tolist(),
                "outlier_flags": bor.outlier_flags,
                "isolation_score": bor.isolation_score.tolist() if bor.isolation_score is not None else None,
                "pca_2d": bor.pca_2d.tolist() if bor.pca_2d is not None else None,
                "umap_2d": bor.umap_2d.tolist() if bor.umap_2d is not None else None,
            }, indent=2),
            encoding="utf-8",
        )
        outliers = [n for n, f in zip(bor.sample_names, bor.outlier_flags) if f]
        typer.echo(f"[volumetric-qc] outlier samples ({len(outliers)}): {outliers}")


@app.command()
def dashboard(
    json_summary: Path = typer.Argument(..., exists=True, help="qc_summary.json produced by `volumetric-qc run`."),
    output: Path = typer.Option(Path("qc_dashboard.html"), "--output", "-o"),
) -> None:
    """Regenerate the HTML dashboard from a JSON summary (no recomputation)."""
    # We rebuild a minimal QCResult shell from the JSON so the HTML writer can reuse its plot builders.
    import dataclasses
    from volumetric_qc.pipeline.runner import QCResult, FlagStatus

    data = json.loads(json_summary.read_text(encoding="utf-8"))
    flags = [FlagStatus(**f) for f in data.get("flags", [])]
    result = QCResult(
        volume_info=data.get("volume", {}),
        metrics=data.get("metrics", {}),
        flags=flags,
        config=data.get("config", {}),
        elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
    )
    p = write_html_report(result, output)
    typer.echo(f"[volumetric-qc] wrote {p}")


@app.command()
def presets() -> None:
    """List available modality presets and their threshold deltas."""
    from volumetric_qc.pipeline.config import _PRESETS
    for name, payload in _PRESETS.items():
        typer.echo(f"\n# preset: {name}")
        thr = payload.get("thresholds", {})
        if thr:
            for k, v in thr.items():
                typer.echo(f"  {k} = {v}")
        else:
            typer.echo("  (default thresholds)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
