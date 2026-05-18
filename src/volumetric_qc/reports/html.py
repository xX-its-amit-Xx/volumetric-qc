"""Standalone HTML QC dashboard.

We build a single self-contained HTML file that bundles all plotly figures for
a single QC run. The file can be emailed, archived, or hosted on a static
server. No JavaScript runtime beyond plotly's bundled JS is required.

Sections
--------
* Header with overall pass/fail summary and run metadata.
* Flags table (color-coded by severity).
* Per-channel: intensity profile, sharpness profile, background/signal, stripe energy.
* Cross-channel: bleed-through heatmap, registration shift bars.
* Histograms / scatter plots where appropriate.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from volumetric_qc.pipeline.runner import QCResult


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _intensity_figure(result: QCResult) -> go.Figure | None:
    metrics = result.metrics.get("intensity")
    if not metrics:
        return None
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Per-slice mean intensity (drift)", "Per-slice std"),
        horizontal_spacing=0.12,
    )
    for label, data in metrics.items():
        z = data["z"]
        fig.add_trace(go.Scatter(x=z, y=data["mean"], mode="lines", name=f"mean {label}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=z, y=data["std"], mode="lines", name=f"std {label}", showlegend=False), row=1, col=2)
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    fig.update_xaxes(title_text="z", row=1, col=1)
    fig.update_xaxes(title_text="z", row=1, col=2)
    return fig


def _sharpness_figure(result: QCResult) -> go.Figure | None:
    metrics = result.metrics.get("sharpness")
    if not metrics:
        return None
    fig = go.Figure()
    for label, data in metrics.items():
        fig.add_trace(go.Scatter(x=data["z"], y=data["relative"], mode="lines+markers", name=label))
        if data.get("outlier_z"):
            ys = [data["relative"][data["z"].index(z)] for z in data["outlier_z"] if z in data["z"]]
            fig.add_trace(go.Scatter(
                x=data["outlier_z"], y=ys, mode="markers",
                marker=dict(color="red", size=10, symbol="x"), name=f"out-of-focus {label}",
            ))
    fig.update_layout(
        title="Sharpness (Laplacian variance, relative to peak slice)",
        xaxis_title="z", yaxis_title="relative sharpness",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _background_figure(result: QCResult) -> go.Figure | None:
    metrics = result.metrics.get("background")
    if not metrics:
        return None
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Background level vs z", "Signal level vs z"),
        horizontal_spacing=0.12,
    )
    for label, data in metrics.items():
        fig.add_trace(go.Scatter(x=data["z"], y=data["background"], mode="lines", name=f"bg {label}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data["z"], y=data["signal"], mode="lines", name=f"sig {label}", showlegend=False), row=1, col=2)
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    fig.update_xaxes(title_text="z", row=1, col=1)
    fig.update_xaxes(title_text="z", row=1, col=2)
    return fig


def _stripes_figure(result: QCResult) -> go.Figure | None:
    metrics = result.metrics.get("stripes")
    if not metrics:
        return None
    fig = go.Figure()
    for label, data in metrics.items():
        fig.add_trace(go.Scatter(x=data["z_sampled"], y=data["energy_ratio"], mode="lines+markers", name=label))
    fig.update_layout(
        title="Stripe artifact energy ratio (FFT, per sampled z-slice)",
        xaxis_title="z", yaxis_title="energy fraction in stripe band",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _bubbles_figure(result: QCResult) -> go.Figure | None:
    metrics = result.metrics.get("bubbles")
    if not metrics:
        return None
    fig = go.Figure()
    for label, data in metrics.items():
        fig.add_trace(go.Bar(x=data["z_sampled"], y=data["counts"], name=label))
    fig.update_layout(
        title="Bubble / blob count per sampled z-slice",
        xaxis_title="z", yaxis_title="count",
        barmode="group",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _clearing_figure(result: QCResult) -> go.Figure | None:
    cr = result.metrics.get("clearing_residue")
    fold = result.metrics.get("folding")
    if not cr and not fold:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Clearing residue (HF energy)", "Folding outlier fraction"), horizontal_spacing=0.12)
    if cr:
        for label, data in cr.items():
            fig.add_trace(go.Scatter(x=data["z_sampled"], y=data["hf_energy_per_slice"], mode="lines+markers", name=f"residue {label}"), row=1, col=1)
    if fold:
        for label, data in fold.items():
            fig.add_trace(go.Scatter(x=data["z_sampled"], y=data["outlier_fraction_per_slice"], mode="lines+markers", name=f"folding {label}"), row=1, col=2)
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=50, b=40))
    fig.update_xaxes(title_text="z", row=1, col=1)
    fig.update_xaxes(title_text="z", row=1, col=2)
    return fig


def _bleed_figure(result: QCResult) -> go.Figure | None:
    bleed = result.metrics.get("channel_bleed")
    if not bleed or not bleed.get("pairwise_corr"):
        return None
    labels = bleed.get("channels") or sorted({p.split("->")[0] for p in bleed["pairwise_corr"]})
    n = len(labels)
    mat = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                mat[i][j] = 1.0
            else:
                mat[i][j] = bleed["pairwise_corr"].get(f"{a}->{b}", 0.0)
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu_r",
        text=[[f"{v:.2f}" for v in row] for row in mat],
        texttemplate="%{text}", textfont={"size": 14},
    ))
    fig.update_layout(
        title="Cross-channel bleed-through (Pearson r on signal pixels)",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _registration_figure(result: QCResult) -> go.Figure | None:
    reg = result.metrics.get("registration")
    if not reg or not reg.get("pairwise_shifts"):
        return None
    pairs = list(reg["pairwise_shifts"].keys())
    dy = [abs(reg["pairwise_shifts"][p][0]) for p in pairs]
    dx = [abs(reg["pairwise_shifts"][p][1]) for p in pairs]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="|dy|", x=pairs, y=dy))
    fig.add_trace(go.Bar(name="|dx|", x=pairs, y=dx))
    fig.update_layout(
        title="Cross-channel registration shift (median voxels)",
        barmode="group", yaxis_title="voxels",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


_BASE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 0; color: #1f2937; background: #f9fafb; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 16px; margin: 24px 0 8px; color: #374151; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px;}
.subtitle { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 13px;}
.badge.pass { background: #d1fae5; color: #065f46; }
.badge.fail { background: #fee2e2; color: #991b1b; }
.badge.warn { background: #fef3c7; color: #92400e; }
.summary-cards { display: flex; gap: 12px; margin: 16px 0; }
.card { flex: 1; background: white; padding: 14px 16px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.card .num { font-size: 22px; font-weight: 600; }
.card .lbl { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 4px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; font-size: 13px; }
th { background: #f3f4f6; font-weight: 600; color: #374151; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
tr.fail td { background: #fef2f2; }
tr.warn td { background: #fffbeb; }
.figure { background: white; border-radius: 6px; padding: 8px; margin-top: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.meta { font-size: 12px; color: #6b7280; }
.meta strong { color: #374151; }
"""


def _figure_html(fig: go.Figure | None, include_js: bool = False) -> str:
    if fig is None:
        return ""
    return to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def _flags_table_html(result: QCResult) -> str:
    rows = []
    for f in result.flags:
        cls = "fail" if f.severity == "fail" else ("warn" if f.severity == "warn" else "")
        badge_cls = f.severity if f.severity in ("pass", "warn", "fail") else "pass"
        rows.append(
            f"<tr class='{cls}'><td>{f.name}</td>"
            f"<td><span class='badge {badge_cls}'>{f.severity.upper()}</span></td>"
            f"<td>{f.value:.4f}</td><td>{f.threshold:.4f}</td>"
            f"<td>{f.message}</td></tr>"
        )
    return "<table><thead><tr><th>Check</th><th>Status</th><th>Value</th><th>Threshold</th><th>Description</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _volume_info_html(result: QCResult) -> str:
    info = result.volume_info
    shape = "×".join(str(s) for s in info.get("shape_czyx", []))
    voxel = info.get("voxel_size_um")
    voxel_str = f"{voxel[0]:.2f} × {voxel[1]:.2f} × {voxel[2]:.2f} µm (z,y,x)" if voxel else "unknown"
    channels = ", ".join(info.get("channel_names") or [])
    return (
        f"<div class='meta'><strong>Source:</strong> {info.get('source', '?')}"
        f" &nbsp;&nbsp;<strong>Format:</strong> {info.get('format', '?')}"
        f" &nbsp;&nbsp;<strong>Shape:</strong> {shape} (C,Z,Y,X)"
        f" &nbsp;&nbsp;<strong>Dtype:</strong> {info.get('dtype', '?')}"
        f" &nbsp;&nbsp;<strong>Voxel:</strong> {voxel_str}"
        + (f" &nbsp;&nbsp;<strong>Channels:</strong> {channels}" if channels else "")
        + "</div>"
    )


def write_html_report(result: QCResult, path: str | Path, *, title: str = "Volumetric QC Report") -> Path:
    """Write a self-contained HTML dashboard for a QC result.

    Parameters
    ----------
    result
        :class:`QCResult` from :func:`volumetric_qc.run_qc`.
    path
        Output file path.
    title
        Page title.

    Returns
    -------
    Path
        The file written.
    """
    figures: list[tuple[str, go.Figure | None]] = [
        ("Intensity (per-slice mean / std, drift)", _intensity_figure(result)),
        ("Sharpness (Laplacian variance)", _sharpness_figure(result)),
        ("Background and signal levels", _background_figure(result)),
        ("Stripe artifact energy", _stripes_figure(result)),
        ("Bubble / debris counts", _bubbles_figure(result)),
        ("Clearing residue and folding", _clearing_figure(result)),
        ("Channel bleed-through", _bleed_figure(result)),
        ("Cross-channel registration shift", _registration_figure(result)),
    ]

    body_parts: list[str] = []
    first = True
    for heading, fig in figures:
        if fig is None:
            continue
        body_parts.append(f"<h2>{heading}</h2><div class='figure'>{_figure_html(fig, include_js=first)}</div>")
        first = False

    status_badge_cls = "pass" if result.overall_pass else "fail"
    status_text = "PASS" if result.overall_pass else "FAIL"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="subtitle">Generated {timestamp} by volumetric-qc</div>
    {_volume_info_html(result)}
    <div class="summary-cards">
      <div class="card"><div class="lbl">Overall</div><div class="num"><span class="badge {status_badge_cls}">{status_text}</span></div></div>
      <div class="card"><div class="lbl">Total checks</div><div class="num">{len(result.flags)}</div></div>
      <div class="card"><div class="lbl">Warnings</div><div class="num">{result.n_warn}</div></div>
      <div class="card"><div class="lbl">Failures</div><div class="num">{result.n_fail}</div></div>
      <div class="card"><div class="lbl">Elapsed</div><div class="num">{result.elapsed_seconds:.1f}s</div></div>
    </div>
    <h2>Flag summary</h2>
    {_flags_table_html(result)}
    {''.join(body_parts)}
  </div>
</body>
</html>
"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
