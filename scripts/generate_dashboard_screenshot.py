"""Render a composite PNG of the QC dashboard from the most recent demo run.

This produces ``assets/dashboard_screenshot.png`` that the README embeds.
The image is built directly from the JSON summary written by
``scripts/generate_demo_dashboard.py`` so it always reflects real metric
output from a real pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


SEVERITY_COLORS = {"pass": "#d1fae5", "warn": "#fef3c7", "fail": "#fee2e2"}
SEVERITY_TEXT = {"pass": "#065f46", "warn": "#92400e", "fail": "#991b1b"}


def _palette(n: int) -> list[str]:
    base = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b", "#0891b2"]
    return [base[i % len(base)] for i in range(n)]


def render(json_path: Path, out_png: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    flags = data["flags"]
    status = data["status"]
    info = data["volume"]

    channels = info.get("channel_names") or sorted(metrics.get("intensity", {}).keys())
    colors = _palette(len(channels))

    # Layout: 11 row grid, more vertical room for the flag table.
    fig = plt.figure(figsize=(16, 22), dpi=110, facecolor="white")
    gs = gridspec.GridSpec(
        11, 4, figure=fig,
        height_ratios=[0.8, 0.7, 4.2, 2.5, 0.25, 2.5, 0.25, 2.5, 0.25, 2.6, 0.4],
        hspace=0.55, wspace=0.4,
        top=0.97, bottom=0.03, left=0.05, right=0.97,
    )

    # ----- Header band -------------------------------------------------------
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    overall = "PASS" if status["overall_pass"] else "FAIL"
    header_color = "#10b981" if status["overall_pass"] else "#ef4444"
    ax_header.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_header.transAxes,
                                       facecolor="#1f2937", edgecolor="none"))
    ax_header.text(0.02, 0.66, "Volumetric QC – Synthetic SHIELD demo",
                   color="white", fontsize=20, fontweight="bold", transform=ax_header.transAxes)
    shape = "×".join(str(s) for s in info.get("shape_czyx", []))
    voxel = info.get("voxel_size_um") or []
    voxel_str = f"voxel {voxel[0]:.2f} × {voxel[1]:.2f} × {voxel[2]:.2f} µm" if voxel else ""
    subtitle = (f"format: {info.get('format', '?')}   shape (C,Z,Y,X): {shape}"
                f"   dtype: {info.get('dtype', '?')}   {voxel_str}")
    ax_header.text(0.02, 0.25, subtitle, color="#cbd5e1", fontsize=12, transform=ax_header.transAxes)
    ax_header.add_patch(plt.Rectangle((0.86, 0.22), 0.12, 0.56, transform=ax_header.transAxes,
                                       facecolor=header_color, edgecolor="none"))
    ax_header.text(0.92, 0.5, overall, ha="center", va="center", color="white",
                    fontsize=22, fontweight="bold", transform=ax_header.transAxes)

    # ----- Summary cards row -------------------------------------------------
    ax_cards = fig.add_subplot(gs[1, :])
    ax_cards.axis("off")
    cards = [
        ("CHECKS", str(len(flags)), "#1f2937"),
        ("WARNINGS", str(status["n_warn"]), "#b45309"),
        ("FAILURES", str(status["n_fail"]), "#b91c1c"),
        ("ELAPSED", f"{data['elapsed_seconds']:.1f}s", "#1f2937"),
        ("PRESET", data["config"].get("preset", "generic").upper(), "#1f2937"),
    ]
    for i, (lbl, val, col) in enumerate(cards):
        x = 0.005 + i * 0.20
        ax_cards.add_patch(plt.Rectangle((x, 0.08), 0.185, 0.86, transform=ax_cards.transAxes,
                                          facecolor="#f3f4f6", edgecolor="#e5e7eb"))
        ax_cards.text(x + 0.0925, 0.72, lbl, transform=ax_cards.transAxes,
                       fontsize=10, color="#6b7280", ha="center", weight="bold")
        ax_cards.text(x + 0.0925, 0.32, val, transform=ax_cards.transAxes,
                       fontsize=24, color=col, ha="center", weight="bold")

    # ----- Flag table --------------------------------------------------------
    ax_flags = fig.add_subplot(gs[2, :])
    ax_flags.axis("off")
    ax_flags.set_title("Flag summary (top failures and warnings)",
                        loc="left", fontsize=14, weight="bold", pad=8)
    ax_flags.set_xlim(0, 1); ax_flags.set_ylim(0, 1)
    failed = [f for f in flags if f["severity"] in ("fail", "warn")]
    failed.sort(key=lambda f: 0 if f["severity"] == "fail" else 1)
    rows = failed[:14]
    headers = ["Check", "Status", "Value", "Threshold", "Description"]
    col_x = [0.01, 0.30, 0.40, 0.50, 0.61]

    # Header row.
    ax_flags.add_patch(plt.Rectangle((0.0, 0.92), 1.0, 0.06, transform=ax_flags.transAxes,
                                      facecolor="#1f2937", edgecolor="none"))
    for cx, h in zip(col_x, headers):
        ax_flags.text(cx, 0.95, h, transform=ax_flags.transAxes, fontsize=10,
                       weight="bold", color="white", va="center")

    row_h = 0.88 / max(1, len(rows))
    for i, row in enumerate(rows):
        y_top = 0.92 - (i + 1) * row_h
        ax_flags.add_patch(plt.Rectangle((0.0, y_top), 1.0, row_h * 0.96,
                                          transform=ax_flags.transAxes,
                                          facecolor=SEVERITY_COLORS[row["severity"]],
                                          edgecolor="#e5e7eb", linewidth=0.5))
        y_mid = y_top + row_h / 2
        ax_flags.text(col_x[0], y_mid, row["name"], transform=ax_flags.transAxes,
                       fontsize=10, color="#111827", family="monospace", va="center")
        ax_flags.text(col_x[1], y_mid, row["severity"].upper(),
                       transform=ax_flags.transAxes, fontsize=10,
                       color=SEVERITY_TEXT[row["severity"]], weight="bold", va="center")
        ax_flags.text(col_x[2], y_mid, f"{row['value']:.4f}",
                       transform=ax_flags.transAxes, fontsize=10,
                       color="#111827", family="monospace", va="center")
        ax_flags.text(col_x[3], y_mid, f"{row['threshold']:.2f}",
                       transform=ax_flags.transAxes, fontsize=10,
                       color="#111827", family="monospace", va="center")
        msg = row.get("message", "")
        if len(msg) > 60:
            msg = msg[:57] + "…"
        ax_flags.text(col_x[4], y_mid, msg, transform=ax_flags.transAxes,
                       fontsize=9, color="#374151", va="center")

    # ----- Intensity drift ---------------------------------------------------
    ax = fig.add_subplot(gs[3, :2])
    for ch, col in zip(channels, colors):
        d = metrics.get("intensity", {}).get(ch)
        if not d:
            continue
        ax.plot(d["z"], d["mean"], color=col, lw=1.8, label=ch)
    ax.set_title("Per-slice mean intensity (drift)", fontsize=12, weight="bold", loc="left")
    ax.set_xlabel("z-slice"); ax.set_ylabel("mean")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)

    # ----- Sharpness ---------------------------------------------------------
    ax = fig.add_subplot(gs[3, 2:])
    for ch, col in zip(channels, colors):
        d = metrics.get("sharpness", {}).get(ch)
        if not d:
            continue
        ax.plot(d["z"], d["relative"], color=col, lw=1.8, label=ch)
        if d.get("outlier_z"):
            ys = []
            xs = []
            for z in d["outlier_z"]:
                if z in d["z"]:
                    xs.append(z)
                    ys.append(d["relative"][d["z"].index(z)])
            if ys:
                ax.scatter(xs, ys, color="red", marker="x", s=50, zorder=10, lw=2)
    ax.set_title("Sharpness (Laplacian variance, normalized)",
                  fontsize=12, weight="bold", loc="left")
    ax.set_xlabel("z-slice"); ax.set_ylabel("relative")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)

    # ----- Background / signal ----------------------------------------------
    ax = fig.add_subplot(gs[5, :2])
    for ch, col in zip(channels, colors):
        d = metrics.get("background", {}).get(ch)
        if not d:
            continue
        ax.plot(d["z"], d["background"], color=col, lw=1.5, label=f"{ch} bg")
        ax.plot(d["z"], d["signal"], color=col, lw=1.5, ls="--", alpha=0.6, label=f"{ch} sig")
    ax.set_title("Background and signal vs z", fontsize=12, weight="bold", loc="left")
    ax.set_xlabel("z-slice"); ax.set_ylabel("intensity")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # ----- Stripe energy ----------------------------------------------------
    ax = fig.add_subplot(gs[5, 2:])
    for ch, col in zip(channels, colors):
        d = metrics.get("stripes", {}).get(ch)
        if not d:
            continue
        ax.plot(d["z_sampled"], d["energy_ratio"], "-o",
                 color=col, lw=1.6, ms=5, label=ch)
    ax.set_title("Stripe artifact energy ratio (FFT)",
                  fontsize=12, weight="bold", loc="left")
    ax.set_xlabel("z-slice"); ax.set_ylabel("energy fraction")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)

    # ----- Bubbles ----------------------------------------------------------
    ax = fig.add_subplot(gs[7, :2])
    width = 0.27
    for i, (ch, col) in enumerate(zip(channels, colors)):
        d = metrics.get("bubbles", {}).get(ch)
        if not d:
            continue
        zs = np.asarray(d["z_sampled"], dtype=float)
        ax.bar(zs + (i - 1) * width, d["counts"], width=width, color=col, label=ch)
    ax.set_title("Bubble / debris counts per sampled z-slice",
                  fontsize=12, weight="bold", loc="left")
    ax.set_xlabel("z-slice"); ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)

    # ----- Clearing residue --------------------------------------------------
    ax = fig.add_subplot(gs[7, 2:])
    for ch, col in zip(channels, colors):
        d = metrics.get("clearing_residue", {}).get(ch)
        if not d:
            continue
        ax.plot(d["z_sampled"], d["hf_energy_per_slice"], "-o",
                 color=col, lw=1.6, ms=5, label=ch)
    ax.set_title("Clearing residue (high-frequency speckle energy)",
                  fontsize=12, weight="bold", loc="left")
    ax.set_xlabel("z-slice"); ax.set_ylabel("HF energy fraction")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)

    # ----- Bleed-through heatmap --------------------------------------------
    ax = fig.add_subplot(gs[9, :2])
    bleed = metrics.get("channel_bleed", {})
    bleed_labels = bleed.get("channels", channels)
    n = len(bleed_labels)
    mat = np.zeros((n, n))
    for i, a in enumerate(bleed_labels):
        for j, b in enumerate(bleed_labels):
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = bleed.get("pairwise_corr", {}).get(f"{a}->{b}", 0.0)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(bleed_labels, fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(bleed_labels, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 0.5 else "#1f2937",
                    fontsize=11, weight="bold")
    ax.set_title("Cross-channel bleed-through (Pearson r, signal pixels)",
                  fontsize=12, weight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # ----- Registration shifts ----------------------------------------------
    ax = fig.add_subplot(gs[9, 2:])
    reg = metrics.get("registration", {})
    pairs = list(reg.get("pairwise_shifts", {}).keys())
    dy = [abs(reg["pairwise_shifts"][p][0]) for p in pairs]
    dx = [abs(reg["pairwise_shifts"][p][1]) for p in pairs]
    x = np.arange(len(pairs))
    ax.bar(x - 0.18, dy, width=0.34, color="#2563eb", label="|dy|")
    ax.bar(x + 0.18, dx, width=0.34, color="#f59e0b", label="|dx|")
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=18, ha="right", fontsize=10)
    ax.set_title("Cross-channel registration shift (voxels)",
                  fontsize=12, weight="bold", loc="left")
    ax.set_ylabel("voxels"); ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(2.0, color="red", lw=1.2, ls="--", alpha=0.6)
    ax.text(0.02, 2.1, "threshold (2 voxels)", color="red", fontsize=9,
             transform=ax.get_yaxis_transform())

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    render(Path("demo_output/qc_summary.json"), Path("assets/dashboard_screenshot.png"))
