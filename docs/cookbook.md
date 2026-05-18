# Cookbook: real-world use cases

End-to-end recipes built around concrete production scenarios. Each recipe is a self-contained script that can be adapted to your pipeline.

---

## 1. QC-gate an atlas registration pipeline

**Scenario.** You're running an atlas-registration job (e.g. brainreg, elastix) on every sample. Bad samples produce silently misaligned registrations that downstream cell counts are then averaged across. Use `volumetric-qc` to fail early.

```python
from pathlib import Path
import sys
from volumetric_qc import open_volume, run_qc, load_preset
from volumetric_qc.reports import write_json_summary

sample = Path(sys.argv[1])
result = run_qc(open_volume(sample), load_preset("idisco"))
write_json_summary(result, sample.parent / "qc_summary.json")

if not result.overall_pass:
    fail_names = [f.name for f in result.flags if f.severity == "fail"]
    print(f"[QC GATE] {sample.name} failed: {fail_names}", file=sys.stderr)
    sys.exit(1)
print(f"[QC GATE] {sample.name} passed in {result.elapsed_seconds:.1f}s")
```

Wire this into your Nextflow / Snakemake DAG as a step that **must** exit 0 before the registration step is allowed to run.

---

## 2. Cohort-wide outlier detection across 240 iDISCO+ samples

**Scenario.** A multi-site cleared-tissue study collected 240 brains over 18 months. One technician's January batch was processed during a clearing-reagent shortage and may have residual lipid. Find which samples were affected without manually opening 240 dashboards.

```python
from pathlib import Path
from volumetric_qc import open_volume, run_qc, load_preset
from volumetric_qc.pipeline.batch import detect_outliers
from volumetric_qc.reports import write_json_summary

study = Path("/data/study_2026/")
samples = sorted(study.glob("*/raw.zarr"))

results, names = [], []
for s in samples:
    r = run_qc(open_volume(s), load_preset("idisco"))
    write_json_summary(r, s.parent / "qc_summary.json")
    results.append(r)
    names.append(s.parent.name)

bor = detect_outliers(results, sample_names=names, robust_z_threshold=4.0)
outliers = [n for n, flag in zip(bor.sample_names, bor.outlier_flags) if flag]
print(f"Outlier samples: {outliers}")
```

The `BatchOutlierResult` also includes `pca_2d` and `umap_2d` arrays for visualization — plot them with matplotlib to see whether outliers cluster (a process-batch effect) or are scattered (individual-sample issues).

---

## 3. Tune thresholds from a curated reference set

**Scenario.** Your lab uses SHIELD with a non-standard reagent kit; the default thresholds are too tight. Calibrate against five known-good and three known-bad samples.

```python
import yaml
from pathlib import Path
from volumetric_qc import open_volume, run_qc, QCConfig

good = list(Path("/data/calibration/good").glob("*.zarr"))
bad  = list(Path("/data/calibration/bad").glob("*.zarr"))

def metric_value(result, name):
    return next((f.value for f in result.flags if f.name == name), float("nan"))

good_results = [run_qc(open_volume(s)) for s in good]
bad_results  = [run_qc(open_volume(s)) for s in bad]

worst_good_drift = max(metric_value(r, f"intensity_drift::channel_0") for r in good_results)
best_bad_drift   = min(metric_value(r, f"intensity_drift::channel_0") for r in bad_results)

# Set threshold halfway between worst-good and best-bad.
threshold = (worst_good_drift + best_bad_drift) / 2
cfg = QCConfig()
cfg.thresholds.intensity_drift_max = threshold
cfg.to_yaml(Path("qc_config.yaml"))
```

Commit `qc_config.yaml` to the analysis repo. Run with `volumetric-qc run sample.zarr --config qc_config.yaml`.

---

## 4. Diff metric distributions before and after a microscope service

**Scenario.** Your light-sheet was serviced; the engineer realigned the illumination optics. You want to verify nothing regressed.

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

before = [json.loads(p.read_text()) for p in Path("qc/before_service").glob("*/qc_summary.json")]
after  = [json.loads(p.read_text()) for p in Path("qc/after_service").glob("*/qc_summary.json")]

def stripe_max(s):
    return max((v.get("mean_ratio", 0) for v in s["metrics"].get("stripes", {}).values()), default=0)

fig, ax = plt.subplots()
ax.hist([stripe_max(s) for s in before], bins=20, alpha=0.5, label="before service")
ax.hist([stripe_max(s) for s in after],  bins=20, alpha=0.5, label="after service")
ax.set_xlabel("stripe energy ratio"); ax.legend(); fig.savefig("service_diff.png")
```

A clean service shows the "after" distribution shifted left (lower stripe energy). A regression shows it shifted right or unchanged — escalate.

---

## 5. Snakemake gating step

```python
# Snakefile excerpt
rule qc:
    input: "samples/{name}/raw.zarr"
    output:
        summary = "samples/{name}/qc_summary.json",
        flag    = "samples/{name}/qc_passed"
    shell:
        """
        volumetric-qc run {input} --preset shield --output samples/{wildcards.name}/
        touch {output.flag}
        """

rule register:
    input:
        zarr = "samples/{name}/raw.zarr",
        flag = "samples/{name}/qc_passed"
    output: "samples/{name}/registered.zarr"
    shell: "brainreg-run --input {input.zarr} --output {output}"
```

If `volumetric-qc run` exits non-zero (QC failed), Snakemake will not produce the `qc_passed` flag and the `register` rule will not run on that sample.

---

## 6. Headless screenshot generation for archival

**Scenario.** Your IRB protocol requires the QC artifact to be a static, immutable artifact (no JS). Generate a PNG alongside the HTML.

```python
from volumetric_qc import open_volume, run_qc, load_preset
from volumetric_qc.reports import write_html_report, write_json_summary
from pathlib import Path
import subprocess

sample = Path("sample.zarr")
result = run_qc(open_volume(sample), load_preset("shield"))
out = Path("qc/")
write_html_report(result, out / "qc.html")
write_json_summary(result, out / "qc.json")

# Reproducible PNG (uses the same matplotlib renderer as the README screenshot).
subprocess.check_call(["python", "scripts/generate_dashboard_screenshot.py"])
```

---

## 7. Detecting a process-batch effect via UMAP

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
from volumetric_qc.pipeline.batch import detect_outliers
from volumetric_qc.pipeline.runner import QCResult, FlagStatus

def load_result(path):
    d = json.loads(Path(path).read_text())
    return QCResult(volume_info=d["volume"], metrics=d["metrics"],
                    flags=[FlagStatus(**f) for f in d["flags"]],
                    config=d["config"], elapsed_seconds=d["elapsed_seconds"])

paths = sorted(Path("study").glob("*/qc_summary.json"))
results = [load_result(p) for p in paths]
batch_ids = [p.parent.name.split("_")[0] for p in paths]  # e.g. "B23_sample01"

bor = detect_outliers(results, sample_names=[p.parent.name for p in paths])
if bor.umap_2d is not None:
    fig, ax = plt.subplots()
    for b in set(batch_ids):
        mask = [bid == b for bid in batch_ids]
        ax.scatter(bor.umap_2d[mask, 0], bor.umap_2d[mask, 1], label=b, s=20)
    ax.legend(); fig.savefig("batch_umap.png")
```

A clean study shows samples mixed regardless of batch. A process effect produces visible batch-colored clusters in UMAP space.

---

## 8. CI workflow for a custom metric

Add a custom metric to `metrics/saturation.py` (see README "Extending"). Verify it's sensitive on synthetic data:

```python
# tests/test_saturation_custom.py
import numpy as np
from volumetric_qc import open_volume
from volumetric_qc.synthetic import clean_volume
from my_metrics import saturation_fraction

def test_saturation_detects_clipping():
    vol = clean_volume(shape=(1, 8, 64, 64), background=100, signal=300, seed=0)
    vol[0, :, 32:40, 32:40] = 65535  # saturate a patch
    result = saturation_fraction(open_volume(vol).channel(0), threshold=65000)
    assert result["max_saturation"] > 0
```

Run as part of `pytest` in CI. This pattern keeps custom metrics' sensitivity regression-tested as the codebase evolves.
