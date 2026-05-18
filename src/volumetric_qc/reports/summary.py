"""JSON summary writer.

The JSON summary is the machine-readable counterpart to the HTML dashboard.
It mirrors the structure of :class:`QCResult.to_dict` with light flattening
designed for easy parsing by downstream batch analysis or LIMS integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from volumetric_qc.pipeline.runner import QCResult


def qc_result_to_summary(result: QCResult) -> dict[str, Any]:
    """Return a dict-of-dicts summary suitable for JSON serialization.

    Keys
    ----
    ``volume`` : volume metadata
    ``config`` : the config used for the run
    ``flags`` : list of pass/fail flags
    ``metrics`` : nested metric outputs (full per-z arrays preserved)
    ``status`` : aggregate summary (overall_pass, n_warn, n_fail)
    ``elapsed_seconds`` : wall time
    """
    return {
        "volume": result.volume_info,
        "config": result.config,
        "flags": [vars(f) for f in result.flags],
        "metrics": result.to_dict()["metrics"],
        "status": {
            "overall_pass": result.overall_pass,
            "n_warn": result.n_warn,
            "n_fail": result.n_fail,
        },
        "elapsed_seconds": result.elapsed_seconds,
    }


def write_json_summary(result: QCResult, path: str | Path) -> Path:
    """Serialize the summary to a JSON file. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(qc_result_to_summary(result), f, indent=2, default=str)
    return p
