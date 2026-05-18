"""Report generation: JSON summary + standalone HTML dashboard."""

from volumetric_qc.reports.html import write_html_report
from volumetric_qc.reports.summary import write_json_summary, qc_result_to_summary

__all__ = ["write_html_report", "write_json_summary", "qc_result_to_summary"]
