"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Report generation for speech corpora validation.
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime


def _format_duration(seconds) -> str:
    """Return e.g. '7384.0 s (2 h 3 min 4 s)'."""
    if seconds == "N/A" or seconds is None:
        return "N/A"
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        friendly = f"{h} h {m} min {s} s"
    elif m > 0:
        friendly = f"{m} min {s} s"
    else:
        friendly = f"{s} s"
    return f"{seconds} s ({friendly})"


class SpeechReportGenerator:
    """
    Generate validation reports (JSON + HTML) for speech corpora.
    """

    def __init__(self, metrics: Dict, errors: Dict, error_summary: Dict):
        self.metrics = metrics
        self.errors = errors
        self.error_summary = error_summary
        self.timestamp = datetime.now().isoformat()

    def generate_json_report(self, output_path: Path) -> None:
        """Write a JSON validation report."""
        report = {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "errors": self.errors,
            "error_summary": self.error_summary,
            "validation_passed": not self.error_summary.get(
                "has_critical_errors", False
            ),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def generate_html_report(self, output_path: Path) -> None:
        """Write an HTML validation report."""
        passed = not self.error_summary.get("has_critical_errors", False)
        status_color = "#28a745" if passed else "#dc3545"
        status_text = "PASSED" if passed else "FAILED"

        critical_rows = self._build_error_rows("critical")
        warning_rows = self._build_error_rows("warning")

        metrics = self.metrics
        dur = metrics.get("duration_seconds", {})
        rms = metrics.get("rms_amplitude", {})

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Speech Corpora Validation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
    h1, h2 {{ color: #333; }}
    .status {{ display: inline-block; padding: 6px 16px; border-radius: 4px;
               color: #fff; font-weight: bold; background: {status_color}; }}
    .card {{ background: #fff; border-radius: 6px; padding: 16px 20px;
             margin: 12px 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }}
    th {{ background: #f0f0f0; }}
    .critical {{ color: #dc3545; }}
    .warning {{ color: #ffc107; }}
    .none {{ color: #6c757d; font-style: italic; }}
  </style>
</head>
<body>
  <h1>Speech Corpora Validation Report</h1>
  <p>Generated: {self.timestamp}</p>
  <p>Status: <span class="status">{status_text}</span></p>

  <div class="card">
    <h2>Summary</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Total dialog files</td><td>{metrics.get('total_files', 0)}</td></tr>
      <tr><td>Total duration</td><td>{_format_duration(dur.get('total', 'N/A'))}</td></tr>
      <tr><td>Min duration</td><td>{dur.get('min', 'N/A')} s</td></tr>
      <tr><td>Max duration</td><td>{dur.get('max', 'N/A')} s</td></tr>
      <tr><td>Mean duration</td><td>{dur.get('mean', 'N/A')} s</td></tr>
      <tr><td>Mean RMS amplitude</td><td>{rms.get('mean', 'N/A')}</td></tr>
      <tr><td>Sample rate(s)</td><td>{metrics.get('sample_rates', 'N/A')}</td></tr>
      <tr><td>Critical errors</td>
          <td class="critical">{self.error_summary.get('critical_count', 0)}</td></tr>
      <tr><td>Warnings</td>
          <td class="warning">{self.error_summary.get('warning_count', 0)}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Critical Errors</h2>
    {critical_rows if critical_rows else '<p class="none">No critical errors.</p>'}
  </div>

  <div class="card">
    <h2>Warnings</h2>
    {warning_rows if warning_rows else '<p class="none">No warnings.</p>'}
  </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    # ------------------------------------------------------------------

    def _build_error_rows(self, severity: str) -> str:
        items = self.errors.get(severity, [])
        if not items:
            return ""
        rows = "".join(
            f"<tr><td>{e.get('check', '')}</td>"
            f"<td>{e.get('message', '')}</td>"
            f"<td>{Path(e.get('path', '')).name}</td></tr>"
            for e in items
        )
        return f"""<table>
      <tr><th>Check</th><th>Message</th><th>File</th></tr>
      {rows}
    </table>"""
