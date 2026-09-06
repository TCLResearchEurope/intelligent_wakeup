"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Report generation for text corpora validation.
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime


class ReportGenerator:
    """
    Generate validation reports in HTML and JSON formats.
    """

    def __init__(self, metrics: Dict, errors: Dict, error_summary: Dict):
        """
        Initialize report generator.

        Args:
            metrics: Metrics dictionary from CorpusMetrics
            errors: Errors dictionary from ErrorDetector
            error_summary: Error summary from ErrorDetector
        """
        self.metrics = metrics
        self.errors = errors
        self.error_summary = error_summary
        self.timestamp = datetime.now().isoformat()

    def generate_json_report(self, output_path: Path):
        """
        Generate JSON report.

        Args:
            output_path: Path to save the JSON report
        """
        report = {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "errors": self.errors,
            "error_summary": self.error_summary,
            "validation_passed": self._determine_validation_status(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def generate_html_report(self, output_path: Path):
        """
        Generate HTML report.

        Args:
            output_path: Path to save the HTML report
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Corpora Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header .timestamp {{
            opacity: 0.9;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .status-pass {{
            background-color: #48bb78;
            color: white;
        }}
        .status-fail {{
            background-color: #f56565;
            color: white;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #2d3748;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .section h3 {{
            color: #4a5568;
            margin-top: 20px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card .label {{
            font-size: 0.85em;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2d3748;
            margin-top: 5px;
        }}
        .error-list {{
            list-style: none;
            padding: 0;
        }}
        .error-item {{
            background: #fff5f5;
            border-left: 4px solid #fc8181;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }}
        .error-item.warning {{
            background: #fffaf0;
            border-left-color: #f6ad55;
        }}
        .error-item.info {{
            background: #ebf8ff;
            border-left-color: #63b3ed;
        }}
        .error-item .error-type {{
            font-weight: bold;
            color: #2d3748;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .error-item .error-message {{
            margin-top: 5px;
            color: #4a5568;
        }}
        .error-item .error-file {{
            font-family: monospace;
            font-size: 0.85em;
            color: #718096;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f7fafc;
            font-weight: 600;
            color: #2d3748;
        }}
        tr:hover {{
            background-color: #f7fafc;
        }}
        a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }}
        a:hover {{
            color: #5568d3;
            text-decoration: underline;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        code {{
            background: #2d3748;
            color: #68d391;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Text Corpora Validation Report</h1>
        <div class="timestamp">Generated: {self.timestamp}</div>
        <span class="status-badge {'status-pass' if self._determine_validation_status() else 'status-fail'}">
            {'VALIDATION PASSED' if self._determine_validation_status() else 'VALIDATION FAILED'}
        </span>
    </div>

    {self._generate_summary_section()}
    {self._generate_coverage_section()}
    {self._generate_quality_section()}
    {self._generate_errors_section()}
    {self._generate_distribution_section()}

</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _determine_validation_status(self) -> bool:
        """
        Determine if validation passed (no critical errors).

        Returns:
            True if no critical errors, False otherwise
        """
        critical_count = self.error_summary.get("by_severity", {}).get("critical", 0)
        return critical_count == 0

    def generate_scenario_json_report(
        self,
        scenario: str,
        scenario_metrics: Dict,
        scenario_errors: Dict,
        output_path: Path,
    ):
        """
        Generate JSON report for a specific scenario.

        Args:
            scenario: Scenario name
            scenario_metrics: Metrics for the scenario
            scenario_errors: Errors for the scenario
            output_path: Path to save the JSON report
        """
        # Calculate error summary for this scenario
        error_summary = {
            "total_errors": sum(len(errs) for errs in scenario_errors.values()),
            "by_severity": {
                "critical": len(scenario_errors.get("critical", [])),
                "warning": len(scenario_errors.get("warning", [])),
                "info": len(scenario_errors.get("info", [])),
            },
            "by_type": {},
        }

        # Count errors by type
        for _, errors in scenario_errors.items():
            for error in errors:
                error_type = error.get("type", "unknown")
                error_summary["by_type"][error_type] = (
                    error_summary["by_type"].get(error_type, 0) + 1
                )

        report = {
            "timestamp": self.timestamp,
            "scenario": scenario,
            "metrics": scenario_metrics,
            "errors": scenario_errors,
            "error_summary": error_summary,
            "validation_passed": error_summary["by_severity"]["critical"] == 0,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def generate_scenario_html_report(
        self,
        scenario: str,
        scenario_metrics: Dict,
        scenario_errors: Dict,
        output_path: Path,
    ):
        """
        Generate HTML report for a specific scenario.

        Args:
            scenario: Scenario name
            scenario_metrics: Metrics for the scenario
            scenario_errors: Errors for the scenario
            output_path: Path to save the HTML report
        """
        # Calculate error summary
        error_summary = {
            "total_errors": sum(len(errs) for errs in scenario_errors.values()),
            "by_severity": {
                "critical": len(scenario_errors.get("critical", [])),
                "warning": len(scenario_errors.get("warning", [])),
                "info": len(scenario_errors.get("info", [])),
            },
        }

        validation_passed = error_summary["by_severity"]["critical"] == 0

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report - {scenario}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .nav-back {{
            margin-bottom: 20px;
        }}
        .nav-back a {{
            display: inline-block;
            padding: 10px 20px;
            background-color: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-back a:hover {{
            background-color: #5568d3;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header .scenario-name {{
            font-size: 1.2em;
            margin-top: 10px;
            opacity: 0.9;
        }}
        .header .timestamp {{
            opacity: 0.9;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .status-pass {{
            background-color: #48bb78;
            color: white;
        }}
        .status-fail {{
            background-color: #f56565;
            color: white;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #2d3748;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card .label {{
            font-size: 0.85em;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2d3748;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f7fafc;
            font-weight: 600;
            color: #2d3748;
        }}
        .error-list {{
            list-style: none;
            padding: 0;
        }}
        .error-item {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        .error-item.critical {{
            background-color: #fff5f5;
            border-color: #f56565;
        }}
        .error-item.warning {{
            background-color: #fffaf0;
            border-color: #ed8936;
        }}
        .error-item.info {{
            background-color: #ebf8ff;
            border-color: #4299e1;
        }}
        .error-type {{
            font-weight: bold;
            color: #2d3748;
        }}
        .error-message {{
            color: #4a5568;
            margin-top: 5px;
        }}
        .error-file {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #718096;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="nav-back">
        <a href="../validation_report.html">← Back to Main Report</a>
    </div>

    <div class="header">
        <h1>Scenario Validation Report</h1>
        <div class="scenario-name">Scenario: {scenario}</div>
        <div class="timestamp">Generated: {self.timestamp}</div>
        <span class="status-badge {'status-pass' if validation_passed else 'status-fail'}">
            {'✓ PASSED' if validation_passed else '✗ FAILED'}
        </span>
    </div>

    {self._generate_scenario_overview_section(scenario_metrics, error_summary)}
    {self._generate_scenario_quality_section(scenario_metrics)}
    {self._generate_scenario_coverage_section(scenario_metrics)}
    {self._generate_scenario_errors_section(scenario_errors)}
    {self._generate_scenario_distribution_section(scenario_metrics)}

</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _generate_scenario_overview_section(
        self, metrics: Dict, error_summary: Dict
    ) -> str:
        """
        Generate overview section for scenario report.

        Args:
            metrics: Metrics for the scenario
            error_summary: Error summary for the scenario

        Returns:
            HTML string for the overview section
        """
        coverage = metrics.get("coverage", {})
        quality_stats = metrics.get("quality", {}).get("statistics", {})

        total_files = coverage.get("total_files", 0)
        variants = len(coverage.get("variants", []))
        avg_turns = quality_stats.get("turn_count", {}).get("mean", 0)
        avg_tokens = quality_stats.get("token_count", {}).get("mean", 0)

        return f"""
    <div class="section">
        <h2>Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Total Files</div>
                <div class="value">{total_files}</div>
            </div>
            <div class="metric-card">
                <div class="label">Variants</div>
                <div class="value">{variants}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Turns</div>
                <div class="value">{avg_turns:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Tokens</div>
                <div class="value">{avg_tokens:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Critical Errors</div>
                <div class="value">{error_summary['by_severity']['critical']}</div>
            </div>
            <div class="metric-card">
                <div class="label">Warnings</div>
                <div class="value">{error_summary['by_severity']['warning']}</div>
            </div>
        </div>
    </div>
"""

    def _generate_scenario_quality_section(self, metrics: Dict) -> str:
        """
        Generate quality metrics section for scenario report.

        Args:
            metrics: Metrics for the scenario

        Returns:
            HTML string for the quality section
        """
        quality = metrics.get("quality", {})
        stats = quality.get("statistics", {})

        turn_stats = stats.get("turn_count", {})
        token_stats = stats.get("token_count", {})
        speaker_stats = stats.get("speaker_count", {})

        return f"""
    <div class="section">
        <h2>Quality Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Turn Count</td>
                    <td>{turn_stats.get('min', 0)}</td>
                    <td>{turn_stats.get('max', 0)}</td>
                    <td>{turn_stats.get('mean', 0):.2f}</td>
                    <td>{turn_stats.get('median', 0):.2f}</td>
                    <td>{turn_stats.get('std_dev', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Token Count</td>
                    <td>{token_stats.get('min', 0)}</td>
                    <td>{token_stats.get('max', 0)}</td>
                    <td>{token_stats.get('mean', 0):.2f}</td>
                    <td>{token_stats.get('median', 0):.2f}</td>
                    <td>{token_stats.get('std_dev', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Speaker Count</td>
                    <td>{speaker_stats.get('min', 0)}</td>
                    <td>{speaker_stats.get('max', 0)}</td>
                    <td>{speaker_stats.get('mean', 0):.2f}</td>
                    <td>{speaker_stats.get('median', 0):.2f}</td>
                    <td>{speaker_stats.get('std_dev', 0):.2f}</td>
                </tr>
            </tbody>
        </table>
    </div>
"""

    def _generate_scenario_coverage_section(self, metrics: Dict) -> str:
        """
        Generate coverage section for scenario report.

        Args:
            metrics: Metrics for the scenario

        Returns:
            HTML string for the coverage section
        """
        coverage = metrics.get("coverage", {})

        variants = coverage.get("variants", [])
        characters = coverage.get("characters_used", [])
        conversation_types = coverage.get("conversation_types", {})

        variant_rows = "\n".join(f"<li>{v}</li>" for v in variants)
        char_rows = "\n".join(f"<li>{c}</li>" for c in characters[:10])

        conv_type_rows = ""
        for conv_type, count in conversation_types.items():
            conv_type_rows += f"""
                <tr>
                    <td>{conv_type}</td>
                    <td>{count}</td>
                </tr>
"""

        return f"""
    <div class="section">
        <h2>Coverage Details</h2>

        <h3>Variants ({len(variants)})</h3>
        <ul>{variant_rows if variant_rows else '<li>No variants</li>'}</ul>

        <h3>Conversation Types</h3>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {conv_type_rows if conv_type_rows else '<tr><td colspan="2">No data</td></tr>'}
            </tbody>
        </table>

        <h3>Characters Used (Top 10)</h3>
        <ul>{char_rows if char_rows else '<li>No characters</li>'}</ul>
    </div>
"""

    def _generate_scenario_errors_section(self, errors: Dict) -> str:
        """
        Generate errors section for scenario report.

        Args:
            errors: Errors for the scenario

        Returns:
            HTML string for the errors section
        """
        critical_errors = errors.get("critical", [])
        warning_errors = errors.get("warning", [])
        info_errors = errors.get("info", [])

        def format_error_list(error_list, severity):
            if not error_list:
                return "<p>No errors found.</p>"

            items = ""
            for error in error_list:
                error_type = error.get("type", "unknown")
                message = error.get("message", "")
                file_path = error.get("file", "")

                items += f"""
                <li class="error-item {severity}">
                    <div class="error-type">{error_type}</div>
                    <div class="error-message">{message}</div>
                    {f'<div class="error-file">File: {file_path}</div>' if file_path else ''}
                </li>
"""

            return f'<ul class="error-list">{items}</ul>'

        return f"""
    <div class="section">
        <h2>Error Report</h2>

        <h3>Critical Errors ({len(critical_errors)})</h3>
        {format_error_list(critical_errors, 'critical')}

        <h3>Warnings ({len(warning_errors)})</h3>
        {format_error_list(warning_errors, 'warning')}

        {f'<h3>Info ({len(info_errors)})</h3>{format_error_list(info_errors, "info")}' if info_errors else ''}
    </div>
"""

    def _generate_scenario_distribution_section(self, metrics: Dict) -> str:
        """
        Generate distribution section for scenario report.

        Args:
            metrics: Metrics for the scenario

        Returns:
            HTML string for the distribution section
        """
        distribution = metrics.get("distribution", {})
        char_freq = distribution.get("character_frequency", {})
        variant_dist = distribution.get("variant_distribution", {})

        # Character frequency
        top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        char_rows = ""
        for char, count in top_chars:
            char_rows += f"""
                <tr>
                    <td>{char}</td>
                    <td>{count}</td>
                </tr>
"""

        # Variant distribution
        variant_rows = ""
        for variant, count in sorted(variant_dist.items()):
            variant_rows += f"""
                <tr>
                    <td>{variant}</td>
                    <td>{count}</td>
                </tr>
"""

        return f"""
    <div class="section">
        <h2>Distribution Metrics</h2>

        <h3>Top 10 Characters by Appearance</h3>
        <table>
            <thead>
                <tr>
                    <th>Character</th>
                    <th>Appearances</th>
                </tr>
            </thead>
            <tbody>
                {char_rows if char_rows else '<tr><td colspan="2">No data</td></tr>'}
            </tbody>
        </table>

        <h3>Variant Distribution</h3>
        <table>
            <thead>
                <tr>
                    <th>Variant</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {variant_rows if variant_rows else '<tr><td colspan="2">No data</td></tr>'}
            </tbody>
        </table>
    </div>
"""

    def _generate_summary_section(self) -> str:
        """
        Generate executive summary section.

        Returns:
            HTML string for the summary section
        """
        coverage = self.metrics.get("coverage", {})
        quality = self.metrics.get("quality", {})

        total_files = coverage.get("total_files", 0)
        total_errors = self.error_summary.get("total_errors", 0)
        critical_errors = self.error_summary.get("by_severity", {}).get("critical", 0)
        warning_errors = self.error_summary.get("by_severity", {}).get("warning", 0)

        turn_stats = quality.get("statistics", {}).get("turn_count", {})

        return f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Total Files</div>
                <div class="value">{total_files}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Errors</div>
                <div class="value" style="color: {'#f56565' if total_errors > 0 else '#48bb78'};">{total_errors}</div>
            </div>
            <div class="metric-card">
                <div class="label">Critical Errors</div>
                <div class="value" style="color: {'#f56565' if critical_errors > 0 else '#48bb78'};">{critical_errors}</div>
            </div>
            <div class="metric-card">
                <div class="label">Warnings</div>
                <div class="value" style="color: {'#f6ad55' if warning_errors > 0 else '#48bb78'};">{warning_errors}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Turns/Dialogue</div>
                <div class="value">{turn_stats.get('mean', 0):.1f}</div>
            </div>
        </div>
    </div>
"""

    def _generate_coverage_section(self) -> str:
        """
        Generate coverage metrics section.

        Returns:
            HTML string for the coverage section
        """
        coverage = self.metrics.get("coverage", {})
        scenarios = coverage.get("scenarios", {})
        conversation_types = coverage.get("conversation_types", {})
        missing = coverage.get("missing_scenarios", [])
        expected_scenarios = coverage.get("expected_scenarios", [])

        scenarios_table = ""
        # Include both generated and expected scenarios
        all_scenarios = set(scenarios.keys()) | set(expected_scenarios)
        for scenario in sorted(all_scenarios):
            data = scenarios.get(scenario, {"total_files": 0, "variants": []})
            is_generated = scenario in scenarios

            scenarios_table += f"""
                <tr>
                    <td>{scenario}</td>
                    <td>{data['total_files']}</td>
                    <td>{', '.join(data['variants']) if data['variants'] else '-'}</td>
                    <td>{'<a href="scenarios/' + scenario + '.html">View Report</a>' if is_generated else '<span style="color: #cbd5e0;">Not generated</span>'}</td>
                </tr>
"""

        missing_html = ""
        if missing:
            missing_html = f"""
            <div class="error-item">
                <div class="error-type">Missing Scenarios</div>
                <div class="error-message">
                    The following scenarios have configurations but no generated files:
                    <ul>
                        {''.join(f'<li>{s}</li>' for s in missing)}
                    </ul>
                </div>
            </div>
"""

        return f"""
    <div class="section">
        <h2>Coverage Metrics</h2>

        <h3>Conversation Types</h3>
        <div class="metric-grid">
            {''.join(f'<div class="metric-card"><div class="label">{ct}</div><div class="value">{count}</div></div>' for ct, count in conversation_types.items())}
        </div>

        <h3>Scenarios</h3>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Files</th>
                    <th>Variants</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                {scenarios_table}
            </tbody>
        </table>

        {missing_html}
    </div>
"""

    def _generate_quality_section(self) -> str:
        """
        Generate quality metrics section.

        Returns:
            HTML string for the quality section
        """
        quality = self.metrics.get("quality", {})
        stats = quality.get("statistics", {})

        def format_stat_row(label, stat_dict):
            return f"""
                <tr>
                    <td>{label}</td>
                    <td>{stat_dict.get('min', 0):.1f}</td>
                    <td>{stat_dict.get('max', 0):.1f}</td>
                    <td>{stat_dict.get('mean', 0):.1f}</td>
                    <td>{stat_dict.get('median', 0):.1f}</td>
                    <td>{stat_dict.get('std_dev', 0):.1f}</td>
                </tr>
"""

        return f"""
    <div class="section">
        <h2>Quality Metrics</h2>

        <h3>Overall Statistics</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                </tr>
            </thead>
            <tbody>
                {format_stat_row('Turn Count', stats.get('turn_count', {}))}
                {format_stat_row('Token Count', stats.get('token_count', {}))}
                {format_stat_row('Speaker Count', stats.get('speaker_count', {}))}
                {format_stat_row('Dialogue Length (chars)', stats.get('dialogue_length', {}))}
            </tbody>
        </table>

        {self._generate_scenario_quality_table(quality.get('scenario_statistics', {}))}
    </div>
"""

    def _generate_scenario_quality_table(self, scenario_stats: Dict) -> str:
        """Generate per-scenario quality table."""
        if not scenario_stats:
            return ""

        rows = ""
        for scenario, stats in sorted(scenario_stats.items()):
            turn_stat = stats.get("turn_count", {})
            token_stat = stats.get("token_count", {})
            rows += f"""
                <tr>
                    <td>{scenario}</td>
                    <td>{stats.get('file_count', 0)}</td>
                    <td>{turn_stat.get('mean', 0):.1f}</td>
                    <td>{token_stat.get('mean', 0):.1f}</td>
                </tr>
"""

        return f"""
        <h3>Per-Scenario Quality</h3>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Files</th>
                    <th>Avg Turns</th>
                    <th>Avg Tokens</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
"""

    def _generate_errors_section(self) -> str:
        """Generate errors section."""
        critical_errors = self.errors.get("critical", [])
        warning_errors = self.errors.get("warning", [])
        info_errors = self.errors.get("info", [])

        def format_error_list(errors, severity_class):
            if not errors:
                return "<p>No errors found.</p>"

            items = ""
            for error in errors:
                error_type = error.get("type", "unknown")
                message = error.get("message", "")
                file_path = error.get("file", error.get("scenario", ""))

                items += f"""
                <li class="error-item {severity_class}">
                    <div class="error-type">{error_type}</div>
                    <div class="error-message">{message}</div>
                    {f'<div class="error-file">{file_path}</div>' if file_path else ''}
                </li>
"""
            return f'<ul class="error-list">{items}</ul>'

        return f"""
    <div class="section">
        <h2>Error Report</h2>

        <h3>Critical Errors ({len(critical_errors)})</h3>
        {format_error_list(critical_errors, 'critical')}

        <h3>Warnings ({len(warning_errors)})</h3>
        {format_error_list(warning_errors, 'warning')}

        {f'<h3>Info ({len(info_errors)})</h3>{format_error_list(info_errors, "info")}' if info_errors else ''}
    </div>
"""

    def _generate_distribution_section(self) -> str:
        """
        Generate distribution metrics section.

        Returns:
            HTML string for the distribution section
        """
        distribution = self.metrics.get("distribution", {})
        char_freq = distribution.get("character_frequency", {})

        # Get top 10 characters
        top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        char_rows = ""
        for char, count in top_chars:
            char_rows += f"""
                <tr>
                    <td>{char}</td>
                    <td>{count}</td>
                </tr>
"""

        return f"""
    <div class="section">
        <h2>Distribution Metrics</h2>

        <h3>Top 10 Characters by Appearance</h3>
        <table>
            <thead>
                <tr>
                    <th>Character</th>
                    <th>Appearances</th>
                </tr>
            </thead>
            <tbody>
                {char_rows}
            </tbody>
        </table>
    </div>
"""
