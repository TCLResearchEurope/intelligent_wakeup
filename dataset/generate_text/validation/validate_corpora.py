"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Main script for validating generated text corpora.
"""

import argparse
import sys
from pathlib import Path
import json

from .metrics import CorpusMetrics
from .error_detectors import ErrorDetector
from .report_generator import ReportGenerator
from ...utils import logger, LoggerConfigurator


def main():
    """
    Main validation function.
    """
    parser = argparse.ArgumentParser(
        description="Validate generated text corpora and generate reports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing generated conversations",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing scenario configurations",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to save validation reports (default: output-dir/validation)",
    )
    parser.add_argument(
        "--min-turn-count",
        type=int,
        default=4,
        help="Minimum number of turns for a valid conversation (default: 4)",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with non-zero status if critical errors are found",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Configure logger
    LoggerConfigurator.configure_logger(args.log_level)

    # Set default report directory
    if args.report_dir is None:
        args.report_dir = args.output_dir / "validation"
    args.report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Text Corpora Validation")
    logger.info("=" * 60)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Config directory: %s", args.config_dir)
    logger.info("Report directory: %s", args.report_dir)
    logger.info("")

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_calculator = CorpusMetrics(args.output_dir, args.config_dir)
    metrics = metrics_calculator.calculate_all_metrics()

    # Log coverage summary
    coverage = metrics.get("coverage", {})
    logger.info("Total files generated: %d", coverage.get("total_files", 0))
    logger.info("Scenarios covered: %d", len(coverage.get("scenarios", {})))
    logger.info(
        "Conversation types: %s", list(coverage.get("conversation_types", {}).keys())
    )

    # Detect errors
    logger.info("")
    logger.info("Detecting errors...")
    error_detector = ErrorDetector(
        args.output_dir, args.config_dir, args.min_turn_count
    )
    errors = error_detector.detect_all_errors()
    error_summary = error_detector.get_error_summary()

    # Log error summary
    logger.info("Total errors found: %d", error_summary.get("total_errors", 0))
    logger.info(
        "  Critical: %d", error_summary.get("by_severity", {}).get("critical", 0)
    )
    logger.info(
        "  Warnings: %d", error_summary.get("by_severity", {}).get("warning", 0)
    )
    logger.info("  Info: %d", error_summary.get("by_severity", {}).get("info", 0))

    # Log error breakdown by type
    if error_summary.get("by_type"):
        logger.info("")
        logger.info("Errors by type:")
        for error_type, count in sorted(
            error_summary.get("by_type", {}).items(), key=lambda x: x[1], reverse=True
        ):
            logger.info("  %s: %d", error_type, count)

    # Generate reports
    logger.info("")
    logger.info("Generating validation reports...")
    report_generator = ReportGenerator(metrics, errors, error_summary)

    # Generate JSON report
    json_report_path = args.report_dir / "validation_report.json"
    report_generator.generate_json_report(json_report_path)
    logger.info("JSON report saved to: %s", json_report_path)

    # Generate HTML report
    html_report_path = args.report_dir / "validation_report.html"
    report_generator.generate_html_report(html_report_path)
    logger.info("HTML report saved to: %s", html_report_path)

    # Generate per-scenario reports
    logger.info("")
    logger.info("Generating per-scenario reports...")
    scenario_report_dir = args.report_dir / "scenarios"
    scenario_report_dir.mkdir(parents=True, exist_ok=True)

    # Get per-scenario data
    scenario_metrics = metrics_calculator.get_all_scenario_metrics()
    scenario_errors = error_detector.get_errors_by_scenario()
    scenario_error_summaries = error_detector.get_scenario_error_summaries()

    # Generate reports for each scenario
    for scenario in sorted(scenario_metrics.keys()):
        logger.info("  Generating report for scenario: %s", scenario)

        # Get data for this scenario
        s_metrics = scenario_metrics[scenario]
        s_errors = scenario_errors.get(scenario, {})

        # Generate JSON report
        json_path = scenario_report_dir / f"{scenario}.json"
        report_generator.generate_scenario_json_report(
            scenario, s_metrics, s_errors, json_path
        )

        # Generate HTML report
        html_path = scenario_report_dir / f"{scenario}.html"
        report_generator.generate_scenario_html_report(
            scenario, s_metrics, s_errors, html_path
        )

    logger.info("Per-scenario reports saved to: %s", scenario_report_dir)

    # Generate manifest file with metadata
    manifest_path = args.report_dir / "manifest.json"
    manifest = {
        "output_dir": str(args.output_dir),
        "config_dir": str(args.config_dir),
        "total_files": coverage.get("total_files", 0),
        "scenarios": list(coverage.get("scenarios", {}).keys()),
        "conversation_types": list(coverage.get("conversation_types", {}).keys()),
        "validation_passed": error_summary.get("by_severity", {}).get("critical", 0)
        == 0,
        "total_errors": error_summary.get("total_errors", 0),
        "critical_errors": error_summary.get("by_severity", {}).get("critical", 0),
        "warnings": error_summary.get("by_severity", {}).get("warning", 0),
        "per_scenario_summaries": scenario_error_summaries,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest saved to: %s", manifest_path)

    # Print final status
    logger.info("")
    logger.info("=" * 60)
    validation_passed = error_summary.get("by_severity", {}).get("critical", 0) == 0
    if validation_passed:
        logger.info("✓ VALIDATION PASSED")
    else:
        logger.error("✗ VALIDATION FAILED (critical errors found)")
    logger.info("=" * 60)

    # Exit with appropriate status code
    if args.fail_on_critical and not validation_passed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
