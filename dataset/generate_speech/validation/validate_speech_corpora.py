"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Main script for validating generated speech corpora.
"""

import argparse
import json
import sys
from pathlib import Path

from .audio_validators import AudioValidator
from .report_generator import SpeechReportGenerator, _format_duration
from ...utils import logger, LoggerConfigurator


def main():
    """
    Main validation function.
    """
    parser = argparse.ArgumentParser(
        description="Validate generated speech corpora and generate reports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory containing per-scenario speech output",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to save validation reports (default: output-dir/validation)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum acceptable dialog duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--silence-rms-threshold",
        type=float,
        default=0.001,
        help="RMS amplitude threshold below which a frame is considered silent (default: 0.001)",
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

    LoggerConfigurator.configure_logger(args.log_level)

    if args.report_dir is None:
        args.report_dir = args.output_dir / "validation"
    args.report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Speech Corpora Validation")
    logger.info("=" * 60)
    logger.info("Output directory:  %s", args.output_dir)
    logger.info("Report directory:  %s", args.report_dir)
    logger.info("Min duration:      %.1f s", args.min_duration)
    logger.info("")

    validator = AudioValidator(
        output_dir=args.output_dir,
        min_duration_seconds=args.min_duration,
        silence_rms_threshold=args.silence_rms_threshold,
    )

    errors = validator.validate_all()
    summary = validator.get_summary()
    metrics = validator.collect_metrics()

    # Log summary
    logger.info("")
    logger.info("Validation Results:")
    logger.info("  Total dialog files : %d", metrics.get("total_files", 0))
    logger.info(
        "  Total duration     : %s",
        _format_duration(metrics.get("duration_seconds", {}).get("total")),
    )
    logger.info("  Critical errors    : %d", summary["critical_count"])
    logger.info("  Warnings           : %d", summary["warning_count"])

    for error in errors.get("critical", []):
        logger.error("[CRITICAL] %s — %s", error.get("check"), error.get("message"))
    for error in errors.get("warning", []):
        logger.warning("[WARNING]  %s — %s", error.get("check"), error.get("message"))

    # Generate reports
    json_report_path = args.report_dir / "validation_report.json"
    html_report_path = args.report_dir / "validation_report.html"
    manifest_path = args.report_dir / "manifest.json"

    reporter = SpeechReportGenerator(
        metrics=metrics, errors=errors, error_summary=summary
    )
    reporter.generate_json_report(json_report_path)
    reporter.generate_html_report(html_report_path)
    logger.info("")
    logger.info("Reports written to: %s", args.report_dir)

    # Write manifest for use by archive jobs
    manifest = {
        "validation_passed": not summary["has_critical_errors"],
        "critical_errors": summary["critical_count"],
        "warnings": summary["warning_count"],
        "total_files": metrics.get("total_files", 0),
        "report": str(json_report_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.fail_on_critical and summary["has_critical_errors"]:
        logger.error("Validation FAILED: critical errors found.")
        sys.exit(1)

    logger.info("Validation completed.")


if __name__ == "__main__":
    main()
