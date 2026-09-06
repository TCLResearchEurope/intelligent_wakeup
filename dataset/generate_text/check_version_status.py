"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Utility script to check version status of generated corpora.
"""

import argparse
from pathlib import Path
from typing import Dict

from .versioning import VersionManager
from .version import get_version
from ..utils import logger, LoggerConfigurator


def print_scenario_status(scenario_key: str, scenario_data: Dict, status: str) -> None:
    """
    Print status information for a scenario.

    Args:
        scenario_key: Scenario identifier
        scenario_data: Scenario metadata
        status: Status string (up-to-date/needs-regeneration)
    """
    print(f"\n{scenario_key}")
    print(f"  Status: {status}")
    print(f"  Code version: {scenario_data.get('code_version', 'N/A')}")
    print(f"  Generated at: {scenario_data.get('generated_at', 'N/A')}")
    print(f"  Conversation count: {scenario_data.get('conversation_count', 'N/A')}")
    print(f"  Conversation type: {scenario_data.get('conversation_type', 'N/A')}")


def main():
    """Check version status of generated corpora."""
    parser = argparse.ArgumentParser(
        description="Check version status of generated text corpora."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing generated text and version file",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information including config hashes",
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

    # Initialize version manager
    version_manager = VersionManager(args.output_dir)

    print("=" * 60)
    print("Text Corpora Version Status")
    print("=" * 60)
    print(f"Current code version: {get_version()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config directory: {args.config_dir}")
    print()

    # Load version data
    version_data = version_manager.load_version_data()

    if not version_data:
        print("No version file found. No scenarios have been generated yet.")
        return

    corpus_version = version_data.get("corpus_version", "N/A")
    last_updated = version_data.get("last_updated", "N/A")

    print(f"Corpus version: {corpus_version}")
    print(f"Last updated: {last_updated}")
    print(f"Total scenarios: {len(version_data.get('scenarios', {}))}")

    # Check each scenario
    scenarios = version_data.get("scenarios", {})
    up_to_date = []
    needs_regeneration = []

    for scenario_key, scenario_data in scenarios.items():
        parts = scenario_key.split("/")
        if len(parts) != 3:
            logger.warning("Invalid scenario key format: %s", scenario_key)
            continue

        scenario_name, variant_type, variant_name = parts

        # Assume single_user for checking status (doesn't matter much for version checking)
        needs_regen, reason, _ = version_manager.check_scenario_version(
            scenario_name, variant_type, variant_name, args.config_dir, "single_user"
        )

        if needs_regen:
            needs_regeneration.append((scenario_key, scenario_data, reason))
        else:
            up_to_date.append((scenario_key, scenario_data))

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Up-to-date scenarios: {len(up_to_date)}")
    print(f"Scenarios needing regeneration: {len(needs_regeneration)}")

    # Print up-to-date scenarios
    if up_to_date:
        print()
        print("=" * 60)
        print("Up-to-date Scenarios")
        print("=" * 60)
        for scenario_key, scenario_data in up_to_date:
            print_scenario_status(scenario_key, scenario_data, "✓ Up-to-date")

            if args.verbose:
                config_hash = scenario_data.get("config_hash", {})
                print(
                    f"  Config hash (composite): {config_hash.get('composite', 'N/A')}"
                )

    # Print scenarios needing regeneration
    if needs_regeneration:
        print()
        print("=" * 60)
        print("Scenarios Needing Regeneration")
        print("=" * 60)
        for scenario_key, scenario_data, reason in needs_regeneration:
            print_scenario_status(scenario_key, scenario_data, f"✗ {reason}")

            if args.verbose:
                config_hash = scenario_data.get("config_hash", {})
                print(
                    f"  Config hash (composite): {config_hash.get('composite', 'N/A')}"
                )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
