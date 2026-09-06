"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Metrics calculation for text corpora validation.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import statistics

from ...utils import logger


class CorpusMetrics:
    """
    Calculate various metrics for generated text corpora.
    """

    def __init__(self, output_dir: Path, config_dir: Path):
        """
        Initialize metrics calculator.

        Args:
            output_dir: Directory containing generated conversations
            config_dir: Directory containing scenario configurations
        """
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        self.metrics = defaultdict(dict)

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all available metrics.

        Returns:
            Dict containing all calculated metrics
        """
        self.metrics["coverage"] = self._calculate_coverage_metrics()
        self.metrics["quality"] = self._calculate_quality_metrics()
        self.metrics["distribution"] = self._calculate_distribution_metrics()

        return dict(self.metrics)

    def _calculate_coverage_metrics(self) -> Dict:
        """
        Calculate coverage metrics (what was generated vs expected).

        Returns:
            Dict containing coverage statistics
        """
        coverage = {
            "total_files": 0,
            "scenarios": {},
            "conversation_types": {},
            "variants": {},
            "characters_used": set(),
            "expected_scenarios": self._get_expected_scenarios(),
            "missing_scenarios": [],
        }

        # Walk through output directory
        for json_file in self.output_dir.rglob("*.json"):
            # Skip validation reports, history, and version files
            parts = json_file.relative_to(self.output_dir).parts
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue

            coverage["total_files"] += 1

            # Parse path structure: {scenario}/{variant_type}/{file}
            if len(parts) >= 3:
                scenario = parts[0]
                variant_type = parts[1]

                # Derive conversation type from variant_type
                # single -> single_user, couple -> multi_user
                conversation_type = (
                    "single_user" if variant_type == "single" else "multi_user"
                )

                # Count by conversation type
                coverage["conversation_types"][conversation_type] = (
                    coverage["conversation_types"].get(conversation_type, 0) + 1
                )

                # Count by scenario
                if scenario not in coverage["scenarios"]:
                    coverage["scenarios"][scenario] = {
                        "total_files": 0,
                        "variants": set(),
                    }
                coverage["scenarios"][scenario]["total_files"] += 1
                coverage["scenarios"][scenario]["variants"].add(variant_type)

                # Count by variant
                variant_key = f"{scenario}/{variant_type}"
                coverage["variants"][variant_key] = (
                    coverage["variants"].get(variant_key, 0) + 1
                )

                # Extract character names from file
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for msg in data.get("conversation", []):
                            speaker = msg.get("speaker", "")
                            if speaker and speaker.lower() not in [
                                "sigma",
                                "assistant",
                            ]:
                                coverage["characters_used"].add(speaker)
                except Exception:
                    pass

        # Convert sets to lists for JSON serialization
        coverage["characters_used"] = sorted(list(coverage["characters_used"]))
        for scenario_data in coverage["scenarios"].values():
            scenario_data["variants"] = sorted(list(scenario_data["variants"]))

        # Find missing scenarios
        actual_scenarios = set(coverage["scenarios"].keys())
        expected_scenarios = set(coverage["expected_scenarios"])
        coverage["missing_scenarios"] = sorted(
            list(expected_scenarios - actual_scenarios)
        )

        return coverage

    def _calculate_quality_metrics(self) -> Dict:
        """
        Calculate quality metrics (dialogue length, turns, tokens).

        Returns:
            Dict containing quality statistics
        """
        quality = {
            "dialogue_lengths": [],
            "turn_counts": [],
            "token_counts": [],
            "speaker_counts": [],
            "by_scenario": defaultdict(
                lambda: {
                    "dialogue_lengths": [],
                    "turn_counts": [],
                    "token_counts": [],
                }
            ),
        }

        for json_file in self.output_dir.rglob("*.json"):
            # Skip validation reports, history, and version files
            parts = json_file.relative_to(self.output_dir).parts
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversation = data.get("conversation", [])

                    if not conversation:
                        continue

                    # Calculate metrics for this conversation
                    turn_count = len(conversation)
                    total_tokens = sum(
                        len(msg.get("content", "").split()) for msg in conversation
                    )
                    unique_speakers = len(
                        set(msg.get("speaker", "") for msg in conversation)
                    )
                    dialogue_length = sum(
                        len(msg.get("content", "")) for msg in conversation
                    )

                    quality["turn_counts"].append(turn_count)
                    quality["token_counts"].append(total_tokens)
                    quality["speaker_counts"].append(unique_speakers)
                    quality["dialogue_lengths"].append(dialogue_length)

                    # Per-scenario metrics
                    parts = json_file.relative_to(self.output_dir).parts
                    if len(parts) >= 3:
                        # Structure can be:
                        # {scenario}/{variant}/{file} or
                        # {model}/{version}/{scenario}/{variant}/{file}
                        scenario_idx = 0 if len(parts) <= 3 else 2
                        scenario = parts[scenario_idx]
                        quality["by_scenario"][scenario]["turn_counts"].append(
                            turn_count
                        )
                        quality["by_scenario"][scenario]["token_counts"].append(
                            total_tokens
                        )
                        quality["by_scenario"][scenario]["dialogue_lengths"].append(
                            dialogue_length
                        )

            except Exception:
                continue

        # Calculate statistics
        quality["statistics"] = {
            "turn_count": self._calculate_stats(quality["turn_counts"]),
            "token_count": self._calculate_stats(quality["token_counts"]),
            "speaker_count": self._calculate_stats(quality["speaker_counts"]),
            "dialogue_length": self._calculate_stats(quality["dialogue_lengths"]),
        }

        # Per-scenario statistics
        quality["scenario_statistics"] = {}
        for scenario, data in quality["by_scenario"].items():
            quality["scenario_statistics"][scenario] = {
                "turn_count": self._calculate_stats(data["turn_counts"]),
                "token_count": self._calculate_stats(data["token_counts"]),
                "dialogue_length": self._calculate_stats(data["dialogue_lengths"]),
                "file_count": len(data["turn_counts"]),
            }

        return quality

    def _calculate_distribution_metrics(self) -> Dict:
        """
        Calculate distribution metrics (character usage, scenario balance).

        Returns:
            Dict containing distribution statistics
        """
        distribution = {
            "character_frequency": Counter(),
            "scenario_balance": {},
            "variant_distribution": Counter(),
        }

        for json_file in self.output_dir.rglob("*.json"):
            # Skip validation reports, history, and version files
            parts = json_file.relative_to(self.output_dir).parts
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversation = data.get("conversation", [])

                    # Count character appearances
                    for msg in conversation:
                        speaker = msg.get("speaker", "")
                        if speaker:
                            distribution["character_frequency"][speaker] += 1

                    # Parse path for scenario/variant
                    parts = json_file.relative_to(self.output_dir).parts
                    if len(parts) >= 3:
                        # Structure can be:
                        # {scenario}/{variant}/{file} or
                        # {model}/{version}/{scenario}/{variant}/{file}
                        scenario_idx = 0 if len(parts) <= 3 else 2
                        scenario = parts[scenario_idx]
                        variant_type = parts[scenario_idx + 1]
                        distribution["variant_distribution"][
                            f"{scenario}/{variant_type}"
                        ] += 1

            except Exception:
                continue

        # Convert Counter to dict for JSON serialization
        distribution["character_frequency"] = dict(distribution["character_frequency"])
        distribution["variant_distribution"] = dict(
            distribution["variant_distribution"]
        )

        return distribution

    def _calculate_stats(self, values: List[float]) -> Dict:
        """
        Calculate statistical summary for a list of values.

        Args:
            values: List of numerical values

        Returns:
            Dict containing min, max, mean, median, std_dev, and count
        """
        if not values:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "std_dev": 0,
                "count": 0,
            }

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "count": len(values),
        }

    def _get_expected_scenarios(self) -> List[str]:
        """
        Get list of expected scenarios from config directory.

        Reads all JSON files in the scenarios directory (including nested ones)
        and extracts the "name" field to get the actual scenario names.

        Returns:
            List of scenario names
        """
        scenarios = set()
        scenarios_dir = self.config_dir / "scenarios"

        if scenarios_dir.exists():
            # Find all JSON files recursively in the scenarios directory
            for json_file in scenarios_dir.rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        # Extract the "name" field from the scenario config
                        if "name" in config:
                            scenarios.add(config["name"])
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    # Log warning but continue - malformed configs will be caught elsewhere
                    logger.warning(
                        "Could not read scenario name from %s: %s",
                        json_file.relative_to(self.config_dir),
                        str(e),
                    )
                    continue

        return sorted(list(scenarios))

    def get_scenario_metrics(self, scenario: str) -> Dict:
        """
        Get comprehensive metrics for a specific scenario.

        Args:
            scenario: Scenario name

        Returns:
            Dict containing all metrics for the scenario
        """
        scenario_metrics = {
            "scenario": scenario,
            "coverage": self._get_scenario_coverage(scenario),
            "quality": self._get_scenario_quality(scenario),
            "distribution": self._get_scenario_distribution(scenario),
        }

        return scenario_metrics

    def get_all_scenario_metrics(self) -> Dict[str, Dict]:
        """
        Get comprehensive metrics for all scenarios.

        Returns:
            Dict mapping scenario names to their metrics
        """
        # First ensure all metrics are calculated
        all_metrics = self.calculate_all_metrics()

        # Get list of all scenarios (both generated and expected)
        generated_scenarios = set(all_metrics["coverage"]["scenarios"].keys())
        expected_scenarios = set(all_metrics["coverage"]["expected_scenarios"])
        all_scenarios = generated_scenarios | expected_scenarios

        scenario_metrics = {}
        for scenario in sorted(all_scenarios):
            scenario_metrics[scenario] = self.get_scenario_metrics(scenario)

        return scenario_metrics

    def _get_scenario_coverage(self, scenario: str) -> Dict:
        """
        Get coverage metrics for a specific scenario.

        Args:
            scenario: Scenario name

        Returns:
            Dict containing coverage metrics for the scenario
        """
        coverage = {
            "total_files": 0,
            "variants": set(),
            "conversation_types": {},
            "characters_used": set(),
            "is_generated": False,
        }

        for json_file in self.output_dir.rglob("*.json"):
            parts = json_file.relative_to(self.output_dir).parts

            # Skip validation reports, history, and version files
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue

            # Determine path structure and extract scenario
            # Structure can be:
            # {scenario}/{variant}/{file} or {model}/{version}/{scenario}/{variant}/{file}
            scenario_idx = 0 if len(parts) <= 3 else 2

            if len(parts) >= 2 and parts[scenario_idx] == scenario:
                coverage["is_generated"] = True
                coverage["total_files"] += 1

                # Extract info based on path structure
                variant_idx = scenario_idx + 1
                if len(parts) > variant_idx:
                    variant_type = parts[variant_idx]
                    coverage["variants"].add(variant_type)

                    # For short paths, variant is the conversation type
                    # For long paths, first part is conversation type
                    conversation_type = parts[0] if scenario_idx > 0 else variant_type
                    coverage["conversation_types"][conversation_type] = (
                        coverage["conversation_types"].get(conversation_type, 0) + 1
                    )

                # Extract character names
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for msg in data.get("conversation", []):
                            speaker = msg.get("speaker", "")
                            if speaker and speaker.lower() not in [
                                "sigma",
                                "assistant",
                            ]:
                                coverage["characters_used"].add(speaker)
                except Exception:
                    pass

        # Convert sets to lists for JSON serialization
        coverage["variants"] = sorted(list(coverage["variants"]))
        coverage["characters_used"] = sorted(list(coverage["characters_used"]))

        return coverage

    def _get_scenario_quality(self, scenario: str) -> Dict:
        """
        Get quality metrics for a specific scenario.

        Args:
            scenario: Scenario name

        Returns:
            Dict containing quality metrics for the scenario
        """
        quality = {
            "dialogue_lengths": [],
            "turn_counts": [],
            "token_counts": [],
            "speaker_counts": [],
        }

        for json_file in self.output_dir.rglob("*.json"):
            parts = json_file.relative_to(self.output_dir).parts

            # Skip validation reports, history, and version files
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue

            # Determine path structure and extract scenario
            scenario_idx = 0 if len(parts) <= 3 else 2

            if len(parts) >= 2 and parts[scenario_idx] == scenario:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        conversation = data.get("conversation", [])

                        if not conversation:
                            continue

                        turn_count = len(conversation)
                        total_tokens = sum(
                            len(msg.get("content", "").split()) for msg in conversation
                        )
                        unique_speakers = len(
                            set(msg.get("speaker", "") for msg in conversation)
                        )
                        dialogue_length = sum(
                            len(msg.get("content", "")) for msg in conversation
                        )

                        quality["turn_counts"].append(turn_count)
                        quality["token_counts"].append(total_tokens)
                        quality["speaker_counts"].append(unique_speakers)
                        quality["dialogue_lengths"].append(dialogue_length)

                except Exception:
                    continue

        # Calculate statistics
        quality["statistics"] = {
            "turn_count": self._calculate_stats(quality["turn_counts"]),
            "token_count": self._calculate_stats(quality["token_counts"]),
            "speaker_count": self._calculate_stats(quality["speaker_counts"]),
            "dialogue_length": self._calculate_stats(quality["dialogue_lengths"]),
        }

        return quality

    def _get_scenario_distribution(self, scenario: str) -> Dict:
        """
        Get distribution metrics for a specific scenario.

        Args:
            scenario: Scenario name

        Returns:
            Dict containing distribution metrics for the scenario
        """
        distribution = {
            "character_frequency": Counter(),
            "variant_distribution": Counter(),
        }

        for json_file in self.output_dir.rglob("*.json"):
            parts = json_file.relative_to(self.output_dir).parts

            # Skip validation reports, history, and version files
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue

            # Determine path structure and extract scenario
            scenario_idx = 0 if len(parts) <= 3 else 2

            if len(parts) >= 2 and parts[scenario_idx] == scenario:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        conversation = data.get("conversation", [])

                        # Count character appearances
                        for msg in conversation:
                            speaker = msg.get("speaker", "")
                            if speaker:
                                distribution["character_frequency"][speaker] += 1

                        # Count variant distribution
                        variant_idx = scenario_idx + 1
                        if len(parts) > variant_idx:
                            variant_type = parts[variant_idx]
                            distribution["variant_distribution"][variant_type] += 1

                except Exception:
                    continue

        # Convert Counter to dict for JSON serialization
        distribution["character_frequency"] = dict(distribution["character_frequency"])
        distribution["variant_distribution"] = dict(
            distribution["variant_distribution"]
        )

        return distribution
