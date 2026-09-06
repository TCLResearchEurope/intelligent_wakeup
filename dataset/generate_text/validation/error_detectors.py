"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Error detection for text corpora validation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import hashlib

from ...utils import logger


class ErrorDetector:
    """
    Detect various types of errors in generated text corpora.
    """

    # Severity levels
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    def __init__(self, output_dir: Path, config_dir: Path, min_turn_count: int = 4):
        """
        Initialize error detector.

        Args:
            output_dir: Directory containing generated conversations
            config_dir: Directory containing scenario configurations
            min_turn_count: Minimum number of turns for a valid conversation
        """
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        self.min_turn_count = min_turn_count
        self.errors = defaultdict(list)

    def detect_all_errors(self) -> Dict[str, List[Dict]]:
        """
        Detect all types of errors.

        Returns:
            Dict mapping severity levels to lists of errors
        """
        self._detect_malformed_json()
        self._detect_missing_files()
        self._detect_short_dialogues()
        self._detect_missing_fields()
        self._detect_empty_utterances()
        self._detect_speaker_issues()
        self._detect_duplicate_dialogues()
        self._detect_sigma_invocation_missing()
        self._detect_sigma_filler_words()

        return dict(self.errors)

    def _detect_malformed_json(self):
        """
        Detect JSON files that cannot be parsed.
        """
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
                    json.load(f)
            except json.JSONDecodeError as e:
                self.errors[self.CRITICAL].append(
                    {
                        "type": "malformed_json",
                        "file": str(json_file.relative_to(self.output_dir)),
                        "message": f"Invalid JSON: {str(e)}",
                        "line": getattr(e, "lineno", None),
                    }
                )
            except Exception as e:
                self.errors[self.CRITICAL].append(
                    {
                        "type": "file_read_error",
                        "file": str(json_file.relative_to(self.output_dir)),
                        "message": f"Cannot read file: {str(e)}",
                    }
                )

    def _detect_missing_files(self):
        """
        Detect scenarios that have config but no generated output.
        """
        expected_scenarios = self._get_expected_scenarios()
        actual_scenarios = self._get_actual_scenarios()

        missing = expected_scenarios - actual_scenarios
        for scenario in sorted(missing):
            self.errors[self.CRITICAL].append(
                {
                    "type": "missing_scenario",
                    "scenario": scenario,
                    "message": f"Scenario '{scenario}' has config but no generated files",
                }
            )

    def _detect_short_dialogues(self):
        """
        Detect conversations with too few turns.
        """
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
                    turn_count = len(conversation)

                    if turn_count == 0:
                        self.errors[self.CRITICAL].append(
                            {
                                "type": "empty_conversation",
                                "file": str(json_file.relative_to(self.output_dir)),
                                "message": "Conversation has no turns",
                            }
                        )
                    elif turn_count < self.min_turn_count:
                        self.errors[self.CRITICAL].append(
                            {
                                "type": "short_dialogue",
                                "file": str(json_file.relative_to(self.output_dir)),
                                "message": (
                                    f"Conversation has only {turn_count} turns "
                                    f"(minimum: {self.min_turn_count})"
                                ),
                                "turn_count": turn_count,
                            }
                        )
            except Exception:
                # Already caught by malformed_json detector
                pass

    def _detect_missing_fields(self):
        """
        Detect messages missing required fields.
        """
        required_fields = {"speaker", "content"}

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

                    for idx, msg in enumerate(conversation):
                        missing = required_fields - set(msg.keys())
                        if missing:
                            self.errors[self.WARNING].append(
                                {
                                    "type": "missing_fields",
                                    "file": str(json_file.relative_to(self.output_dir)),
                                    "message": f"Turn {idx} missing fields: {', '.join(missing)}",
                                    "turn_index": idx,
                                    "missing_fields": list(missing),
                                }
                            )
            except Exception:
                # Already caught by malformed_json detector
                pass

    def _detect_empty_utterances(self):
        """
        Detect messages with empty content.
        """
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

                    for idx, msg in enumerate(conversation):
                        content = msg.get("content", "").strip()
                        if not content:
                            self.errors[self.WARNING].append(
                                {
                                    "type": "empty_utterance",
                                    "file": str(json_file.relative_to(self.output_dir)),
                                    "message": f"Turn {idx} has empty content",
                                    "turn_index": idx,
                                    "speaker": msg.get("speaker", "unknown"),
                                }
                            )
            except Exception:
                pass

    def _detect_speaker_issues(self):
        """
        Detect speaker name issues.
        """
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

                    speakers = [msg.get("speaker", "") for msg in conversation]

                    # Check for missing speaker names
                    for idx, speaker in enumerate(speakers):
                        if not speaker or speaker.strip() == "":
                            self.errors[self.WARNING].append(
                                {
                                    "type": "missing_speaker",
                                    "file": str(json_file.relative_to(self.output_dir)),
                                    "message": f"Turn {idx} has no speaker name",
                                    "turn_index": idx,
                                }
                            )

                    # Check for single-speaker dominance in multi-user conversations
                    parts = json_file.relative_to(self.output_dir).parts
                    if len(parts) > 0 and "multi_user" in parts[0]:
                        unique_speakers = set(s for s in speakers if s)
                        if len(unique_speakers) == 1:
                            self.errors[self.WARNING].append(
                                {
                                    "type": "single_speaker_dominance",
                                    "file": str(json_file.relative_to(self.output_dir)),
                                    "message": "Multi-user conversation has only one speaker",
                                    "speaker": (
                                        list(unique_speakers)[0]
                                        if unique_speakers
                                        else "unknown"
                                    ),
                                }
                            )
            except Exception:
                pass

    def _detect_duplicate_dialogues(self):
        """
        Detect potentially duplicate conversations using content hashing.
        """
        content_hashes = defaultdict(list)

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

                    # Create hash from conversation content
                    content = " ".join(msg.get("content", "") for msg in conversation)
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    content_hashes[content_hash].append(
                        str(json_file.relative_to(self.output_dir))
                    )
            except Exception:
                pass

        # Report duplicates
        for content_hash, files in content_hashes.items():
            if len(files) > 1:
                self.errors[self.WARNING].append(
                    {
                        "type": "duplicate_dialogue",
                        "files": files,
                        "message": f"Potentially duplicate content found in {len(files)} files",
                        "count": len(files),
                    }
                )

    def _get_expected_scenarios(self) -> Set[str]:
        """
        Get set of expected scenarios from config directory.

        Reads all JSON files in the scenarios directory (including nested ones)
        and extracts the "name" field to get the actual scenario names.

        Returns:
            Set of scenario names
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

        return scenarios

    def _get_actual_scenarios(self) -> Set[str]:
        """
        Get set of scenarios that have generated files.

        Returns:
            Set of scenario names
        """
        scenarios = set()

        for json_file in self.output_dir.rglob("*.json"):
            parts = json_file.relative_to(self.output_dir).parts

            # Skip validation reports, history, and version files
            if (
                "validation" in parts
                or ".history" in parts
                or json_file.name == "corpora_version.json"
            ):
                continue

            if len(parts) >= 2:
                # Structure can be:
                # {scenario}/{variant}/{file} or {model}/{version}/{scenario}/{variant}/{file}
                scenario_idx = 0 if len(parts) <= 3 else 2
                scenario = parts[scenario_idx]
                scenarios.add(scenario)

        return scenarios

    def get_error_summary(self) -> Dict:
        """
        Get summary of errors by severity and type.

        Returns:
            Dict containing error counts by severity and type
        """
        summary = {
            "total_errors": sum(len(errors) for errors in self.errors.values()),
            "by_severity": {
                self.CRITICAL: len(self.errors.get(self.CRITICAL, [])),
                self.WARNING: len(self.errors.get(self.WARNING, [])),
                self.INFO: len(self.errors.get(self.INFO, [])),
            },
            "by_type": defaultdict(int),
        }

        for _, errors in self.errors.items():
            for error in errors:
                error_type = error.get("type", "unknown")
                summary["by_type"][error_type] += 1

        summary["by_type"] = dict(summary["by_type"])
        return summary

    def get_errors_by_scenario(self) -> Dict[str, Dict]:
        """
        Group errors by scenario.

        Returns:
            Dict mapping scenario names to error dictionaries (by severity)
        """
        scenario_errors = defaultdict(lambda: defaultdict(list))

        for severity, errors in self.errors.items():
            for error in errors:
                # Extract scenario from file path if available
                scenario = self._extract_scenario_from_error(error)
                if scenario:
                    scenario_errors[scenario][severity].append(error)

        # Convert defaultdict to regular dict for JSON serialization
        return {scenario: dict(errors) for scenario, errors in scenario_errors.items()}

    def get_scenario_error_summaries(self) -> Dict[str, Dict]:
        """
        Get error summaries for each scenario.

        Returns:
            Dict mapping scenario names to their error summaries
        """
        scenario_errors = self.get_errors_by_scenario()
        summaries = {}

        for scenario, errors in scenario_errors.items():
            summaries[scenario] = {
                "total_errors": sum(len(errs) for errs in errors.values()),
                "by_severity": {
                    self.CRITICAL: len(errors.get(self.CRITICAL, [])),
                    self.WARNING: len(errors.get(self.WARNING, [])),
                    self.INFO: len(errors.get(self.INFO, [])),
                },
                "by_type": defaultdict(int),
            }

            for _, error_list in errors.items():
                for error in error_list:
                    error_type = error.get("type", "unknown")
                    summaries[scenario]["by_type"][error_type] += 1

            summaries[scenario]["by_type"] = dict(summaries[scenario]["by_type"])

        return summaries

    def _detect_sigma_invocation_missing(self):
        """
        Detect if first Sigma turn is not preceded by 'sigma' mention.
        """
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

                    # Find first Sigma turn
                    first_sigma_idx = None
                    for idx, msg in enumerate(conversation):
                        speaker = msg.get("speaker", "")
                        if speaker.lower() == "sigma":
                            first_sigma_idx = idx
                            break

                    # If Sigma appears and it's not the first turn
                    if first_sigma_idx is not None and first_sigma_idx > 0:
                        # Check if previous turn contains 'sigma' (case-insensitive)
                        previous_msg = conversation[first_sigma_idx - 1]
                        previous_content = previous_msg.get("content", "").lower()

                        if "sigma" not in previous_content:
                            self.errors[self.CRITICAL].append(
                                {
                                    "type": "sigma_invocation_missing",
                                    "file": str(json_file.relative_to(self.output_dir)),
                                    "message": (
                                        f"First Sigma turn at index {first_sigma_idx} is not "
                                        "preceded by 'sigma' mention in previous turn"
                                    ),
                                    "turn_index": first_sigma_idx,
                                    "previous_turn_index": first_sigma_idx - 1,
                                    "previous_speaker": previous_msg.get(
                                        "speaker", "unknown"
                                    ),
                                }
                            )
            except Exception:
                # Already caught by malformed_json detector
                pass

    def _detect_sigma_filler_words(self):
        """
        Detect filler words (um, mm, uh) in Sigma turns.
        """
        # Define filler words to check (case-insensitive)
        filler_words = ["um", "mm", "uh"]

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

                    # Check all Sigma turns
                    for idx, msg in enumerate(conversation):
                        speaker = msg.get("speaker", "")
                        if speaker.lower() == "sigma":
                            content = msg.get("content", "")
                            content_lower = content.lower()

                            # Check for filler words (as whole words, not substrings)
                            found_fillers = []
                            for filler in filler_words:
                                # Use word boundaries to match whole words
                                pattern = r"\b" + re.escape(filler) + r"\b"
                                if re.search(pattern, content_lower):
                                    found_fillers.append(filler)

                            if found_fillers:
                                self.errors[self.CRITICAL].append(
                                    {
                                        "type": "sigma_filler_words",
                                        "file": str(
                                            json_file.relative_to(self.output_dir)
                                        ),
                                        "message": (
                                            f"Sigma turn at index {idx} contains filler words: "
                                            f"{', '.join(found_fillers)}"
                                        ),
                                        "turn_index": idx,
                                        "filler_words": found_fillers,
                                        "content_preview": (
                                            content[:100] + "..."
                                            if len(content) > 100
                                            else content
                                        ),
                                    }
                                )
            except Exception:
                # Already caught by malformed_json detector
                pass

    def _extract_scenario_from_error(self, error: Dict) -> str:
        """
        Extract scenario name from error information.

        Args:
            error: Error dictionary

        Returns:
            Scenario name or empty string if not found
        """
        # Try to extract from file path
        file_path = error.get("file")
        if file_path:
            parts = Path(file_path).parts
            # Structure can be:
            # {scenario}/{variant}/{file} or {model}/{version}/{scenario}/{variant}/{file}
            scenario_idx = 0 if len(parts) <= 3 else 2
            if len(parts) >= 2:
                return parts[scenario_idx]

        # Try to extract from files list (for duplicate errors)
        files = error.get("files", [])
        if files and len(files) > 0:
            parts = Path(files[0]).parts
            scenario_idx = 0 if len(parts) <= 3 else 2
            if len(parts) >= 2:
                return parts[scenario_idx]

        # Try to extract from scenario field (for missing scenario errors)
        scenario = error.get("scenario")
        if scenario:
            return scenario

        return ""
