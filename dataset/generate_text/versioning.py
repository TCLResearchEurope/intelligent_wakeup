"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Versioning system for text generation pipeline with config hashing and history tracking.
"""

import hashlib
import json
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
import contextlib

from .version import get_version, should_regenerate
from ..utils import logger


class VersionManager:
    """
    Manages versioning for text generation pipeline.

    Tracks code versions and config hashes to determine when regeneration is needed.
    Supports concurrent access via file locking and maintains generation history.
    """

    VERSION_FILE = "corpora_version.json"
    HISTORY_DIR = ".history"

    def __init__(self, output_dir: Path):
        """
        Initialize version manager.

        Args:
            output_dir: Root output directory for generated text
        """
        self.output_dir = Path(output_dir)
        self.version_file_path = self.output_dir / self.VERSION_FILE
        self.history_dir = self.output_dir / self.HISTORY_DIR

    def compute_config_hash(
        self,
        scenario_name: str,
        variant_type: str,
        variant_name: str,
        config_dir: Path,
    ) -> Dict[str, str]:
        """
        Compute hash of all relevant configuration files for a scenario.

        Args:
            scenario_name: Name of the scenario
            variant_type: Type of variant (single/couple)
            variant_name: Name of the variant
            config_dir: Base configuration directory

        Returns:
            Dictionary with individual file hashes and a composite hash
        """
        config_dir = Path(config_dir)
        file_hashes = {}

        # Hash the specific variant config file
        variant_config_path = (
            config_dir
            / "scenarios"
            / scenario_name
            / variant_type
            / f"{variant_name}.json"
        )
        if variant_config_path.exists():
            file_hashes["variant_config"] = self._hash_file(variant_config_path)
        else:
            logger.warning("Variant config not found: %s", variant_config_path)

        # Hash global generation config
        generation_config_path = config_dir / "generation.json"
        if generation_config_path.exists():
            file_hashes["generation_config"] = self._hash_file(generation_config_path)

        # Hash postprocessing config
        postprocessing_config_path = config_dir / "postprocessing.json"
        if postprocessing_config_path.exists():
            file_hashes["postprocessing_config"] = self._hash_file(
                postprocessing_config_path
            )

        # Hash character files referenced in the variant config
        if variant_config_path.exists():
            character_hashes = self._hash_referenced_characters(
                variant_config_path, config_dir
            )
            file_hashes.update(character_hashes)

        # Create composite hash from all individual hashes
        composite_input = json.dumps(file_hashes, sort_keys=True).encode()
        composite_hash = hashlib.md5(composite_input).hexdigest()
        file_hashes["composite"] = composite_hash

        return file_hashes

    def _hash_file(self, file_path: Path) -> str:
        """
        Compute MD5 hash of a file.

        Args:
            file_path: Path to file to hash

        Returns:
            Hexadecimal MD5 hash string
        """
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _hash_referenced_characters(
        self, variant_config_path: Path, config_dir: Path
    ) -> Dict[str, str]:
        """
        Hash all character files referenced in a variant config.

        Args:
            variant_config_path: Path to variant configuration file
            config_dir: Base configuration directory

        Returns:
            Dictionary mapping character names to their file hashes
        """
        character_hashes = {}

        try:
            with open(variant_config_path, encoding="utf-8") as f:
                config = json.load(f)

            # Extract character backstory filenames from default_characters and agent_configs
            backstory_files: Set[str] = set()

            # From default_characters (nested structure: conversation_type -> agent_name -> config)
            default_chars = config.get("default_characters", {})
            if isinstance(default_chars, dict):
                for _, agents in default_chars.items():
                    if isinstance(agents, dict):
                        for _, agent_config in agents.items():
                            if isinstance(agent_config, dict):
                                backstory_file = agent_config.get("backstory_file")
                                if backstory_file:
                                    # Remove .txt extension if present to get character name
                                    char_name = backstory_file.replace(".txt", "")
                                    backstory_files.add(char_name)

            # From agent_configs (dict of agent_name -> config)
            agent_configs = config.get("agent_configs", {})
            if isinstance(agent_configs, dict):
                for _, agent_config in agent_configs.items():
                    if isinstance(agent_config, dict):
                        char_name = agent_config.get("character")
                        if char_name:
                            backstory_files.add(char_name)

            # Hash each referenced character file
            characters_dir = config_dir / "characters"
            for char_name in backstory_files:
                char_file = characters_dir / f"{char_name}.txt"
                if char_file.exists():
                    character_hashes[f"character_{char_name}"] = self._hash_file(
                        char_file
                    )

        except Exception as e:
            logger.warning(
                "Error extracting characters from %s: %s", variant_config_path, str(e)
            )

        return character_hashes

    @contextlib.contextmanager
    def _lock_version_file(self):
        """
        Context manager for exclusive file locking.

        Ensures only one process can modify the version file at a time.
        """
        lock_file = self.output_dir / ".version_lock"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(lock_file, "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                logger.debug("Acquired lock on version file")
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                logger.debug("Released lock on version file")

    def load_version_data(self) -> Dict:
        """
        Load version data from file with locking.

        Returns:
            Version data dictionary, or empty dict if file doesn't exist
        """
        with self._lock_version_file():
            if not self.version_file_path.exists():
                logger.debug("No version file found, returning empty data")
                return {}

            try:
                with open(self.version_file_path, encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug("Loaded version data from %s", self.version_file_path)
                return data
            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse version file %s: %s",
                    self.version_file_path,
                    str(e),
                )
                return {}

    def save_version_data(self, version_data: Dict) -> None:
        """
        Save version data to file with locking.

        Also saves a timestamped copy to history directory.

        Args:
            version_data: Version data to save
        """
        with self._lock_version_file():
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save current version file
            with open(self.version_file_path, "w", encoding="utf-8") as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)
            logger.debug("Saved version data to %s", self.version_file_path)

            # Save to history
            self._save_to_history(version_data)

    def _save_to_history(self, version_data: Dict) -> None:
        """
        Save a timestamped copy of version data to history directory.

        Args:
            version_data: Version data to save
        """
        self.history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.history_dir / f"corpora_version_{timestamp}.json"

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2, ensure_ascii=False)

        logger.debug("Saved version history to %s", history_file)

    def scenario_files_exist(
        self,
        scenario_name: str,
        variant_type: str,
        variant_name: str,
        conversation_type: str,
    ) -> bool:
        """
        Check if generated files exist for a scenario.

        Args:
            scenario_name: Name of the scenario
            variant_type: Type of variant (single/couple)
            variant_name: Name of the variant
            conversation_type: Type of conversation (not used, kept for compatibility)

        Returns:
            True if generated conversation files exist, False otherwise
        """
        # Check in the output directory structure (simplified, no conversation_type/custom layers)
        scenario_dir = self.output_dir / scenario_name / variant_type

        if not scenario_dir.exists():
            return False

        # Check if there are any JSON files (conversations) in the directory
        json_files = list(scenario_dir.glob("*.json"))
        # Filter out review files
        conversation_files = [f for f in json_files if not f.stem.endswith("-review")]

        return len(conversation_files) > 0

    def check_scenario_version(
        self,
        scenario_name: str,
        variant_type: str,
        variant_name: str,
        config_dir: Path,
        conversation_type: str = "single_user",
    ) -> tuple[bool, Optional[str], bool]:
        """
        Check if a scenario needs regeneration based on version changes.

        Args:
            scenario_name: Name of the scenario
            variant_type: Type of variant (single/couple)
            variant_name: Name of the variant
            config_dir: Base configuration directory
            conversation_type: Type of conversation (single_user/multi_user/background_noise)

        Returns:
            Tuple of (needs_regeneration, reason, is_first_generation)
            - needs_regeneration: True if regeneration is needed
            - reason: Human-readable reason for regeneration, or None
            - is_first_generation: True if this is the first time generating (no version data
                AND no files)
        """
        scenario_key = f"{scenario_name}/{variant_type}/{variant_name}"
        current_version = get_version()
        current_config_hashes = self.compute_config_hash(
            scenario_name, variant_type, variant_name, config_dir
        )

        version_data = self.load_version_data()

        # Check if generated files exist
        files_exist = self.scenario_files_exist(
            scenario_name, variant_type, variant_name, conversation_type
        )

        # Check if scenario exists in version file
        if "scenarios" not in version_data:
            if files_exist:
                # Files exist but no version data - needs regeneration
                return True, "No version data found (but files exist)", False

            # First time generation - no files, no version data
            return True, "First time generation", True

        scenario_data = version_data["scenarios"].get(scenario_key)
        if not scenario_data:
            if files_exist:
                # Files exist but scenario not in version file
                return (
                    True,
                    f"Scenario {scenario_key} not tracked in version file",
                    False,
                )

            # First time generation for this scenario
            return True, "First time generation", True

        # Check code version
        stored_version = scenario_data.get("code_version")
        if not stored_version:
            return True, "No code version stored", False

        if should_regenerate(stored_version, current_version):
            return (
                True,
                f"Code version changed: {stored_version} → {current_version}",
                False,
            )

        # Check config hash
        stored_hash = scenario_data.get("config_hash", {}).get("composite")
        current_hash = current_config_hashes.get("composite")

        if stored_hash != current_hash:
            # Identify which config file changed
            stored_file_hashes = scenario_data.get("config_hash", {})
            changed_files = []
            for key, current_file_hash in current_config_hashes.items():
                if key == "composite":
                    continue
                stored_file_hash = stored_file_hashes.get(key)
                if stored_file_hash != current_file_hash:
                    changed_files.append(key)

            if changed_files:
                return True, f"Config changed: {', '.join(changed_files)}", False
            return True, "Config hash mismatch", False

        # No regeneration needed
        return False, None, False

    def update_scenario_version(
        self,
        scenario_name: str,
        variant_type: str,
        variant_name: str,
        config_dir: Path,
        conversation_count: int,
        conversation_type: str,
    ) -> None:
        """
        Update version data for a scenario after successful generation.

        Args:
            scenario_name: Name of the scenario
            variant_type: Type of variant (single/couple)
            variant_name: Name of the variant
            config_dir: Base configuration directory
            conversation_count: Number of conversations generated
            conversation_type: Type of conversation (single_user/multi_user/background_noise)
        """
        scenario_key = f"{scenario_name}/{variant_type}/{variant_name}"
        current_version = get_version()
        current_config_hashes = self.compute_config_hash(
            scenario_name, variant_type, variant_name, config_dir
        )

        version_data = self.load_version_data()

        # Initialize structure if needed
        if "scenarios" not in version_data:
            version_data["scenarios"] = {}

        # Update scenario data
        version_data["scenarios"][scenario_key] = {
            "code_version": current_version,
            "config_hash": current_config_hashes,
            "generated_at": datetime.now().isoformat(),
            "conversation_count": conversation_count,
            "conversation_type": conversation_type,
        }

        # Update corpus-level metadata
        version_data["corpus_version"] = current_version
        version_data["last_updated"] = datetime.now().isoformat()

        self.save_version_data(version_data)

        logger.info("Updated version data for scenario: %s", scenario_key)

    def get_scenarios_to_regenerate(self, config_dir: Path) -> List[Dict]:
        """
        Get list of all scenarios that need regeneration.

        Args:
            config_dir: Base configuration directory

        Returns:
            List of dictionaries with scenario information and regeneration reasons
        """
        scenarios_to_regenerate = []
        version_data = self.load_version_data()

        if "scenarios" not in version_data:
            return []

        for scenario_key, scenario_data in version_data["scenarios"].items():
            parts = scenario_key.split("/")
            if len(parts) != 3:
                logger.warning("Invalid scenario key format: %s", scenario_key)
                continue

            scenario_name, variant_type, variant_name = parts

            # Get conversation type from stored data
            conversation_type = scenario_data.get("conversation_type", "single_user")

            needs_regen, reason, _ = self.check_scenario_version(
                scenario_name, variant_type, variant_name, config_dir, conversation_type
            )

            if needs_regen:
                scenarios_to_regenerate.append(
                    {
                        "scenario_key": scenario_key,
                        "scenario_name": scenario_name,
                        "variant_type": variant_type,
                        "variant_name": variant_name,
                        "reason": reason,
                        "old_version": scenario_data.get("code_version"),
                        "generated_at": scenario_data.get("generated_at"),
                    }
                )

        return scenarios_to_regenerate
