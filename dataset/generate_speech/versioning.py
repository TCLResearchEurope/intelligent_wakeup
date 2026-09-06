"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Versioning system for speech generation pipeline with input hashing and history tracking.
"""

import hashlib
import json
import fcntl
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

from .version import get_version, should_regenerate
from ..utils import logger


class SpeechVersionManager:
    """
    Manages versioning for the speech generation pipeline.

    Tracks code versions and input hashes (text file + TTS config + voice mapping)
    to skip re-generation of already up-to-date dialogs.  Supports concurrent
    access via file locking and maintains a generation history.
    """

    VERSION_FILE = "speech_version.json"
    HISTORY_DIR = ".history"

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Root output directory for generated speech.
        """
        self.output_dir = Path(output_dir)
        self.version_file_path = self.output_dir / self.VERSION_FILE
        self.history_dir = self.output_dir / self.HISTORY_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_dialog_version(
        self,
        dialog_key: str,
        input_file: Path,
        tts_config: Dict,
        voice_mapping: Dict,
        output_wav: Path,
        sound_effects_config: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Check whether a dialog needs to be (re-)generated, and how much of it.

        Args:
            dialog_key: Unique key for this dialog, e.g.
                        ``"WritingWorkshop_couple_evening_session"``.
            input_file: Path to the source text JSON file.
            tts_config: TTS configuration dict for the scenario.
            voice_mapping: Loaded voice mapping dict (may be empty).
            output_wav: Expected output .wav path.
            sound_effects_config: Background/ambience config for the scenario.

        Returns:
            Tuple of (needs_generation, reason, mode).
            *needs_generation* is ``True`` if audio should be generated.
            *reason* is a human-readable explanation, or ``None`` when
            generation can be skipped.
            *mode* is ``"full"`` when speech must be re-synthesised,
            ``"background_only"`` when the cached per-utterance speech can be
            reused and only the background needs re-mixing, or ``"none"`` when
            nothing needs doing.
        """
        if not output_wav.exists():
            return True, "Output file does not exist", "full"

        current_hash = self._compute_composite_hash(
            input_file, tts_config, voice_mapping
        )
        current_bg_hash = self._compute_background_hash(sound_effects_config)
        current_version = get_version()
        version_data = self._load_version_data()
        stored = version_data.get("dialogs", {}).get(dialog_key)

        if stored is None:
            return True, "No version record found", "full"

        stored_version = stored.get("code_version")
        if not stored_version or should_regenerate(stored_version, current_version):
            return (
                True,
                f"Code version changed: {stored_version} → {current_version}",
                "full",
            )

        if stored.get("composite_hash") != current_hash:
            return True, "Input or speech config changed", "full"

        # Speech inputs are unchanged. Records written before background
        # tracking existed have no background_hash; treat those as a background
        # change so the ambience is refreshed without paying for TTS again.
        stored_bg_hash = stored.get("background_hash")
        if stored_bg_hash is None:
            return (
                True,
                "No background hash recorded (pre-existing dialog)",
                "background_only",
            )
        if stored_bg_hash != current_bg_hash:
            return True, "Background config changed", "background_only"

        return False, None, "none"

    def update_dialog_version(
        self,
        dialog_key: str,
        input_file: Path,
        tts_config: Dict,
        voice_mapping: Dict,
        output_wav: Path,
        sound_effects_config: Optional[Dict] = None,
    ) -> None:
        """
        Record that a dialog was successfully generated.

        Args:
            dialog_key: Unique key for this dialog.
            input_file: Path to the source text JSON file.
            tts_config: TTS configuration dict for the scenario.
            voice_mapping: Loaded voice mapping dict.
            output_wav: Path where the audio was written.
            sound_effects_config: Background/ambience config for the scenario.
        """
        composite_hash = self._compute_composite_hash(
            input_file, tts_config, voice_mapping
        )
        background_hash = self._compute_background_hash(sound_effects_config)
        current_version = get_version()

        version_data = self._load_version_data()
        version_data.setdefault("dialogs", {})[dialog_key] = {
            "code_version": current_version,
            "composite_hash": composite_hash,
            "background_hash": background_hash,
            "input_file": str(input_file),
            "output_file": str(output_wav),
            "generated_at": datetime.now().isoformat(),
        }
        version_data["last_updated"] = datetime.now().isoformat()
        self._save_version_data(version_data)
        logger.debug("Updated version record for dialog: %s", dialog_key)

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    def _compute_composite_hash(
        self, input_file: Path, tts_config: Dict, voice_mapping: Dict
    ) -> str:
        """Hash the inputs that determine the synthesised *speech*.

        Deliberately excludes ``sound_effects_config``: the background is mixed
        in after synthesis, so a background-only change does not invalidate the
        cached per-utterance audio. See ``_compute_background_hash``.
        """
        parts = {
            "input_file": self._hash_file(input_file),
            "tts_config": self._hash_dict(tts_config),
            "voice_mapping": self._hash_dict(voice_mapping),
        }
        composite_input = json.dumps(parts, sort_keys=True).encode()
        return hashlib.md5(composite_input).hexdigest()

    def _compute_background_hash(self, sound_effects_config: Optional[Dict]) -> str:
        """Hash the config that only affects the background/ambience mix."""
        return self._hash_dict(sound_effects_config or {})

    @staticmethod
    def _hash_file(path: Path) -> str:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _hash_dict(data: Dict) -> str:
        serialised = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.md5(serialised).hexdigest()

    # ------------------------------------------------------------------
    # Version file I/O
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _lock_version_file(self):
        """Exclusive file lock so parallel jobs don't corrupt the version file."""
        lock_file = self.output_dir / ".speech_version_lock"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_version_data(self) -> Dict:
        with self._lock_version_file():
            if not self.version_file_path.exists():
                return {}
            try:
                with open(self.version_file_path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse speech version file: %s", exc)
                return {}

    def _save_version_data(self, version_data: Dict) -> None:
        with self._lock_version_file():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.version_file_path, "w", encoding="utf-8") as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)
            self._save_to_history(version_data)

    def _save_to_history(self, version_data: Dict) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.history_dir / f"speech_version_{timestamp}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2, ensure_ascii=False)
