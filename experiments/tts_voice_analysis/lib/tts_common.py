"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Shared helpers for the tts_voice_analysis scripts: loading voice_mapping.json /
character backstories, and calling the ElevenLabs TTS REST API directly (the
shared st_synthetic_audio_generation package is not used here since its git
submodule isn't checked out in this repo).

DEFAULT_OUTPUT_DIR honors the TTS_OUTPUT_DIR environment variable so a test run
can redirect every script's output elsewhere (e.g. output_test/) without
passing --output-dir through every single step - see pipeline/run_pipeline.sh
and pipeline/pipeline_config.test.env.
"""

import os
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "dataset" / "config"
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("TTS_OUTPUT_DIR") or (Path(__file__).resolve().parent.parent / "output")
)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_TTS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def load_voice_mapping(config_dir: Path) -> dict:
    """Load the characters dict from voice_mapping.json.

    Args:
        config_dir: Path to the config directory containing voice_mapping.json.

    Returns:
        The "characters" mapping loaded from voice_mapping.json.
    """
    import json

    with open(config_dir / "voice_mapping.json", encoding="utf-8") as f:
        return json.load(f)["characters"]


def load_backstory(config_dir: Path, name: str) -> Optional[str]:
    """Load a character's backstory file contents.

    Args:
        config_dir: Path to the config directory containing the characters subdirectory.
        name: Character name; matched case-insensitively to a backstory filename.

    Returns:
        The backstory file's text contents, or None if no such file exists.
    """
    path = config_dir / "characters" / f"{name.lower()}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def sanitize(name: str) -> str:
    """Make a name safe for use as a filename component.

    Args:
        name: Raw name to sanitize.

    Returns:
        name with any character that isn't alphanumeric, "-", or "_" replaced by "_".
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def elevenlabs_tts(text: str, voice_id: str, api_key: str, model_id: str) -> bytes:
    """Synthesize speech for text with the given ElevenLabs voice.

    Args:
        text: Text to synthesize.
        voice_id: ElevenLabs voice ID to synthesize with.
        api_key: ElevenLabs API key.
        model_id: ElevenLabs TTS model ID to use.

    Returns:
        The synthesized audio as MP3 bytes.
    """
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    max_retries = 5
    for attempt in range(max_retries):
        response = requests.post(url, json=body, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.content
        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(2.0 * (2**attempt))
            continue
        raise RuntimeError(
            f"ElevenLabs TTS failed for voice {voice_id} ({response.status_code}): "
            f"{response.text[:300]}"
        )
    raise RuntimeError(f"ElevenLabs TTS failed for voice {voice_id} after retries (rate limited)")
