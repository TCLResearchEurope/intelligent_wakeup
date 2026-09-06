"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Thin client for the internally-hosted Qwen3-TTS server (see qwen-tts-openapi.json).
Two endpoints:
  POST /generate-audio         voice cloning from a reference WAV (multipart form)
  POST /generate-voice-design  voice design from a natural-language style instruction,
                                no reference audio (x-www-form-urlencoded)

ICL mode (passing ref_text, the reference clip's exact transcript) is used by
default for cloning since the API's own docs say it gives better quality than
x_vector_only_mode, and LibriTTS gives us exact transcripts for free.

Note: /generate-voice-design has no persistent "voice" concept - each call
independently interprets the same instruct string. Getting several utterances
"of the same voice" means calling it repeatedly with an identical instruct and
different text; how consistent that turns out to be across calls is itself
part of what the diversity analysis measures.
"""

import os
import time
from pathlib import Path
from typing import Optional

import requests

# The Qwen3-TTS server is internally hosted and has no public endpoint, so there
# is no sensible default to ship here. Configure it out of band: set QWEN_BASE_URL
# (see pipeline/pipeline_config.env) or pass --qwen-base-url / base_url explicitly.
DEFAULT_BASE_URL = os.environ.get("QWEN_BASE_URL", "")


def _require_base_url(base_url: str) -> str:
    """Return base_url, or fail with an actionable message if it is unset."""
    if not base_url:
        raise RuntimeError(
            "No Qwen3-TTS base URL configured. Set the QWEN_BASE_URL environment "
            "variable or pass an explicit base_url / --qwen-base-url. The server is "
            "internally hosted, so this package ships no default endpoint."
        )
    return base_url


def get_version(base_url: str = DEFAULT_BASE_URL) -> str:
    """Query the Qwen3-TTS server for its version.

    Args:
        base_url: Base URL of the Qwen3-TTS server.

    Returns:
        The server's reported version string.
    """
    base_url = _require_base_url(base_url)
    response = requests.get(f"{base_url}/version", timeout=10)
    response.raise_for_status()
    return response.json()


def generate_audio(
    ref_audio_path: Path,
    text: str,
    ref_text: Optional[str] = None,
    language: str = "English",
    x_vector_only_mode: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 180.0,
    max_retries: int = 3,
) -> bytes:
    """Call /generate-audio to clone a reference voice and synthesize text.

    ICL mode (passing ref_text, the reference clip's exact transcript) is used by
    default since the API's own docs say it gives better quality than
    x_vector_only_mode.

    Args:
        ref_audio_path: Path to the reference WAV file whose voice is cloned.
        text: The text to synthesize in the cloned voice.
        ref_text: Exact transcript of the reference clip; enables ICL mode when given.
        language: Language of the text to synthesize.
        x_vector_only_mode: Whether to use x-vector-only cloning instead of ICL mode.
        base_url: Base URL of the Qwen3-TTS server.
        timeout: Per-request timeout, in seconds.
        max_retries: Number of attempts before giving up, with exponential backoff
            between retries.

    Returns:
        The synthesized audio as raw WAV bytes.
    """
    base_url = _require_base_url(base_url)
    data = {
        "text": text,
        "language": language,
        "x_vector_only_mode": str(x_vector_only_mode).lower(),
    }
    if ref_text:
        data["ref_text"] = ref_text

    last_exception: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with open(ref_audio_path, "rb") as f:
                files = {"file": (ref_audio_path.name, f, "audio/wav")}
                response = requests.post(
                    f"{base_url}/generate-audio",
                    files=files,
                    data=data,
                    timeout=timeout,
                )
            if response.status_code == 200:
                return response.content
            raise RuntimeError(
                f"Qwen3-TTS generate-audio failed ({response.status_code}): {response.text[:300]}"
            )
        except (requests.RequestException, RuntimeError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(2.0 * (2**attempt))
    raise RuntimeError(f"Qwen3-TTS generate-audio failed after retries: {last_exception}")


def generate_voice_design(
    text: str,
    instruct: str,
    language: str = "English",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 180.0,
    max_retries: int = 3,
) -> bytes:
    """Call /generate-voice-design to synthesize speech from a style instruction.

    Style-instruction based, no reference audio, no persistent voice object - every
    call is independently re-interpreted.

    Args:
        text: The text to synthesize.
        instruct: Natural-language voice/style instruction (e.g. "Speak in a calm,
            warm, elderly female voice").
        language: Language of the text to synthesize.
        base_url: Base URL of the Qwen3-TTS server.
        timeout: Per-request timeout, in seconds.
        max_retries: Number of attempts before giving up, with exponential backoff
            between retries.

    Returns:
        The synthesized audio as raw WAV bytes.
    """
    base_url = _require_base_url(base_url)
    data = {"text": text, "instruct": instruct, "language": language}

    last_exception: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/generate-voice-design", data=data, timeout=timeout
            )
            if response.status_code == 200:
                return response.content
            raise RuntimeError(
                f"Qwen3-TTS generate-voice-design failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        except (requests.RequestException, RuntimeError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(2.0 * (2**attempt))
    raise RuntimeError(f"Qwen3-TTS generate-voice-design failed after retries: {last_exception}")
