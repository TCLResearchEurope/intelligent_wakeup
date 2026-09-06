"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0
"""

from enum import Enum


class TextToSpeechModel(Enum):
    """Enum representing available Text-to-Speech models."""

    # pylint: disable=invalid-name
    ElevenLabs: str = "ElevenLabs"
    OpenAI_TTS: str = "OpenAI_TTS"
    OpenAI_Audio: str = "OpenAI_Audio"
    OpenAI_Audio_mini: str = "OpenAI_Audio_mini"
    Kokoro: str = "Kokoro"
