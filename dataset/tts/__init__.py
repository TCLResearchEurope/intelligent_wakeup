"""Copyright © 2026 TCL Research Europe.

SPDX-License-Identifier: Apache-2.0
"""

from .backends import elevenlabs
from .backends.kokoro import Kokoro
from .backends import openai_audio
from .backends import openai_tts
from .utils import save_audio, add_background_audio
from .generate_background_noise import (
    generate_sound_effect_elevenlabs,
    generate_sound_effect_stable_audio,
)
from .text_to_speech import TextToSpeechModel
