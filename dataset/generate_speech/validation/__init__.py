"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Validation module for generated speech corpora.
"""

from .validate_speech_corpora import main as validate_speech_corpora
from .audio_validators import AudioValidator
from .report_generator import SpeechReportGenerator

__all__ = [
    "validate_speech_corpora",
    "AudioValidator",
    "SpeechReportGenerator",
]
