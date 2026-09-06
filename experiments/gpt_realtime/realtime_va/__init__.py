"""
OpenAI Realtime API Virtual Assistant Package

This package provides a modular implementation of a virtual assistant
using OpenAI's Realtime API.
"""

from realtime_va.core import RealtimeVACore
from realtime_va.audio_io import AudioInput, AudioOutput
from realtime_va.interface import LiveInterface, BatchInterface

__all__ = [
    "RealtimeVACore",
    "AudioInput",
    "AudioOutput",
    "LiveInterface",
    "BatchInterface",
]
