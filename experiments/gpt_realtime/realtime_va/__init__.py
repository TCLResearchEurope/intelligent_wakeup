"""
OpenAI Realtime API Virtual Assistant Package

This package provides a modular implementation of a virtual assistant
using OpenAI's Realtime API.

The names below are resolved on first use rather than at import time. Only the
microphone and WAV playback paths need PyAudio, and corpus evaluation needs
neither, so importing this package must not drag PortAudio onto a machine that
will never open an audio device.
"""

# pylint: disable=undefined-all-variable
# These are resolved by __getattr__ below (PEP 562), which pylint does not model.
__all__ = [
    "RealtimeVACore",
    "AudioInput",
    "AudioOutput",
    "LiveInterface",
    "BatchInterface",
]

_MODULES = {
    "RealtimeVACore": "realtime_va.core",
    "AudioInput": "realtime_va.audio_io",
    "AudioOutput": "realtime_va.audio_io",
    "LiveInterface": "realtime_va.interface",
    "BatchInterface": "realtime_va.interface",
}


def __getattr__(name):
    """Import the module backing an exported name, on first access.

    Args:
        name: Attribute being looked up.

    Returns:
        The requested class.

    Raises:
        AttributeError: If the name is not part of the public API.
    """
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__():
    """List the public API, so tab completion still works.

    Returns:
        Sorted attribute names.
    """
    return sorted(__all__)
