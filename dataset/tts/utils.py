"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Set of auxiliary functions.
"""

import os
import io
import wave
from typing import Callable, Any
import time
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
import langcodes


def resample_audio(
    audio_bytes: bytes, orig_sampling_rate: int, target_sampling_rate: int
) -> bytes:
    """Resamples audio from one sampling rate to another."""
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_tensor = torch.tensor(audio).unsqueeze(0).float()
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sampling_rate,
        new_freq=target_sampling_rate,
        lowpass_filter_width=16,
    )
    resampled_audio = resampler(audio_tensor)
    return resampled_audio.squeeze(0).numpy().astype(np.int16).tobytes()


def save_audio(audio_bytes: bytes, sample_rate: int, output_file_path: str) -> None:
    """Saves audio data as a .wav file to the specified file path."""
    with wave.open(output_file_path, "wb") as wav_file:
        wav_file: wave.Wave_write
        n_channels = 1
        sampwidth = 2
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


def add_background_audio(
    foreground_audio: bytes,
    background_audio: bytes,
    foreground_sampling_rate: int,
    background_sampling_rate: int,
    decibels_diff: int = 15,
) -> bytes:
    """Adds background audio to foreground audio with adjustable volume difference."""
    background_audio = resample_audio(
        audio_bytes=background_audio,
        orig_sampling_rate=background_sampling_rate,
        target_sampling_rate=foreground_sampling_rate,
    )

    foreground_audio_segment = AudioSegment.from_raw(
        io.BytesIO(foreground_audio),
        frame_rate=foreground_sampling_rate,
        sample_width=2,
        channels=1,
    )
    background_audio_segment = AudioSegment.from_raw(
        io.BytesIO(background_audio),
        frame_rate=foreground_sampling_rate,
        sample_width=2,
        channels=1,
    )
    background_audio_segment -= decibels_diff - (
        foreground_audio_segment.dBFS - background_audio_segment.dBFS
    )
    combined_audio = foreground_audio_segment.overlay(
        background_audio_segment, loop=True
    )
    return combined_audio.raw_data


def create_directory(path: str) -> None:
    """Creates specified directory if it does not yet exist."""
    dir_path = path
    if os.path.isfile(dir_path):
        dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def measure_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Measures execution time of specified function in milliseconds."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        elapsed_time *= 1000.0

    return result, elapsed_time


def convert_language_code(language_code: str) -> str:
    """Converts a language code to its corresponding language name."""
    return langcodes.Language.make(language=language_code).display_name()
