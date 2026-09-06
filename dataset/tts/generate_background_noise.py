"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Functions for background sound generation.
"""

import os
import io
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import numpy as np
import torch
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def generate_sound_effect_elevenlabs(prompt: str, duration: float) -> tuple[bytes, int]:
    """
    Generate a sound effect from text using the ElevenLabs API.

    Args:
        prompt (str): The textual description of the sound effect.
        duration (float): Desired duration of the audio in seconds.

    Returns:
        tuple[bytes, int]: A tuple containing:
            - The raw audio bytes in PCM format.
            - The effective sampling rate in Hz.

    Raises:
        ValueError: If the ElevenLabs API key is not set in the environment.
    """
    api_key: str = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key hasn't been provided.")

    client = ElevenLabs(api_key=api_key)

    response = client.text_to_sound_effects.convert(
        text=prompt,
        duration_seconds=duration,
        loop=True,
    )

    audio_bytes_io = io.BytesIO()
    for chunk in response:
        if chunk:
            audio_bytes_io.write(chunk)

    audio_bytes_io.seek(0)
    audio_segment = AudioSegment.from_mp3(audio_bytes_io)
    return audio_segment.raw_data, audio_segment.frame_rate * 2


def generate_sound_effect_stable_audio(
    prompt: str, duration: float
) -> tuple[bytes, int]:
    """
    Generate a sound effect using Stability AI's `stable-audio-open-1.0` diffusion model.

    Args:
        prompt (str): A textual description of the desired sound.
        duration (float): Length of the output audio in seconds.

    Returns:
        tuple[bytes, int]: A tuple containing:
            - The raw PCM audio bytes (int16, interleaved channels).
            - The sample rate in Hz.

    Raises:
        ValueError: If the generated audio duration significantly differs from the
            requested duration.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)

    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration}]

    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )

    output = rearrange(output, "b d n -> d (b n)")

    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
        .numpy()
    )

    audio_duration: float = round(np.shape(output)[1] / sample_rate, 2)
    duration_epsilon: float = 0.5
    if duration - audio_duration > duration_epsilon:
        raise ValueError(
            f"Generated sound has duration of {audio_duration} seconds. "
            f"While requested duration is {duration}"
        )

    num_samples: int = int(sample_rate * duration)
    output = output[:, :num_samples]
    return output.tobytes(), sample_rate
