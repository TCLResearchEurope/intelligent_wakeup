"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Audio generation using OpenAI's TTS model.
"""

import os
import io
import random
from openai import AsyncOpenAI
from pydub import AudioSegment
from .openai_audio import voices


async def generate_audio(text: str, voice: str) -> tuple[bytes, int]:
    """
    Generates speech from text using OpenAI's TTS model with a specified voice.

    Args:
        text (str): The text to be converted into speech.
        voice (str): The voice to be used for speech synthesis.

    Returns:
        tuple[bytes, int]:
            A tuple containing the generated audio in bytes and the sample rate.

    Raises:
        ValueError: If the OpenAI API key is not provided.
    """
    api_key: str = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key hasn't been provided.")

    client = AsyncOpenAI(api_key=api_key)

    response = await client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
    )

    audio_bytes_io = io.BytesIO()
    for chunk in response.iter_bytes():
        if chunk:
            audio_bytes_io.write(chunk)
    audio_bytes_io.seek(0)
    audio_segment = AudioSegment.from_file(audio_bytes_io, format="mp3")
    return audio_segment.raw_data, audio_segment.frame_rate


async def generate_audio_adapter(text: str) -> tuple[bytes, int, dict[str]]:
    """
    Adapter function to generate audio from text using OpenAI's TTS model with a random voice.

    Args:
        text (str): The text to be converted into speech.

    Returns:
        tuple[bytes, int, dict[str]]:
            A tuple containing the generated audio in bytes, the sampling rate and
            for the debugging purposes parameters provided for the synthesis.
    """
    voice: str = random.choice(voices)
    parameters: dict[str] = {
        "voice": voice,
    }
    return *(await generate_audio(text=text, voice=voice)), parameters
