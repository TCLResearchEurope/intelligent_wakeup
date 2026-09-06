"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Audio generation using OpenAI's GPT-4o audio model.
"""

import os
import base64
import random
from openai import AsyncOpenAI
from ..utils import convert_language_code
from .prompts import prompts


voices: list[str] = [
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
]


async def generate_audio(
    model: str, text: str, voice: str, prompt: str, language_code: str
) -> tuple[bytes, int]:
    """
    Generates audio from text using OpenAI's GPT-4o audio model with specified voice and modulation.

    Args:
        model (str): GPT audio model which is to be used.
        text (str): The text to be converted into speech.
        voice (str): The voice to be used for speech synthesis.
        prompt (str): A description of how the voice should be modulated.
        language_code (str): The language code to adjust the language of the text.

    Returns:
        tuple[bytes, int]:
            A tuple containing the generated audio in bytes and the sample rate (24000Hz).

    Raises:
        ValueError: If the OpenAI API key is not provided.
    """
    api_key: str = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key hasn't been provided.")
    language = convert_language_code(language_code=language_code)

    client = AsyncOpenAI(api_key=api_key)

    completion = await client.chat.completions.create(
        model=model,
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": f"""
                    Could you repeat this sentence without modifying anything in {language}?: {text}
                    Modulate the voice according to the following characteristics: {prompt}
                """,
            }
        ],
    )

    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    return wav_bytes, 24000


async def generate_audio_adapter(
    model: str, text: str, language_code: str
) -> tuple[bytes, int, dict[str]]:
    """
    Adapter function to generate audio from text using OpenAI's GPT-4 model
    with a random voice and modulation prompt.

    Args:
        model (str): GPT audio model which is to be used.
        text (str): The text to be converted into speech.
        language_code (str): The language code to select the appropriate language for the voice.

    Returns:
        tuple[bytes, int, dict[str]]:
            A tuple containing the generated audio in bytes, the sampling rate and
            for the debugging purposes parameters provided for the synthesis.
    """
    voice: str = random.choice(voices)
    prompt: str = random.choice(prompts)
    parameters: dict[str] = {"voice": voice, "prompt": prompt}

    return (
        *(
            await generate_audio(
                model=model,
                text=text,
                voice=voice,
                prompt=prompt,
                language_code=language_code,
            )
        ),
        parameters,
    )
