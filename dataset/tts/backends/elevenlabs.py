"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Audio generation using ElevenLabs API.
"""

import os
import random
import io
import re
from pydub import AudioSegment
import aiohttp


class RateLimitError(Exception):
    """Raised when ElevenLabs API returns a 429 rate limit / concurrency error.

    Attributes:
        retry_after: Seconds the API asked us to wait, when it supplied a
            Retry-After header. None when absent.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# eleven_v3 is a large, high-fidelity model that ElevenLabs excludes from
# real-time use because of its generation latency. A Flash-era timeout of a few
# seconds will abort valid v3 requests, so the default here is generous.
DEFAULT_TIMEOUT_SECONDS = 180.0

# Sample rate of the mp3 the API returns, used when we synthesise silence
# locally instead of calling out.
FALLBACK_SAMPLE_RATE = 44100
# A turn with nothing speakable ("...", "[sighs]") still occupies a beat, so it
# becomes a short pause rather than a zero-length gap.
UNSPEAKABLE_SILENCE_MS = 600
# Bracketed audio tags are direction for the model, not words.
_TAGS = re.compile(r"\[[^\]]*\]")


def _is_speakable(text: str) -> bool:
    """Report whether a turn contains anything the TTS can voice.

    ElevenLabs answers 200 with a body that is not decodable mp3 when asked to
    speak nothing, which surfaces far downstream as an opaque ffmpeg error, so
    these turns are caught before the request is made.

    Args:
        text: The turn's text.

    Returns:
        True when at least one alphanumeric character remains after audio tags
        are removed.
    """
    return bool(re.search(r"[^\W_]", _TAGS.sub("", text or "")))


def _silence(duration_ms: int = UNSPEAKABLE_SILENCE_MS) -> tuple[bytes, int]:
    """Build a short silent segment in the same shape as a synthesised one.

    Args:
        duration_ms: Length of the silence.

    Returns:
        Tuple of (raw pcm bytes, sample rate).
    """
    seg = AudioSegment.silent(duration=duration_ms, frame_rate=FALLBACK_SAMPLE_RATE)
    return seg.raw_data, seg.frame_rate


async def generate_audio(
    text: str, voice_id: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> tuple[bytes, int]:
    """
    Generates audio from text using the ElevenLabs Text-to-Speech API.

    Args:
        text (str): The text to be converted into speech.
        voice_id (str): The voice ID to use for generating the speech.
        timeout (float): Total request timeout in seconds. Must accommodate the
            configured model — eleven_v3 is far slower than the Flash models.

    Returns:
        tuple[bytes, int]: A tuple containing the audio in raw bytes and the sampling rate.

    Raises:
        ValueError: If the ElevenLabs API key is missing or the request fails.
        RateLimitError: If the API returns 429 (concurrency limit exceeded).
    """
    if not _is_speakable(text):
        # e.g. Sigma's "..." in the silent-failure test scenario.
        return _silence()

    api_key: str = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key hasn't been provided.")

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    data = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }

    chunk_size = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=data, headers=headers, timeout=timeout
        ) as response:
            if response.status == 429:
                # Surface Retry-After so the caller can back off as instructed
                # rather than guessing.
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    f"ElevenLabs rate limit exceeded (429): {await response.text()}",
                    retry_after=(
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else None
                    ),
                )
            if response.status != 200:
                raise ValueError(
                    f"Request to ElveneLabs API failed with status code {response.status},"
                    f" content: {await response.text()}"
                )

            audio_bytes_io = io.BytesIO()
            async for chunk in response.content.iter_chunked(chunk_size):
                if chunk:
                    audio_bytes_io.write(chunk)

            audio_bytes_io.seek(0)
            payload = audio_bytes_io.getvalue()
            # A 200 with a body that is not mp3 otherwise fails inside ffmpeg
            # with no indication of which utterance or voice caused it.
            if len(payload) < 512 or not payload[:4].startswith(
                (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
            ):
                raise ValueError(
                    "ElevenLabs returned 200 but the body is not decodable mp3 "
                    f"({len(payload)} bytes, head={payload[:16]!r}). "
                    f"voice_id={voice_id} text={text[:120]!r}"
                )
            audio_bytes_io.seek(0)
            audio_segment = AudioSegment.from_mp3(audio_bytes_io)
            return audio_segment.raw_data, audio_segment.frame_rate


async def generate_audio_adapter(
    text: str, language_code: str
) -> tuple[bytes, int, dict[str]]:
    """
    Adapter function to generate audio using the ElevenLabs API
    with a random voice for the specified language.

    List of available languages is available under: https://elevenlabs.io/app/voice-lab

    Args:
        text (str): The text to be converted into speech.
        language_code (str): The language code to select a voice (e.g., 'en', 'es').

    Returns:
        tuple[bytes, int, dict[str]]:
            A tuple containing the generated audio in bytes, the sampling rate and
            for the debugging purposes parameters provided for the synthesis.

    Raises:
        ValueError: If the specified language is not supported by the ElevenLabs API.
    """
    voices = {
        "en": [
            "cgSgspJ2msm6clMCkdW9",  # Jessica
            "9BWtsMINqrJLrRacOk9x",  # Aria
            "CwhRBWXzGAHq8TQ4Fs17",  # Roger
            "EXAVITQu4vr4xnSDxMaL",  # Sarah
            "IKne3meq5aSn9XLyUdCD",  # Charlie
        ],
        "es": [
            "SKjgN71N3MeGl4r2JbRt",  # Bruno Torres
            "RgXx32WYOGrd7gFNifSf",  # Eva Dorado
            "l1zE9xgNpUTaQCZzpNJa",  # Alberto Rodriguez
        ],
        "zh": [
            "4VZIsMPtgggwNg7OXbPY",  # James Gao
            "fQj4gJSexpu8RDE2Ii5m",  # Yu (Chinese)
            "Ca5bKgudqKJzq8YRFoAz",  # Coco Li
        ],
    }

    if language_code not in voices:
        raise ValueError(
            f"ElevenLabs does not handle specified language: {language_code}"
        )

    voice_id: str = random.choice(voices[language_code])
    parameters: dict[str] = {"voice_id": voice_id}
    return *(await generate_audio(text=text, voice_id=voice_id)), parameters
