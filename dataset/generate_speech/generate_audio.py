"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Main script for creating audio files from text corpus.
"""

import os
import sys
import json
import wave
import random
import base64
import argparse
from pathlib import Path
from typing import Optional, Tuple
import asyncio
import requests
import soundfile as sf
import numpy as np
import pyroomacoustics as pra

from ..tts import (
    openai_audio,
    openai_tts,
    elevenlabs,
    save_audio,
    Kokoro,
    generate_sound_effect_elevenlabs,
    generate_sound_effect_stable_audio,
    add_background_audio,
    TextToSpeechModel,
)
from ..tts.backends.elevenlabs import RateLimitError
from ..utils import logger, LoggerConfigurator
from .version import get_version
from .versioning import SpeechVersionManager


async def retry_with_backoff(func, *func_args, max_retries=3, base_delay=1.0, **kwargs):
    """
    Retry an async function with exponential backoff on timeout/connection errors.

    Args:
        func: The async function to retry
        *func_args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return await func(*func_args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if attempt < max_retries:
                # Prefer the server's own Retry-After when it sends one.
                delay = base_delay * (2**attempt) * 5  # Longer backoff for rate limits
                if getattr(e, "retry_after", None):
                    delay = max(delay, e.retry_after)
                logger.warning(
                    "Attempt %d rate-limited by ElevenLabs, retrying in %.1f seconds...",
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("All retry attempts failed (rate limit)")
        except (
            asyncio.TimeoutError,
            asyncio.exceptions.CancelledError,
            ConnectionError,
        ) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    "Attempt %d failed with %s, retrying in %.1f seconds...",
                    attempt + 1,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("All retry attempts failed")
        except Exception as e:
            # Don't retry on non-network errors
            logger.error("Non-retryable error: %s", str(e))
            raise

    # If we get here, all retries failed
    raise last_exception


# Default sampling rates for different TTS models
TTS_MODEL_SAMPLING_RATES = {
    "Kokoro": 24000,
    "ElevenLabs": 44100,
    "OpenAI_TTS": 24000,
    "OpenAI_Audio": 24000,
    "OpenAI_Audio_mini": 24000,
}

# Fallback length of an ambient/silence segment when neither the utterance nor
# sound_effects_config specifies a duration
DEFAULT_BACKGROUND_NOISE_DURATION = 8.0

# Sentinel speaker name marking a silence/ambient-only segment in a conversation
BACKGROUND_NOISE_SPEAKER = "background_noise"


def design_elevenlabs_voice(
    voice_description: str,
    character_name: str,
    api_key: Optional[str] = None,
    model_id: str = "eleven_multilingual_ttv_v2",
) -> Tuple[str, bytes]:
    """
    Design and create a permanent voice using ElevenLabs Voice Design API.

    This function performs two steps:
    1. Design a voice preview using /v1/text-to-voice/design
    2. Create a permanent voice from the preview using /v1/text-to-voice

    Args:
        voice_description: Description of the voice to design (e.g., "A patient mother
            helping with studies")
        character_name: Name of the character (used for logging and voice naming)
        api_key: ElevenLabs API key (if None, will use ELEVENLABS_API_KEY env var)
        model_id: Model to use for voice generation

    Returns:
        Tuple of (permanent_voice_id, audio_preview_bytes)

    Raises:
        Exception if API call fails
    """
    if api_key is None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    # Step 1: Design the voice preview
    design_url = "https://api.elevenlabs.io/v1/text-to-voice/design"
    design_payload = {
        "voice_description": voice_description,
        "model_id": model_id,
        "auto_generate_text": True,
    }

    logger.info(
        "Designing ElevenLabs voice for %s: %s", character_name, voice_description
    )

    try:
        # Design voice
        response = requests.post(
            design_url, headers=headers, json=design_payload, timeout=30
        )
        response.raise_for_status()

        data = response.json()
        previews = data.get("previews", [])

        if not previews:
            raise ValueError("No voice previews returned from API")

        # Use the first preview
        first_preview = previews[0]
        generated_voice_id = first_preview["generated_voice_id"]
        audio_base64 = first_preview["audio_base_64"]
        audio_bytes = base64.b64decode(audio_base64)

        logger.info(
            "Successfully designed voice preview for %s: generated_voice_id=%s, duration=%.2fs",
            character_name,
            generated_voice_id,
            first_preview.get("duration_secs", 0),
        )

        # Step 2: Create permanent voice from preview
        create_url = "https://api.elevenlabs.io/v1/text-to-voice"
        create_payload = {
            "voice_name": f"{character_name}_voice",
            "voice_description": voice_description,
            "generated_voice_id": generated_voice_id,
        }

        logger.info("Creating permanent voice for %s from preview...", character_name)

        create_response = requests.post(
            create_url, headers=headers, json=create_payload, timeout=30
        )
        create_response.raise_for_status()

        voice_data = create_response.json()
        permanent_voice_id = voice_data.get("voice_id")

        if not permanent_voice_id:
            raise ValueError("No voice_id returned from voice creation")

        logger.info(
            "Successfully created permanent voice for %s: voice_id=%s",
            character_name,
            permanent_voice_id,
        )

        return permanent_voice_id, audio_bytes

    except requests.exceptions.RequestException as e:
        logger.error("Failed to design/create voice for %s: %s", character_name, str(e))
        raise


def load_voice_mapping(config_dir: Path) -> dict:
    """Load voice mapping configuration."""
    voice_mapping_path = config_dir / "voice_mapping.json"
    if not voice_mapping_path.exists():
        logger.warning("Voice mapping file not found: %s", voice_mapping_path)
        return {}

    with open(voice_mapping_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_character_voice(
    voice_mapping: dict,
    character_name: str,
    tts_model: str,
    conversation_mode: str = None,
) -> dict:
    """
    Get voice configuration for a character based on the voice mapping.

    Args:
        voice_mapping: Loaded voice mapping configuration
        character_name: Name of the character
        tts_model: TTS model being used (e.g., "Kokoro", "ElevenLabs")
        conversation_mode: Mode like "single_user", "multi_user" for generic characters

    Returns:
        Dict with voice_id and tts_prompt, or empty dict if not found
    """
    # First check if it's a specific character
    if character_name in voice_mapping.get("characters", {}):
        character_voices = voice_mapping["characters"][character_name].get("voices", {})
        if tts_model in character_voices:
            return character_voices[tts_model]

    # Check generic voices if conversation_mode is provided
    if conversation_mode and character_name in voice_mapping.get("generic_voices", {}):
        generic_char = voice_mapping["generic_voices"][character_name]
        if (
            conversation_mode in generic_char
            and tts_model in generic_char[conversation_mode]
        ):
            return generic_char[conversation_mode][tts_model]

    logger.warning(
        "Voice not found for character: %s, model: %s, mode: %s",
        character_name,
        tts_model,
        conversation_mode,
    )
    return {}


async def synthesize_speech(
    utterances: list[dict[str, str]],
    characters: dict,
    tts_config: dict,
    elevenlabs_semaphore: asyncio.Semaphore | None = None,
    elevenlabs_timeout: float | None = None,
    sound_effects_config: Optional[dict] = None,
) -> list[tuple[bytes, int]]:
    """
    Generate audio for each utterance using the specified text-to-speech model.

    Utterances whose speaker is the BACKGROUND_NOISE_SPEAKER sentinel are ambient
    segments rather than speech: they are synthesized as sound effects and are not
    routed through the TTS backends.

    Args:
        utterances (list[dict[str, str]]): List of utterance dictionaries containing speaker and
            content.
        characters (dict): Character configuration including names, voice IDs and tts prompts.
        tts_config (dict): Configuration for the TTS model including model type, language, and
            assistant's voice and prompt.
        elevenlabs_semaphore (asyncio.Semaphore | None): Caps in-flight ElevenLabs
            requests so the account's concurrency limit is not exceeded.
        elevenlabs_timeout (float | None): Per-request timeout in seconds for
            ElevenLabs calls. None uses the backend default.
        sound_effects_config (dict, optional): Background sound generation configuration, used
            for ambient segments.

    Returns:
        list[tuple[bytes, int]]: List of tuples containing generated audio bytes and sampling rates
            for each utterance.
    """
    logger.info(
        "Synthesizing speech for %d utterances using %s",
        len(utterances),
        tts_config.get("model", "unknown"),
    )

    # Build name→character lookup to avoid hardcoded user1/user2 assumptions
    char_by_name: dict = {c["name"]: c for c in characters.values() if "name" in c}

    # The scene is assembled at a single rate, so ambient segments must match it
    scene_sampling_rate: int = TTS_MODEL_SAMPLING_RATES.get(
        tts_config.get("model", "Kokoro"), 24000
    )

    # Results are filled by slot, so synchronously produced utterances and awaited
    # tasks both land at their conversation position
    tasks: list[asyncio.Task] = []
    task_slots: list[int] = []
    results: list[Optional[tuple[bytes, int]]] = [None] * len(utterances)
    for i, utterance in enumerate(utterances):
        speaker = utterance["speaker"]
        content = utterance["content"]
        logger.info(
            "Utterance %d/%d: %s says: %s",
            i + 1,
            len(utterances),
            speaker,
            content[:100] + ("..." if len(content) > 100 else ""),
        )
        # Ambient segments carry no speech: synthesize sound effects instead of TTS
        if speaker.lower() == BACKGROUND_NOISE_SPEAKER:
            results[i] = await get_synthesize_background_noise_task(
                description=content,
                sound_effects_config=sound_effects_config,
                default_sampling_rate=scene_sampling_rate,
                duration=utterance.get("duration"),
            )
            continue

        if utterance["speaker"] == tts_config["assistant"]["name"]:
            voice: str = tts_config["assistant"]["voice_id"]
        else:
            char = char_by_name.get(utterance["speaker"], characters.get("user1", {}))
            voice: str = char.get("voice_id", characters["user1"]["voice_id"])

        if TextToSpeechModel(tts_config["model"]) == TextToSpeechModel.OpenAI_Audio:
            if utterance["speaker"] == tts_config["assistant"]["name"]:
                prompt: str = tts_config["assistant"]["tts_prompt"]
            else:
                prompt: str = char.get(
                    "tts_prompt", characters["user1"].get("tts_prompt", "")
                )
            task = openai_audio.generate_audio(
                "gpt-4o-audio-preview",
                utterance["content"],
                voice,
                prompt,
                tts_config["language"],
            )
            tasks.append(task)
            task_slots.append(i)
        elif (
            TextToSpeechModel(tts_config["model"])
            == TextToSpeechModel.OpenAI_Audio_mini
        ):
            if utterance["speaker"] == tts_config["assistant"]["name"]:
                prompt: str = tts_config["assistant"]["tts_prompt"]
            else:
                prompt: str = char.get(
                    "tts_prompt", characters["user1"].get("tts_prompt", "")
                )

            task = openai_audio.generate_audio(
                "gpt-4o-mini-audio-preview",
                utterance["content"],
                voice,
                prompt,
                tts_config["language"],
            )
            tasks.append(task)
            task_slots.append(i)
        elif TextToSpeechModel(tts_config["model"]) == TextToSpeechModel.OpenAI_TTS:
            task = openai_tts.generate_audio(
                utterance["content"],
                voice,
            )
            tasks.append(task)
            task_slots.append(i)
        elif TextToSpeechModel(tts_config["model"]) == TextToSpeechModel.ElevenLabs:

            async def _el_task(content=utterance["content"], v=voice):
                sem = elevenlabs_semaphore
                kwargs = {"max_retries": 4, "base_delay": 2.0}
                if elevenlabs_timeout is not None:
                    kwargs["timeout"] = elevenlabs_timeout
                if sem is not None:
                    async with sem:
                        return await retry_with_backoff(
                            elevenlabs.generate_audio, content, v, **kwargs
                        )
                return await retry_with_backoff(
                    elevenlabs.generate_audio, content, v, **kwargs
                )

            tasks.append(_el_task())
            task_slots.append(i)
        elif TextToSpeechModel(tts_config["model"]) == TextToSpeechModel.Kokoro:
            logger.debug("Generating Kokoro TTS for %s with voice %s", speaker, voice)
            kokoro = Kokoro(language_code=tts_config["language"])
            audio, sampling_rate = kokoro.generate_audio(
                text=utterance["content"], voice=voice, speed=1.0
            )
            audio_duration = (
                len(audio) / sampling_rate if hasattr(audio, "__len__") else 0
            )
            logger.debug(
                "Generated audio: %.2fs duration, %d bytes, %dHz sampling rate",
                audio_duration,
                len(audio) if hasattr(audio, "__len__") else 0,
                sampling_rate,
            )

            # Check for potential silent/empty audio
            if hasattr(audio, "__len__") and len(audio) == 0:
                logger.warning(
                    "WARNING: Empty audio generated for %s: '%s'", speaker, content[:50]
                )
            elif (
                hasattr(audio, "__len__") and len(audio) < sampling_rate * 0.1
            ):  # Less than 0.1 seconds
                logger.warning(
                    "WARNING: Very short audio (%.2fs) for %s: '%s'",
                    audio_duration,
                    speaker,
                    content[:50],
                )

            # Check audio amplitude to detect silent audio
            if hasattr(audio, "__len__") and len(audio) > 0:
                if isinstance(audio, np.ndarray):
                    max_amp = np.max(np.abs(audio))
                elif isinstance(audio, bytes):
                    audio_samples = np.frombuffer(audio, dtype=np.int16)
                    max_amp = np.max(np.abs(audio_samples)) / 32767.0
                else:
                    max_amp = float("unknown")

                logger.debug(
                    "Audio amplitude check for %s: max=%.4f, type=%s",
                    speaker,
                    max_amp,
                    type(audio).__name__,
                )

                if isinstance(max_amp, float) and max_amp < 0.001:  # Very quiet audio
                    logger.warning(
                        "WARNING: Very quiet audio (max=%.6f) for %s: '%s'",
                        max_amp,
                        speaker,
                        content[:50],
                    )

            results[i] = (audio, sampling_rate)
        else:
            raise ValueError(
                "Following model isn't supported. Choose from among: 'OpenAI_Audio', "
                "'OpenAI_TTS', 'ElevenLabs', 'Kokoro'"
            )

    if tasks:
        task_results = await asyncio.gather(*tasks)
        for slot, task_result in zip(task_slots, task_results):
            results[slot] = task_result

    missing = [i + 1 for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"No audio produced for utterances: {missing}")

    # Log final results summary
    logger.info("TTS synthesis completed: %d utterances generated", len(results))
    for i, (audio, sr) in enumerate(results):
        audio_duration = len(audio) / sr if hasattr(audio, "__len__") else 0
        logger.debug(
            "Result %d: %.2fs duration, %d bytes",
            i + 1,
            audio_duration,
            len(audio) if hasattr(audio, "__len__") else 0,
        )

    return results


def resample_pcm_audio(
    audio: bytes, source_sampling_rate: int, target_sampling_rate: int
) -> bytes:
    """
    Resample 16-bit PCM audio to a different sampling rate.

    Args:
        audio (bytes): Raw 16-bit PCM audio bytes.
        source_sampling_rate (int): Sampling rate of the input audio in Hz.
        target_sampling_rate (int): Desired sampling rate in Hz.

    Returns:
        bytes: Resampled audio as raw 16-bit PCM bytes. Returned unchanged when the
            rates already match.
    """
    if source_sampling_rate == target_sampling_rate or len(audio) == 0:
        return audio

    # Backends differ: sound effects hand back raw bytes, Kokoro an int16 array
    samples = (
        audio.astype(np.int16)
        if isinstance(audio, np.ndarray)
        else np.frombuffer(audio, dtype=np.int16)
    )
    if samples.size == 0:
        return audio

    duration = samples.size / source_sampling_rate
    target_size = max(1, int(round(duration * target_sampling_rate)))
    # Linear interpolation keeps the segment duration exact, which is what callers
    # rely on for ambient timing
    source_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    resampled = np.interp(target_positions, source_positions, samples.astype(np.float32))
    return resampled.astype(np.int16).tobytes()


def generate_silent_audio_bytes(
    duration_seconds: float, sampling_rate: int, channels: int = 1
) -> bytes:
    """
    Generate silence as raw 16-bit PCM bytes.

    Args:
        duration_seconds (float): Duration of silence in seconds.
        sampling_rate (int): Sampling rate in Hz.
        channels (int, optional): Number of audio channels. Defaults to 1.

    Returns:
        bytes: Raw audio bytes representing silence.
    """
    samples_per_channel = int(duration_seconds * sampling_rate)
    bytes_per_sample = 2  # 16-bit PCM
    return bytes(samples_per_channel * channels * bytes_per_sample)


async def get_synthesize_background_noise_task(
    description: str,
    sound_effects_config: Optional[dict] = None,
    default_sampling_rate: int = 24000,
    duration: Optional[float] = None,
) -> tuple[bytes, int]:
    """
    Generate a background noise audio segment from a description.

    Args:
        description (str): Text that describes the ambient background noise.
        sound_effects_config (dict, optional): Background sound generation configuration.
        default_sampling_rate (int): Sampling rate of the audio scene. The generated
            segment is resampled to it, since the scene is assembled at a single rate.
        duration (float, optional): Duration of the audio sample. Falls back to the
            duration from sound_effects_config, then to
            DEFAULT_BACKGROUND_NOISE_DURATION.

    Returns:
        tuple[bytes, int]: Audio bytes and sampling rate. The rate is always
            default_sampling_rate.
    """
    config = sound_effects_config or {}
    # A phase describes what should be audible. No description means a plain pause:
    # a gap in the conversation with nothing to hear, so nothing is synthesized.
    prompt = description.strip() if description else ""
    silent_gap = not prompt
    if not prompt:
        prompt = config.get("prompt") or config.get(
            "default_background", "Ambient background noise."
        )
    model = config.get("model", "elevenlabs").strip()
    # A scenario may omit duration entirely, in which case neither the utterance nor
    # the config resolves it and the module default applies
    if duration is None:
        duration = config.get("duration") or config.get("default_duration")
    if duration is None:
        duration = DEFAULT_BACKGROUND_NOISE_DURATION
    duration = float(duration)

    if silent_gap:
        logger.info("Silent gap of %.1fs (no sound requested)", duration)
        return (
            generate_silent_audio_bytes(duration, default_sampling_rate),
            default_sampling_rate,
        )

    logger.info(
        "Synthesizing background noise with model=%s duration=%.1fs prompt=%s",
        model or "none",
        duration,
        prompt[:120],
    )

    audio: Optional[bytes] = None
    sampling_rate: int = default_sampling_rate
    try:
        if model.lower() == "elevenlabs":
            audio, sampling_rate = generate_sound_effect_elevenlabs(prompt, duration)
        elif model.lower() == "stableaudio":
            audio, sampling_rate = generate_sound_effect_stable_audio(prompt, duration)
        else:
            logger.warning("Unknown sound effects model %s, using silence", model)
    except Exception as e:  # noqa: BLE001 - ambience must never fail the whole scene
        logger.error("Background noise generation failed: %s", e)

    if audio is None:
        logger.warning(
            "Falling back to a silent ambient segment for prompt: %s", prompt[:120]
        )
        return (
            generate_silent_audio_bytes(duration, default_sampling_rate),
            default_sampling_rate,
        )

    if sampling_rate != default_sampling_rate:
        logger.info(
            "Resampling background noise from %d Hz to scene rate %d Hz",
            sampling_rate,
            default_sampling_rate,
        )
        audio = resample_pcm_audio(audio, sampling_rate, default_sampling_rate)

    return audio, default_sampling_rate


def fit_background_audio_to_duration(
    background_audio: bytes, sample_rate: int, target_duration: float
) -> bytes:
    """
    Loop or trim background audio so ambience stays present for the whole scene.

    Args:
        background_audio (bytes): Raw mono 16-bit PCM background audio.
        sample_rate (int): Sampling rate in Hz.
        target_duration (float): Desired output duration in seconds.

    Returns:
        bytes: Background audio fitted to the target duration.
    """
    target_samples = max(0, int(round(target_duration * sample_rate)))
    if target_samples == 0 or not background_audio:
        return b""

    background_samples = np.frombuffer(background_audio, dtype=np.int16)
    if len(background_samples) == 0:
        return b""

    if len(background_samples) >= target_samples:
        max_offset = len(background_samples) - target_samples
        start_sample = random.randint(0, max_offset) if max_offset > 0 else 0
        return background_samples[
            start_sample : start_sample + target_samples
        ].tobytes()

    # Rotate the loop so we do not always start from the potentially artificial intro.
    if len(background_samples) > 1:
        start_sample = random.randint(0, len(background_samples) - 1)
        background_samples = np.concatenate(
            (background_samples[start_sample:], background_samples[:start_sample])
        )

    repeat_count = int(np.ceil(target_samples / len(background_samples)))
    return np.tile(background_samples, repeat_count)[:target_samples].tobytes()


async def add_background_effects(
    audio: bytes,
    sampling_rate: int,
    model: Optional[str],
    duration: Optional[float],
    prompt: Optional[str],
    decibels_diff: int = 12,
    debug_background_output_path: Optional[Path] = None,
    padding_range: Tuple[float, float] = (2.0, 5.0),
) -> bytes:
    """
    Add generated background sound effects to an audio track.

    The background is generated to be longer than the speech so that a random
    amount of ambient audio plays before and after the conversation.

    Args:
        audio (bytes): Foreground audio bytes.
        sampling_rate (int): Sampling rate of the audio in Hz.
        model (Optional[str]): Background effect generation model ('ElevenLabs' or 'StableAudio').
        duration (Optional[float]): Desired background effect duration.
        prompt (Optional[str]): Prompt to use when generating the background effect.
        decibels_diff (int): Target loudness gap between foreground and background.
        debug_background_output_path (Optional[Path]): Optional path where the raw generated
            background should be saved for debugging.
        padding_range (Tuple[float, float]): Min/max seconds of ambient audio to add before
            and after the speech (each pad drawn independently). Default: (2.0, 5.0).

    Returns:
        bytes: Audio with background effects mixed in, padded with ambient sound.
    """
    bytes_per_sample: int = 2
    audio_duration: float = len(audio) / (sampling_rate * bytes_per_sample)

    # Draw independent random pads for before and after the conversation
    pad_min, pad_max = padding_range
    pad_before: float = random.uniform(pad_min, pad_max)
    pad_after: float = random.uniform(pad_min, pad_max)
    total_duration: float = audio_duration + pad_before + pad_after
    logger.info(
        "Background padding: %.2fs before + %.2fs speech + %.2fs after = %.2fs total",
        pad_before,
        audio_duration,
        pad_after,
        total_duration,
    )

    audio_background: Optional[bytes] = None
    sampling_rate_background: int = sampling_rate
    if model == "ElevenLabs":
        # ElevenLabs API limits duration to 0.5-22 seconds
        audio_background_duration: float = min(duration, total_duration, 22.0)
        logger.info(
            "Generating ElevenLabs background with prompt: %s",
            prompt or "<empty prompt>",
        )
        logger.info(
            "ElevenLabs background duration request: %.2fs", audio_background_duration
        )
        # Generate ElevenLabs sound effects (synchronous function)
        try:
            (
                audio_background,
                sampling_rate_background,
            ) = generate_sound_effect_elevenlabs(prompt, audio_background_duration)
            if (
                debug_background_output_path is not None
                and audio_background is not None
            ):
                debug_background_output_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(
                    audio_background,
                    sampling_rate_background,
                    str(debug_background_output_path),
                )
                logger.info(
                    "Saved raw ElevenLabs background debug audio to: %s",
                    debug_background_output_path,
                )
        except Exception as e:
            logger.error(
                "Failed to generate background sound effects with ElevenLabs: %s",
                str(e),
            )
            audio_background, sampling_rate_background = None, sampling_rate
    elif model == "StableAudio":
        audio_background_duration: float = min(duration, total_duration)
        audio_background, sampling_rate_background = generate_sound_effect_stable_audio(
            prompt, audio_background_duration
        )
    elif model is not None:
        raise ValueError(
            "Following model isn't supported. Choose from among: 'ElevenLabs', 'StableAudio'"
        )

    if audio_background is not None:
        # Pad the foreground with silence so overlay produces background-only audio
        # at the start and end of the scene
        silence_before = bytes(int(pad_before * sampling_rate) * bytes_per_sample)
        silence_after = bytes(int(pad_after * sampling_rate) * bytes_per_sample)
        padded_audio = silence_before + audio + silence_after

        audio_background = fit_background_audio_to_duration(
            background_audio=audio_background,
            sample_rate=sampling_rate_background,
            target_duration=total_duration,
        )
        audio_augmented: bytes = add_background_audio(
            foreground_audio=padded_audio,
            background_audio=audio_background,
            foreground_sampling_rate=sampling_rate,
            background_sampling_rate=sampling_rate_background,
            decibels_diff=decibels_diff,
        )
    else:
        audio_augmented: bytes = audio[:]

    return audio_augmented


# Overlap duration ranges (seconds) for config-authored interjection turns —
# see Message.interjection / scenario_template speaker tagging in
# conversation_manager.py. Disagreement gets a beat before the assertive
# entry; "add" is eager so it overlaps the most.
_INTERJECTION_OVERLAP_RANGES = {
    "agree": (0.1, 0.4),
    "disagree": (0.2, 0.6),
    "add": (0.3, 0.7),
}


def sample_inter_utterance_gaps(
    utterances: list[dict],
    overlap_config: Optional[dict] = None,
) -> list[float]:
    """Sample inter-utterance gap durations in seconds, one per consecutive pair.

    A negative value means the next utterance starts before the previous one
    ends — i.e. overlapping speech, as is common in real multi-speaker
    meetings (people talking over each other, backchannels like "yeah"/"right"
    landing mid-sentence).

    A turn tagged with ``"interjection"`` (set deterministically by the scenario
    config via a speaker slot like ``{"speaker": "user5", "interjection": "agree"}``,
    see ``conversation_manager.py``) always overlaps the previous turn, with a
    duration drawn from ``_INTERJECTION_OVERLAP_RANGES`` for its type,
    regardless of the ambient overlap settings below. Untagged turns fall back
    to ambient overlap, which is off by default; scenarios opt in via
    ``conversation_dynamics.overlapping_speech`` in their config.

    Args:
        utterances (list[dict]): Conversation turns in order. Each turn's word
            count feeds the short-utterance overlap bias, and an optional
            ``"interjection"`` field ("agree"/"disagree"/"add") forces overlap
            onto the previous turn.
        overlap_config (Optional[dict]): The scenario's ``conversation_dynamics``
            block. Recognized keys: ``overlapping_speech`` (bool), ``overlap_probability``,
            ``short_utterance_overlap_multiplier``, ``short_utterance_word_limit``,
            ``overlap_duration_range``, ``gap_duration_range``.

    Returns:
        list[float]: One gap duration per consecutive utterance pair, in seconds.
            Positive values are silence; negative values are overlap.
    """
    n_gaps = max(0, len(utterances) - 1)
    overlap_config = overlap_config or {}
    ambient_overlap_enabled = overlap_config.get("overlapping_speech", False)

    overlap_probability = overlap_config.get("overlap_probability", 0.2)
    short_utterance_overlap_multiplier = overlap_config.get(
        "short_utterance_overlap_multiplier", 2.0
    )
    short_utterance_word_limit = overlap_config.get("short_utterance_word_limit", 4)
    overlap_min, overlap_max = overlap_config.get("overlap_duration_range", [0.15, 0.8])
    gap_min, gap_max = overlap_config.get("gap_duration_range", [0.3, 1.2])

    gaps = []
    for i in range(n_gaps):
        next_utterance = utterances[i + 1]

        interjection_type = next_utterance.get("interjection")
        interjection_range = _INTERJECTION_OVERLAP_RANGES.get(interjection_type)
        if interjection_range is not None:
            gaps.append(-random.uniform(*interjection_range))
            continue

        if not ambient_overlap_enabled:
            gaps.append(random.uniform(0.3, 1.2))
            continue

        next_word_count = len(str(next_utterance.get("content", "")).split())
        probability = overlap_probability
        if next_word_count <= short_utterance_word_limit:
            probability *= short_utterance_overlap_multiplier
        if random.random() < probability:
            gaps.append(-random.uniform(overlap_min, overlap_max))
        else:
            gaps.append(random.uniform(gap_min, gap_max))
    return gaps


def load_cached_utterances(
    utterances_path: Path,
    expected_count: int,
) -> Optional[Tuple[list[bytes], int]]:
    """Load previously synthesised per-utterance audio from disk.

    Used when only the background configuration changed: the speech itself is
    still valid, so it is read back instead of paying for TTS again.

    Args:
        utterances_path: Directory holding the numbered per-utterance WAVs.
        expected_count: Number of utterances the conversation should have.

    Returns:
        Tuple of (list of raw PCM16 byte strings, sampling rate), or None if the
        cache is missing, incomplete, or unreadable — in which case the caller
        must fall back to full synthesis.
    """
    if not utterances_path.exists():
        logger.warning("No cached utterances directory: %s", utterances_path)
        return None

    wav_files = sorted(utterances_path.glob("*.wav"))
    if len(wav_files) != expected_count:
        logger.warning(
            "Cached utterance count mismatch in %s: found %d, expected %d",
            utterances_path,
            len(wav_files),
            expected_count,
        )
        return None

    audio_list: list[bytes] = []
    sampling_rate: Optional[int] = None
    for wav_path in wav_files:
        try:
            with wave.open(str(wav_path), "rb") as wf:
                if sampling_rate is None:
                    sampling_rate = wf.getframerate()
                elif wf.getframerate() != sampling_rate:
                    logger.warning(
                        "Inconsistent sampling rate in cached utterances (%s)", wav_path
                    )
                    return None
                audio_list.append(wf.readframes(wf.getnframes()))
        except Exception as e:  # noqa: BLE001 - any unreadable file forces a full run
            logger.warning("Could not read cached utterance %s: %s", wav_path, e)
            return None

    logger.info(
        "Reusing %d cached utterances from %s (%d Hz)",
        len(audio_list),
        utterances_path,
        sampling_rate,
    )
    return audio_list, sampling_rate


def mix_utterances_sequential(
    utterance_audio_list: list[bytes],
    inter_utterance_gap_samples: list[int],
) -> bytes:
    """Combine utterances (PCM16 mono) into one track, honoring per-boundary gaps.

    Positive gaps insert silence; negative gaps make the next utterance start
    before the previous one finishes, and the overlapping region is mixed by
    addition (clipped to the valid PCM16 range) rather than concatenated.

    Args:
        utterance_audio_list (list[bytes]): Per-utterance PCM16 mono audio, in order.
        inter_utterance_gap_samples (list[int]): One gap per consecutive pair, in
            samples. Positive values insert silence; negative values overlap the
            next utterance onto the previous one's tail.

    Returns:
        bytes: The combined PCM16 mono track.
    """
    signals = [np.frombuffer(a, dtype=np.int16).astype(np.float32) for a in utterance_audio_list]

    total_samples = sum(len(s) for s in signals) + sum(inter_utterance_gap_samples or [])
    output = np.zeros(max(total_samples, 0), dtype=np.float32)

    current_sample = 0
    for i, signal in enumerate(signals):
        end_sample = current_sample + len(signal)
        if end_sample > len(output):
            extended = np.zeros(end_sample, dtype=np.float32)
            extended[: len(output)] = output
            output = extended
        output[current_sample:end_sample] += signal
        current_sample = end_sample
        if inter_utterance_gap_samples and i < len(inter_utterance_gap_samples):
            current_sample = max(0, current_sample + inter_utterance_gap_samples[i])

    output = np.clip(output, -32768, 32767)
    return output.astype(np.int16).tobytes()


def create_room(room_dim: list[float], sample_rate: int) -> pra.Room:
    """Create a pyroomacoustics room with minimal reverberation for natural sound"""
    corners = np.array(
        [
            [0, 0],  # bottom-left
            [0, room_dim[1]],  # top-left
            [room_dim[0], room_dim[1]],  # top-right
            [room_dim[0], 0],  # bottom-right
        ]
    ).T

    # Use high absorption (0.8) and low reflection order (1) for minimal reverberation
    room = pra.Room.from_corners(
        corners,
        fs=sample_rate,
        materials=pra.Material(energy_absorption=0.8),  # Much higher absorption
        max_order=1,  # Lower reflection order
        ray_tracing=False,
        air_absorption=True,  # Enable air absorption for further damping
    )

    # Add the microphone array
    mic_loc = np.array([[room_dim[0] / 2], [room_dim[1] / 2]])  # Center of the room
    room.add_microphone(mic_loc)

    return room


def create_spatial_audio_scene(
    utterance_audio_list: list[bytes],
    utterance_speakers: list[str],
    background_audio: Optional[bytes],
    sampling_rate: int,
    room_dim: list[float] = None,
    characters: dict = None,
    inter_utterance_gap_samples: list[int] = None,
) -> bytes:
    """
    Create a spatial audio scene using pyroomacoustics.

    Args:
        utterance_audio_list: List of audio bytes for each utterance
        utterance_speakers: List of speaker names for each utterance
        background_audio: Background audio bytes (optional)
        sampling_rate: Audio sampling rate
        room_dim: Room dimensions [width, height] in meters
        characters: Character configuration for speaker positioning

    Returns:
        Combined spatial audio as bytes
    """
    logger.info(
        "Creating spatial audio scene with %d utterances", len(utterance_audio_list)
    )
    logger.info("Room dimensions: %.1f x %.1f meters", room_dim[0], room_dim[1])

    if room_dim is None:
        room_dim = [5.0, 4.0]

    # Log detailed info about each utterance
    for i, (audio_bytes, speaker) in enumerate(
        zip(utterance_audio_list, utterance_speakers)
    ):
        audio_duration = len(audio_bytes) / (sampling_rate * 2)  # 2 bytes per sample
        logger.info(
            "Utterance %d: Speaker=%s, Duration=%.2fs, Size=%d bytes",
            i + 1,
            speaker,
            audio_duration,
            len(audio_bytes),
        )
        if len(audio_bytes) == 0:
            logger.warning(
                "WARNING: Utterance %d (%s) has zero length!", i + 1, speaker
            )

    # Create room
    create_room(room_dim, sampling_rate)

    # Calculate total duration from all utterances (plus inter-utterance gaps)
    total_samples = sum(
        len(audio) // 2 for audio in utterance_audio_list
    )  # 2 bytes per sample
    if inter_utterance_gap_samples:
        total_samples += sum(inter_utterance_gap_samples)
    total_duration = total_samples / sampling_rate
    logger.info(
        "Total conversation duration: %.2fs (%d samples)", total_duration, total_samples
    )

    # Initialize output array
    output_signal = np.zeros(total_samples, dtype=np.float32)

    # Process each utterance with spatial positioning
    current_sample = 0
    for i, (audio_bytes, speaker) in enumerate(
        zip(utterance_audio_list, utterance_speakers)
    ):
        logger.debug(
            "Processing utterance %d/%d: %s", i + 1, len(utterance_audio_list), speaker
        )

        # Skip empty audio
        if len(audio_bytes) == 0:
            logger.warning("Skipping empty audio for utterance %d (%s)", i + 1, speaker)
            continue

        # Convert bytes to numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0

        # Log audio statistics
        max_amplitude = np.max(np.abs(audio_float))
        logger.debug(
            "Audio stats - Length: %d samples, Max amplitude: %.4f",
            len(audio_float),
            max_amplitude,
        )

        # Debug character configuration
        logger.debug("Character config for positioning:")
        if characters:
            for key, char_info in characters.items():
                logger.debug("  %s: name='%s'", key, char_info.get("name", "unknown"))
        else:
            logger.debug("  No characters config provided")

        # Ambient segments are already mixed for the room: adding them through the
        # simulation would impose a speaker position on non-directional sound
        if speaker.lower() == BACKGROUND_NOISE_SPEAKER:
            logger.debug(
                "Background noise segment at sample %d, inserted without spatial simulation",
                current_sample,
            )
            if len(audio_float) == 0:
                logger.warning(
                    "Skipping zero-length background noise audio for utterance %d", i + 1
                )
                continue
            end_sample = current_sample + len(audio_float)
            if end_sample > len(output_signal):
                logger.debug(
                    "Extending output signal from %d to %d samples for background noise",
                    len(output_signal),
                    end_sample,
                )
                extended = np.zeros(end_sample, dtype=np.float32)
                extended[: len(output_signal)] = output_signal
                output_signal = extended
            output_signal[current_sample:end_sample] += audio_float
            current_sample = end_sample
            continue

        # Determine speaker position based on character configuration
        user_keys = sorted(k for k in (characters or {}) if k.startswith("user"))
        user_idx = next((i for i, k in enumerate(user_keys) if characters[k]["name"] == speaker), None)

        if user_idx is not None:
            n = len(user_keys)
            x = 1.0 + user_idx * (room_dim[0] - 2.0) / max(n - 1, 1)
            position = [x, room_dim[1] / 2]
            position_desc = user_keys[user_idx]
            logger.debug("Matched %s as %s", speaker, user_keys[user_idx])
        else:
            # Assistant or unknown speaker - slightly off-center to avoid mathematical issues
            position = [
                room_dim[0] / 2 + 0.1,
                room_dim[1] / 2 + 0.1,
            ]  # Slightly off-center
            position_desc = "center-assistant"
            logger.debug(
                "Matched %s as assistant/unknown (fallback to center)", speaker
            )

        logger.debug(
            "Positioning %s at [%.1f, %.1f] (%s)",
            speaker,
            position[0],
            position[1],
            position_desc,
        )

        # Create a room for this utterance
        utterance_room = create_room(room_dim, sampling_rate)
        utterance_room.add_source(position, signal=audio_float)

        # Compute RIR and simulate
        try:
            utterance_room.compute_rir()
            utterance_room.simulate()

            # Add to output at the correct time position
            utterance_signal = utterance_room.mic_array.signals[0]

            # Validate the simulated signal
            signal_max = np.max(np.abs(utterance_signal))
            signal_length = len(utterance_signal)
            logger.debug(
                "Room simulation result for %s: length=%d, max_amplitude=%.4f",
                speaker,
                signal_length,
                signal_max,
            )

            # Normalize individual utterances to prevent one speaker from being much louder
            if signal_max > 0:
                # Target amplitude around 0.5 for consistent volume across speakers
                target_amplitude = 0.5
                utterance_signal = utterance_signal * (target_amplitude / signal_max)
                logger.debug(
                    "Normalized %s from %.4f to %.4f",
                    speaker,
                    signal_max,
                    target_amplitude,
                )

            if signal_max < 0.001:
                logger.warning(
                    "WARNING: Room simulation produced very quiet signal for %s (max=%.6f)",
                    speaker,
                    signal_max,
                )

            if signal_length == 0:
                logger.error(
                    "ERROR: Room simulation produced empty signal for %s", speaker
                )
                continue  # Skip this utterance

        except Exception as e:
            logger.error("ERROR: Room simulation failed for %s: %s", speaker, str(e))
            # Fallback: use original audio without room simulation
            utterance_signal = audio_float
            logger.warning(
                "Using fallback audio without room simulation for %s", speaker
            )

        end_sample = current_sample + len(utterance_signal)

        logger.debug(
            "Adding utterance %d: samples %d-%d (duration: %.2fs)",
            i + 1,
            current_sample,
            end_sample,
            len(utterance_signal) / sampling_rate,
        )

        if end_sample > len(output_signal):
            # Extend output if needed
            logger.debug(
                "Extending output signal from %d to %d samples",
                len(output_signal),
                end_sample,
            )
            new_output = np.zeros(end_sample, dtype=np.float32)
            new_output[: len(output_signal)] = output_signal
            output_signal = new_output

        output_signal[current_sample:end_sample] += utterance_signal
        current_sample = end_sample

        # Advance past the inter-utterance gap; a negative gap overlaps the
        # next utterance onto the tail of this one (clamped so it never
        # rewinds past the start of the signal).
        if inter_utterance_gap_samples and i < len(inter_utterance_gap_samples):
            current_sample = max(0, current_sample + inter_utterance_gap_samples[i])

    # Add background sound if provided
    if background_audio is not None:
        logger.debug("Adding background audio")
        bg_int16 = np.frombuffer(background_audio, dtype=np.int16)
        bg_float = bg_int16.astype(np.float32) / 32767.0

        bg_max_amplitude = np.max(np.abs(bg_float))
        logger.debug(
            "Background audio: %d samples, max amplitude: %.4f",
            len(bg_float),
            bg_max_amplitude,
        )

        # Position background at center-back
        bg_position = [room_dim[0] / 2, room_dim[1] - 1.0]
        logger.debug(
            "Positioning background at [%.1f, %.1f]", bg_position[0], bg_position[1]
        )

        # Create room for background
        bg_room = create_room(room_dim, sampling_rate)
        bg_room.add_source(bg_position, signal=bg_float)
        bg_room.compute_rir()
        bg_room.simulate()

        # Mix background with conversation
        bg_signal = bg_room.mic_array.signals[0]
        min_length = min(len(output_signal), len(bg_signal))
        logger.debug(
            "Mixing background: conversation=%d samples, background=%d samples, mixing=%d samples",
            len(output_signal),
            len(bg_signal),
            min_length,
        )

        # Check output signal before background mixing
        conv_max_before = np.max(np.abs(output_signal))
        logger.debug(
            "Conversation signal max amplitude before background: %.4f", conv_max_before
        )

        output_signal[:min_length] += (
            bg_signal[:min_length] * 0.6
        )  # 60% background volume

        # Check output signal after background mixing
        conv_max_after = np.max(np.abs(output_signal))
        logger.debug(
            "Combined signal max amplitude after background: %.4f", conv_max_after
        )

    # Normalize output
    max_amplitude = np.max(np.abs(output_signal))
    logger.debug("Final signal max amplitude before normalization: %.4f", max_amplitude)

    if max_amplitude > 0:
        output_signal = output_signal / max_amplitude * 0.9  # Leave headroom
        logger.debug("Normalized to max amplitude: 0.9")
    else:
        logger.warning("WARNING: Output signal has zero amplitude!")

    # Check for silence in different parts of the signal
    signal_quarters = len(output_signal) // 4
    for i in range(4):
        start_idx = i * signal_quarters
        end_idx = (i + 1) * signal_quarters if i < 3 else len(output_signal)
        quarter_max = np.max(np.abs(output_signal[start_idx:end_idx]))
        logger.debug("Signal quarter %d amplitude: %.4f", i + 1, quarter_max)

    # Convert back to bytes
    output_int16 = (output_signal * 32767).astype(np.int16)
    logger.debug(
        "Final output: %d samples (%.2fs)",
        len(output_int16),
        len(output_int16) / sampling_rate,
    )
    return output_int16.tobytes()


def load_background_sound(
    background_sounds_dir: Path,
    audio_duration: float,
    sampling_rate: int,
    scenario_name: str = None,
) -> Optional[bytes]:
    """
    Load a random background sound file from the background_sounds directory,
    filtered by scenario type if possible.

    Args:
        background_sounds_dir: Directory containing background sound files
        audio_duration: Duration of the main audio in seconds
        sampling_rate: Target sampling rate
        scenario_name: Name of the scenario to match appropriate background sounds

    Returns:
        Background audio bytes or None if no suitable file found
    """
    if not background_sounds_dir or not background_sounds_dir.exists():
        return None

    # Support either the prepared freesound layout or a plain directory of wav files.
    processed_dir = background_sounds_dir / "freesound_db" / "processed"
    direct_wav_files = list(background_sounds_dir.glob("*.wav"))
    if processed_dir.exists():
        source_dir = processed_dir
        using_processed_layout = True
    elif direct_wav_files:
        source_dir = background_sounds_dir
        using_processed_layout = False
    else:
        logger.warning(
            "No background WAV files found in %s or %s",
            processed_dir,
            background_sounds_dir,
        )
        return None

    # Load background sound metadata if available
    metadata_file = background_sounds_dir / "freesound_db" / "smart_cooking.json"
    available_fragments = []

    if using_processed_layout and metadata_file.exists() and scenario_name:
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Map scenario types to background sound categories
            scenario_mappings = {
                # Kitchen/Cooking related scenarios
                "HomeTasks": ["cooking", "kitchen", "preparing", "frying", "soup"],
                "MealPlanning": ["cooking", "kitchen", "preparing", "frying", "soup"],
                "ContentRecommendation": ["general"],  # More general ambient sounds
                "KnowledgeQueries": ["general"],
                "EducationLearning": ["general"],
                "WorkMeetings": ["general"],
                "TimePlanning": ["general"],
                # Add more mappings as needed
            }

            keywords = scenario_mappings.get(scenario_name, ["general"])

            # Find audio fragments that match the scenario keywords
            for audio_info in metadata.get("audio_files", []):
                title = audio_info.get("title", "").lower()
                description = audio_info.get("description", "").lower()

                # Check if any keywords match the title or description
                if any(
                    keyword in title or keyword in description for keyword in keywords
                ):
                    available_fragments.extend(audio_info.get("audio_fragments", []))

            if available_fragments:
                logger.info(
                    "Found %d matching background sounds for scenario %s",
                    len(available_fragments),
                    scenario_name,
                )
            else:
                logger.info(
                    "No specific background sounds found for scenario %s, using all available",
                    scenario_name,
                )

        except Exception as e:
            logger.warning("Error loading background sound metadata: %s", str(e))

    # If no specific fragments found, use all available files
    if not available_fragments:
        audio_files = list(source_dir.glob("*.wav"))
    else:
        # Filter to only use the matched fragments
        audio_files = []
        for fragment in available_fragments:
            fragment_path = source_dir / fragment
            if fragment_path.exists():
                audio_files.append(fragment_path)

    if not audio_files:
        logger.warning("No WAV files found in %s", processed_dir)
        return None

    # Select a random background audio file
    selected_file = random.choice(audio_files)
    logger.info("Selected background sound: %s", selected_file.name)

    try:
        # Load the background audio
        bg_audio, bg_sr = sf.read(selected_file)

        # Convert to mono if stereo
        if len(bg_audio.shape) > 1:
            bg_audio = np.mean(bg_audio, axis=1)

        # Resample if necessary
        if bg_sr != sampling_rate:
            import librosa

            bg_audio = librosa.resample(
                bg_audio, orig_sr=bg_sr, target_sr=sampling_rate
            )

        # Convert to 16-bit PCM bytes
        bg_audio_int16 = (bg_audio * 32767).astype(np.int16)
        bg_audio_bytes = bg_audio_int16.tobytes()

        # Adjust background audio duration to match or be longer than main audio
        bg_duration = len(bg_audio_bytes) / (
            sampling_rate * 2
        )  # 2 bytes per sample for int16

        if bg_duration < audio_duration:
            # Loop the background audio to match the required duration
            repeat_count = int(np.ceil(audio_duration / bg_duration))
            bg_audio_repeated = np.tile(bg_audio, repeat_count)
            # Trim to exact duration needed
            samples_needed = int(audio_duration * sampling_rate)
            bg_audio_trimmed = bg_audio_repeated[:samples_needed]
            bg_audio_int16 = (bg_audio_trimmed * 32767).astype(np.int16)
            bg_audio_bytes = bg_audio_int16.tobytes()
        elif bg_duration > audio_duration:
            # Trim the background audio to match the main audio duration
            samples_needed = int(audio_duration * sampling_rate)
            bg_audio_trimmed = bg_audio[:samples_needed]
            bg_audio_int16 = (bg_audio_trimmed * 32767).astype(np.int16)
            bg_audio_bytes = bg_audio_int16.tobytes()

        return bg_audio_bytes

    except Exception as e:
        logger.error("Error loading background sound %s: %s", selected_file, str(e))
        return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for configuration, logging, and input/output paths.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Synthesize conversations from text corpus into audio files.",
        epilog=(
            "Example usage:\n"
            "  python -m dataset.generate_speech.generate_audio \\\n"
            "    --config-dir dataset/config \\\n"
            "    --input-dir dataset/data/generated_text \\\n"
            "    --output-dir dataset/data/generated_audio\n\n"
            "  # Run a specific scenario without confirmation prompts:\n"
            "  python -m dataset.generate_speech.generate_audio \\\n"
            "    --config-dir dataset/config \\\n"
            "    --input-dir dataset/data/generated_text \\\n"
            "    --output-dir dataset/data/generated_audio \\\n"
            "    --scenario my_scenario --one-by-one"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--scenario", default="all", help='Specific scenario to generate or "all"'
    )
    parser.add_argument(
        "--tts-model",
        choices=["Kokoro", "ElevenLabs", "OpenAI_TTS", "OpenAI_Audio"],
        help="Override TTS model for audio generation (overrides scenario config)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with generated conversations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--background-sounds-dir",
        type=Path,
        help="Directory containing background sound files (overrides generated background effects)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed audio processing information",
    )
    parser.add_argument(
        "--design-el-voice",
        action="store_true",
        help="Use ElevenLabs Voice Design API to dynamically create voices",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate all dialogs even if inputs have not changed",
    )
    parser.add_argument(
        "--one-by-one",
        action="store_true",
        help="After each dialog is generated, pause and ask whether to continue",
    )
    parser.add_argument(
        "--elevenlabs-concurrency",
        type=int,
        default=12,
        help=(
            "Max simultaneous ElevenLabs TTS requests. Their Text-to-Speech "
            "concurrency limit is per-plan — Free 2, Starter 3, Creator 5, Pro 10, "
            "Scale/Business 15 — and exceeding it returns 429 system_busy. "
            "Set this at or just below your plan's limit. Default 12 suits Scale "
            "or Business; lower it to 8 for Pro, 4 for Creator, 2 for Starter"
        ),
    )
    parser.add_argument(
        "--elevenlabs-timeout",
        type=float,
        default=180.0,
        help=(
            "Per-request timeout in seconds for ElevenLabs TTS. eleven_v3 is a "
            "large model with high generation latency, so this must be generous; "
            "Flash models can use a much smaller value (default: 180)"
        ),
    )
    parsed_args = parser.parse_args()
    return parsed_args


async def main(args: argparse.Namespace) -> None:
    """
    Main execution pipeline to synthesize audio for conversations:
    loads configurations, reads input files, generates utterance and dialog audio,
    and applies optional background effects.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Use debug level if --debug flag is set
    log_level = "debug" if args.debug else args.log_level
    LoggerConfigurator.configure_logger(log_level)

    # Limit concurrent ElevenLabs requests. Text-to-Speech concurrency is far
    # lower than the Agents limit: Free 2, Starter 3, Creator 5, Pro 10,
    # Scale/Business 15. Exceeding it returns 429 system_busy.
    logger.info(
        "ElevenLabs concurrency=%d, per-request timeout=%.0fs",
        args.elevenlabs_concurrency,
        args.elevenlabs_timeout,
    )
    elevenlabs_semaphore = asyncio.Semaphore(args.elevenlabs_concurrency)

    # Suppress HuggingFace, urllib3, and numba debug logging
    import logging

    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
    logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)
    utterances_base_path: Path = args.output_dir / "utterances"

    scenarios_dir = args.config_dir / "scenarios"
    scenarios = {}

    # Load scenarios from both direct .json files and nested directories
    if args.scenario == "all":
        # Load direct .json files
        for json_file in scenarios_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as file:
                scenario: dict = json.load(file)
                scenarios[scenario["name"]] = scenario

        # Load nested scenarios (like content_recommendation/couple/*.json)
        for nested_json in scenarios_dir.rglob("*/*.json"):
            try:
                logger.info("Loading nested scenario config: %s", nested_json)
                with open(nested_json, "r", encoding="utf-8") as file:
                    scenario: dict = json.load(file)
                    variant_name = nested_json.stem
                    scenarios[f"{scenario['name']}_{variant_name}"] = scenario
                    # Also store under plain name as fallback (last write wins, kept for
                    # backwards-compat with code paths that don't use variant_name)
                    scenarios[scenario["name"]] = scenario
            except json.JSONDecodeError as e:
                logger.error("JSON decode error in file %s: %s", nested_json, str(e))
                raise
    else:
        # First try direct .json file
        config_path = scenarios_dir / f"{args.scenario}.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                scenario: dict = json.load(file)
                scenarios[scenario["name"]] = scenario
        else:
            # Try to find all matching variants in nested directories
            found = False
            for nested_json in scenarios_dir.rglob("*.json"):
                with open(nested_json, "r", encoding="utf-8") as file:
                    scenario: dict = json.load(file)
                    if scenario["name"] == args.scenario:
                        found = True
                        variant_name = nested_json.stem
                        scenarios[f"{scenario['name']}_{variant_name}"] = scenario
                        # Keep a plain-name fallback for inputs that do not carry variant_name.
                        scenarios.setdefault(scenario["name"], scenario)

            if not found:
                logger.warning("Scenario config not found: %s", args.scenario)

    logger.info("Loaded scenarios: %s", list(scenarios.keys()))

    # Load voice mapping configuration
    voice_mapping = load_voice_mapping(args.config_dir)

    # Initialize version manager
    version_manager = SpeechVersionManager(args.output_dir)
    logger.info("Speech generation version: %s", get_version())
    if args.force_regenerate:
        logger.info("Force regeneration enabled - skipping version checks")

    for json_file in Path(args.input_dir).rglob("*.json"):
        if (
            json_file.name.startswith("corpora_version")
            or ".history" in json_file.parts
        ):
            logger.info("Skipping version file: %s", json_file)
            continue

        # Director-mode artefacts (--save-takes): intermediate takes and review
        # notes live beside the final conversation. Only the final file should
        # be synthesised — the takes would produce duplicate audio and the notes
        # are not conversations at all.
        stem = json_file.stem
        if "-take" in stem or "-director_notes" in stem:
            logger.info("Skipping director artefact: %s", json_file)
            continue
        logger.info("Processing file: %s", json_file)
        logger.info("Dir: %s", os.path.dirname(json_file))

        with open(json_file, "r", encoding="utf-8") as file:
            conversation: dict = json.load(file)

        # Not every JSON in the tree is a conversation — skip anything without
        # the fields synthesis needs rather than failing the whole run on it.
        if not isinstance(conversation, dict) or not conversation.get("conversation"):
            logger.info("Skipping non-conversation JSON: %s", json_file)
            continue
        if "scenario_type" not in conversation:
            logger.warning(
                "Skipping %s: no scenario_type field", json_file
            )
            continue

        # Extract the correct mode from the file path structure
        # Path: .../ContentRecommendation/couple/evening_movie.json
        # We want "couple" which is at position -2
        mode: str = str(json_file).split("/")[-2]

        # Map the mode to the character configuration key
        mode_mapping = {"couple": "multi_user", "single": "single_user"}
        character_mode = mode_mapping.get(mode, mode)

        logger.info("Mode extracted: %s -> %s", mode, character_mode)
        logger.info("Looking for scenario: %s", conversation["scenario_type"])

        variant_key = (
            f"{conversation['scenario_type']}_{conversation.get('variant_name', '')}"
        )
        scenario_config = (
            scenarios.get(variant_key) or scenarios[conversation["scenario_type"]]
        )
        logger.info("Resolved scenario config key: %s", variant_key)

        characters: dict = scenario_config["default_characters"][
            character_mode
        ].copy()  # Make a copy to avoid modifying original
        tts_config: dict = scenario_config["tts_config"].copy()

        # Override TTS model if specified via command line
        if args.tts_model:
            logger.info(
                "Overriding TTS model: %s -> %s",
                tts_config.get("model"),
                args.tts_model,
            )
            tts_config["model"] = args.tts_model

        # Compute dialog key and expected output path for version checks
        dialog_key = (
            f"{conversation.get('scenario_type', 'unknown')}_"
            f"{conversation.get('conversation_type', 'unknown')}_"
            f"{conversation.get('variant_name', 'unknown')}"
        )
        output_wav = args.output_dir / json_file.relative_to(
            args.input_dir
        ).with_suffix(".wav")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        utterances_path = utterances_base_path / json_file.relative_to(
            args.input_dir
        ).with_suffix("")
        utterances_path.mkdir(parents=True, exist_ok=True)

        # Override character voices using voice mapping or design API
        tts_model = tts_config.get("model", "")

        # If --design-el-voice flag is set and using ElevenLabs, design voices dynamically
        if args.design_el_voice and tts_model == "ElevenLabs":
            logger.info("Using ElevenLabs Voice Design API for character voices")

            # Design voices for user characters
            for _, char_data in characters.items():
                character_name = char_data.get("name", "")
                tts_prompt = char_data.get("tts_prompt", "")

                if tts_prompt:
                    try:
                        voice_id, _ = design_elevenlabs_voice(
                            voice_description=tts_prompt,
                            character_name=character_name,
                        )
                        logger.info(
                            "Designed voice for %s: %s (prompt: %s)",
                            character_name,
                            voice_id,
                            tts_prompt,
                        )
                        char_data["voice_id"] = voice_id
                    except Exception as e:
                        logger.error(
                            "Failed to design voice for %s, will use mapping: %s",
                            character_name,
                            str(e),
                        )

            # Design voice for assistant if present
            if "assistant" in tts_config and "name" in tts_config["assistant"]:
                assistant_name = tts_config["assistant"]["name"]
                assistant_prompt = tts_config["assistant"].get("tts_prompt", "")

                if assistant_prompt:
                    try:
                        voice_id, _ = design_elevenlabs_voice(
                            voice_description=assistant_prompt,
                            character_name=assistant_name,
                        )
                        logger.info(
                            "Designed assistant voice for %s: %s (prompt: %s)",
                            assistant_name,
                            voice_id,
                            assistant_prompt,
                        )
                        tts_config["assistant"]["voice_id"] = voice_id
                    except Exception as e:
                        logger.error(
                            "Failed to design assistant voice for %s, will use mapping: %s",
                            assistant_name,
                            str(e),
                        )

        # Otherwise use voice mapping if available
        elif voice_mapping:
            # Override user character voices
            for char_key, char_data in characters.items():
                character_name = char_data.get("name", "")
                voice_override = get_character_voice(
                    voice_mapping, character_name, tts_model, character_mode
                )
                # Fall back to generic voice by position (user1 → User1) when
                # the specific character name is not in the mapping
                if not voice_override or "voice_id" not in voice_override:
                    generic_key = char_key.replace("user", "User")  # "user1" → "User1"
                    voice_override = get_character_voice(
                        voice_mapping, generic_key, tts_model, character_mode
                    )
                if voice_override and "voice_id" in voice_override:
                    logger.info(
                        "Overriding voice for %s: %s -> %s",
                        character_name,
                        char_data.get("voice_id"),
                        voice_override.get("voice_id"),
                    )
                    # Only override voice_id, keep scenario-specific tts_prompt
                    char_data["voice_id"] = voice_override["voice_id"]

            # Override assistant voice if present
            if "assistant" in tts_config and "name" in tts_config["assistant"]:
                assistant_name = tts_config["assistant"]["name"]
                assistant_voice_override = get_character_voice(
                    voice_mapping, assistant_name, tts_model, character_mode
                )
                if assistant_voice_override and "voice_id" in assistant_voice_override:
                    logger.info(
                        "Overriding assistant voice for %s: %s -> %s",
                        assistant_name,
                        tts_config["assistant"].get("voice_id"),
                        assistant_voice_override.get("voice_id"),
                    )
                    tts_config["assistant"]["voice_id"] = assistant_voice_override[
                        "voice_id"
                    ]
        sound_effects_config: dict = scenario_config["sound_effects_config"]

        # Skip generation if inputs have not changed since last run.
        # Check here — after all tts_config mutations (voice mapping / voice design) —
        # so the hash is computed from the same tts_config state as update_dialog_version.
        generation_mode = "full"
        if not args.force_regenerate:
            needs_generation, reason, generation_mode = (
                version_manager.check_dialog_version(
                    dialog_key=dialog_key,
                    input_file=json_file,
                    tts_config=tts_config,
                    voice_mapping=voice_mapping,
                    output_wav=output_wav,
                    sound_effects_config=sound_effects_config,
                )
            )
            if not needs_generation:
                logger.info("Dialog %s is up-to-date, skipping generation", dialog_key)
                continue
            logger.info("Generating dialog %s (%s): %s", dialog_key, generation_mode, reason)

        # Get default sampling rate based on TTS model
        tts_model = tts_config.get("model", "Kokoro")
        default_sampling_rate = TTS_MODEL_SAMPLING_RATES.get(tts_model, 24000)
        sampling_rate: int = default_sampling_rate  # Initialize with default

        audio_dialog: list[bytes] = []
        cached = None
        if generation_mode == "background_only":
            cached = load_cached_utterances(
                utterances_path, len(conversation["conversation"])
            )
            if cached is None:
                logger.warning(
                    "Cached speech unusable for %s — falling back to full synthesis",
                    dialog_key,
                )
                generation_mode = "full"

        if cached is not None:
            # Reuse existing speech; only the background will be re-mixed.
            audio_dialog, sampling_rate = cached
        else:
            conversation_updated: list[dict] = []
            results: list[tuple[bytes, int]] = await synthesize_speech(
                utterances=conversation["conversation"],
                characters=characters,
                tts_config=tts_config,
                elevenlabs_semaphore=elevenlabs_semaphore,
                elevenlabs_timeout=args.elevenlabs_timeout,
                sound_effects_config=sound_effects_config,
            )

            for utterance_idx, (utterance, (audio, current_sampling_rate)) in enumerate(
                zip(conversation["conversation"], results), start=1
            ):
                sampling_rate = current_sampling_rate  # Update with actual sampling rate
                audio_path: str = os.path.join(
                    utterances_path, f"{utterance_idx:06}.wav"
                )
                save_audio(audio, sampling_rate, audio_path)
                audio_dialog.append(audio)
                utterance["audio_path"] = audio_path
                conversation_updated.append(utterance)
            conversation["conversation"] = conversation_updated
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(conversation, f, indent=4)

        # Compute inter-utterance gaps (one between each consecutive pair).
        # Negative gaps overlap the next utterance onto the previous one's
        # tail — see conversation_dynamics.overlapping_speech in the scenario config.
        bytes_per_sample = 2  # 16-bit PCM
        n_gaps = max(0, len(audio_dialog) - 1)
        gap_durations = sample_inter_utterance_gaps(
            conversation["conversation"],
            scenario_config.get("conversation_dynamics"),
        )
        gap_samples_list = [int(g * sampling_rate) for g in gap_durations]
        if gap_durations:
            logger.info(
                "Inter-utterance gaps: %d gaps, total %.2fs (min %.2fs, max %.2fs)",
                n_gaps,
                sum(gap_durations),
                min(gap_durations),
                max(gap_durations),
            )

        # Use spatial audio mixing with pyroomacoustics
        if args.background_sounds_dir:
            # Calculate audio duration including inter-utterance gaps
            speech_bytes = sum(len(a) for a in audio_dialog)
            gap_bytes = sum(gap_samples_list) * bytes_per_sample
            audio_duration = (speech_bytes + gap_bytes) / (sampling_rate * bytes_per_sample)

            # Load background sound from directory
            background_audio = load_background_sound(
                args.background_sounds_dir,
                audio_duration,
                sampling_rate,
                conversation["scenario_type"],
            )

            if background_audio:
                logger.info("Creating spatial audio scene with background sounds")
                # Extract speaker names for spatial positioning
                utterance_speakers = [
                    utterance["speaker"] for utterance in conversation["conversation"]
                ]

                # Create spatial audio scene
                audio_augmented = create_spatial_audio_scene(
                    utterance_audio_list=audio_dialog,
                    utterance_speakers=utterance_speakers,
                    background_audio=background_audio,
                    sampling_rate=sampling_rate,
                    room_dim=[5.0, 4.0],  # Room dimensions: 5m x 4m
                    characters=characters,
                    inter_utterance_gap_samples=gap_samples_list,
                )
            else:
                logger.warning(
                    "Failed to load background sound, using simple concatenation"
                )
                # Still insert gaps (or overlaps) even without background
                audio_augmented = mix_utterances_sequential(
                    audio_dialog, gap_samples_list
                )
        else:
            # Build combined audio with inter-utterance gaps for ElevenLabs background
            audio_combined = mix_utterances_sequential(audio_dialog, gap_samples_list)

            # Handle different field name conventions in sound_effects_config
            duration = sound_effects_config.get("duration") or sound_effects_config.get(
                "default_duration", 20.0
            )
            prompt = sound_effects_config.get("prompt") or sound_effects_config.get(
                "default_background", "ambient background"
            )

            audio_augmented: bytes = await add_background_effects(
                audio=audio_combined,
                sampling_rate=sampling_rate,
                model=sound_effects_config["model"],
                duration=duration,
                prompt=prompt,
                decibels_diff=int(sound_effects_config.get("decibels_diff", 12)),
                debug_background_output_path=(
                    output_wav.parent
                    / "_background_debug"
                    / f"{output_wav.stem}__elevenlabs_background.wav"
                    if sound_effects_config["model"] == "ElevenLabs"
                    else None
                ),
            )

        save_audio(
            audio_augmented,
            sampling_rate,
            str(output_wav),
        )

        # Record successful generation so this dialog can be skipped next run
        version_manager.update_dialog_version(
            dialog_key=dialog_key,
            input_file=json_file,
            tts_config=tts_config,
            voice_mapping=voice_mapping,
            output_wav=output_wav,
            sound_effects_config=sound_effects_config,
        )

        if args.one_by_one:
            print(f"\n[one-by-one] Done: {json_file.name} → {output_wav}")
            answer = input("Continue? [Y/n] ").strip().lower()
            if answer in ("n", "no"):
                logger.info("Stopped by user after %s", json_file.name)
                break


if __name__ == "__main__":
    cli_args = parse_arguments()
    try:
        asyncio.run(main(cli_args))
    except KeyboardInterrupt:
        sys.exit(130)
