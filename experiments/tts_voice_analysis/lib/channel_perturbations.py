"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Controlled channel/recording-condition perturbations (experiment-setup.md, step 5).

Each function takes a mono float32 waveform at 16kHz (the sample rate used
throughout this pipeline) and returns a perturbed waveform at the same rate,
safety peak-normalized only if needed to avoid digital clipping (loudness
itself is left alone here - that's what step 6's normalization control tests).

No microphone impulse responses are applied: the doc lists these as optional
"if suitable data are available", and none are available in this repo.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = 16000


def _safety_peak_normalize(wav: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Scale a waveform down only if a perturbation pushed it past digital full-scale.

    Args:
        wav: Mono float32 waveform.
        ceiling: Maximum allowed absolute sample value before scaling is applied.

    Returns:
        The waveform, unchanged if already within the ceiling or scaled down to it
        otherwise.
    """
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak > ceiling:
        wav = wav * (ceiling / peak)
    return wav.astype(np.float32)


def bandwidth_limit(wav: np.ndarray, low_hz: float = 300, high_hz: float = 3400) -> np.ndarray:
    """Apply a telephone-band bandpass that emulates a narrowband recording/transmission
    channel.

    Args:
        wav: Mono float32 waveform at 16kHz.
        low_hz: High-pass cutoff frequency, in Hz.
        high_hz: Low-pass cutoff frequency, in Hz.

    Returns:
        The bandwidth-limited waveform, safety peak-normalized if needed.
    """
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=SR, output="sos")
    return _safety_peak_normalize(sosfiltfilt(sos, wav))


def eq_tilt(wav: np.ndarray, coefficient: float = 0.95) -> np.ndarray:
    """Apply first-order pre-emphasis, producing a frequency-response tilt (brighter/
    thinner timbre).

    Args:
        wav: Mono float32 waveform at 16kHz.
        coefficient: Pre-emphasis coefficient controlling the strength of the tilt.

    Returns:
        The tilted waveform, safety peak-normalized if needed.
    """
    tilted = np.append(wav[0], wav[1:] - coefficient * wav[:-1])
    return _safety_peak_normalize(tilted)


def codec_degrade(wav: np.ndarray, codec: str = "gsm") -> np.ndarray:
    """Round-trip a waveform through a lossy codec via ffmpeg.

    Supported codecs are gsm (narrowband, ~13kbit/s) and opus (low-bitrate).

    Args:
        wav: Mono float32 waveform at 16kHz.
        codec: Which codec to round-trip through, either "gsm" or "opus".

    Returns:
        The codec-degraded waveform, safety peak-normalized if needed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_wav = tmp / "src.wav"
        sf.write(src_wav, wav, SR)

        if codec == "gsm":
            encoded = tmp / "encoded.wav"
            # GSM 06.10 requires 8kHz mono; ffmpeg resamples on the way in/out.
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_wav),
                    "-ar",
                    "8000",
                    "-ab",
                    "13k",
                    "-ac",
                    "1",
                    str(encoded),
                ],
                check=True,
                capture_output=True,
            )
        elif codec == "opus":
            encoded = tmp / "encoded.opus"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_wav),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "8k",
                    str(encoded),
                ],
                check=True,
                capture_output=True,
            )
        else:
            raise ValueError(f"Unknown codec: {codec}")

        decoded = tmp / "decoded.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(encoded),
                "-ar",
                str(SR),
                "-ac",
                "1",
                str(decoded),
            ],
            check=True,
            capture_output=True,
        )
        out, _ = sf.read(decoded, dtype="float32")
    if len(out) < len(wav):
        out = np.pad(out, (0, len(wav) - len(out)))
    return _safety_peak_normalize(out[: len(wav)])


def add_noise(
    wav: np.ndarray, snr_db: float = 15.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add white noise at a target signal-to-noise ratio.

    Args:
        wav: Mono float32 waveform.
        snr_db: Target signal-to-noise ratio, in dB.
        rng: Random generator to draw noise from; a new default generator is used if
            not provided.

    Returns:
        The noisy waveform, safety peak-normalized if needed.
    """
    rng = rng or np.random.default_rng()
    signal_power = float(np.mean(wav**2))
    noise = rng.normal(0.0, 1.0, size=len(wav)).astype(np.float32)
    noise_power_target = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power_target / (np.mean(noise**2) + 1e-12))
    return _safety_peak_normalize(wav + noise)


def add_reverb(wav: np.ndarray, rt60: float = 0.6, room_dim: tuple = (6.0, 5.0, 3.0)) -> np.ndarray:
    """Convolve a waveform with a synthetic shoebox-room impulse response at a target
    RT60.

    Args:
        wav: Mono float32 waveform at 16kHz.
        rt60: Target reverberation time, in seconds.
        room_dim: Shoebox room dimensions (length, width, height), in meters.

    Returns:
        The reverberant waveform, safety peak-normalized if needed.
    """
    absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(absorption), max_order=max_order)
    room.add_source([1.5, 1.5, 1.5])
    room.add_microphone([room_dim[0] - 1.5, room_dim[1] - 1.5, 1.5])
    room.compute_rir()
    rir = room.rir[0][0]
    out = fftconvolve(wav, rir)[: len(wav)]
    return _safety_peak_normalize(out)


PERTURBATIONS = {
    "bandwidth_limit": bandwidth_limit,
    "eq_tilt": eq_tilt,
    "codec_gsm": lambda wav: codec_degrade(wav, codec="gsm"),
    "codec_opus": lambda wav: codec_degrade(wav, codec="opus"),
    "noise_15db": lambda wav: add_noise(wav, snr_db=15.0),
    "noise_5db": lambda wav: add_noise(wav, snr_db=5.0),
    "reverb": lambda wav: add_reverb(wav, rt60=0.6),
}
