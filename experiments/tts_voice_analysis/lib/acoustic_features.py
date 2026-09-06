"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Interpretable acoustic descriptors (experiment-setup.md, step 7): a small set of
speaker-related and channel-related measurements, computed per utterance, used
both to characterize how ElevenLabs voices differ and to test whether
speaker-embedding distances are just tracking recording-quality differences.

Reverberation estimation (RT60-style decay) is not implemented: doc lists it as
optional "where reliable", and a robust blind estimate needs more than a single
short utterance to be trustworthy.
"""

import librosa
import numpy as np
import parselmouth

SR = 16000


def f0_stats(wav: np.ndarray) -> dict:
    """Compute median F0, F0 range, std, coefficient of variation, and voiced fraction.

    All values are in Hz unless noted. f0_cv is intonation range normalized by pitch
    level (flat/monotone delivery -> low cv); voiced_fraction is a cheap pause/breathing
    proxy (fraction of analysis frames with a detected pitch).

    Args:
        wav: Audio samples at 16kHz, mono.

    Returns:
        Dict with median_f0_hz, f0_range_hz (p5-p95), f0_std_hz, f0_cv, and
        voiced_fraction (all None but voiced_fraction if no voiced frames are found).
    """
    snd = parselmouth.Sound(wav, sampling_frequency=SR)
    all_f0 = snd.to_pitch().selected_array["frequency"]
    voiced_fraction = float(np.mean(all_f0 > 0)) if len(all_f0) else None
    f0 = all_f0[all_f0 > 0]
    if len(f0) == 0:
        return {
            "median_f0_hz": None,
            "f0_range_hz": None,
            "f0_std_hz": None,
            "f0_cv": None,
            "voiced_fraction": voiced_fraction,
        }
    median = float(np.median(f0))
    std = float(np.std(f0))
    return {
        "median_f0_hz": median,
        "f0_range_hz": float(np.percentile(f0, 95) - np.percentile(f0, 5)),
        "f0_std_hz": std,
        "f0_cv": std / median if median > 0 else None,
        "voiced_fraction": voiced_fraction,
    }


def hnr_db(wav: np.ndarray) -> float | None:
    """Compute mean harmonics-to-noise ratio in dB (a voice-quality descriptor).

    Args:
        wav: Audio samples at 16kHz, mono.

    Returns:
        Mean HNR in dB, or None if no voiced frames are found.
    """
    snd = parselmouth.Sound(wav, sampling_frequency=SR)
    values = snd.to_harmonicity().values
    values = values[values != -200]  # praat's "undefined" sentinel for unvoiced frames
    return float(np.mean(values)) if len(values) else None


def jitter_shimmer(wav: np.ndarray) -> dict:
    """Compute local jitter and shimmer (%): cycle-to-cycle F0/amplitude perturbation.

    A voice-quality/naturalness cue distinct from HNR - unnaturally low values here
    are a classic synthetic-speech tell (real voices always have some
    micro-instability).

    Args:
        wav: Audio samples at 16kHz, mono.

    Returns:
        Dict with jitter_local_pct and shimmer_local_pct (None if either is undefined).
    """
    snd = parselmouth.Sound(wav, sampling_frequency=SR)
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return {"jitter_local_pct": None, "shimmer_local_pct": None}
    return {
        "jitter_local_pct": (float(jitter_local) * 100 if not np.isnan(jitter_local) else None),
        "shimmer_local_pct": (float(shimmer_local) * 100 if not np.isnan(shimmer_local) else None),
    }


def formant_freqs(wav: np.ndarray) -> dict:
    """Compute median F1/F2/F3 (Hz) over voiced frames.

    These vocal-tract resonances are the main acoustic correlate of speaker identity
    that's independent of pitch.

    Args:
        wav: Audio samples at 16kHz, mono.

    Returns:
        Dict with f1_hz, f2_hz, and f3_hz (all None if no voiced frames are found).
    """
    snd = parselmouth.Sound(wav, sampling_frequency=SR)
    pitch = snd.to_pitch()
    voiced_times = pitch.ts()[pitch.selected_array["frequency"] > 0]
    if len(voiced_times) == 0:
        return {"f1_hz": None, "f2_hz": None, "f3_hz": None}
    formant = snd.to_formant_burg(maximum_formant=5500)
    values: dict[int, list] = {1: [], 2: [], 3: []}
    for t in voiced_times:
        for n in (1, 2, 3):
            v = formant.get_value_at_time(n, t)
            if not np.isnan(v):
                values[n].append(v)
    return {
        "f1_hz": float(np.median(values[1])) if values[1] else None,
        "f2_hz": float(np.median(values[2])) if values[2] else None,
        "f3_hz": float(np.median(values[3])) if values[3] else None,
    }


def speaking_rate(text: str, duration_s: float) -> float | None:
    """Compute words per second from the utterance's known text and audio duration.

    Args:
        text: The utterance's transcript.
        duration_s: Audio duration in seconds.

    Returns:
        Words per second, or None if text is empty or duration_s is non-positive.
    """
    if not text or duration_s <= 0:
        return None
    return len(text.split()) / duration_s


def effective_bandwidth_hz(wav: np.ndarray, energy_fraction: float = 0.99) -> float:
    """Compute the frequency below which most of the spectral energy is contained.

    Args:
        wav: Audio samples at 16kHz, mono.
        energy_fraction: Fraction of total spectral energy to capture below the cutoff.

    Returns:
        The frequency (Hz) below which `energy_fraction` of spectral energy is
        contained.
    """
    spectrum = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(len(wav), d=1 / SR)
    cumulative = np.cumsum(spectrum) / np.sum(spectrum)
    idx = int(np.searchsorted(cumulative, energy_fraction))
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_slope_db(wav: np.ndarray, low_edge: float = 1000, high_edge: float = 4000) -> float:
    """Compute the low/high-frequency energy ratio in dB.

    Positive values are low-frequency-heavy, a channel cue.

    Args:
        wav: Audio samples at 16kHz, mono.
        low_edge: Upper bound (Hz) of the low-frequency band.
        high_edge: Lower bound (Hz) of the high-frequency band.

    Returns:
        Low/high-frequency energy ratio in dB.
    """
    spectrum = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(len(wav), d=1 / SR)
    low_energy = spectrum[freqs < low_edge].sum()
    high_energy = spectrum[freqs > high_edge].sum()
    if high_energy <= 0:
        return float("inf")
    return float(10 * np.log10((low_energy + 1e-12) / (high_energy + 1e-12)))


def spectral_flatness(wav: np.ndarray) -> float:
    """Compute mean spectral flatness, a channel/noise-quality cue.

    Args:
        wav: Audio samples at 16kHz, mono.

    Returns:
        Mean spectral flatness (0=tonal, 1=noise-like).
    """
    return float(np.mean(librosa.feature.spectral_flatness(y=wav)))


def estimated_snr_db(
    wav: np.ndarray, frame_length: int = 512, hop_length: int = 256
) -> float | None:
    """Compute a crude frame-energy-based SNR estimate.

    Compares the loudest half of frames against the quietest 10% as a noise-floor
    proxy.

    Args:
        wav: Audio samples at 16kHz, mono.
        frame_length: Frame length (samples) used for the RMS energy computation.
        hop_length: Hop length (samples) between frames.

    Returns:
        Estimated SNR in dB, or None if there are too few frames to estimate.
    """
    frame_rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    if len(frame_rms) < 10:
        return None
    noise_floor = np.percentile(frame_rms, 10)
    speech_level = np.percentile(frame_rms, 50)
    if noise_floor <= 0:
        return None
    return float(20 * np.log10((speech_level + 1e-12) / (noise_floor + 1e-12)))


def clipping_rate(wav: np.ndarray, threshold: float = 0.99) -> float:
    """Compute the fraction of samples at or near digital full-scale.

    Args:
        wav: Audio samples at 16kHz, mono.
        threshold: Absolute sample value above which a sample counts as clipped.

    Returns:
        Fraction of samples with |wav| >= threshold.
    """
    return float(np.mean(np.abs(wav) >= threshold))


def compute_features(
    wav: np.ndarray, text: str | None = None, duration_s: float | None = None
) -> dict:
    """Compute all speaker-related and channel-related descriptors for one utterance.

    Args:
        wav: Audio samples at 16kHz, mono.
        text: The utterance's transcript, used for speaking rate (optional).
        duration_s: Audio duration in seconds; computed from wav if not provided.

    Returns:
        Dict combining all per-utterance acoustic feature dicts, plus duration_s.
    """
    duration_s = duration_s if duration_s is not None else len(wav) / SR
    features = {
        **f0_stats(wav),
        "hnr_db": hnr_db(wav),
        **jitter_shimmer(wav),
        **formant_freqs(wav),
        "speaking_rate_wps": speaking_rate(text, duration_s) if text else None,
        "effective_bandwidth_hz": effective_bandwidth_hz(wav),
        "spectral_slope_db": spectral_slope_db(wav),
        "spectral_flatness": spectral_flatness(wav),
        "estimated_snr_db": estimated_snr_db(wav),
        "clipping_rate": clipping_rate(wav),
        "duration_s": duration_s,
    }
    return features
