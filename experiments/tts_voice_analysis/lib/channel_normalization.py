"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Conservative channel normalization (experiment-setup.md, step 6): removes obvious
recording-condition differences (loudness, DC offset/rumble, an upper bandwidth
ceiling) without attempting to reshape the spectral envelope that carries speaker
identity. No aggressive spectral whitening/deconvolution is applied.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 16000
TARGET_DBFS = -20.0
LOW_HZ = 50.0
HIGH_HZ = 7900.0  # just under the 8kHz Nyquist ceiling of our common 16kHz pipeline


def _dbfs(wav: np.ndarray) -> float:
    """Compute the RMS level of a waveform in dBFS.

    Args:
        wav: Mono float32 waveform.

    Returns:
        The RMS level in dB relative to full scale.
    """
    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    return float(20 * np.log10(rms + 1e-12))


def normalize_bandwidth(
    wav: np.ndarray, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ
) -> np.ndarray:
    """Apply a common bandwidth ceiling via gentle high-pass and low-pass filtering.

    The high-pass removes rumble/DC offset and the low-pass caps the spectrum at a
    shared limit, so recordings with different upper bandwidths are made comparable.

    Args:
        wav: Mono float32 waveform at 16kHz.
        low_hz: High-pass cutoff frequency, in Hz.
        high_hz: Low-pass cutoff frequency, in Hz.

    Returns:
        The bandwidth-limited waveform.
    """
    sos = butter(2, [low_hz, high_hz], btype="bandpass", fs=SR, output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def normalize_loudness(wav: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Apply simple whole-utterance RMS normalization to a common loudness target.

    Args:
        wav: Mono float32 waveform.
        target_dbfs: Desired RMS loudness, in dBFS.

    Returns:
        The loudness-normalized waveform, safety peak-limited if needed to avoid clipping.
    """
    gain_db = target_dbfs - _dbfs(wav)
    out = wav * (10 ** (gain_db / 20))
    peak = np.max(np.abs(out)) if len(out) else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out.astype(np.float32)


def normalize_channel(wav: np.ndarray) -> np.ndarray:
    """Apply the full conservative normalization: common bandwidth, then common loudness.

    Args:
        wav: Mono float32 waveform at 16kHz.

    Returns:
        The fully normalized waveform.
    """
    return normalize_loudness(normalize_bandwidth(wav))
