"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Audio validation checks for generated speech corpora.
"""

from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import soundfile as sf

from ...utils import logger


class AudioValidator:
    """
    Validate audio files in generated speech corpora.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    def __init__(
        self,
        output_dir: Path,
        min_duration_seconds: float = 2.0,
        silence_rms_threshold: float = 0.001,
        silence_ratio_threshold: float = 0.95,
    ):
        """
        Initialize audio validator.

        Args:
            output_dir: Root directory containing per-scenario speech output
            min_duration_seconds: Minimum acceptable audio duration in seconds
            silence_rms_threshold: RMS amplitude below which a frame is considered silent
            silence_ratio_threshold: Max fraction of silent frames before flagging as silent
        """
        self.output_dir = Path(output_dir)
        self.min_duration_seconds = min_duration_seconds
        self.silence_rms_threshold = silence_rms_threshold
        self.silence_ratio_threshold = silence_ratio_threshold
        self.errors: Dict[str, List[Dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_all(self) -> Dict[str, List[Dict]]:
        """
        Run all validation checks against the output directory.

        Returns:
            Dict mapping severity levels to lists of error dicts.
        """
        wav_files = [
            p
            for p in self.output_dir.rglob("*.wav")
            if "utterances" not in p.parts and "validation" not in p.parts
        ]

        if not wav_files:
            self.errors[self.CRITICAL].append(
                {
                    "check": "files_exist",
                    "message": "No dialog .wav files found under output directory",
                    "path": str(self.output_dir),
                }
            )
            return dict(self.errors)

        logger.info("Found %d dialog audio file(s) to validate", len(wav_files))

        for wav_path in wav_files:
            self._validate_file(wav_path)

        return dict(self.errors)

    def get_summary(self) -> Dict:
        """Return a summary of validation results."""
        return {
            "critical_count": len(self.errors.get(self.CRITICAL, [])),
            "warning_count": len(self.errors.get(self.WARNING, [])),
            "info_count": len(self.errors.get(self.INFO, [])),
            "has_critical_errors": len(self.errors.get(self.CRITICAL, [])) > 0,
        }

    def collect_metrics(self) -> Dict:
        """Collect audio quality metrics across all dialog files."""
        wav_files = [
            p
            for p in self.output_dir.rglob("*.wav")
            if "utterances" not in p.parts and "validation" not in p.parts
        ]
        durations = []
        rms_values = []
        sample_rates = set()

        for wav_path in wav_files:
            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                duration = len(audio) / sr
                durations.append(duration)
                rms_values.append(float(np.sqrt(np.mean(audio**2))))
                sample_rates.add(sr)
            except Exception:  # pylint: disable=broad-except
                pass

        if not durations:
            return {"total_files": 0}

        return {
            "total_files": len(durations),
            "duration_seconds": {
                "min": round(min(durations), 2),
                "max": round(max(durations), 2),
                "mean": round(float(np.mean(durations)), 2),
                "total": round(sum(durations), 2),
            },
            "rms_amplitude": {
                "min": round(min(rms_values), 5),
                "max": round(max(rms_values), 5),
                "mean": round(float(np.mean(rms_values)), 5),
            },
            "sample_rates": sorted(sample_rates),
        }

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _validate_file(self, wav_path: Path) -> None:
        """Run all checks on a single audio file."""
        try:
            audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        except Exception as exc:  # pylint: disable=broad-except
            self.errors[self.CRITICAL].append(
                {
                    "check": "readable",
                    "message": f"Cannot read audio file: {exc}",
                    "path": str(wav_path),
                }
            )
            return

        self._check_duration(wav_path, audio, sample_rate)
        self._check_silence(wav_path, audio, sample_rate)
        self._check_sample_rate(wav_path, sample_rate)
        self._check_clipping(wav_path, audio)

    def _check_duration(
        self, wav_path: Path, audio: np.ndarray, sample_rate: int
    ) -> None:
        """Flag files shorter than the minimum duration."""
        duration = len(audio) / sample_rate
        if duration < self.min_duration_seconds:
            self.errors[self.CRITICAL].append(
                {
                    "check": "min_duration",
                    "message": (
                        f"Audio too short: {duration:.2f}s < {self.min_duration_seconds}s"
                    ),
                    "path": str(wav_path),
                    "duration_seconds": round(duration, 3),
                }
            )

    def _check_silence(
        self, wav_path: Path, audio: np.ndarray, sample_rate: int
    ) -> None:
        """Flag files where most of the audio is silent."""
        frame_size = max(1, sample_rate // 100)  # 10 ms frames
        frames = [
            audio[i : i + frame_size]
            for i in range(0, len(audio) - frame_size + 1, frame_size)
        ]
        if not frames:
            return

        silent_frames = sum(
            1 for f in frames if np.sqrt(np.mean(f**2)) < self.silence_rms_threshold
        )
        silence_ratio = silent_frames / len(frames)

        if silence_ratio >= self.silence_ratio_threshold:
            self.errors[self.CRITICAL].append(
                {
                    "check": "non_silent",
                    "message": (
                        f"Audio is mostly silent: {silence_ratio:.1%} silent frames"
                    ),
                    "path": str(wav_path),
                    "silence_ratio": round(silence_ratio, 3),
                }
            )
        elif silence_ratio > 0.5:
            self.errors[self.WARNING].append(
                {
                    "check": "high_silence_ratio",
                    "message": (
                        f"High proportion of silent frames: {silence_ratio:.1%}"
                    ),
                    "path": str(wav_path),
                    "silence_ratio": round(silence_ratio, 3),
                }
            )

    def _check_sample_rate(self, wav_path: Path, sample_rate: int) -> None:
        """Warn if the sample rate is unexpectedly low."""
        known_rates = {8000, 16000, 22050, 24000, 44100, 48000}
        if sample_rate not in known_rates:
            self.errors[self.WARNING].append(
                {
                    "check": "sample_rate",
                    "message": f"Unexpected sample rate: {sample_rate} Hz",
                    "path": str(wav_path),
                    "sample_rate": sample_rate,
                }
            )

    def _check_clipping(self, wav_path: Path, audio: np.ndarray) -> None:
        """Warn if a meaningful number of samples are clipped (|amplitude| >= 1.0).

        A handful of samples at exactly 1.0 is normal encoder behaviour (e.g.
        ElevenLabs normalisation) and is inaudible.  Only warn when the ratio
        exceeds 0.01% (roughly >44 samples in a 44 kHz file) to avoid noise.
        """
        clipped = int(np.sum(np.abs(audio) >= 1.0))
        if clipped == 0:
            return
        ratio = clipped / len(audio)
        if ratio >= 0.0001:
            self.errors[self.WARNING].append(
                {
                    "check": "clipping",
                    "message": (
                        f"Audio contains {clipped} clipped sample(s) ({ratio:.2%})"
                    ),
                    "path": str(wav_path),
                    "clipped_samples": clipped,
                    "clipping_ratio": round(ratio, 5),
                }
            )
