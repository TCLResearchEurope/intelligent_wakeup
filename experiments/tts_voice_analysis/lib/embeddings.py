"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Speaker-embedding extraction shared by features/extract_embeddings.py, used for both the
ElevenLabs voices and the natural (VCTK) reference corpus so the two
are directly comparable.

Two independently trained speaker-verification encoders are used (per
experiment-setup.md) so diversity results aren't an artifact of one
representation:
  - ECAPA-TDNN (SpeechBrain, trained on VoxCeleb)
  - WavLM-SV   (HuggingFace, WavLM base+ fine-tuned for x-vector speaker verification)

Both run on CPU; this environment has no GPU.
"""

from pathlib import Path

import librosa
import numpy as np
import torch

MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"
TARGET_SR = 16000


def load_audio(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file as a mono float32 waveform at a target sample rate.

    Args:
        path: Path to the audio file. Any format ffmpeg/librosa can decode is accepted.
        sr: Target sample rate to resample to.

    Returns:
        The decoded mono float32 waveform.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


class EcapaEncoder:
    """ECAPA-TDNN speaker encoder producing 192-dim embeddings."""

    name = "ecapa"

    def __init__(self):
        """Load the pretrained ECAPA-TDNN speaker-verification model from SpeechBrain."""
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODEL_CACHE_DIR / "ecapa"),
        )

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Compute an L2-normalized speaker embedding for a waveform.

        Args:
            wav: Mono float32 waveform at 16kHz.

        Returns:
            The L2-normalized embedding vector.
        """
        tensor = torch.tensor(wav).unsqueeze(0)
        with torch.no_grad():
            vec = self.model.encode_batch(tensor).flatten().numpy()
        return vec / np.linalg.norm(vec)


class WavlmSvEncoder:
    """WavLM-SV speaker encoder producing 512-dim x-vector embeddings."""

    name = "wavlm_sv"
    CHECKPOINT = "microsoft/wavlm-base-plus-sv"

    def __init__(self):
        """Load the pretrained WavLM base+ x-vector speaker-verification model."""
        from transformers import AutoFeatureExtractor, WavLMForXVector

        cache_dir = str(MODEL_CACHE_DIR / "wavlm")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.CHECKPOINT, cache_dir=cache_dir
        )
        self.model = WavLMForXVector.from_pretrained(self.CHECKPOINT, cache_dir=cache_dir)
        self.model.eval()

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Compute an L2-normalized speaker embedding for a waveform.

        Args:
            wav: Mono float32 waveform at 16kHz.

        Returns:
            The L2-normalized embedding vector.
        """
        inputs = self.feature_extractor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            vec = self.model(**inputs).embeddings[0].numpy()
        return vec / np.linalg.norm(vec)


ENCODERS = {
    "ecapa": EcapaEncoder,
    "wavlm_sv": WavlmSvEncoder,
}
