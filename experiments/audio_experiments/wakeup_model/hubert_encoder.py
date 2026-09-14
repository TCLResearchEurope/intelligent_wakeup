"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

HuBERT utterance encoder — a self-supervised alternative to the Whisper and
Zipformer ones.

Same contract as the others: waveforms in, one pooled vector per utterance out,
attention pooling on top, so nothing downstream changes.

What makes it a different point in the comparison is the pre-training objective,
not the size or the shape:

* **Whisper** — weakly supervised ASR over web-scale audio. Optimised for *what
  words were said*.
* **Zipformer** — supervised ASR on LibriSpeech, or audio tagging on AudioSet.
* **HuBERT** — self-supervised masked prediction over LibriSpeech-960 with no
  transcripts at all. Optimised for *predictable structure in speech*, which
  keeps prosody and speaker/turn-taking cues that an ASR objective has no reason
  to preserve.

That last point is the reason to try it here. `direct` turns are a word-spotting
problem and ASR pre-training already solves them (recall 0.985). `contextual`
turns — follow-ups with no wake word — have been stuck near 0.76 across every
encoder tried, and they plausibly depend on exactly the prosodic "is this aimed
at me" signal a masked-prediction objective is more likely to keep.

Like Zipformer and unlike Whisper, it runs on the utterance's real length, so it
inherits the same window bounds — see ``config.max_utterance_seconds``.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoFeatureExtractor, HubertModel

from .config import WakeupModelConfig
from .encoder import AttentionPooling

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class HubertUtteranceEncoder(nn.Module):
    """HuBERT → attention pooling → one vector per utterance.

    Args:
        config: Model config; ``hubert_model`` names the checkpoint and
            ``encoder_output_dim`` must match its hidden size.

    Raises:
        ValueError: If the configured output dim does not match the checkpoint.
    """

    def __init__(self, config: WakeupModelConfig):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.hubert_model
        )
        self.hubert = HubertModel.from_pretrained(config.hubert_model)

        hidden = self.hubert.config.hidden_size
        if config.encoder_output_dim != hidden:
            raise ValueError(
                f"{config.hubert_model} is {hidden}-d, but "
                f"model.encoder_output_dim is {config.encoder_output_dim}."
            )

        self.pooling = AttentionPooling(config.encoder_output_dim, config.pooling)
        self.max_samples = int(config.max_utterance_seconds * SAMPLE_RATE)
        self.min_samples = int(config.min_utterance_seconds * SAMPLE_RATE)

        # The convolutional front end is always frozen when fine-tuning a
        # wav2vec2-family model: it is a fixed feature extractor and updating it
        # is the documented way to destabilise training.
        self.hubert.feature_extractor._freeze_parameters()  # pylint: disable=protected-access
        self._apply_freeze(config.freeze_encoder, config.unfreeze_top_layers)

    def _apply_freeze(self, freeze: bool, unfreeze_top: int) -> None:
        """Freeze the transformer, optionally leaving the top layers trainable.

        Args:
            freeze: Whether to freeze at all.
            unfreeze_top: How many of the last layers to leave trainable.

        Returns:
            None.
        """
        if not freeze:
            return
        for param in self.hubert.parameters():
            param.requires_grad = False
        if unfreeze_top > 0:
            self.unfreeze_top_layers(unfreeze_top)

    def unfreeze_top_layers(self, n: int) -> None:
        """Unfreeze the last ``n`` transformer layers.

        Args:
            n: How many layers to unfreeze.

        Returns:
            None.
        """
        for layer in self.hubert.encoder.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_values: torch.Tensor,  # (B, T_samples) 16 kHz
        attention_mask: torch.Tensor = None,  # (B, T_samples) bool, True = real
    ) -> torch.Tensor:  # (B, encoder_output_dim)
        """Encode raw waveforms into one embedding each.

        Args:
            input_values: (B, T_samples) 16 kHz audio.
            attention_mask: (B, T_samples) bool, True where a sample is real.

        Returns:
            (B, encoder_output_dim) utterance embeddings.
        """
        device = input_values.device
        lengths = (
            attention_mask.sum(dim=1)
            if attention_mask is not None
            else torch.full(
                (input_values.shape[0],), input_values.shape[1], device=device
            )
        )

        trimmed = []
        for wav, length in zip(input_values, lengths.tolist()):
            usable = wav[: min(int(length), self.max_samples)]
            if usable.numel() < self.min_samples:
                usable = torch.nn.functional.pad(
                    usable, (0, self.min_samples - usable.numel())
                )
            trimmed.append(usable.detach().cpu().numpy())

        # Ask for the mask explicitly. This checkpoint's extractor defaults to
        # return_attention_mask=False, which is right for the uniform-length
        # batches it was pre-trained on; here the batch is ragged, and without
        # the mask the per-utterance normalisation would take its statistics
        # over the padding too.
        batch = self.feature_extractor(
            trimmed,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        values = batch.input_values.to(device)
        mask = batch.attention_mask.to(device)

        hidden = self.hubert(values, attention_mask=mask).last_hidden_state

        frame_lens = self.hubert._get_feat_extract_output_lengths(  # pylint: disable=protected-access
            mask.sum(dim=-1)
        ).to(
            device
        )
        frames = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
        # AttentionPooling takes True == padding.
        return self.pooling(hidden, frames >= frame_lens.unsqueeze(1))
