"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

import torch
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel

from .config import WakeupModelConfig


class AttentionPooling(nn.Module):
    """Learnable attention pooling: collapses frame sequence → single utterance vector."""

    def __init__(self, dim: int, mode: str = "attention"):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.mode = mode

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T_frames, D)
        padding_mask: torch.Tensor = None,  # (B, T_frames) bool, True = pad
    ) -> torch.Tensor:  # (B, D)
        """Collapse a frame sequence to one vector by learned attention.

        Args:
            hidden_states: (B, T_frames, D) encoder output.
            padding_mask: (B, T_frames) bool, True where a frame is padding.

        Returns:
            (B, D) pooled utterance embedding.
        """
        logits = self.score(hidden_states).squeeze(-1)  # (B, T_frames)
        if padding_mask is not None:
            # A row that is entirely padding would be softmax(-inf, ..., -inf),
            # which is NaN, and one NaN embedding poisons every batch the turn
            # appears in as context. Leave such a row unmasked instead: its
            # frames are all padding anyway, so the result is unused, but it
            # stays finite. Reached when an encoder returns zero frames for a
            # degenerately short utterance.
            all_pad = padding_mask.all(dim=-1, keepdim=True)
            logits = logits.masked_fill(padding_mask & ~all_pad, float("-inf"))
        weights = logits.softmax(dim=-1)  # (B, T_frames)
        pooled = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)  # (B, D)

        if self.mode != "mean_max":
            return pooled

        # Attention pooling is a weighted average, so a brief event -- a wake
        # word inside a 30 s window -- is diluted by everything around it. A max
        # over frames keeps the peak instead; the two are complementary.
        masked = hidden_states
        if padding_mask is not None:
            masked = hidden_states.masked_fill(
                (padding_mask & ~all_pad).unsqueeze(-1), float("-inf")
            )
        return torch.cat([pooled, masked.amax(dim=1)], dim=-1)  # (B, 2D)


class UtteranceEncoder(nn.Module):
    """
    Whisper encoder + attention pooling → fixed-dim utterance embedding.

    Whisper is trained with ASR supervision so its encoder representations encode
    both acoustic (phonetics, prosody) and semantic (what was said) content,
    making it more suitable for wakeword + contextual trigger detection than
    self-supervised acoustic-only models.

    Phase 1 (freeze_encoder=True, unfreeze_top_layers=0):
        All Whisper encoder weights frozen. Used to pre-cache embeddings offline.
        Only the attention pooling layer has gradients.

    Phase 2 (unfreeze_top_layers > 0):
        Top N Whisper encoder layers unfrozen. Fine-tune with lower LR.
    """

    def __init__(self, config: WakeupModelConfig):
        super().__init__()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.whisper_model
        )
        whisper = WhisperModel.from_pretrained(config.whisper_model)
        self.encoder = whisper.encoder
        self.num_encoder_layers = len(self.encoder.layers)
        self.pooling = AttentionPooling(config.encoder_output_dim, config.pooling)
        self._apply_freeze(config.freeze_encoder, config.unfreeze_top_layers)

    def _apply_freeze(self, freeze: bool, unfreeze_top: int) -> None:
        if not freeze:
            return
        for param in self.encoder.parameters():
            param.requires_grad = False
        if unfreeze_top > 0:
            self.unfreeze_top_layers(unfreeze_top)

    def unfreeze_top_layers(self, n: int) -> None:
        """Call this when transitioning from phase 1 → phase 2."""
        for i in range(self.num_encoder_layers - n, self.num_encoder_layers):
            for param in self.encoder.layers[i].parameters():
                param.requires_grad = True

    def forward(
        self,
        input_values: torch.Tensor,  # (B, T_samples)  16kHz audio
        attention_mask: torch.Tensor = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # (B, encoder_output_dim)
        """Encode raw waveforms into one embedding each.

        Args:
            input_values: (B, T_samples) 16 kHz audio.
            attention_mask: Accepted for API compatibility and ignored — the
                feature extractor pads every utterance to a fixed 30 s window.

        Returns:
            (B, encoder_output_dim) utterance embeddings.
        """
        # Convert raw waveforms → log-mel spectrograms (B, 80, 3000).
        # WhisperFeatureExtractor pads / truncates each utterance to 30 s.
        # Silence frames will produce distinct encoder outputs that the
        # attention pooling learns to downweight automatically.
        features = self.feature_extractor(
            list(input_values.cpu().numpy()),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        input_features = features.input_features.to(
            input_values.device
        )  # (B, 80, 3000)
        outputs = self.encoder(input_features=input_features)
        return self.pooling(outputs.last_hidden_state)  # (B, D)
