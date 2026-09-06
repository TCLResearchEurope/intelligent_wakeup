"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WakeupModelConfig
from .context_transformer import CausalContextTransformer
from .encoder import UtteranceEncoder


class OfflineWakeupDetector(nn.Module):
    """
    Offline, turn-level contextual wakeup detector.

    Decides whether the current utterance should trigger the assistant,
    using both its own acoustic content and the conversational history.

    Two forward modes:

    1. encode_utterance(audio) → embedding
       Pre-compute and cache embeddings for all turns (phase 1 training / inference prep).

    2. forward_from_embeddings(context_embs, current_emb, ...) → logits
       Run only the context transformer + heads. Fast inner loop for phase 1 training
       when encoder is fully frozen.

    3. forward(current_audio, context_embs, ...) → logits
       Full forward pass. Used for phase 2 (encoder fine-tuning) and inference.
    """

    # Trigger type indices — keep in sync with dataset labelling
    TYPE_NONE = 0
    TYPE_DIRECT = 1
    TYPE_CONTEXTUAL = 2

    def __init__(self, config: WakeupModelConfig):
        super().__init__()
        self.config = config

        self.utterance_encoder = UtteranceEncoder(config)
        self.speaker_embedding = nn.Embedding(
            config.num_speakers,
            config.speaker_embed_dim,
            padding_idx=0,  # 0 = unknown/padding speaker
        )
        self.context_transformer = CausalContextTransformer(config)

        self.trigger_head = nn.Sequential(
            nn.Linear(config.context_dim, config.context_dim // 2),
            nn.GELU(),
            nn.Dropout(config.context_dropout),
            nn.Linear(config.context_dim // 2, 1),
        )
        # Auxiliary head: predicts trigger type (direct / contextual / none)
        # Used as extra training signal; not needed at inference.
        self.type_head = nn.Sequential(
            nn.Linear(config.context_dim, config.context_dim // 2),
            nn.GELU(),
            nn.Dropout(config.context_dropout),
            nn.Linear(config.context_dim // 2, config.num_trigger_types),
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_utterance(
        self,
        audio: torch.Tensor,              # (B, T_samples)
        audio_mask: torch.Tensor = None,  # (B, T_samples) bool, True = valid
    ) -> torch.Tensor:                    # (B, encoder_output_dim)
        return self.utterance_encoder(audio, audio_mask)

    # ------------------------------------------------------------------
    # Classification (context transformer + heads)
    # ------------------------------------------------------------------

    def forward_from_embeddings(
        self,
        current_emb: torch.Tensor,           # (B, encoder_output_dim)
        context_embs: torch.Tensor,          # (B, C, encoder_output_dim)
        current_speaker: torch.Tensor,       # (B,)  int speaker id
        context_speakers: torch.Tensor,      # (B, C) int speaker ids
        context_mask: torch.Tensor = None,   # (B, C) bool, True = pad (ignore)
    ) -> dict:
        B, C, _ = context_embs.shape

        current_spk = self.speaker_embedding(current_speaker)    # (B, spk_dim)
        context_spk = self.speaker_embedding(context_speakers)   # (B, C, spk_dim)

        # Build turn sequence: [context turns ... | current turn]
        ctx_tokens = torch.cat([context_embs, context_spk], dim=-1)             # (B, C, D+spk)
        cur_token = torch.cat([current_emb, current_spk], dim=-1).unsqueeze(1)  # (B, 1, D+spk)
        turn_seq = torch.cat([ctx_tokens, cur_token], dim=1)                    # (B, C+1, D+spk)

        # Padding mask: extend context mask with False (current turn is always valid)
        if context_mask is not None:
            cur_valid = context_mask.new_zeros(B, 1)
            full_mask = torch.cat([context_mask, cur_valid], dim=1)  # (B, C+1)
        else:
            full_mask = None

        ctx_out = self.context_transformer(turn_seq, padding_mask=full_mask)  # (B, C+1, ctx_dim)

        # Use the last position = current turn's contextualized representation
        current_ctx = ctx_out[:, -1, :]  # (B, ctx_dim)

        return {
            "trigger_logit": self.trigger_head(current_ctx).squeeze(-1),  # (B,)
            "type_logits": self.type_head(current_ctx),                    # (B, num_types)
        }

    # ------------------------------------------------------------------
    # Full forward (encode + classify)
    # ------------------------------------------------------------------

    def forward(
        self,
        current_audio: torch.Tensor,         # (B, T_samples)
        context_embs: torch.Tensor,          # (B, C, encoder_output_dim)
        current_speaker: torch.Tensor,       # (B,)
        context_speakers: torch.Tensor,      # (B, C)
        context_mask: torch.Tensor = None,   # (B, C) bool, True = pad
        audio_mask: torch.Tensor = None,     # (B, T_samples) bool, True = valid
    ) -> dict:
        current_emb = self.encode_utterance(current_audio, audio_mask)
        out = self.forward_from_embeddings(
            current_emb, context_embs, current_speaker, context_speakers, context_mask
        )
        # Expose the embedding so callers can cache it without a second encode pass
        out["current_embedding"] = current_emb
        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        outputs: dict,
        trigger_labels: torch.Tensor,    # (B,) float {0, 1}
        type_labels: torch.Tensor,       # (B,) int {0, 1, 2}
    ) -> dict:
        pos_weight = trigger_labels.new_tensor([self.config.pos_weight])
        if self.config.label_smoothing > 0.0:
            s = self.config.label_smoothing
            smooth_labels = trigger_labels * (1.0 - s) + 0.5 * s
        else:
            smooth_labels = trigger_labels
        trigger_loss = F.binary_cross_entropy_with_logits(
            outputs["trigger_logit"], smooth_labels, pos_weight=pos_weight
        )
        type_loss = F.cross_entropy(outputs["type_logits"], type_labels)

        total = trigger_loss + self.config.aux_loss_weight * type_loss
        return {"loss": total, "trigger_loss": trigger_loss, "type_loss": type_loss}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def predict(self, outputs: dict, threshold: float = 0.5) -> torch.Tensor:
        """Binary prediction from logits."""
        return (outputs["trigger_logit"].sigmoid() > threshold).long()

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}
