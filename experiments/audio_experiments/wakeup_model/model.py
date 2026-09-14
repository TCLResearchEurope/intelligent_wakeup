"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .config import WakeupModelConfig
from .context_transformer import CausalContextTransformer
from .encoder import UtteranceEncoder
from .hubert_encoder import HubertUtteranceEncoder
from .zipformer_encoder import ZipformerUtteranceEncoder


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

        encoders = {
            "zipformer": ZipformerUtteranceEncoder,
            "hubert": HubertUtteranceEncoder,
        }
        self.utterance_encoder = encoders.get(config.encoder_type, UtteranceEncoder)(
            config
        )
        self.speaker_embedding = nn.Embedding(
            config.num_speakers,
            config.speaker_embed_dim,
            padding_idx=0,  # 0 = unknown/padding speaker
        )
        # Wake-word presence per turn: absent / present. See the note in
        # configs/model/default.yaml -- the frozen encoder loses the keyword,
        # so this hands it to the context transformer directly.
        self.wake_embedding = (
            nn.Embedding(2, config.wake_embed_dim) if config.wake_embed_dim else None
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
        audio: torch.Tensor,  # (B, T_samples)
        audio_mask: torch.Tensor = None,  # (B, T_samples) bool, True = valid
    ) -> torch.Tensor:  # (B, encoder_output_dim)
        """Encode one batch of utterances.

        Args:
            audio: (B, T_samples) 16 kHz waveforms.
            audio_mask: (B, T_samples) bool, True where a sample is real.

        Returns:
            (B, encoder_output_dim) utterance embeddings.
        """
        return self.utterance_encoder(audio, audio_mask)

    # ------------------------------------------------------------------
    # Classification (context transformer + heads)
    # ------------------------------------------------------------------

    def forward_from_embeddings(  # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        self,
        current_emb: torch.Tensor,  # (B, encoder_output_dim)
        context_embs: torch.Tensor,  # (B, C, encoder_output_dim)
        current_speaker: torch.Tensor,  # (B,)  int speaker id
        context_speakers: torch.Tensor,  # (B, C) int speaker ids
        context_mask: torch.Tensor = None,  # (B, C) bool, True = pad (ignore)
        current_wake: torch.Tensor = None,  # (B,)  int {0, 1}
        context_wakes: torch.Tensor = None,  # (B, C) int {0, 1}
    ) -> dict:
        """Classify the current turn from embeddings, skipping the encoder.

        Args:
            current_emb: (B, encoder_output_dim) current turn.
            context_embs: (B, C, encoder_output_dim) preceding turns.
            current_speaker: (B,) speaker role id of the current turn.
            context_speakers: (B, C) speaker role ids of the context.
            context_mask: (B, C) bool, True where a context slot is padding.

        Returns:
            Dict with ``trigger_logit`` and ``type_logits``.
        """
        # pylint: disable=too-many-locals
        batch_size = context_embs.shape[0]

        current_spk = self.speaker_embedding(current_speaker)  # (B, spk_dim)
        context_spk = self.speaker_embedding(context_speakers)  # (B, C, spk_dim)

        ctx_parts = [context_embs, context_spk]
        cur_parts = [current_emb, current_spk]
        if self.wake_embedding is not None:
            # Absent when the caller supplies nothing, so a model trained with
            # the feature still runs on batches that lack it.
            zeros_c = context_speakers.new_zeros(context_speakers.shape)
            zeros_b = current_speaker.new_zeros(current_speaker.shape)
            ctx_parts.append(
                self.wake_embedding(zeros_c if context_wakes is None else context_wakes)
            )
            cur_parts.append(
                self.wake_embedding(zeros_b if current_wake is None else current_wake)
            )

        # Build turn sequence: [context turns ... | current turn]
        ctx_tokens = torch.cat(ctx_parts, dim=-1)  # (B, C, D+spk+wake)
        cur_token = torch.cat(cur_parts, dim=-1).unsqueeze(1)  # (B, 1, D+spk+wake)
        turn_seq = torch.cat([ctx_tokens, cur_token], dim=1)  # (B, C+1, D+spk)

        # Padding mask: extend context mask with False (current turn is always valid)
        if context_mask is not None:
            cur_valid = context_mask.new_zeros(batch_size, 1)
            full_mask = torch.cat([context_mask, cur_valid], dim=1)  # (B, C+1)
        else:
            full_mask = None

        ctx_out = self.context_transformer(
            turn_seq, padding_mask=full_mask
        )  # (B, C+1, ctx_dim)

        # Use the last position = current turn's contextualized representation
        current_ctx = ctx_out[:, -1, :]  # (B, ctx_dim)

        return {
            "trigger_logit": self.trigger_head(current_ctx).squeeze(-1),  # (B,)
            "type_logits": self.type_head(current_ctx),  # (B, num_types)
        }

    # ------------------------------------------------------------------
    # Full forward (encode + classify)
    # ------------------------------------------------------------------

    def forward(
        self,
        current_audio: torch.Tensor,  # (B, T_samples)
        context_embs: torch.Tensor,  # (B, C, encoder_output_dim)
        current_speaker: torch.Tensor,  # (B,)
        context_speakers: torch.Tensor,  # (B, C)
        context_mask: torch.Tensor = None,  # (B, C) bool, True = pad
        audio_mask: torch.Tensor = None,  # (B, T_samples) bool, True = valid
    ) -> dict:
        """Encode the current turn and classify it against its context.

        Args:
            current_audio: (B, T_samples) waveform of the current turn.
            context_embs: (B, C, encoder_output_dim) preceding turns.
            current_speaker: (B,) speaker role id of the current turn.
            context_speakers: (B, C) speaker role ids of the context.
            context_mask: (B, C) bool, True where a context slot is padding.
            audio_mask: (B, T_samples) bool, True where a sample is real.

        Returns:
            Dict with ``trigger_logit``, ``type_logits`` and the computed
            ``current_embedding``, so callers can cache it without re-encoding.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        trigger_labels: torch.Tensor,  # (B,) float {0, 1}
        type_labels: torch.Tensor,  # (B,) int {0, 1, 2}
    ) -> dict:
        """Binary trigger loss plus the auxiliary trigger-type loss.

        Args:
            outputs: Model output dict.
            trigger_labels: (B,) float {0, 1} — should the VA respond.
            type_labels: (B,) int {0, 1, 2} — trigger type.

        Returns:
            Dict with ``loss``, ``trigger_loss`` and ``type_loss``.
        """
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
        """Count parameters.

        Returns:
            Dict with total and trainable counts, in millions.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}


def forward_batch(model: OfflineWakeupDetector, batch: dict, device: torch.device):
    """Run one batch through the model, whichever phase it belongs to.

    The phase is read off the batch itself: ``current_embedding`` means the
    encoder is frozen and the current turn came from the cache (Phase 1),
    ``current_audio`` means it runs with gradient (Phase 2).

    Args:
        model: The detector.
        batch: A collated batch.
        device: Device to move tensors to.

    Returns:
        The model's output dict.
    """
    context_embs = batch["context_embeddings"].to(device)
    context_spks = batch["context_speakers"].to(device)
    context_mask = batch["context_mask"].to(device)
    current_spk = batch["current_speaker_id"].to(device)

    wake = batch.get("current_wake")
    wakes = batch.get("context_wakes")
    if wake is not None:
        wake, wakes = wake.to(device), wakes.to(device)

    if "current_embedding" in batch:
        return model.forward_from_embeddings(
            batch["current_embedding"].to(device),
            context_embs,
            current_spk,
            context_spks,
            context_mask,
            wake,
            wakes,
        )

    audio_mask = batch.get("audio_mask")
    if audio_mask is not None:
        audio_mask = audio_mask.to(device)
    return model.forward(
        batch["current_audio"].to(device),
        context_embs,
        current_spk,
        context_spks,
        context_mask,
        audio_mask,
    )
