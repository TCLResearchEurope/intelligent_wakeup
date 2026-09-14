"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

import torch
from torch import nn

from .config import WakeupModelConfig


class CausalContextTransformer(nn.Module):
    """
    Small transformer that reasons over a sequence of turn tokens (one per utterance).
    Causal masking ensures turn t only attends to turns 0..t — no future leakage.

    NaN-safety: when a position can only attend to padding positions (e.g. the first
    turn in a conversation has no context history), softmax(-inf, ..., -inf) = NaN.
    We run layers manually and zero-out padding positions after each layer so NaN
    values never propagate as keys into the next layer.
    """

    def __init__(self, config: WakeupModelConfig):
        super().__init__()
        # pooled_dim, not encoder_output_dim: mean_max pooling concatenates
        # two vectors and doubles the embedding width.
        in_dim = config.pooled_dim + config.speaker_embed_dim + config.wake_embed_dim

        self.input_proj = nn.Linear(in_dim, config.context_dim)
        self.pos_embedding = nn.Embedding(
            config.max_context_turns + 1, config.context_dim
        )

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.context_dim,
                    nhead=config.context_heads,
                    dim_feedforward=config.context_dim * config.context_ffn_multiplier,
                    dropout=config.context_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.context_layers)
            ]
        )

    def forward(
        self,
        turn_tokens: torch.Tensor,  # (B, T, encoder_dim + speaker_dim)
        padding_mask: torch.Tensor = None,  # (B, T) bool, True = pad
    ) -> torch.Tensor:  # (B, T, context_dim)
        """Contextualise a sequence of turn tokens.

        Args:
            turn_tokens: (B, T, encoder_dim + speaker_dim) per-turn features.
            padding_mask: (B, T) bool, True where a slot is padding.

        Returns:
            (B, T, context_dim) contextualised turns.
        """
        _, n_turns, _ = turn_tokens.shape

        x = self.input_proj(turn_tokens)
        positions = torch.arange(n_turns, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Zero-out padded inputs so their keys start as 0 not garbage
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Bool, not the float mask generate_square_subsequent_mask returns:
        # padding_mask is bool, and torch deprecates mixing the two types on
        # the same call ("Support for mismatched src_key_padding_mask and
        # src_mask is deprecated").
        causal_mask = torch.triu(
            torch.ones(n_turns, n_turns, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )
            # Zero-out padded positions after every layer.
            # Padding positions whose entire causal window is masked produce NaN via
            # softmax(-inf,...,-inf). Zeroing here prevents those NaNs becoming keys
            # in the next layer and corrupting non-padded positions.
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x
