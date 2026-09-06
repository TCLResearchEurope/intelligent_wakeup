"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

from dataclasses import dataclass


@dataclass
class WakeupModelConfig:
    # --- Encoder ---
    whisper_model: str = "openai/whisper-base"
    encoder_output_dim: int = 512  # whisper-base hidden dim
    freeze_encoder: bool = True
    # Phase 2: unfreeze top N Whisper encoder layers for fine-tuning
    # whisper-base has 6 layers; whisper-small has 12
    unfreeze_top_layers: int = 0  # 0 = fully frozen; 2-4 = fine-tune top layers

    # --- Speaker embedding ---
    num_speakers: int = 16  # max distinct speakers + 1 for unknown
    speaker_embed_dim: int = 64

    # --- Causal context transformer ---
    context_dim: int = 256
    context_layers: int = 4
    context_heads: int = 4
    context_ffn_multiplier: int = 4
    context_dropout: float = 0.1
    max_context_turns: int = 20  # max conversation history kept

    # --- Classification ---
    # Trigger types: 0=none, 1=direct, 2=contextual
    num_trigger_types: int = 3

    # --- Training ---
    # Weight for the auxiliary type-classification loss
    aux_loss_weight: float = 0.3
    # Upweight positive (trigger) samples to counter class imbalance (~1:4 ratio)
    pos_weight: float = 4.0
    # Label smoothing for BCE loss (0 = off; e.g. 0.1 prevents overconfident predictions)
    label_smoothing: float = 0.0
