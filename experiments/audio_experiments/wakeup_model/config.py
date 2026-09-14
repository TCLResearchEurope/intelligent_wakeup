"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

from dataclasses import dataclass


@dataclass
class WakeupModelConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters for :class:`OfflineWakeupDetector`.

    Mirrors ``configs/model/default.yaml``; Hydra builds it from that file.
    """

    # --- Encoder ---
    #: "whisper", "zipformer" or "hubert" — which per-turn encoder to use. Changes the
    #: embedding width and the feature frontend, so a cache and a checkpoint
    #: belong to one setting.
    encoder_type: str = "whisper"
    #: HuBERT checkpoint, when encoder_type is "hubert". Base is 768-d.
    hubert_model: str = "facebook/hubert-base-ls960"
    #: Zipformer stack size, when encoder_type is "zipformer".
    zipformer_size: str = "small"
    #: Start from the published icefall ASR weights rather than random ones.
    zipformer_pretrained: bool = True
    #: Longest utterance the Zipformer encoder looks at, in seconds.
    #:
    #: WhisperFeatureExtractor pads *and truncates* to a fixed 30 s window, so
    #: every Whisper run in this repo only ever saw the first 30 s of a turn.
    #: Zipformer has no such window, so without this it would see more of a long
    #: turn than Whisper did and the comparison would not be like-for-like.
    #:
    #: It is also what keeps memory bounded: cost scales with the longest turn
    #: in the batch, and this corpus has turns up to 97 s against a median of
    #: 5.6 s, so one of them padded a batch of 64 to 9,680 frames and OOM'd a
    #: 24 GB card. Only 0.12% of turns are affected by the cap.
    max_utterance_seconds: float = 30.0
    #: Shortest utterance the Zipformer encoder will encode, in seconds;
    #: anything briefer is zero-padded up to it.
    #:
    #: Conv2dSubsampling needs a minimum number of frames: feed it a 0.006 s
    #: turn and it returns *zero* output frames, so the pooling softmax runs
    #: over an entirely masked row and yields NaN — which then propagates
    #: through every batch that turn appears in as context. The v1.0.1 corpus
    #: has exactly two such turns (one of them zero-length), and they were
    #: enough to make the whole run NaN from epoch 1. Whisper never hit this
    #: because it pads everything to a fixed 30 s window.
    min_utterance_seconds: float = 0.3

    whisper_model: str = "openai/whisper-base"
    encoder_output_dim: int = 512  # whisper-base hidden dim
    freeze_encoder: bool = True
    # Phase 2: unfreeze top N Whisper encoder layers for fine-tuning
    # whisper-base has 6 layers; whisper-small has 12
    unfreeze_top_layers: int = 0  # 0 = fully frozen; 2-4 = fine-tune top layers

    # --- Speaker embedding ---
    num_speakers: int = 16  # max distinct speakers + 1 for unknown
    speaker_embed_dim: int = 64

    # Width of the wake-word presence embedding added to each turn token.
    # 0 disables the feature entirely and restores the original input width.
    wake_embed_dim: int = 0

    # How frame sequences collapse to one utterance vector:
    #   attention - learned softmax over frames (original)
    #   mean_max  - attention pooling concatenated with max over frames,
    #               doubling the output width. Attention averages, so a ~0.5 s
    #               keyword inside a 30 s window is diluted; max keeps peaks.
    # Changing this changes the cached embeddings -- use a separate cache dir.
    pooling: str = "attention"

    @property
    def pooled_dim(self) -> int:
        """Width of one pooled utterance embedding.

        Not the same as ``encoder_output_dim``: mean_max pooling concatenates
        two vectors, so everything sized by the embedding -- the cache, the
        context transformer's input -- must use this instead.

        Returns:
            Embedding width after pooling.
        """
        return self.encoder_output_dim * (2 if self.pooling == "mean_max" else 1)

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
