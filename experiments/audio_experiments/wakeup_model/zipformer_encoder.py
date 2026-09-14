"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Zipformer utterance encoder — a drop-in alternative to the Whisper one.

Same contract as :class:`wakeup_model.encoder.UtteranceEncoder`: waveforms in,
one pooled vector per utterance out, with the same attention pooling on top so
everything downstream is unchanged.

Two things differ from Whisper and both matter here:

* **No fixed window.** Whisper pads every utterance to 30 s, so a 2 s turn costs
  the same as a 30 s one and the pooling has to learn to ignore 28 s of silence.
  Zipformer runs on the real length, which is a better fit for turn-level work.
* **Kaldi filterbank features**, not Whisper's log-mel — the pretrained ASR
  weights expect them, so the frontend is part of the encoder rather than a
  detail.

Pretrained weights come from the k2-fsa/icefall LibriSpeech recipes on the Hub.
The stack geometry is not stored with them, so each size is spelled out in
:data:`ZIPFORMER_SIZES`; a mismatch surfaces immediately as missing or
unexpected keys rather than as silently wrong numbers.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .config import WakeupModelConfig
from .encoder import AttentionPooling
from .zipformer import Conv2dSubsampling, Zipformer2, fbank

log = logging.getLogger(__name__)

#: Feature frontend, matching what the icefall recipes train with.
NUM_MEL_BINS = 80
FRAME_SHIFT_MS = 10.0
FRAME_LENGTH_MS = 25.0

#: Stack geometry per published size. `output_dim` is max(encoder_dim), which is
#: what Zipformer2 emits.
ZIPFORMER_SIZES = {
    "small": {
        "repo": "Zengwei/icefall-asr-librispeech-zipformer-small-cr-ctc-20241018",
        "output_dim": 256,
        "kwargs": {
            "downsampling_factor": (1, 2, 4, 8, 4, 2),
            "num_encoder_layers": (2, 2, 2, 2, 2, 2),
            "encoder_dim": (192, 256, 256, 256, 256, 256),
            "encoder_unmasked_dim": (192, 192, 192, 192, 192, 192),
            "query_head_dim": (32,),
            "pos_head_dim": (4,),
            "value_head_dim": (12,),
            "num_heads": (4, 4, 4, 8, 4, 4),
            "feedforward_dim": (512, 768, 768, 768, 768, 768),
            "cnn_module_kernel": (31, 31, 15, 15, 15, 31),
            "pos_dim": 48,
        },
        "embed": {"layer1_channels": 8, "layer2_channels": 32, "layer3_channels": 128},
    },
    # Same geometry as "small", different pretraining: AudioSet tagging rather
    # than LibriSpeech ASR. Device-directed detection mixes acoustic and
    # semantic cues, so a general audio-event prior is a genuinely different
    # starting point, not just another size.
    "small-audioset": {
        "repo": "marcoyang/icefall-audio-tagging-audioset-zipformer-small-2024-04-23",
        "output_dim": 256,
        "kwargs": {
            "downsampling_factor": (1, 2, 4, 8, 4, 2),
            "num_encoder_layers": (2, 2, 2, 2, 2, 2),
            "encoder_dim": (192, 256, 256, 256, 256, 256),
            "encoder_unmasked_dim": (192, 192, 192, 192, 192, 192),
            "query_head_dim": (32,),
            "pos_head_dim": (4,),
            "value_head_dim": (12,),
            "num_heads": (4, 4, 4, 8, 4, 4),
            "feedforward_dim": (512, 768, 768, 768, 768, 768),
            "cnn_module_kernel": (31, 31, 15, 15, 15, 31),
            "pos_dim": 48,
        },
        "embed": {"layer1_channels": 8, "layer2_channels": 32, "layer3_channels": 128},
    },
    "medium": {
        "repo": "Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018",
        "output_dim": 512,
        "kwargs": {
            "downsampling_factor": (1, 2, 4, 8, 4, 2),
            "num_encoder_layers": (2, 2, 3, 4, 3, 2),
            "encoder_dim": (192, 256, 384, 512, 384, 256),
            "encoder_unmasked_dim": (192, 192, 256, 256, 256, 192),
            "query_head_dim": (32,),
            "pos_head_dim": (4,),
            "value_head_dim": (12,),
            "num_heads": (4, 4, 4, 8, 4, 4),
            "feedforward_dim": (512, 768, 1024, 1536, 1024, 768),
            "cnn_module_kernel": (31, 31, 15, 15, 15, 31),
            "pos_dim": 48,
        },
        "embed": {"layer1_channels": 8, "layer2_channels": 32, "layer3_channels": 128},
    },
}


class ZipformerUtteranceEncoder(nn.Module):
    """Kaldi fbank → Conv2dSubsampling → Zipformer2 → attention pooling.

    Args:
        config: Model config; ``zipformer_size`` selects the stack and
            ``encoder_output_dim`` must equal that stack's output width.

    Raises:
        ValueError: If the configured size is unknown, or the configured
            output dim does not match it.
    """

    def __init__(self, config: WakeupModelConfig):
        super().__init__()
        size = config.zipformer_size
        if size not in ZIPFORMER_SIZES:
            raise ValueError(
                f"zipformer_size={size!r}; known sizes: {sorted(ZIPFORMER_SIZES)}"
            )
        spec = ZIPFORMER_SIZES[size]
        if config.encoder_output_dim != spec["output_dim"]:
            raise ValueError(
                f"zipformer-{size} emits {spec['output_dim']}-d, but "
                f"model.encoder_output_dim is {config.encoder_output_dim}."
            )

        self.encoder_embed = Conv2dSubsampling(
            in_channels=NUM_MEL_BINS,
            out_channels=spec["kwargs"]["encoder_dim"][0],
            **spec["embed"],
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=2, causal=False, **spec["kwargs"]
        )
        self.pooling = AttentionPooling(config.encoder_output_dim, config.pooling)
        self.max_samples = int(config.max_utterance_seconds * 16000)
        self.min_samples = int(config.min_utterance_seconds * 16000)

        if config.zipformer_pretrained:
            self._load_pretrained(spec["repo"])
        self._apply_freeze(config.freeze_encoder, config.unfreeze_top_layers)

    def _load_pretrained(self, repo: str) -> None:
        """Load published ASR weights into the encoder.

        Args:
            repo: Hub repo holding ``exp/pretrained.pt``.

        Returns:
            None.

        Raises:
            RuntimeError: If the checkpoint does not match the stack geometry.
        """
        # pylint: disable=import-outside-toplevel
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "exp/pretrained.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        for prefix, module in (
            ("encoder.", self.encoder),
            ("encoder_embed.", self.encoder_embed),
        ):
            part = {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }
            missing, unexpected = module.load_state_dict(part, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"{repo} does not match this stack geometry under {prefix!r}: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected "
                    f"(first: {(missing + unexpected)[:3]})"
                )
        log.info("Loaded pretrained Zipformer weights from %s", repo)

    def _apply_freeze(self, freeze: bool, unfreeze_top: int) -> None:
        """Freeze the encoder, optionally leaving the top layers trainable.

        Args:
            freeze: Whether to freeze at all.
            unfreeze_top: How many of the last layers to leave trainable.

        Returns:
            None.
        """
        if not freeze:
            return
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_embed.parameters():
            param.requires_grad = False
        if unfreeze_top > 0:
            self.unfreeze_top_layers(unfreeze_top)

    def unfreeze_top_layers(self, n: int) -> None:
        """Unfreeze the last ``n`` encoder layers.

        Zipformer is a stack of stacks, so "the last n layers" is counted over
        the flattened sequence of layers from the output end — the closest
        equivalent to what this means for Whisper.

        Args:
            n: How many layers to unfreeze.

        Returns:
            None.
        """
        # Downsampled stacks wrap the real encoder one level down, so the
        # layers have to be gathered rather than read off `.layers` directly.
        layers = []
        for stack in self.encoder.encoders:
            inner = getattr(stack, "encoder", stack)
            layers.extend(getattr(inner, "layers", []))
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(  # pylint: disable=too-many-locals
        self,
        input_values: torch.Tensor,  # (B, T_samples) 16 kHz
        attention_mask: torch.Tensor = None,  # (B, T_samples) bool, True = real
    ) -> torch.Tensor:  # (B, encoder_output_dim)
        """Encode raw waveforms into one embedding each.

        Args:
            input_values: (B, T_samples) 16 kHz audio.
            attention_mask: (B, T_samples) bool, True where a sample is real.
                Used to give each utterance its true length, which is the point
                of this encoder — unlike Whisper there is no fixed window.

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

        # fbank is per-utterance: it takes (1, T) and Kaldi's framing is defined
        # on a single signal, so batching here would mean framing the padding too.
        feats = []
        for wav, length in zip(input_values, lengths.tolist()):
            # Truncate like Whisper does -- see config.max_utterance_seconds.
            usable = wav[: min(int(length), self.max_samples)]
            if usable.numel() < self.min_samples:
                # Too short for the conv frontend, which would return zero
                # frames -- see config.min_utterance_seconds.
                usable = torch.nn.functional.pad(
                    usable, (0, self.min_samples - usable.numel())
                )
            usable = usable.unsqueeze(0).to(torch.float32)
            feats.append(
                fbank(
                    usable,
                    num_mel_bins=NUM_MEL_BINS,
                    frame_shift=FRAME_SHIFT_MS,
                    frame_length=FRAME_LENGTH_MS,
                    dither=0.0,
                    energy_floor=1e-10,
                    snip_edges=False,
                    sample_frequency=16000,
                )
            )

        feat_lens = torch.tensor([f.shape[0] for f in feats], device=device)
        padded = torch.zeros(
            len(feats), int(feat_lens.max()), NUM_MEL_BINS, device=device
        )
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f.to(device)

        embedded, embedded_lens = self.encoder_embed(padded, feat_lens)
        # Zipformer2 works time-first.
        encoder_out, out_lens = self.encoder(
            embedded.permute(1, 0, 2),
            embedded_lens,
            src_key_padding_mask=self._padding_mask(embedded_lens),
        )
        hidden = encoder_out.permute(1, 0, 2)  # (B, T', D)

        # AttentionPooling takes True == padding, so this is the complement of
        # "valid frame".
        frames = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
        return self.pooling(hidden, frames >= out_lens.unsqueeze(1))

    @staticmethod
    def _padding_mask(lengths: torch.Tensor) -> torch.Tensor:
        """Build the True-is-padding mask Zipformer2 expects.

        Args:
            lengths: (B,) valid frame counts.

        Returns:
            (B, T) bool mask, True where a frame is padding.
        """
        frames = torch.arange(int(lengths.max()), device=lengths.device)
        return frames.unsqueeze(0) >= lengths.unsqueeze(1)
