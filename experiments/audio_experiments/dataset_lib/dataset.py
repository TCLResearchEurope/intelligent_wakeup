"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

WakeupDataset — one sample per evaluable (non-Sigma) turn.

Modes (controlled by embedding_cache + current_from_cache):

  No cache (pre-caching pass)
      Returns raw current audio only. The pre-caching script iterates
      dataset.all_audio_items (ALL turns incl. Sigma) to build the cache.

  Cache, current_from_cache=False  (Phase 2 / fine-tuning)
      Returns raw current audio + cached context embeddings.
      Encoder runs with gradient for the current turn only.

  Cache, current_from_cache=True  (Phase 1 / frozen encoder)
      Returns cached current embedding + cached context embeddings.
      The encoder never runs during training — fastest training path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .cache import EmbeddingCache
from .labels import (
    SPEAKER_PAD,
    TRIGGER_TYPE_MAP,
    build_speaker_map,
    label_turns,
)

# Absolute prefix that audio paths carry *inside the corpus JSON files*, stripped
# so the remainder can be re-rooted at a local copy. This is not a location to
# read from - it must keep matching whatever prefix was baked into the recorded
# metadata, so only override it if your corpus was written with a different one.
CORPUS_PATH_PREFIX = os.environ.get(
    "CORPUS_PATH_PREFIX", "/nas/projects/intelligent_wakeup/"
)


def _resolve_audio_path(audio_path: str, local_root: Path) -> Path:
    stripped = audio_path.removeprefix(CORPUS_PATH_PREFIX)
    return local_root / stripped


def discover_json_files(root: Path) -> list[Path]:
    """Return all conversation JSON files under root, sorted."""
    return sorted(
        p for p in root.rglob("*.json")
        if p.name not in ("corpora_version.json",)
        and "conversation" in json.loads(p.read_text())
    )


@dataclass
class TurnSample:
    """Metadata for one evaluable (non-Sigma) turn. Audio loaded lazily."""

    utterance_id: str   # e.g. "HomeTasks/couple/weekday_cooking/turn_003"
    audio_path: Path
    speaker_id: int

    # Preceding turns ordered oldest → newest (up to max_context_turns)
    context_ids: list[str] = field(default_factory=list)
    context_audio_paths: list[Path] = field(default_factory=list)
    context_speaker_ids: list[int] = field(default_factory=list)

    trigger_label: float = 0.0
    type_label: int = 0
    trigger_type: str = "non-assistance"


class WakeupDataset(Dataset):
    def __init__(
        self,
        text_corpora_root: Path,
        local_audio_root: Path,
        max_context_turns: int = 20,
        sample_rate: int = 16_000,
        # Supply an explicit list to enable conversation-level train/val splitting.
        # If None, all JSON files under text_corpora_root are used.
        json_files: Optional[list[Path]] = None,
        # Cache settings
        embedding_cache: Optional[EmbeddingCache] = None,
        # Phase 1 fast path: load current turn embedding from cache too.
        # Requires the cache to be fully populated (precompute_embeddings.py).
        current_from_cache: bool = False,
    ):
        self.local_audio_root = Path(local_audio_root)
        self.max_context_turns = max_context_turns
        self.sample_rate = sample_rate
        self.embedding_cache = embedding_cache
        self.current_from_cache = current_from_cache and (embedding_cache is not None)

        self.samples: list[TurnSample] = []
        # ALL turns (incl. Sigma) → needed by the pre-caching script
        self._all_audio_items: list[tuple[str, Path]] = []

        files = json_files if json_files is not None else discover_json_files(Path(text_corpora_root))
        for p in files:
            self._add_file(p)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _add_file(self, json_path: Path) -> None:
        corpus = json.loads(json_path.read_text())
        conv = corpus["conversation"]
        labelled = label_turns(conv)
        speaker_map = build_speaker_map(conv)

        scenario = corpus.get("scenario_type", json_path.parent.parent.name)
        variant_type = corpus.get("variant_type", json_path.parent.name)
        variant_name = corpus.get("variant_name", json_path.stem)
        conv_id = f"{scenario}/{variant_type}/{variant_name}"

        for idx, turn in enumerate(labelled):
            uid = f"{conv_id}/turn_{idx:03d}"
            ap = _resolve_audio_path(turn["audio_path"], self.local_audio_root)
            self._all_audio_items.append((uid, ap))  # register for pre-caching

            if turn["speaker"] == "Sigma":
                continue  # VA turns are not classified, but are used as context

            ctx_start = max(0, idx - self.max_context_turns)
            ctx_indices = list(range(ctx_start, idx))

            self.samples.append(TurnSample(
                utterance_id=uid,
                audio_path=ap,
                speaker_id=speaker_map.get(turn["speaker"], SPEAKER_PAD),
                context_ids=[f"{conv_id}/turn_{ci:03d}" for ci in ctx_indices],
                context_audio_paths=[
                    _resolve_audio_path(labelled[ci]["audio_path"], self.local_audio_root)
                    for ci in ctx_indices
                ],
                context_speaker_ids=[
                    speaker_map.get(labelled[ci]["speaker"], SPEAKER_PAD)
                    for ci in ctx_indices
                ],
                trigger_label=float(turn["expected"]),
                type_label=TRIGGER_TYPE_MAP[turn["trigger_type"]],
                trigger_type=turn["trigger_type"],
            ))

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load a WAV → mono float32 tensor resampled to self.sample_rate."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)  # (T, channels) → (T,)
        if sr != self.sample_rate:
            g = gcd(self.sample_rate, sr)
            mono = resample_poly(mono, self.sample_rate // g, sr // g).astype(np.float32)
        return torch.from_numpy(mono)  # (T_samples,)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        if self.embedding_cache is None:
            # Pre-caching pass: just current audio + metadata
            return {
                "current_audio": self._load_audio(s.audio_path),
                "current_speaker_id": torch.tensor(s.speaker_id),
                "trigger_label": torch.tensor(s.trigger_label),
                "type_label": torch.tensor(s.type_label, dtype=torch.long),
                "utterance_id": s.utterance_id,
                "trigger_type": s.trigger_type,
            }

        # --- context embeddings (always from cache) ---
        D = self.embedding_cache.dim
        C = self.max_context_turns
        n = len(s.context_ids)

        context_embeddings = torch.zeros(C, D)
        context_speakers = torch.zeros(C, dtype=torch.long)
        context_mask = torch.ones(C, dtype=torch.bool)  # True = pad

        fill_start = C - n  # left-pad: most recent context at position C-1
        for i, (uid, spk) in enumerate(zip(s.context_ids, s.context_speaker_ids)):
            emb = self.embedding_cache.load(uid)
            if emb is not None:
                context_embeddings[fill_start + i] = emb
            context_speakers[fill_start + i] = spk
            context_mask[fill_start + i] = False

        base = {
            "current_speaker_id": torch.tensor(s.speaker_id),
            "context_embeddings": context_embeddings,   # (C, D)
            "context_speakers": context_speakers,        # (C,)
            "context_mask": context_mask,                # (C,) bool
            "trigger_label": torch.tensor(s.trigger_label),
            "type_label": torch.tensor(s.type_label, dtype=torch.long),
            "utterance_id": s.utterance_id,
            "trigger_type": s.trigger_type,
        }

        if self.current_from_cache:
            # Phase 1: load current embedding from cache too — encoder never runs
            emb = self.embedding_cache.load(s.utterance_id)
            base["current_embedding"] = emb if emb is not None else torch.zeros(D)
        else:
            # Phase 2: load raw audio — encoder runs with gradient
            base["current_audio"] = self._load_audio(s.audio_path)

        return base

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def all_audio_items(self) -> list[tuple[str, Path]]:
        """All (utterance_id, audio_path) pairs including Sigma turns.
        Used by precompute_embeddings.py to build a complete cache."""
        return self._all_audio_items

    def label_stats(self) -> dict:
        from collections import Counter
        counts = Counter(s.trigger_type for s in self.samples)
        total = len(self.samples)
        return {k: {"n": v, "pct": round(100 * v / total, 1)} for k, v in counts.items()}
