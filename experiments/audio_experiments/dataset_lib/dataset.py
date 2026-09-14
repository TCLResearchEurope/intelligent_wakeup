"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

WakeupDataset — one sample per evaluable (non-Sigma) turn, sourced from the
published corpus on the Hugging Face Hub.

Turns are cut out of the session render at their recorded onsets, so a turn's
audio carries the room tone, ambience and speaker overlap it was mixed with.
Earlier revisions of this experiment read clean per-utterance wavs from a local
NAS copy; those were never published, and results are not comparable.

Splits come from the dataset itself and are assigned by group, keeping a
scenario's ``_long`` sibling, ``_variantN`` re-takes and ``no_va_`` twin
together. Do not re-split on conversation id: those near-duplicates share cast,
voices and room tone, so splitting them apart puts effectively-seen audio in
validation.

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

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset

from .cache import EmbeddingCache
from .hf_corpus import (
    ASSISTANT_SPEAKER,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    SessionAudio,
    load_split,
    turn_spans,
)
from .labels import (
    SPEAKER_PAD,
    TRIGGER_TYPE_MAP,
    build_speaker_map,
    has_wake_word,
    label_turns,
)


@dataclass
class TurnSample:  # pylint: disable=too-many-instance-attributes
    """Metadata for one evaluable (non-Sigma) turn. Audio sliced lazily."""

    utterance_id: str  # e.g. "HomeTasks/couple/weekday_cooking/turn_003"
    row_index: int  # which conversation in the split
    span: tuple[float, float]
    speaker_id: int

    # Preceding turns ordered oldest → newest (up to max_context_turns)
    context_ids: list[str] = field(default_factory=list)
    context_speaker_ids: list[int] = field(default_factory=list)
    wake: int = 0
    context_wakes: list[int] = field(default_factory=list)

    trigger_label: float = 0.0
    type_label: int = 0
    trigger_type: str = "non-assistance"


class WakeupDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    """One sample per non-assistant turn of one split of the corpus.

    Args:
        split: ``train``, ``validation`` or ``test``.
        repo_id: Hub dataset id.
        revision: Dataset revision; v1.0.1 or later is required.
        max_context_turns: How many preceding turns to carry as context.
        sample_rate: Rate to decode audio at.
        embedding_cache: Cache of utterance embeddings, or None for the
            pre-caching pass.
        current_from_cache: Phase 1 fast path — take the current turn's
            embedding from the cache too, so the encoder never runs. Requires a
            fully populated cache.
        session_cache_size: Decoded sessions to keep in memory. One decode
            serves every turn of a conversation, so this only matters when
            turns are visited out of order.
        dataset: An already-loaded split, to avoid loading it twice.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        split: str = "train",
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = DEFAULT_REVISION,
        max_context_turns: int = 20,
        sample_rate: int = 16_000,
        embedding_cache: Optional[EmbeddingCache] = None,
        current_from_cache: bool = False,
        session_cache_size: int = 2,
        dataset=None,
        wake_word_source: str = "none",
    ):
        self.split = split
        # "oracle" reads the flag off the corpus transcript. That is ground
        # truth no deployed system has, so it measures the ceiling of the
        # feature rather than a shippable result. "none" zeroes it.
        self.wake_word_source = wake_word_source
        self.wake_flags = wake_word_source == "oracle"
        self.max_context_turns = max_context_turns
        self.sample_rate = sample_rate
        self.embedding_cache = embedding_cache
        self.current_from_cache = current_from_cache and (embedding_cache is not None)

        self.dataset = (
            dataset
            if dataset is not None
            else load_split(repo_id, revision, split, sample_rate)
        )
        self.audio = SessionAudio(self.dataset, sample_rate, session_cache_size)

        self.samples: list[TurnSample] = []
        # ALL turns (incl. Sigma) → needed by the pre-caching script. Kept in
        # conversation order so the caching pass decodes each session once.
        self._all_audio_items: list[tuple[str, int, tuple[float, float]]] = []
        # `id` is not unique in v1.0.1: five FitnessHealth/*/meal_planning ids
        # each name two different conversations, because two scenario
        # directories resolve to the same category and variant name. Left
        # alone, their turns collide in the embedding cache and one silently
        # overwrites the other. Counting occurrences keeps the first one's keys
        # unchanged -- so an existing cache stays valid -- and disambiguates
        # the rest.
        self._id_counts: Counter = Counter()
        self.duplicate_ids: dict[str, int] = {}

        for row_index in range(len(self.dataset)):
            self._add_conversation(row_index)

        self.duplicate_ids = {i: n for i, n in self._id_counts.items() if n > 1}

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _add_conversation(  # pylint: disable=too-many-locals
        self, row_index: int
    ) -> None:
        """Register every turn of one conversation.

        Args:
            row_index: Row index within the split.

        Returns:
            None.
        """
        # Only the metadata columns are touched here; the audio column stays
        # undecoded until a turn is actually asked for.
        row = self.dataset.select_columns(["id", "turns", "duration_seconds"])[
            row_index
        ]
        turns = row["turns"]
        if not turns:
            return

        labelled = label_turns(turns)
        speaker_map = build_speaker_map(turns)
        spans = turn_spans(turns, row["duration_seconds"])

        # See the note in __init__: a repeated id gets a "#n" suffix so its
        # turns cannot overwrite the first conversation's cache entries.
        conv_id = row["id"]
        seen = self._id_counts[conv_id]
        self._id_counts[conv_id] += 1
        if seen:
            conv_id = f"{conv_id}#{seen}"

        for idx, (turn, span) in enumerate(zip(labelled, spans)):
            content = turn.get("content")
            uid = f"{conv_id}/turn_{idx:03d}"
            self._all_audio_items.append((uid, row_index, span))

            if turn["speaker"] == ASSISTANT_SPEAKER:
                continue  # VA turns are not classified, but are used as context

            ctx_start = max(0, idx - self.max_context_turns)
            ctx_indices = list(range(ctx_start, idx))

            self.samples.append(
                TurnSample(
                    utterance_id=uid,
                    row_index=row_index,
                    span=span,
                    speaker_id=speaker_map.get(turn["speaker"], SPEAKER_PAD),
                    context_ids=[f"{conv_id}/turn_{ci:03d}" for ci in ctx_indices],
                    context_speaker_ids=[
                        speaker_map.get(labelled[ci]["speaker"], SPEAKER_PAD)
                        for ci in ctx_indices
                    ],
                    wake=int(self.wake_flags and has_wake_word(content)),
                    context_wakes=[
                        int(
                            self.wake_flags
                            and has_wake_word(labelled[ci].get("content"))
                        )
                        for ci in ctx_indices
                    ],
                    trigger_label=float(turn["expected"]),
                    type_label=TRIGGER_TYPE_MAP[turn["trigger_type"]],
                    trigger_type=turn["trigger_type"],
                )
            )

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, row_index: int, span: tuple[float, float]) -> torch.Tensor:
        """Cut one turn's audio out of its session.

        Args:
            row_index: Row index within the split.
            span: (start, end) in seconds.

        Returns:
            Mono float32 tensor at ``self.sample_rate``.
        """
        return self.audio.turn(row_index, span[0], span[1])

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:  # pylint: disable=too-many-locals
        s = self.samples[idx]

        if self.embedding_cache is None:
            # Pre-caching pass: just current audio + metadata
            return {
                "current_audio": self._load_audio(s.row_index, s.span),
                "current_speaker_id": torch.tensor(s.speaker_id),
                "trigger_label": torch.tensor(s.trigger_label),
                "type_label": torch.tensor(s.type_label, dtype=torch.long),
                "utterance_id": s.utterance_id,
                "trigger_type": s.trigger_type,
            }

        # --- context embeddings (always from cache) ---
        dim = self.embedding_cache.dim
        n_slots = self.max_context_turns
        n = len(s.context_ids)

        context_embeddings = torch.zeros(n_slots, dim)
        context_speakers = torch.zeros(n_slots, dtype=torch.long)
        context_mask = torch.ones(n_slots, dtype=torch.bool)  # True = pad

        # left-pad: most recent context sits at the last slot
        context_wakes = torch.zeros(n_slots, dtype=torch.long)

        fill_start = n_slots - n
        for i, wake in enumerate(s.context_wakes):
            context_wakes[fill_start + i] = wake
        for i, (uid, spk) in enumerate(zip(s.context_ids, s.context_speaker_ids)):
            emb = self.embedding_cache.load(uid)
            if emb is not None:
                context_embeddings[fill_start + i] = emb
            context_speakers[fill_start + i] = spk
            context_mask[fill_start + i] = False

        base = {
            "current_speaker_id": torch.tensor(s.speaker_id),
            "context_embeddings": context_embeddings,  # (C, D)
            "context_speakers": context_speakers,  # (C,)
            "context_mask": context_mask,  # (C,) bool
            "trigger_label": torch.tensor(s.trigger_label),
            "type_label": torch.tensor(s.type_label, dtype=torch.long),
            "utterance_id": s.utterance_id,
            "trigger_type": s.trigger_type,
            "current_wake": torch.tensor(s.wake, dtype=torch.long),
            "context_wakes": context_wakes,
        }

        if self.current_from_cache:
            # Phase 1: load current embedding from cache too — encoder never runs
            emb = self.embedding_cache.load(s.utterance_id)
            base["current_embedding"] = emb if emb is not None else torch.zeros(dim)
        else:
            # Phase 2: slice raw audio — encoder runs with gradient
            base["current_audio"] = self._load_audio(s.row_index, s.span)

        return base

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def all_audio_items(self) -> list[tuple[str, int, tuple[float, float]]]:
        """All (utterance_id, row_index, span) triples including Sigma turns.

        Ordered by conversation, so a sequential pass over them decodes each
        session once. Used by precompute_embeddings.py to build a complete
        cache.
        """
        return self._all_audio_items

    def label_stats(self) -> dict:
        """Count samples per trigger type.

        Returns:
            Mapping of trigger type to its count and percentage share.
        """
        counts = Counter(s.trigger_type for s in self.samples)
        total = len(self.samples)
        return {
            k: {"n": v, "pct": round(100 * v / total, 1)} for k, v in counts.items()
        }


def dataset_kwargs(dataset_cfg) -> dict:
    """Map the ``dataset`` config block onto WakeupDataset's arguments.

    One definition for all three entry points, so adding a knob to the config
    cannot silently reach training but not evaluation.

    Args:
        dataset_cfg: The ``dataset`` config block.

    Returns:
        Keyword arguments for :class:`WakeupDataset`, without the split or the
        cache — callers supply those.
    """
    return {
        "wake_word_source": dataset_cfg.get("wake_word_source", "none"),
        "repo_id": dataset_cfg.repo_id,
        "revision": dataset_cfg.revision,
        "max_context_turns": dataset_cfg.max_context_turns,
        "sample_rate": dataset_cfg.sample_rate,
        "session_cache_size": dataset_cfg.session_cache_size,
    }
