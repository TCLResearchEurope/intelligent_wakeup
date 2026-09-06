"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

File-based embedding cache: utterance_id (str) → Tensor (D,).

Used in Phase 1 training where the Whisper encoder is frozen.
All utterances are encoded once, saved to disk, then loaded
during training without ever touching the encoder again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class EmbeddingCache:
    def __init__(self, cache_dir: Path, dim: int = 768):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim

    def _path(self, utterance_id: str) -> Path:
        # utterance_id contains slashes (e.g. "HomeTasks/couple/weekday_cooking/turn_003")
        safe = utterance_id.replace("/", "__")
        return self.cache_dir / f"{safe}.pt"

    def has(self, utterance_id: str) -> bool:
        return self._path(utterance_id).exists()

    def save(self, utterance_id: str, embedding: torch.Tensor) -> None:
        torch.save(embedding.cpu(), self._path(utterance_id))

    def load(self, utterance_id: str) -> Optional[torch.Tensor]:
        p = self._path(utterance_id)
        return torch.load(p, weights_only=True) if p.exists() else None

    def coverage(self, utterance_ids: list[str]) -> float:
        """Fraction of IDs that are already cached."""
        if not utterance_ids:
            return 1.0
        return sum(self.has(uid) for uid in utterance_ids) / len(utterance_ids)
