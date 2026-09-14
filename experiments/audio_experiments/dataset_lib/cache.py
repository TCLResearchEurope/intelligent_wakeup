"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

File-based embedding cache: utterance_id (str) → Tensor (D,).

Used in Phase 1 training where the Whisper encoder is frozen.
All utterances are encoded once, saved to disk, then loaded
during training without ever touching the encoder again.

The attention-pooling head sits between the frozen Whisper encoder and the
cached vector, and it is randomly initialised. So a cache is only meaningful
alongside the head that produced it: two precompute runs with different random
heads give embeddings that differ by a few percent, and a model whose own head
differs from the cache's is reading a different space than it was trained on.
The head is therefore stored in the cache directory and loaded back by anything
that encodes audio itself -- see :meth:`EmbeddingCache.save_pooling`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class EmbeddingCache:
    """Maps an utterance id to its encoder embedding, one file per utterance.

    Args:
        cache_dir: Directory to store embeddings in.
        dim: Embedding width, used to size zero-fills for missing entries.
    """

    def __init__(self, cache_dir: Path, dim: int = 768):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._overlay: dict[str, torch.Tensor] = {}

    #: The pooling head that produced every embedding in this directory.
    POOLING_FILE = "pooling_head.pt"

    def _path(self, utterance_id: str) -> Path:
        # utterance_id contains slashes (e.g. "HomeTasks/couple/weekday_cooking/turn_003")
        safe = utterance_id.replace("/", "__")
        return self.cache_dir / f"{safe}.pt"

    def has(self, utterance_id: str) -> bool:
        """Return whether this utterance is already cached.

        Args:
            utterance_id: Utterance identifier.

        Returns:
            True if an embedding is on disk.
        """
        return self._path(utterance_id).exists()

    def save(self, utterance_id: str, embedding: torch.Tensor) -> None:
        """Write one embedding to disk.

        Args:
            utterance_id: Utterance identifier.
            embedding: The embedding to store.

        Returns:
            None.
        """
        # clone(), or torch.save writes the whole batch the row is a view of:
        # a 512-float vector then costs 64 KiB on disk instead of 2 KiB.
        torch.save(embedding.detach().cpu().clone(), self._path(utterance_id))

    def load(self, utterance_id: str) -> Optional[torch.Tensor]:
        """Read one embedding back.

        Args:
            utterance_id: Utterance identifier.

        Returns:
            The embedding, or None when it is not cached.
        """
        hot = self._overlay.get(utterance_id)
        if hot is not None:
            return hot
        p = self._path(utterance_id)
        return torch.load(p, weights_only=True) if p.exists() else None

    def is_empty(self) -> bool:
        """Return whether the directory holds no embeddings yet.

        Returns:
            True when nothing has been encoded into this cache.
        """
        return not any(p.name != self.POOLING_FILE for p in self.cache_dir.glob("*.pt"))

    def save_pooling(self, module: torch.nn.Module) -> None:
        """Record the pooling head that produced this cache.

        Args:
            module: The pooling module used during the encoder pass.

        Returns:
            None.
        """
        # to CPU first: saved straight off the GPU this file cannot be loaded
        # on a CPU-only machine without an explicit map_location.
        state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
        torch.save(state, self.cache_dir / self.POOLING_FILE)

    def pooling_state(self) -> Optional[dict]:
        """Return the recorded pooling head's state, if there is one.

        Returns:
            The state dict on CPU, or None when no head was recorded.
        """
        path = self.cache_dir / self.POOLING_FILE
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def load_pooling(self, module: torch.nn.Module) -> bool:
        """Restore the pooling head that produced this cache, if recorded.

        Args:
            module: The pooling module to load into.

        Returns:
            True if a head was found and loaded.
        """
        path = self.cache_dir / self.POOLING_FILE
        if not path.exists():
            return False
        module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return True

    def overlay(self, utterance_id: str, embedding: torch.Tensor) -> None:
        """Override one entry in memory, without touching disk.

        Used during Phase 2 to keep the context embeddings in step with the
        encoder that is being fine-tuned — see ``training.phase2_refresh_context``.
        In-memory rather than on disk because the cache on disk is shared between
        runs and must keep belonging to the pooling head recorded next to it.

        Args:
            utterance_id: Utterance identifier.
            embedding: The embedding to serve for it from now on.

        Returns:
            None.
        """
        self._overlay[utterance_id] = embedding.detach().cpu().clone()

    def overlay_size(self) -> int:
        """Return how many entries are currently overridden in memory."""
        return len(self._overlay)

    def coverage(self, utterance_ids: list[str]) -> float:
        """Fraction of IDs that are already cached."""
        if not utterance_ids:
            return 1.0
        return sum(self.has(uid) for uid in utterance_ids) / len(utterance_ids)
