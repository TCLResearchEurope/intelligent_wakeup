"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Collate functions for DataLoader.

  training_collate  — Phase 1 and Phase 2.
                      Detects mode from batch keys:
                        "current_embedding" → Phase 1 (from cache, no audio)
                        "current_audio"     → Phase 2 (raw audio, encoder runs)

  caching_collate   — Pre-caching pass. Audio only, no context.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def training_collate(batch: list[dict]) -> dict:
    is_phase1 = "current_embedding" in batch[0]

    out: dict = {
        "current_speaker_id": torch.stack([s["current_speaker_id"] for s in batch]),
        "context_embeddings": torch.stack([s["context_embeddings"] for s in batch]),
        "context_speakers": torch.stack([s["context_speakers"] for s in batch]),
        "context_mask": torch.stack([s["context_mask"] for s in batch]),
        "trigger_label": torch.stack([s["trigger_label"] for s in batch]),
        "type_label": torch.stack([s["type_label"] for s in batch]),
        "utterance_id": [s["utterance_id"] for s in batch],
        "trigger_type": [s["trigger_type"] for s in batch],
    }

    if is_phase1:
        out["current_embedding"] = torch.stack([s["current_embedding"] for s in batch])
    else:
        audios = [s["current_audio"] for s in batch]
        max_len = max(a.shape[0] for a in audios)
        out["current_audio"] = torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in audios])
        out["audio_mask"] = torch.stack([
            F.pad(torch.ones(a.shape[0], dtype=torch.bool), (0, max_len - a.shape[0]))
            for a in audios
        ])

    return out


def caching_collate(batch: list[dict]) -> dict:
    audios = [s["current_audio"] for s in batch]
    max_len = max(a.shape[0] for a in audios)
    return {
        "current_audio": torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in audios]),
        "audio_mask": torch.stack([
            F.pad(torch.ones(a.shape[0], dtype=torch.bool), (0, max_len - a.shape[0]))
            for a in audios
        ]),
        "utterance_id": [s["utterance_id"] for s in batch],
    }
