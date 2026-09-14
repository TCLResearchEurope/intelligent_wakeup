"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.
"""

from .cache import EmbeddingCache
from .collate import caching_collate, training_collate
from .dataset import WakeupDataset, dataset_kwargs
from .hf_corpus import (
    ASSISTANT_SPEAKER,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    SessionAudio,
    load_split,
    turn_spans,
)
from .metrics import compute_metrics, log_metrics
from .labels import (
    TRIGGER_TYPE_CONTEXTUAL,
    TRIGGER_TYPE_DIRECT,
    TRIGGER_TYPE_NONE,
)

__all__ = [
    "WakeupDataset",
    "dataset_kwargs",
    "EmbeddingCache",
    "SessionAudio",
    "load_split",
    "turn_spans",
    "compute_metrics",
    "log_metrics",
    "training_collate",
    "caching_collate",
    "ASSISTANT_SPEAKER",
    "DEFAULT_REPO_ID",
    "DEFAULT_REVISION",
    "TRIGGER_TYPE_NONE",
    "TRIGGER_TYPE_DIRECT",
    "TRIGGER_TYPE_CONTEXTUAL",
]
