"""
Pre-caching pass: encode every utterance in the corpus (incl. Sigma turns)
through the Whisper encoder and save the resulting embeddings to disk.

Run this once before training:
    python precompute_embeddings.py

Override any config value on the command line:
    python precompute_embeddings.py dataset.embedding_cache_dir=.cache/embeddings_v2
"""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_lib import EmbeddingCache, WakeupDataset, caching_collate
from wakeup_model import OfflineWakeupDetector, WakeupModelConfig

log = logging.getLogger(__name__)


class AllTurnsDataset(Dataset):
    """Thin wrapper exposing all_audio_items (incl. Sigma) as a Dataset."""

    def __init__(self, base: WakeupDataset):
        self.items = base.all_audio_items   # list of (utterance_id, Path)
        self._load_audio = base._load_audio

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        uid, path = self.items[idx]
        return {
            "current_audio": self._load_audio(path),
            "utterance_id": uid,
        }


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cwd = Path(get_original_cwd())
    log.info("Device: %s | CWD: %s", device, cwd)

    # ── model (encoder only) ────────────────────────────────────────────
    model_cfg = WakeupModelConfig(**dict(cfg.model))
    model = OfflineWakeupDetector(model_cfg)
    model.utterance_encoder.eval()
    model.utterance_encoder.to(device)
    log.info("Whisper encoder loaded. Encoder params: %dM", sum(
        p.numel() for p in model.utterance_encoder.parameters()) // 1_000_000)

    # ── dataset (audio-only mode, all turns) ───────────────────────────
    base_ds = WakeupDataset(
        text_corpora_root=cwd / cfg.dataset.text_corpora_root,
        local_audio_root=cwd / cfg.dataset.local_audio_root,
        max_context_turns=cfg.dataset.max_context_turns,
        sample_rate=cfg.dataset.sample_rate,
        embedding_cache=None,
    )
    all_turns_ds = AllTurnsDataset(base_ds)
    log.info("Total turns to encode (incl. Sigma): %d", len(all_turns_ds))

    # ── cache ────────────────────────────────────────────────────────────
    cache = EmbeddingCache(
        cache_dir=cwd / cfg.dataset.embedding_cache_dir,
        dim=model_cfg.encoder_output_dim,
    )
    already_cached = sum(cache.has(uid) for uid, _ in base_ds.all_audio_items)
    log.info("Already cached: %d / %d", already_cached, len(all_turns_ds))
    if already_cached == len(all_turns_ds):
        log.info("Cache is complete — nothing to do.")
        return

    loader = DataLoader(
        all_turns_ds,
        batch_size=32,
        collate_fn=caching_collate,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── encode ───────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding utterances"):
            audio = batch["current_audio"].to(device)
            mask = batch["audio_mask"].to(device)
            embeddings = model.encode_utterance(audio, mask)  # (B, D)
            for uid, emb in zip(batch["utterance_id"], embeddings):
                if not cache.has(uid):
                    cache.save(uid, emb)

    coverage = cache.coverage([uid for uid, _ in base_ds.all_audio_items])
    log.info("Cache coverage: %.1f%%", coverage * 100)


if __name__ == "__main__":
    main()
