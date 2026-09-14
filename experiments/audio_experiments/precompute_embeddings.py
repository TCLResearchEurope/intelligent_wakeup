"""
Pre-caching pass: encode every utterance in the corpus (incl. Sigma turns)
through the Whisper encoder and save the resulting embeddings to disk.

Turns are cut out of the session render on the fly, so the pass walks them in
conversation order: each session is decoded once and serves all of its turns.
Shuffling here would decode the same session once per turn, so don't.

Run this once before training, for each split you intend to use:
    python precompute_embeddings.py
    python precompute_embeddings.py splits=[train,validation]

Override any config value on the command line:
    python precompute_embeddings.py dataset.embedding_cache_dir=.cache/embeddings_v2

Pass ``checkpoint=`` to encode with a trained model's encoder rather than a
freshly initialised one. That is what inference outside this repo does — there
is no cache there, so every turn is encoded by the model itself — so it is also
how to check that a shipped checkpoint reproduces its reported numbers.
"""

import logging
import random
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_lib import (
    EmbeddingCache,
    WakeupDataset,
    caching_collate,
    dataset_kwargs,
)
from wakeup_model import OfflineWakeupDetector, WakeupModelConfig

log = logging.getLogger(__name__)


class AllTurnsDataset(Dataset):
    """The turns still to encode (incl. Sigma), as a Dataset.

    Items keep the corpus's conversation order, so a sequential DataLoader
    decodes each session once.

    Already-cached turns are dropped here rather than skipped at save time.
    Skipping later would still decode and slice every session, which is the
    expensive part — filling a handful of gaps would cost as much as building
    the cache from scratch.

    Args:
        base: The dataset whose turns to encode.
        cache: Existing cache, used to drop turns already in it.
    """

    def __init__(self, base: WakeupDataset, cache=None):
        items = base.all_audio_items  # (utterance_id, row_index, span)
        if cache is not None:
            items = [item for item in items if not cache.has(item[0])]
        self.items = items
        self._load_audio = base._load_audio

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        uid, row_index, span = self.items[idx]
        return {
            "current_audio": self._load_audio(row_index, span),
            "utterance_id": uid,
        }


def encode_split(  # pylint: disable=too-many-locals
    cfg: DictConfig, split: str, model, device, cache
) -> None:
    """Encode and cache every utterance of one split.

    Args:
        cfg: Hydra config.
        split: Split name to encode.
        model: The detector, used for its utterance encoder.
        device: Torch device.
        cache: Embedding cache to fill.

    Returns:
        None.
    """
    base_ds = WakeupDataset(
        split=split,
        embedding_cache=None,
        **dataset_kwargs(cfg.dataset),
    )
    ids = [uid for uid, _, _ in base_ds.all_audio_items]
    all_turns_ds = AllTurnsDataset(base_ds, cache)

    if base_ds.duplicate_ids:
        log.warning(
            "[%s] %d conversation ids are not unique in this revision and were "
            "disambiguated with a '#n' suffix: %s",
            split,
            len(base_ds.duplicate_ids),
            ", ".join(sorted(base_ds.duplicate_ids)),
        )

    log.info(
        "[%s] %d conversations, %d turns total (incl. Sigma); %d already cached, "
        "%d to encode (batch %d, %d workers)",
        split,
        len(base_ds.dataset),
        len(ids),
        len(ids) - len(all_turns_ds),
        len(all_turns_ds),
        cfg.training.precompute_batch_size,
        cfg.training.num_workers,
    )
    if not all_turns_ds:
        log.info("[%s] cache is complete — nothing to do.", split)
        return

    loader = DataLoader(
        all_turns_ds,
        batch_size=cfg.training.precompute_batch_size,
        shuffle=False,  # conversation order — see module docstring
        collate_fn=caching_collate,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {split}"):
            audio = batch["current_audio"].to(device)
            mask = batch["audio_mask"].to(device)
            embeddings = model.encode_utterance(audio, mask)  # (B, D)
            for uid, emb in zip(batch["utterance_id"], embeddings):
                if not cache.has(uid):
                    cache.save(uid, emb)

    log.info("[%s] cache coverage: %.1f%%", split, cache.coverage(ids) * 100)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Fill the embedding cache for every configured split.

    Args:
        cfg: Hydra config. ``splits`` overrides which splits are encoded.

    Returns:
        None.
    """
    # The pooling head is randomly initialised, so an unseeded pass would give
    # this cache a different embedding space from every other one. Seeded here
    # with the same value train.py uses, and pinned to the cache below.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cwd = Path(get_original_cwd())
    log.info("Device: %s | CWD: %s", device, cwd)
    log.info("Corpus: %s @ %s", cfg.dataset.repo_id, cfg.dataset.revision)

    # ── model (encoder only) ────────────────────────────────────────────
    model_cfg = WakeupModelConfig(**dict(cfg.model))
    model = OfflineWakeupDetector(model_cfg)

    # A trained encoder, if one was given. Loaded before the cache's own head is
    # considered, because a checkpoint's head is the one its context transformer
    # expects — see the pooling head section of README.md.
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint:
        ckpt = torch.load(cwd / checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(
            "Encoding with %s (epoch %d, val_f1 %.4f)",
            checkpoint,
            ckpt.get("epoch", -1),
            ckpt.get("val_f1", 0),
        )

    model.utterance_encoder.eval()
    model.utterance_encoder.to(device)
    log.info(
        "Whisper encoder loaded. Encoder params: %dM",
        sum(p.numel() for p in model.utterance_encoder.parameters()) // 1_000_000,
    )

    # ── cache ────────────────────────────────────────────────────────────
    cache = EmbeddingCache(
        cache_dir=cwd / cfg.dataset.embedding_cache_dir,
        dim=model_cfg.pooled_dim,
    )

    # A cache and the head that produced it are one artefact. Topping up an
    # existing cache must reuse its head; a new cache records the one it starts
    # with, so training and Phase 2 can encode into the same space later.
    if checkpoint:
        # The checkpoint's head wins, but it must be the one already in the
        # cache if the cache has any -- otherwise a resumed or topped-up run
        # would mix two spaces. Comparing beats forbidding: an interrupted
        # encode is normal and should be resumable.
        existing = cache.pooling_state()
        current = {
            k: v.detach().cpu()
            for k, v in model.utterance_encoder.pooling.state_dict().items()
        }
        if existing is None:
            if not cache.is_empty():
                raise RuntimeError(
                    f"{cache.cache_dir} holds embeddings but records no pooling "
                    f"head, so they were not produced by {checkpoint}. Encode "
                    f"into a fresh directory."
                )
            cache.save_pooling(model.utterance_encoder.pooling)
            log.info("Recorded the checkpoint's pooling head in %s", cache.cache_dir)
        elif any(not torch.equal(existing[k], current[k]) for k in current):
            raise RuntimeError(
                f"{cache.cache_dir} was built with a different pooling head "
                f"than {checkpoint} carries. Encode into a fresh directory."
            )
        else:
            log.info("Resuming %s — same pooling head.", cache.cache_dir)
    elif cache.load_pooling(model.utterance_encoder.pooling):
        log.info("Reusing the pooling head recorded in %s", cache.cache_dir)
    elif cache.is_empty():
        cache.save_pooling(model.utterance_encoder.pooling)
        log.info("Recorded this run's pooling head in %s", cache.cache_dir)
    else:
        # A cache from before the head was pinned. Stamping this run's head on
        # it would be a lie -- the embeddings in there came from a different,
        # unrecoverable one -- and a convincing one, since everything
        # downstream trusts the file.
        log.warning(
            "%s holds embeddings but no pooling head, so it predates the head "
            "being pinned and the one that produced it is unrecoverable. Topping "
            "it up mixes two embedding spaces, and fine-tuning against it mixes "
            "them inside every turn token. Encode into a fresh directory instead.",
            cache.cache_dir,
        )

    splits = cfg.get("splits", None) or [
        cfg.dataset.train_split,
        cfg.dataset.val_split,
    ]
    for split in splits:
        encode_split(cfg, split, model, device, cache)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
