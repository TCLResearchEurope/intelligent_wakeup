"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Training script for OfflineWakeupDetector.

Usage:
    # Phase 1 (frozen encoder, fast):
    python train.py

    # Override config on CLI:
    python train.py training.batch_size=16 training.num_epochs=80

    # Phase 2 (unfreeze top 4 Whisper encoder layers at epoch 40):
    python train.py model.unfreeze_top_layers=4 training.phase2_start_epoch=40
"""

import logging
import random
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dataset_lib import (
    EmbeddingCache,
    WakeupDataset,
    compute_metrics,
    dataset_kwargs,
    log_metrics,
    training_collate,
)
from wakeup_model import (
    OfflineWakeupDetector,
    WakeupModelConfig,
    forward_batch,
)

log = logging.getLogger(__name__)


# ── Train / eval loops ───────────────────────────────────────────────────────


def train_one_epoch(  # pylint: disable=too-many-locals,too-many-arguments
    # pylint: disable=too-many-positional-arguments
    model: OfflineWakeupDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    schedule: tuple[float, int, int],
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    refresh_cache=None,
) -> float:
    """Run one training epoch.

    The scheduler is stepped once per optimiser step, not once per epoch:
    warmup is defined in steps, so stepping per epoch would hold the rate fixed
    for a whole epoch and leave the first one at zero.

    Args:
        model: The detector.
        loader: Training data loader.
        optimizer: Optimiser to step.
        device: Device to train on.
        schedule: (grad_clip, log_every, epoch) — the epoch number is used only
            for logging.
        scheduler: Stepped after each optimiser step.

    Returns:
        Mean loss over the epoch.
    """
    grad_clip, log_every, epoch = schedule
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        trigger_labels = batch["trigger_label"].to(device)
        type_labels = batch["type_label"].to(device)
        outputs = forward_batch(model, batch, device)

        losses = model.compute_loss(outputs, trigger_labels, type_labels)
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if refresh_cache is not None:
            # Phase 2 re-encodes the current turn every step, and that same turn
            # is context for the twenty that follow it. Writing the fresh vector
            # back means the history the model reads keeps pace with the encoder
            # being fine-tuned, instead of staying frozen at what the pre-training
            # encoder produced. Free: the embedding is already computed.
            for uid, emb in zip(batch["utterance_id"], outputs["current_embedding"]):
                refresh_cache.overlay(uid, emb)

        total_loss += loss.item()
        if (step + 1) % log_every == 0:
            log.info(
                "  epoch %d  step %d/%d  loss=%.4f  (trigger=%.4f  type=%.4f)",
                epoch,
                step + 1,
                len(loader),
                loss.item(),
                losses["trigger_loss"].item(),
                losses["type_loss"].item(),
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(  # pylint: disable=too-many-locals
    model: OfflineWakeupDetector,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Score the loader and return metrics at a fixed threshold.

    Args:
        model: The detector.
        loader: Validation data loader.
        device: Device to run on.
        threshold: Trigger probability above which a turn counts as predicted.

    Returns:
        Mapping of category name to its metrics.
    """
    model.eval()
    results = []

    for batch in loader:
        outputs = forward_batch(model, batch, device)
        scores = outputs["trigger_logit"].sigmoid()
        preds = (scores > threshold).long()
        expected = (batch["trigger_label"] > 0.5).tolist()

        for exp, pred, score, ttype in zip(
            expected, preds.tolist(), scores.tolist(), batch["trigger_type"]
        ):
            results.append(
                {
                    "expected": exp,
                    "predicted": bool(pred),
                    "score": score,
                    "trigger_type": ttype,
                }
            )

    # Log score distribution to diagnose threshold issues
    pos_scores = [r["score"] for r in results if r["expected"]]
    neg_scores = [r["score"] for r in results if not r["expected"]]
    if pos_scores:
        log.info(
            "  score dist — pos: mean=%.3f min=%.3f max=%.3f | neg: mean=%.3f",
            sum(pos_scores) / len(pos_scores),
            min(pos_scores),
            max(pos_scores),
            sum(neg_scores) / len(neg_scores) if neg_scores else 0,
        )

    return compute_metrics(results)


# ── Checkpointing ─────────────────────────────────────────────────────────────


class CheckpointManager:
    """Keeps the top-k checkpoints by validation F1 and deletes the rest.

    Args:
        checkpoint_dir: Where to write checkpoints.
        keep_k: How many to keep.
    """

    def __init__(self, checkpoint_dir: Path, keep_k: int = 3):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_k = keep_k
        self._history: list[tuple[float, Path]] = []  # (val_f1, path)

    def save(self, model: OfflineWakeupDetector, epoch: int, val_f1: float) -> Path:
        """Write a checkpoint and prune any that fall outside the top-k.

        Args:
            model: Model whose weights to save.
            epoch: Epoch number, used in the filename.
            val_f1: Validation F1, used for ranking and in the filename.

        Returns:
            Path the checkpoint was written to.
        """
        path = self.dir / f"epoch{epoch:03d}_f1{val_f1:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_f1": val_f1,
                "model_state_dict": model.state_dict(),
            },
            path,
        )
        self._history.append((val_f1, path))
        self._history.sort(key=lambda x: x[0], reverse=True)

        # Remove checkpoints beyond top-k
        while len(self._history) > self.keep_k:
            _, old_path = self._history.pop()
            if old_path.exists():
                old_path.unlink()
                log.info("Removed old checkpoint: %s", old_path.name)

        return path

    @property
    def best_f1(self) -> float:
        """Best validation F1 seen so far, or 0.0 before anything is saved."""
        return self._history[0][0] if self._history else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the detector and checkpoint the best epochs by validation F1.

    Args:
        cfg: Hydra config.

    Returns:
        None.
    """
    # pylint: disable=too-many-locals,too-many-statements
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Resolve all relative paths from the original working directory,
    # not Hydra's auto-created output directory.
    cwd = Path(get_original_cwd())

    # ── cache ────────────────────────────────────────────────────────────
    cache = EmbeddingCache(
        cache_dir=cwd / cfg.dataset.embedding_cache_dir,
        dim=WakeupModelConfig(**dict(cfg.model)).pooled_dim,
    )

    # ── datasets ─────────────────────────────────────────────────────────
    # Splits come from the dataset, which assigns them by group so a
    # scenario's _long sibling, _variantN re-takes and no_va_ twin never
    # straddle train and validation.
    log.info("Corpus: %s @ %s", cfg.dataset.repo_id, cfg.dataset.revision)
    ds_kwargs = {
        **dataset_kwargs(cfg.dataset),
        "embedding_cache": cache,
        "current_from_cache": True,  # Phase 1 by default
    }
    train_ds = WakeupDataset(split=cfg.dataset.train_split, **ds_kwargs)
    val_ds = WakeupDataset(split=cfg.dataset.val_split, **ds_kwargs)

    log.info(
        "Split: %d train / %d val conversations",
        len(train_ds.dataset),
        len(val_ds.dataset),
    )
    log.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    log.info("Train label stats: %s", train_ds.label_stats())
    log.info("Val   label stats: %s", val_ds.label_stats())

    # Warn if cache is incomplete
    all_ids = [uid for uid, _, _ in train_ds.all_audio_items + val_ds.all_audio_items]
    coverage = cache.coverage(all_ids)
    if coverage < 1.0:
        raise RuntimeError(
            f"Embedding cache coverage is {coverage*100:.1f}% — "
            "run `python precompute_embeddings.py` first, then re-run training."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=training_collate,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=training_collate,
        num_workers=cfg.training.num_workers,
    )

    # ── model ─────────────────────────────────────────────────────────────
    model_cfg = WakeupModelConfig(**dict(cfg.model))
    model = OfflineWakeupDetector(model_cfg).to(device)

    # The cached context embeddings were produced by the pooling head recorded
    # next to them. Phase 2 encodes the current turn with the model's own head,
    # so unless it is this one the two halves of every turn token come from
    # different spaces. A resumed checkpoint's head wins over this, since it
    # descends from the cache's and may have been fine-tuned in Phase 2.
    if cache.load_pooling(model.utterance_encoder.pooling):
        log.info("Loaded the pooling head recorded with the embedding cache.")
    else:
        log.warning(
            "The embedding cache in %s has no pooling head recorded — it predates "
            "this being pinned. Phase 1 is unaffected (it never encodes audio), "
            "but Phase 2 would mix two different embedding spaces. Re-run "
            "precompute_embeddings.py into a fresh directory before fine-tuning.",
            cache.cache_dir,
        )

    resume_from = cfg.get("resume_from", None)
    start_epoch = 1
    if resume_from:
        ckpt_path = cwd / resume_from
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as exc:
            # Almost always an architecture mismatch rather than a corrupt
            # file, and the shapes alone do not say which knob moved.
            raise RuntimeError(
                f"{ckpt_path.name} does not match the current architecture.\n"
                f"Checkpoints are only loadable under the settings they were "
                f"trained with -- model.wake_embed_dim (now "
                f"{cfg.model.wake_embed_dim}), encoder_output_dim, context_dim, "
                f"speaker_embed_dim and max_context_turns all change the shapes."
                f"\n\n{exc}"
            ) from exc
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info(
            "Resumed from %s  (epoch=%d  val_f1=%.4f)",
            ckpt_path.name,
            ckpt.get("epoch", 0),
            ckpt.get("val_f1", 0),
        )
        # When resuming into Phase 2, trigger the switch on the very first epoch
        if cfg.training.phase2_start_epoch and cfg.model.unfreeze_top_layers > 0:
            OmegaConf.update(
                cfg, "training.phase2_start_epoch", start_epoch, merge=True
            )

    counts = model.parameter_count()
    log.info(
        "Model: %.1fM total, %.1fM trainable", counts["total_M"], counts["trainable_M"]
    )

    # ── optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = cfg.training.num_epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        # step + 1: LambdaLR evaluates this at construction with step 0, and a
        # factor of zero there would freeze the first batch -- and, when the
        # scheduler was stepped per epoch, the entire first epoch.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(
            0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_manager = CheckpointManager(
        cfg.training.checkpoint_dir, cfg.training.keep_best_k
    )

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.num_epochs + start_epoch):

        # Phase 2 transition
        if cfg.training.phase2_start_epoch and epoch == cfg.training.phase2_start_epoch:
            log.info(
                "=== Switching to Phase 2: unfreezing top %d Whisper encoder layers ===",
                cfg.model.unfreeze_top_layers,
            )
            model.utterance_encoder.unfreeze_top_layers(cfg.model.unfreeze_top_layers)
            for ds in (train_ds, val_ds):
                ds.current_from_cache = False  # load raw audio from now on
            # Rebuild loaders with updated dataset mode and the smaller
            # Phase 2 batch -- see configs/training/default.yaml.
            log.info(
                "Phase 2 batch size: %d (was %d)",
                cfg.training.phase2_batch_size,
                cfg.training.batch_size,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.training.phase2_batch_size,
                shuffle=True,
                collate_fn=training_collate,
                num_workers=cfg.training.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.training.phase2_batch_size,
                shuffle=False,
                collate_fn=training_collate,
                num_workers=cfg.training.num_workers,
            )
            # Lower LR for fine-tuning. base_lrs has to change too: LambdaLR
            # recomputes lr as base_lr * lambda on every step, so setting the
            # group's lr alone is overwritten by the very next step.
            scheduler.base_lrs = [cfg.training.phase2_lr for _ in scheduler.base_lrs]
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.training.phase2_lr
            counts = model.parameter_count()
            log.info(
                "Trainable params after phase 2 switch: %.1fM", counts["trainable_M"]
            )

        refresh = (
            cache
            if (
                cfg.training.get("phase2_refresh_context", False)
                and cfg.training.phase2_start_epoch
                and epoch >= cfg.training.phase2_start_epoch
            )
            else None
        )
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            (cfg.training.grad_clip, cfg.training.log_every_steps, epoch),
            scheduler,
            refresh,
        )
        if refresh is not None:
            log.info(
                "Context embeddings refreshed by the fine-tuned encoder: %d of %d turns",
                cache.overlay_size(),
                len(train_ds.all_audio_items),
            )
        log.info(
            "Epoch %d/%d — avg train loss: %.4f  lr: %.2e",
            epoch,
            cfg.training.num_epochs,
            avg_loss,
            optimizer.param_groups[0]["lr"],
        )

        if epoch % cfg.training.eval_every_epochs == 0:
            metrics = evaluate(model, val_loader, device)
            val_f1 = metrics.get("overall", {}).get("F1", 0.0)
            log_metrics(metrics, f"── Val epoch {epoch} ──")

            ckpt_path = ckpt_manager.save(model, epoch, val_f1)
            log.info(
                "Checkpoint saved: %s  (best F1 so far: %.4f)",
                ckpt_path.name,
                ckpt_manager.best_f1,
            )

    log.info("Training complete. Best val F1: %.4f", ckpt_manager.best_f1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
