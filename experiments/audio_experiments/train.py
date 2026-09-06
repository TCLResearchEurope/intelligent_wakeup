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
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_lib import (
    EmbeddingCache,
    WakeupDataset,
    caching_collate,
    training_collate,
)
from dataset_lib.dataset import discover_json_files
from wakeup_model import OfflineWakeupDetector, WakeupModelConfig

log = logging.getLogger(__name__)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """TP/FP/TN/FN + precision/recall/F1, broken down by trigger type."""
    buckets = defaultdict(list)
    for r in results:
        buckets["overall"].append(r)
        buckets[r["trigger_type"]].append(r)

    out = {}
    for name, group in buckets.items():
        tp = sum(r["expected"] and r["predicted"] for r in group)
        fp = sum(not r["expected"] and r["predicted"] for r in group)
        tn = sum(not r["expected"] and not r["predicted"] for r in group)
        fn = sum(r["expected"] and not r["predicted"] for r in group)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[name] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "precision": round(prec, 3), "recall": round(rec, 3),
                     "F1": round(f1, 3), "n": len(group)}
    return out


def log_metrics(metrics: dict, header: str) -> None:
    log.info(header)
    log.info("  %-18s  %4s %4s %4s %4s   prec   rec    F1", "category", "TP", "FP", "TN", "FN")
    log.info("  %s", "-" * 60)
    for name in ("overall", "direct", "contextual", "non-assistance"):
        if name not in metrics:
            continue
        m = metrics[name]
        log.info("  %-18s  %4d %4d %4d %4d   %.3f  %.3f  %.3f",
                 name, m["TP"], m["FP"], m["TN"], m["FN"],
                 m["precision"], m["recall"], m["F1"])


# ── Train / eval loops ───────────────────────────────────────────────────────

def train_one_epoch(
    model: OfflineWakeupDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    log_every: int,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    is_phase1 = True  # will be determined from first batch

    for step, batch in enumerate(loader):
        is_phase1 = "current_embedding" in batch

        trigger_labels = batch["trigger_label"].to(device)
        type_labels = batch["type_label"].to(device)
        context_embs = batch["context_embeddings"].to(device)
        context_spks = batch["context_speakers"].to(device)
        context_mask = batch["context_mask"].to(device)
        current_spk = batch["current_speaker_id"].to(device)

        if is_phase1:
            current_emb = batch["current_embedding"].to(device)
            outputs = model.forward_from_embeddings(
                current_emb, context_embs, current_spk, context_spks, context_mask
            )
        else:
            audio = batch["current_audio"].to(device)
            audio_mask = batch.get("audio_mask")
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            outputs = model.forward(
                audio, context_embs, current_spk, context_spks, context_mask, audio_mask
            )

        losses = model.compute_loss(outputs, trigger_labels, type_labels)
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        if (step + 1) % log_every == 0:
            log.info("  epoch %d  step %d/%d  loss=%.4f  (trigger=%.4f  type=%.4f)",
                     epoch, step + 1, len(loader), loss.item(),
                     losses["trigger_loss"].item(), losses["type_loss"].item())

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: OfflineWakeupDetector,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    results = []

    for batch in loader:
        is_phase1 = "current_embedding" in batch
        context_embs = batch["context_embeddings"].to(device)
        context_spks = batch["context_speakers"].to(device)
        context_mask = batch["context_mask"].to(device)
        current_spk = batch["current_speaker_id"].to(device)

        if is_phase1:
            current_emb = batch["current_embedding"].to(device)
            outputs = model.forward_from_embeddings(
                current_emb, context_embs, current_spk, context_spks, context_mask
            )
        else:
            audio = batch["current_audio"].to(device)
            audio_mask = batch.get("audio_mask")
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            outputs = model.forward(
                audio, context_embs, current_spk, context_spks, context_mask, audio_mask
            )

        scores = outputs["trigger_logit"].sigmoid()
        preds = (scores > threshold).long()
        expected = (batch["trigger_label"] > 0.5).tolist()

        for exp, pred, score, ttype in zip(
            expected, preds.tolist(), scores.tolist(), batch["trigger_type"]
        ):
            results.append({
                "expected": exp, "predicted": bool(pred),
                "score": score, "trigger_type": ttype,
            })

    # Log score distribution to diagnose threshold issues
    pos_scores = [r["score"] for r in results if r["expected"]]
    neg_scores = [r["score"] for r in results if not r["expected"]]
    if pos_scores:
        log.info("  score dist — pos: mean=%.3f min=%.3f max=%.3f | neg: mean=%.3f",
                 sum(pos_scores)/len(pos_scores), min(pos_scores), max(pos_scores),
                 sum(neg_scores)/len(neg_scores) if neg_scores else 0)

    return compute_metrics(results)


# ── Checkpointing ─────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, keep_k: int = 3):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_k = keep_k
        self._history: list[tuple[float, Path]] = []  # (val_f1, path)

    def save(self, model: OfflineWakeupDetector, epoch: int, val_f1: float) -> Path:
        path = self.dir / f"epoch{epoch:03d}_f1{val_f1:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "val_f1": val_f1,
            "model_state_dict": model.state_dict(),
        }, path)
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
        return self._history[0][0] if self._history else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Resolve all relative paths from the original working directory,
    # not Hydra's auto-created output directory.
    cwd = Path(get_original_cwd())

    # ── conversation-level train/val split ──────────────────────────────
    all_files = discover_json_files(cwd / cfg.dataset.text_corpora_root)
    rng = random.Random(cfg.dataset.split_seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * cfg.dataset.val_split))
    val_files, train_files = shuffled[:n_val], shuffled[n_val:]
    log.info("Split: %d train / %d val conversations", len(train_files), len(val_files))

    # ── cache ────────────────────────────────────────────────────────────
    cache = EmbeddingCache(
        cache_dir=cwd / cfg.dataset.embedding_cache_dir,
        dim=cfg.model.encoder_output_dim,
    )

    # ── datasets ─────────────────────────────────────────────────────────
    ds_kwargs = dict(
        local_audio_root=cwd / cfg.dataset.local_audio_root,
        max_context_turns=cfg.dataset.max_context_turns,
        sample_rate=cfg.dataset.sample_rate,
        embedding_cache=cache,
        current_from_cache=True,   # Phase 1 by default
    )
    train_ds = WakeupDataset(
        text_corpora_root=cwd / cfg.dataset.text_corpora_root,
        json_files=train_files, **ds_kwargs
    )
    val_ds = WakeupDataset(
        text_corpora_root=cwd / cfg.dataset.text_corpora_root,
        json_files=val_files, **ds_kwargs
    )

    log.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    log.info("Train label stats: %s", train_ds.label_stats())
    log.info("Val   label stats: %s", val_ds.label_stats())

    # Warn if cache is incomplete
    all_ids = [uid for uid, _ in train_ds.all_audio_items + val_ds.all_audio_items]
    coverage = cache.coverage(all_ids)
    if coverage < 1.0:
        raise RuntimeError(
            f"Embedding cache coverage is {coverage*100:.1f}% — "
            "run `python precompute_embeddings.py` first, then re-run training."
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=training_collate, num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=training_collate, num_workers=cfg.training.num_workers,
    )

    # ── model ─────────────────────────────────────────────────────────────
    model_cfg = WakeupModelConfig(**dict(cfg.model))
    model = OfflineWakeupDetector(model_cfg).to(device)

    resume_from = cfg.get("resume_from", None)
    start_epoch = 1
    if resume_from:
        ckpt_path = cwd / resume_from
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed from %s  (epoch=%d  val_f1=%.4f)",
                 ckpt_path.name, ckpt.get("epoch", 0), ckpt.get("val_f1", 0))
        # When resuming into Phase 2, trigger the switch on the very first epoch
        if cfg.training.phase2_start_epoch and cfg.model.unfreeze_top_layers > 0:
            OmegaConf.update(cfg, "training.phase2_start_epoch", start_epoch, merge=True)

    counts = model.parameter_count()
    log.info("Model: %.1fM total, %.1fM trainable", counts["total_M"], counts["trainable_M"])

    # ── optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = cfg.training.num_epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_manager = CheckpointManager(cfg.training.checkpoint_dir, cfg.training.keep_best_k)

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.num_epochs + start_epoch):

        # Phase 2 transition
        if cfg.training.phase2_start_epoch and epoch == cfg.training.phase2_start_epoch:
            log.info("=== Switching to Phase 2: unfreezing top %d Whisper encoder layers ===",
                     cfg.model.unfreeze_top_layers)
            model.utterance_encoder.unfreeze_top_layers(cfg.model.unfreeze_top_layers)
            for ds in (train_ds, val_ds):
                ds.current_from_cache = False  # load raw audio from now on
            # Rebuild loaders with updated dataset mode
            train_loader = DataLoader(
                train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                collate_fn=training_collate, num_workers=cfg.training.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                collate_fn=training_collate, num_workers=cfg.training.num_workers,
            )
            # Lower LR for fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.training.phase2_lr
            counts = model.parameter_count()
            log.info("Trainable params after phase 2 switch: %.1fM", counts["trainable_M"])

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            cfg.training.grad_clip, cfg.training.log_every_steps, epoch,
        )
        scheduler.step(epoch * len(train_loader))
        log.info("Epoch %d/%d — avg train loss: %.4f  lr: %.2e",
                 epoch, cfg.training.num_epochs, avg_loss,
                 optimizer.param_groups[0]["lr"])

        if epoch % cfg.training.eval_every_epochs == 0:
            metrics = evaluate(model, val_loader, device)
            val_f1 = metrics.get("overall", {}).get("F1", 0.0)
            log_metrics(metrics, f"── Val epoch {epoch} ──")

            ckpt_path = ckpt_manager.save(model, epoch, val_f1)
            log.info("Checkpoint saved: %s  (best F1 so far: %.4f)",
                     ckpt_path.name, ckpt_manager.best_f1)

    log.info("Training complete. Best val F1: %.4f", ckpt_manager.best_f1)


if __name__ == "__main__":
    main()
