"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Threshold sweep evaluation for OfflineWakeupDetector.

Loads a checkpoint, runs the val split, and prints metrics at every threshold
from 0.05 to 0.95 in 0.05 steps. Highlights the threshold that maximises F1.

Usage:
    python evaluate.py checkpoint=checkpoints/epoch018_f10.4710.pt

    # Evaluate on the full dataset instead of val only:
    python evaluate.py checkpoint=checkpoints/epoch018_f10.4710.pt eval_on=all
"""

import logging
import random
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset_lib import EmbeddingCache, WakeupDataset, training_collate
from dataset_lib.dataset import discover_json_files
from wakeup_model import OfflineWakeupDetector, WakeupModelConfig

log = logging.getLogger(__name__)


# ── Metrics (same as train.py) ────────────────────────────────────────────────


def compute_metrics(results: list[dict]) -> dict:
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
        out[name] = {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "F1": round(f1, 3), "n": len(group),
        }
    return out


@torch.no_grad()
def collect_scores(model, loader, device) -> list[dict]:
    model.eval()
    records = []
    for batch in loader:
        context_embs = batch["context_embeddings"].to(device)
        context_spks = batch["context_speakers"].to(device)
        context_mask = batch["context_mask"].to(device)
        current_spk = batch["current_speaker_id"].to(device)
        current_emb = batch["current_embedding"].to(device)

        outputs = model.forward_from_embeddings(
            current_emb, context_embs, current_spk, context_spks, context_mask
        )
        scores = outputs["trigger_logit"].sigmoid()
        expected = (batch["trigger_label"] > 0.5).tolist()

        for exp, score, ttype in zip(expected, scores.tolist(), batch["trigger_type"]):
            records.append({"expected": exp, "score": score, "trigger_type": ttype})
    return records


def threshold_sweep(records: list[dict]) -> None:
    thresholds = [round(t / 20, 2) for t in range(1, 20)]  # 0.05 … 0.95

    header = f"  {'thresh':>6}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}  {'prec':>6} {'rec':>6} {'F1':>6}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        results = [{"expected": r["expected"], "predicted": r["score"] > t,
                    "trigger_type": r["trigger_type"]} for r in records]
        m = compute_metrics(results)["overall"]
        marker = " <-- best" if m["F1"] > best_f1 else ""
        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            best_thresh = t
        log.info("  %6.2f  %4d %4d %4d %4d  %6.3f %6.3f %6.3f%s",
                 t, m["TP"], m["FP"], m["TN"], m["FN"],
                 m["precision"], m["recall"], m["F1"], marker)

    log.info("")
    log.info("Best threshold: %.2f  →  F1=%.3f", best_thresh, best_f1)

    # Per-category breakdown at best threshold
    results = [{"expected": r["expected"], "predicted": r["score"] > best_thresh,
                "trigger_type": r["trigger_type"]} for r in records]
    metrics = compute_metrics(results)
    log.info("")
    log.info("Per-category breakdown at threshold=%.2f:", best_thresh)
    log.info("  %-18s  %4s %4s %4s %4s   prec   rec    F1    n", "category", "TP", "FP", "TN", "FN")
    log.info("  " + "-" * 62)
    for name in ("overall", "direct", "contextual", "non-assistance"):
        if name not in metrics:
            continue
        m = metrics[name]
        log.info("  %-18s  %4d %4d %4d %4d   %.3f  %.3f  %.3f  %4d",
                 name, m["TP"], m["FP"], m["TN"], m["FN"],
                 m["precision"], m["recall"], m["F1"], m["n"])

    # Score distribution
    pos_scores = [r["score"] for r in records if r["expected"]]
    neg_scores = [r["score"] for r in records if not r["expected"]]
    log.info("")
    if pos_scores:
        log.info("Score distribution — pos (n=%d): mean=%.3f  min=%.3f  max=%.3f",
                 len(pos_scores), sum(pos_scores) / len(pos_scores),
                 min(pos_scores), max(pos_scores))
    if neg_scores:
        log.info("Score distribution — neg (n=%d): mean=%.3f  min=%.3f  max=%.3f",
                 len(neg_scores), sum(neg_scores) / len(neg_scores),
                 min(neg_scores), max(neg_scores))


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", None)
    eval_on = cfg.get("eval_on", "val")   # "val" | "train" | "all"

    if checkpoint is None:
        raise ValueError("Provide checkpoint=<path>  e.g.  checkpoint=checkpoints/epoch018_f10.4710.pt")

    cwd = Path(get_original_cwd())
    ckpt_path = cwd / checkpoint
    log.info("Loading checkpoint: %s", ckpt_path)

    device = torch.device("cpu")  # evaluation is fast on CPU

    # ── model ────────────────────────────────────────────────────────────
    model_cfg = WakeupModelConfig(**dict(cfg.model))
    model = OfflineWakeupDetector(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Checkpoint epoch=%d  val_f1=%.4f", ckpt.get("epoch", -1), ckpt.get("val_f1", 0))

    # ── dataset split (must match training) ──────────────────────────────
    all_files = discover_json_files(cwd / cfg.dataset.text_corpora_root)
    rng = random.Random(cfg.dataset.split_seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * cfg.dataset.val_split))
    val_files, train_files = shuffled[:n_val], shuffled[n_val:]

    if eval_on == "val":
        eval_files = val_files
    elif eval_on == "train":
        eval_files = train_files
    else:
        eval_files = shuffled  # all

    log.info("Evaluating on %s split: %d conversations", eval_on, len(eval_files))

    cache = EmbeddingCache(
        cache_dir=cwd / cfg.dataset.embedding_cache_dir,
        dim=cfg.model.encoder_output_dim,
    )
    ds = WakeupDataset(
        text_corpora_root=cwd / cfg.dataset.text_corpora_root,
        local_audio_root=cwd / cfg.dataset.local_audio_root,
        max_context_turns=cfg.dataset.max_context_turns,
        sample_rate=cfg.dataset.sample_rate,
        json_files=eval_files,
        embedding_cache=cache,
        current_from_cache=True,
    )
    log.info("Samples: %d   label stats: %s", len(ds), ds.label_stats())

    loader = DataLoader(
        ds, batch_size=128, shuffle=False,
        collate_fn=training_collate, num_workers=0,
    )

    # ── collect scores + sweep ────────────────────────────────────────────
    records = collect_scores(model, loader, device)
    log.info("")
    log.info("=== Threshold sweep (%d samples) ===", len(records))
    threshold_sweep(records)


if __name__ == "__main__":
    main()
