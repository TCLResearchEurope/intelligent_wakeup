"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Threshold sweep over one or more checkpoints, reported in false accepts per hour.

F1 hides what an always-on detector is judged on: a split's false positives
divided by its audio hours is the number the product cares about, so FA/hour is
a column of the sweep rather than something computed afterwards.

Several checkpoints can be given at once, in which case their per-turn
probabilities are averaged before thresholding. The error analysis found both
failure modes are confident rather than borderline (FN median 0.001, FP median
0.941), and confident-but-wrong predictions tend not to agree across
checkpoints — so averaging can cancel them where a threshold cannot.

**The embedding cache must be the one the checkpoint was trained against.**
Its attention-pooling head is part of the cache, and a cache built by a
different precompute run is a different embedding space: scoring a model
against the wrong one cost 0.12 F1 in testing without any error being raised.

Usage:
    python evaluate.py checkpoint=checkpoints/epoch018_f10.4710.pt
    python evaluate.py checkpoints=[ckpt/a.pt,ckpt/b.pt] eval_on=test
"""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset_lib import (
    EmbeddingCache,
    WakeupDataset,
    compute_metrics,
    dataset_kwargs,
    log_metrics,
    training_collate,
)
from wakeup_model import OfflineWakeupDetector, WakeupModelConfig, forward_batch

log = logging.getLogger(__name__)


@torch.no_grad()
def score_models(models: list, loader, device) -> tuple[list[list[float]], list[dict]]:
    """Run every model over the loader in a single pass.

    One pass, not one per model: reading a turn means reading its context
    embeddings too, and over a networked cache that I/O dominates — three
    models scored separately took 70 minutes where one pass takes 25.

    Args:
        models: Detectors to score with.
        loader: Data loader to score.
        device: Device to run on.

    Returns:
        ``(per_model_probabilities, records)``; records carry ``expected`` and
        ``trigger_type`` and are shared by every model.
    """
    for model in models:
        model.eval()
    probs: list[list[float]] = [[] for _ in models]
    records = []
    for batch in loader:
        for i, model in enumerate(models):
            out = forward_batch(model, batch, device)
            probs[i].extend(out["trigger_logit"].sigmoid().tolist())
        for exp, ttype in zip(
            (batch["trigger_label"] > 0.5).tolist(), batch["trigger_type"]
        ):
            records.append({"expected": exp, "trigger_type": ttype})
    return probs, records


def sweep(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    records: list[dict],
    probs: list[float],
    hours: float,
    neg_hours: float,
    report_at: float | None = None,
) -> dict:
    """Print the sweep and return the best operating point.

    "Best" is F1, but both FA/h columns are printed for every threshold so a
    higher-threshold, lower-FA point can be chosen by eye.

    Two denominators, because two are in use and they differ by ~27%:
    ``FA/h`` divides by the whole recording, which is what an always-on
    detector is exposed to; ``FA/h_sp`` divides by the duration of the
    not-addressed-to-the-assistant turns only, which is what
    ``experiments/gpt_realtime`` reports. Compare like with like.

    Args:
        records: Per-turn ``expected``/``trigger_type`` records.
        probs: Per-turn trigger probability, aligned with ``records``.
        hours: Total session hours in the split.
        neg_hours: Hours of negative-turn speech in the split.

    Args (cont.):
        report_at: Threshold to print the per-category breakdown at. Pass the
            threshold chosen on validation when scoring test — reading the
            breakdown off test's own best threshold is selecting on the test
            set. Defaults to this sweep's best-F1 threshold.

    Returns:
        ``{"threshold", "F1", "fa_per_hour", "fa_per_hour_speech"}`` at the
        best-F1 threshold.
    """
    log.info(
        "  %6s  %4s %4s %4s %4s  %6s %6s %6s  %8s %8s",
        "thresh",
        "TP",
        "FP",
        "TN",
        "FN",
        "prec",
        "rec",
        "F1",
        "FA/hour",
        "FA/h_sp",
    )
    log.info("  %s", "-" * 77)

    best = {
        "threshold": 0.5,
        "F1": 0.0,
        "fa_per_hour": float("inf"),
        "fa_per_hour_speech": float("inf"),
    }
    for step in range(1, 20):
        t = round(step / 20, 2)
        results = [{**r, "predicted": p > t} for r, p in zip(records, probs)]
        m = compute_metrics(results)["overall"]
        fa = m["FP"] / hours if hours else float("nan")
        fa_sp = m["FP"] / neg_hours if neg_hours else float("nan")
        better = m["F1"] > best["F1"]
        log.info(
            "  %6.2f  %4d %4d %4d %4d  %6.3f %6.3f %6.3f  %8.1f %8.1f%s",
            t,
            m["TP"],
            m["FP"],
            m["TN"],
            m["FN"],
            m["precision"],
            m["recall"],
            m["F1"],
            fa,
            fa_sp,
            " <-- best F1" if better else "",
        )
        if better:
            best = {
                "threshold": t,
                "F1": m["F1"],
                "fa_per_hour": round(fa, 2),
                "fa_per_hour_speech": round(fa_sp, 2),
            }

    at = best["threshold"] if report_at is None else report_at
    results = [{**r, "predicted": p > at} for r, p in zip(records, probs)]
    chosen = compute_metrics(results)["overall"]
    log.info("")
    log_metrics(
        compute_metrics(results),
        f"Per-category breakdown at threshold={at:.2f}"
        f"{'' if report_at is None else ' (chosen on validation)'} — F1 "
        f"{chosen['F1']:.3f}, {chosen['FP'] / hours:.2f} FA/hour over "
        f"{hours:.2f} h of recording, {chosen['FP'] / neg_hours:.2f} over "
        f"{neg_hours:.2f} h of not-addressed speech:",
        log,
        show_n=True,
    )
    return best


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-statements
    """Score one or more checkpoints and report them at one threshold.

    With ``threshold_from`` set, the threshold is swept on that split and then
    applied unchanged to ``eval_on`` — the only defensible way to report a test
    number, and worth having in one process so the two cannot drift apart.

    Args:
        cfg: Hydra config. Requires ``checkpoints=[a.pt,b.pt]`` or
            ``checkpoint=``; ``eval_on`` selects the split to report on, and
            ``threshold_from`` the split to choose the threshold on.

    Returns:
        None.

    Raises:
        ValueError: If no checkpoint was given.
        RuntimeError: If the embedding cache does not cover a split.
    """
    # pylint: disable=too-many-locals
    paths = cfg.get("checkpoints", None) or (
        [cfg.checkpoint] if cfg.get("checkpoint", None) else None
    )
    if not paths:
        raise ValueError("Provide checkpoints=[a.pt,b.pt] or checkpoint=a.pt")

    cwd = Path(get_original_cwd())
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model_cfg = WakeupModelConfig(**dict(cfg.model))
    aliases = {"val": cfg.dataset.val_split, "train": cfg.dataset.train_split}

    def resolve(name: str) -> str:
        """Map the "val"/"train" shorthands to the configured split names."""
        return aliases.get(name, name)

    # A Phase 2 checkpoint fine-tuned the encoder, so its current turn has to be
    # encoded by the model, not read from a cache the frozen encoder wrote —
    # which is also how training measured it. Context turns still come from the
    # cache either way. Default follows the checkpoint's own config.
    from_audio = cfg.get("from_audio", None)
    if from_audio is None:
        from_audio = cfg.model.unfreeze_top_layers > 0

    cache = EmbeddingCache(
        cache_dir=cwd / cfg.dataset.embedding_cache_dir, dim=model_cfg.pooled_dim
    )
    log.info(
        "Current turn: %s", "encoded from audio" if from_audio else "read from cache"
    )

    def build(split_name: str):
        """Dataset, loader, recorded hours and not-addressed-speech hours."""
        dataset = WakeupDataset(
            split=split_name,
            embedding_cache=cache,
            current_from_cache=not from_audio,
            **dataset_kwargs(cfg.dataset),
        )
        covered = cache.coverage([uid for uid, _, _ in dataset.all_audio_items])
        if covered < 1.0:
            raise RuntimeError(
                f"Embedding cache coverage for '{split_name}' is {covered*100:.1f}% — "
                f"run `python precompute_embeddings.py splits=[{split_name}]` first."
            )
        total = sum(dataset.dataset["duration_seconds"]) / 3600.0
        negative = (
            sum(s.span[1] - s.span[0] for s in dataset.samples if s.trigger_label < 0.5)
            / 3600.0
        )
        log.info(
            "Split '%s': %d conversations, %.2f h recorded (%.2f h of it "
            "not-addressed speech), %d scored turns",
            split_name,
            len(dataset.dataset),
            total,
            negative,
            len(dataset),
        )
        loader = DataLoader(
            dataset,
            batch_size=8 if from_audio else 128,
            shuffle=False,
            collate_fn=training_collate,
            num_workers=0,
        )
        return loader, total, negative

    models = []
    for path in paths:
        model = OfflineWakeupDetector(model_cfg).to(device)
        ckpt = torch.load(cwd / path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(
            "  %-50s epoch=%3d val_f1=%.4f",
            path,
            ckpt.get("epoch", -1),
            ckpt.get("val_f1", 0),
        )
        models.append(model)

    report_at = cfg.get("report_threshold", None)
    chooser = cfg.get("threshold_from", None)
    if chooser:
        chooser = resolve(chooser)
        log.info("")
        log.info("=== choosing the threshold on '%s' ===", chooser)
        loader, hours, neg_hours = build(chooser)
        probs, records = score_models(models, loader, device)
        mean = [sum(col) / len(col) for col in zip(*probs)]
        report_at = sweep(records, mean, hours, neg_hours)["threshold"]
        log.info("Chosen threshold: %.2f", report_at)

    split = resolve(cfg.get("eval_on", "val"))
    log.info("")
    log.info("=== scoring '%s' ===", split)
    loader, hours, neg_hours = build(split)
    all_probs, records = score_models(models, loader, device)
    mean_probs = [sum(col) / len(col) for col in zip(*all_probs)]

    if len(all_probs) > 1:
        for path, probs in zip(paths, all_probs):
            log.info("")
            log.info("=== %s alone ===", path)
            sweep(records, probs, hours, neg_hours, report_at)
        log.info("")
        log.info("=== ensemble of %d (mean probability) ===", len(all_probs))
    else:
        log.info("")
    best = sweep(records, mean_probs, hours, neg_hours, report_at)
    log.info("")
    log.info(
        "Best: threshold=%.2f  F1=%.3f  FA/hour=%.2f (%.2f per hour of "
        "not-addressed speech)",
        best["threshold"],
        best["F1"],
        best["fa_per_hour"],
        best["fa_per_hour_speech"],
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
