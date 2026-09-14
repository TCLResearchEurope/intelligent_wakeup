"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Metrics shared by training and evaluation.

Kept in one place so the two can never drift: a threshold sweep that scored
turns differently from the training loop would make its "best threshold"
meaningless. The breakdown mirrors the GPT Realtime baseline
(``experiments/gpt_realtime``) so numbers from the two are comparable.
"""

from __future__ import annotations

import logging
from collections import defaultdict

CATEGORY_ORDER = ("overall", "direct", "contextual", "non-assistance")


def compute_metrics(results: list[dict]) -> dict:
    """TP/FP/TN/FN + precision/recall/F1, broken down by trigger type.

    Args:
        results: Per-turn records carrying ``expected``, ``predicted`` and
            ``trigger_type``.

    Returns:
        Mapping of category name to its metrics.
    """
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
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "F1": round(f1, 3),
            "n": len(group),
        }
    return out


def log_metrics(
    metrics: dict,
    header: str,
    log: logging.Logger | None = None,
    show_n: bool = False,
) -> None:
    """Print a metrics table.

    Args:
        metrics: Output of :func:`compute_metrics`.
        header: Line printed above the table.
        log: Logger to print through; defaults to this module's.
        show_n: Append the sample count for each category.

    Returns:
        None.
    """
    log = log or logging.getLogger(__name__)
    count_header = "     n" if show_n else ""
    log.info(header)
    log.info(
        "  %-18s  %4s %4s %4s %4s   prec   rec    F1%s",
        "category",
        "TP",
        "FP",
        "TN",
        "FN",
        count_header,
    )
    log.info("  %s", "-" * (62 if show_n else 60))
    for name in CATEGORY_ORDER:
        m = metrics.get(name)
        if m is None:
            continue
        log.info(
            "  %-18s  %4d %4d %4d %4d   %.3f  %.3f  %.3f%s",
            name,
            m["TP"],
            m["FP"],
            m["TN"],
            m["FN"],
            m["precision"],
            m["recall"],
            m["F1"],
            f"  {m['n']:4d}" if show_n else "",
        )
