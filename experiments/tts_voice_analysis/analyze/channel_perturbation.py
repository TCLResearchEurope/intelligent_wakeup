#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Channel perturbation control (experiment-setup.md, step 5): does changing the
recording channel move a voice's embedding by less than changing its identity?

For each perturbation type, measures "same voice, modified channel" - the cosine
distance between an utterance's original embedding and its own perturbed variant
- and compares that distribution against two phase-1 baselines:
  - within-elevenlabs: same voice, different (unperturbed) utterance
  - between-elevenlabs: different voice

If channel-shift distances sit well below the between-voice baseline (ideally
comparable to or below the within-voice baseline), embedding separation is
primarily identity, not channel.

Requires: features/extract_embeddings.py already run on both output/embeddings/ (the full
diversity set) and output/channel_variants_embeddings/ (channel/apply_perturbations.py's
output). See README.md.

Usage (from experiments/tts_voice_analysis/):
    python3 -m analyze.channel_perturbation
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

from lib.diversity_metrics import (  # pylint: disable=wrong-import-position
    between_group_distances,
    centroids_of,
    cosine_distance,
    fmt_stat,
    load_embeddings,
    summarize,
    within_group_distances,
)
from lib.plot_style import DOUBLE_COL_WIDTH_IN  # pylint: disable=wrong-import-position
from lib.tts_common import DEFAULT_OUTPUT_DIR  # pylint: disable=wrong-import-position

ENCODER_NAMES = ["ecapa", "wavlm_sv"]
PERTURBATION_DISPLAY_NAMES = {
    "bandwidth_limit": "Bandwidth Limit",
    "eq_tilt": "EQ Tilt",
    "codec_gsm": "GSM Codec",
    "codec_opus": "Opus Codec",
    "noise_15db": "Noise (15 dB)",
    "noise_5db": "Noise (5 dB)",
    "reverb": "Reverb",
}


def flatten(by_group: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Flatten a per-group embedding mapping into a single per-utterance mapping.

    Variant pairing is done by utterance_id, so the group/voice structure is not
    needed for the lookup - this collapses {group: {uid: vec}} down to {uid: vec}.

    Args:
        by_group: Mapping from group name to a mapping of utterance id to embedding
            vector.

    Returns:
        Mapping from utterance id directly to embedding vector, across all groups.
    """
    return {
        uid: vec for utterances in by_group.values() for uid, vec in utterances.items()
    }


def load_variant_records(variants_dir: Path) -> list[dict]:
    """Load the flat index of channel-perturbation variant records.

    Args:
        variants_dir: Directory containing the index.json written by
            channel/apply_perturbations.py.

    Returns:
        List of variant record dicts, one per perturbed utterance.
    """
    return json.loads((variants_dir / "index.json").read_text(encoding="utf-8"))


def same_voice_channel_shift(
    variant_records: list[dict],
    original_vectors: dict[str, np.ndarray],
    variant_vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute, per perturbation type, the distance between each utterance and its variant.

    Args:
        variant_records: Variant records loaded from index.json, each linking a
            perturbed utterance id back to its original utterance id and
            perturbation type.
        original_vectors: Mapping from original utterance id to its embedding vector.
        variant_vectors: Mapping from perturbed utterance id to its embedding vector.

    Returns:
        Mapping from perturbation type to an array of cosine distances between each
        original utterance and its own perturbed variant.
    """
    by_perturbation: dict[str, list[float]] = {}
    for rec in variant_records:
        orig_id, uid, pert = (
            rec["original_utterance_id"],
            rec["utterance_id"],
            rec["perturbation"],
        )
        if orig_id not in original_vectors or uid not in variant_vectors:
            continue
        d = cosine_distance(original_vectors[orig_id], variant_vectors[uid])
        by_perturbation.setdefault(pert, []).append(d)
    return {p: np.array(v) for p, v in by_perturbation.items()}


def plot_comparison(
    channel_shifts: dict[str, np.ndarray],
    within_voice: np.ndarray,
    between_voice: np.ndarray,
    out_path: Path,
) -> None:
    """Plot a boxplot of each perturbation's channel-shift distances against baselines.

    Args:
        channel_shifts: Mapping from perturbation type to an array of channel-shift
            cosine distances.
        within_voice: Array of intra-speaker baseline cosine distances.
        between_voice: Array of inter-speaker baseline cosine distances.
        out_path: Path to write the PNG (a matching PDF is written alongside it).
    """
    labels = [PERTURBATION_DISPLAY_NAMES.get(p, p) for p in channel_shifts] + [
        "Intra-Speaker\n(Baseline)",
        "Inter-Speaker\n(Baseline)",
    ]
    data = list(channel_shifts.values()) + [within_voice, between_voice]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH_IN, 3.6))
    box = ax.boxplot(data, tick_labels=labels, showmeans=True)
    n_pert = len(channel_shifts)
    for patch_idx in range(n_pert, len(data)):
        box["boxes"][patch_idx].set_color("firebrick")
    ax.set_ylabel("Cosine Distance")
    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def analyze_encoder(
    encoder: str,
    embeddings_dir: Path,
    variants_dir: Path,
    variant_embeddings_dir: Path,
    output_dir: Path,
) -> dict:
    """Run the channel-perturbation comparison for one encoder and write its outputs.

    Args:
        encoder: Name of the speaker-embedding encoder to analyze.
        embeddings_dir: Directory with embeddings for the full diversity set.
        variants_dir: Directory with the channel-perturbation variant records.
        variant_embeddings_dir: Directory with embeddings for the perturbed variants.
        output_dir: Directory to write the per-encoder summary JSON and plot to.

    Returns:
        Summary dict with the within/between-voice baselines and channel-shift stats
        per perturbation type, including whether each stays below the between-voice
        baseline.
    """
    el_by_group = load_embeddings(embeddings_dir, encoder)
    original_vectors = flatten(el_by_group)
    variant_by_group = load_embeddings(variant_embeddings_dir, encoder)
    variant_vectors = flatten(variant_by_group)
    variant_records = load_variant_records(variants_dir)

    within_voice = within_group_distances(el_by_group)
    between_voice = between_group_distances(centroids_of(el_by_group))
    channel_shifts = same_voice_channel_shift(
        variant_records, original_vectors, variant_vectors
    )

    summary = {
        "encoder": encoder,
        "within_voice_baseline": summarize(within_voice),
        "between_voice_baseline": summarize(between_voice),
        "channel_shift_by_perturbation": {
            p: summarize(d) for p, d in channel_shifts.items()
        },
    }
    for p, d in channel_shifts.items():
        summary["channel_shift_by_perturbation"][p]["below_between_voice_baseline"] = (
            bool(np.mean(d) < between_voice.mean()) if len(between_voice) else None
        )

    with open(
        output_dir / f"{encoder}_channel_perturbation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_comparison(
        channel_shifts,
        within_voice,
        between_voice,
        out_path=output_dir / f"{encoder}_channel_perturbation.png",
    )
    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable console summary for one encoder's results.

    Args:
        summary: Summary dict produced by analyze_encoder.
    """
    print(f"\n=== {summary['encoder']} ===")
    print(
        f"  within-voice baseline:  mean={fmt_stat(summary['within_voice_baseline'], 'mean')}"
    )
    print(
        f"  between-voice baseline: mean={fmt_stat(summary['between_voice_baseline'], 'mean')}"
    )
    for p, s in summary["channel_shift_by_perturbation"].items():
        flag = (
            "OK (below between-voice)"
            if s.get("below_between_voice_baseline")
            else "WARNING (>= between-voice!)"
        )
        print(
            f"  {p:15s} mean={fmt_stat(s, 'mean')}  median={fmt_stat(s, 'median')}  "
            f"n={s.get('n')}   {flag}"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--embeddings-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "embeddings"
    )
    parser.add_argument(
        "--variants-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "channel_variants"
    )
    parser.add_argument(
        "--variant-embeddings-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "channel_variants_embeddings",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis"
    )
    parser.add_argument(
        "--encoders", nargs="+", choices=ENCODER_NAMES, default=ENCODER_NAMES
    )
    return parser.parse_args()


def main() -> None:
    """Run the channel-perturbation analysis for every requested encoder."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for encoder in args.encoders:
        summary = analyze_encoder(
            encoder,
            args.embeddings_dir,
            args.variants_dir,
            args.variant_embeddings_dir,
            args.output_dir,
        )
        all_summaries[encoder] = summary
        print_summary(summary)
    with open(
        args.output_dir / "channel_perturbation_combined.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nWrote channel-perturbation summaries and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
