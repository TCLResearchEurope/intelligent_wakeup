#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Channel-normalized replication (experiment-setup.md, step 6): does speaker
separation survive conservative channel normalization (common bandwidth +
loudness, see channel_normalization.py)?

Compares within/between-voice (and within/between natural-speaker) cosine
distances before vs. after normalization. Reports a separation ratio
(between-group mean distance / within-group mean distance) for each condition;
if ElevenLabs-voice separation holds up after normalization (ratio doesn't
collapse toward 1), the original diversity isn't primarily a channel artifact.

Requires features/extract_embeddings.py already run on the four embedding sets: the
original and normalized ElevenLabs diversity utterances, and the original and
normalized VCTK reference. See README.md.

Usage (from experiments/tts_voice_analysis/):
    python3 -m analyze.channel_normalization
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
    load_embeddings,
    summarize,
    within_group_distances,
)
from lib.plot_style import (  # pylint: disable=wrong-import-position
    DOUBLE_COL_WIDTH_IN,
    SYSTEM_COLORS,
    system_display_name,
)
from lib.tts_common import DEFAULT_OUTPUT_DIR  # pylint: disable=wrong-import-position

ENCODER_NAMES = ["ecapa", "wavlm_sv"]
# Display label, bar color, and hatch (solid=unnormalized, hatched=normalized - so
# the before/after contrast survives black-only printing, not just color) per condition.
_CONDITION_STYLE = {
    "elevenlabs_before": (
        f"{system_display_name('elevenlabs')}\n(Unnormalized)",
        "elevenlabs",
        "",
    ),
    "elevenlabs_after": (
        f"{system_display_name('elevenlabs')}\n(Normalized)",
        "elevenlabs",
        "//",
    ),
    "natural_before": (
        f"{system_display_name('natural')}\n(Unnormalized)",
        "natural",
        "",
    ),
    "natural_after": (
        f"{system_display_name('natural')}\n(Normalized)",
        "natural",
        "//",
    ),
}


def separation_ratio(within: np.ndarray, between: np.ndarray) -> float | None:
    """Compute the between-mean / within-mean separation ratio.

    Higher values indicate more separation between groups.

    Args:
        within: Pairwise distances within groups.
        between: Pairwise distances between group centroids.

    Returns:
        The separation ratio, or None if undefined (empty inputs or zero within-mean).
    """
    if len(within) == 0 or len(between) == 0 or np.mean(within) == 0:
        return None
    return float(np.mean(between) / np.mean(within))


def analyze_condition(embeddings_dir: Path, encoder: str) -> dict:
    """Compute within/between distances and separation ratio for one embeddings dir.

    Args:
        embeddings_dir: Directory containing the embeddings to load.
        encoder: Name of the speaker encoder whose embeddings to use.

    Returns:
        Dict with within, between (raw pairwise distances), within_summary,
        between_summary, and separation_ratio.
    """
    by_group = load_embeddings(embeddings_dir, encoder)
    within = within_group_distances(by_group)
    between = between_group_distances(centroids_of(by_group))
    return {
        "within": within,
        "between": between,
        "within_summary": summarize(within),
        "between_summary": summarize(between),
        "separation_ratio": separation_ratio(within, between),
    }


def plot_ratios(results: dict, out_path: Path) -> None:
    """Plot a bar chart of separation ratio across the 4 system x normalization conditions.

    Args:
        results: Mapping from condition name to its analyze_condition result.
        out_path: Path to save the PNG (a matching PDF is also written alongside it).
    """
    keys = list(results)
    values = [results[k]["separation_ratio"] or 0.0 for k in keys]
    labels = [_CONDITION_STYLE[k][0] for k in keys]
    colors = [SYSTEM_COLORS[_CONDITION_STYLE[k][1]] for k in keys]
    hatches = [_CONDITION_STYLE[k][2] for k in keys]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH_IN, 3.2))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="No Separation (Ratio = 1)",
    )
    ax.set_ylabel("Separation Ratio")
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def analyze_encoder(
    encoder: str,
    elevenlabs_dir: Path,
    elevenlabs_normalized_dir: Path,
    natural_dir: Path,
    natural_normalized_dir: Path,
    output_dir: Path,
) -> dict:
    """Run the before/after normalization comparison for one encoder.

    Args:
        encoder: Name of the speaker encoder to use.
        elevenlabs_dir: Directory with original (unnormalized) ElevenLabs embeddings.
        elevenlabs_normalized_dir: Directory with channel-normalized ElevenLabs
            embeddings.
        natural_dir: Directory with original (unnormalized) natural-speaker embeddings.
        natural_normalized_dir: Directory with channel-normalized natural-speaker
            embeddings.
        output_dir: Directory to write the summary JSON and plot to.

    Returns:
        Summary dict with per-condition results and the ElevenLabs separation-retained
        fraction (if computable).
    """
    conditions = {
        "elevenlabs_before": analyze_condition(elevenlabs_dir, encoder),
        "elevenlabs_after": analyze_condition(elevenlabs_normalized_dir, encoder),
        "natural_before": analyze_condition(natural_dir, encoder),
        "natural_after": analyze_condition(natural_normalized_dir, encoder),
    }

    summary = {
        "encoder": encoder,
        "conditions": {
            k: {
                "within": v["within_summary"],
                "between": v["between_summary"],
                "separation_ratio": v["separation_ratio"],
            }
            for k, v in conditions.items()
        },
    }
    el_before, el_after = (
        conditions["elevenlabs_before"]["separation_ratio"],
        conditions["elevenlabs_after"]["separation_ratio"],
    )
    if el_before and el_after:
        summary["elevenlabs_separation_retained_fraction"] = el_after / el_before

    with open(
        output_dir / f"{encoder}_channel_normalization_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_ratios(
        conditions,
        out_path=output_dir / f"{encoder}_channel_normalization.png",
    )
    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable console summary.

    Args:
        summary: Summary dict as returned by analyze_encoder.
    """
    print(f"\n=== {summary['encoder']} ===")
    for name, c in summary["conditions"].items():
        ratio = c["separation_ratio"]
        print(
            f"  {name:18s} within_mean={c['within'].get('mean'):.3f}  "
            f"between_mean={c['between'].get('mean'):.3f}  separation_ratio={ratio:.2f}"
            if ratio
            else f"  {name:18s} n/a"
        )
    retained = summary.get("elevenlabs_separation_retained_fraction")
    if retained is not None:
        verdict = (
            "HOLDS"
            if retained >= 0.8
            else "WEAKENED" if retained >= 0.5 else "COLLAPSED"
        )
        print(
            f"  ElevenLabs separation retained after normalization: {retained:.1%}  -> {verdict}"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--elevenlabs-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "embeddings"
    )
    parser.add_argument(
        "--elevenlabs-normalized-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "embeddings_normalized",
    )
    parser.add_argument(
        "--natural-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "vctk_embeddings"
    )
    parser.add_argument(
        "--natural-normalized-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "vctk_embeddings_normalized",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis"
    )
    parser.add_argument(
        "--encoders", nargs="+", choices=ENCODER_NAMES, default=ENCODER_NAMES
    )
    return parser.parse_args()


def main() -> None:
    """Run the channel-normalization replication for every requested encoder."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for encoder in args.encoders:
        summary = analyze_encoder(
            encoder,
            args.elevenlabs_dir,
            args.elevenlabs_normalized_dir,
            args.natural_dir,
            args.natural_normalized_dir,
            args.output_dir,
        )
        all_summaries[encoder] = summary
        print_summary(summary)
    with open(
        args.output_dir / "channel_normalization_combined.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nWrote channel-normalization summaries and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
