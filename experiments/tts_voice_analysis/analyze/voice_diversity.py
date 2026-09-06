#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Core voice-diversity analysis (experiment-setup.md, steps 1, 2, 3, 4 - channel
controls and acoustic features are a separate follow-up pass).

For each speaker-embedding encoder, and separately for the ElevenLabs voices
and the natural VCTK reference corpus:
  - within-group cosine distances (same voice/speaker, different utterances)
  - between-group cosine distances (different voices/speakers, centroid-to-centroid)

Then, on the ElevenLabs voices only:
  - nearest-neighbour distance per voice (closest other ElevenLabs voice centroid),
    flagged as a possible collision if it falls at or below the natural corpus's
    own nearest-neighbour-centroid distribution's 5th percentile (i.e. as close as
    the closest few percent of genuinely-different natural speakers get - compared
    centroid-to-centroid on both sides, not mixed with utterance-pair distances)
  - nearest-centroid identification accuracy: for each utterance, does its
    nearest centroid (among all ElevenLabs voices) match its own voice?

Requires embeddings already extracted by features/extract_embeddings.py for both the
ElevenLabs diversity utterances and the VCTK reference (see README.md).

Usage (from experiments/tts_voice_analysis/):
    python3 -m analyze.voice_diversity
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
    fmt_stat,
    identification_accuracy,
    load_embeddings,
    nearest_neighbor_analysis,
    summarize,
    within_group_distances,
)
from lib.plot_style import (  # pylint: disable=wrong-import-position
    SINGLE_COL_WIDTH_IN,
    system_display_name,
)
from lib.tts_common import DEFAULT_OUTPUT_DIR  # pylint: disable=wrong-import-position

ENCODER_NAMES = ["ecapa", "wavlm_sv"]
# Display label and color per distribution key, built from the shared system names
# (Intra-/Inter-Speaker is the standard speaker-verification-literature term for
# what this codebase's within-/between-group distances measure).
_DISTRIBUTION_STYLE = {
    "within_elevenlabs": (
        f"{system_display_name('elevenlabs')} - Intra-Speaker",
        "#0072B2",
    ),
    "between_elevenlabs": (
        f"{system_display_name('elevenlabs')} - Inter-Speaker",
        "#56B4E9",
    ),
    "within_natural": (f"{system_display_name('natural')} - Intra-Speaker", "#D55E00"),
    "between_natural": (f"{system_display_name('natural')} - Inter-Speaker", "#E69F00"),
}


def plot_distributions(dists: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot overlaid histograms of the intra-/inter-speaker distance distributions.

    Args:
        dists: Mapping from distribution key (e.g. "within_elevenlabs") to its
            array of cosine distances. Up to 4 distributions are expected, but
            fewer are handled too.
        out_path: Path to write the PNG (a matching PDF is written alongside it).
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, 2.8))
    for name, values in dists.items():
        if len(values) == 0:
            continue
        label, color = _DISTRIBUTION_STYLE.get(name, (name, None))
        ax.hist(
            values,
            bins=40,
            alpha=0.55,
            density=True,
            label=f"{label} (n = {len(values):,})",
            color=color,
        )
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_ylim(
        top=ax.get_ylim()[1] * 1.5
    )  # headroom so the legend clears the tallest peak
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def analyze_encoder(
    encoder: str, elevenlabs_dir: Path, natural_dir: Path, output_dir: Path
) -> dict:
    """Run the full within/between-distance and collision analysis for one encoder.

    Args:
        encoder: Name of the speaker-embedding encoder to analyze.
        elevenlabs_dir: Directory containing the ElevenLabs voice embeddings.
        natural_dir: Directory containing the natural VCTK reference embeddings.
        output_dir: Directory to write the nearest-neighbour table, summary JSON,
            and plot to.

    Returns:
        Summary dict with voice/speaker counts, the collision threshold and count,
        identification accuracy, and the within/between distance distributions for
        both the ElevenLabs voices and the natural reference.
    """
    el_by_group = load_embeddings(elevenlabs_dir, encoder)
    nat_by_group = load_embeddings(natural_dir, encoder)

    el_centroids = centroids_of(el_by_group)
    nat_centroids = centroids_of(nat_by_group)

    within_elevenlabs = within_group_distances(el_by_group)
    between_elevenlabs = between_group_distances(el_centroids)
    within_natural = within_group_distances(nat_by_group)
    between_natural = between_group_distances(nat_centroids)

    # Calibrate the collision threshold against natural speakers' own nearest-neighbour
    # centroid distances (not the within-speaker utterance-pair distances): both sides of
    # the comparison must be centroid-to-centroid, or "close" gets systematically
    # overcounted since centroids are denoised averages and sit closer together than raw
    # utterance pairs do. The threshold is the 5th percentile of natural NN distances -
    # i.e. as close as the closest few percent of genuinely-different natural speakers get.
    natural_nn_distances = [
        r["distance"]
        for r in nearest_neighbor_analysis(nat_centroids, collision_threshold=0.0)
    ]
    collision_threshold = (
        float(np.percentile(natural_nn_distances, 5)) if natural_nn_distances else 0.0
    )
    nn_results = nearest_neighbor_analysis(el_centroids, collision_threshold)
    id_acc = identification_accuracy(el_by_group, el_centroids)

    summary = {
        "encoder": encoder,
        "n_elevenlabs_voices": len(el_centroids),
        "n_natural_speakers": len(nat_centroids),
        "collision_threshold_p5_natural_nn": collision_threshold,
        "n_possible_collisions": sum(r["possible_collision"] for r in nn_results),
        "identification_accuracy": id_acc,
        "distributions": {
            "within_elevenlabs": summarize(within_elevenlabs),
            "between_elevenlabs": summarize(between_elevenlabs),
            "within_natural": summarize(within_natural),
            "between_natural": summarize(between_natural),
        },
    }

    with open(
        output_dir / f"{encoder}_nearest_neighbor.json", "w", encoding="utf-8"
    ) as f:
        json.dump(nn_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / f"{encoder}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_distributions(
        {
            "within_elevenlabs": within_elevenlabs,
            "between_elevenlabs": between_elevenlabs,
            "within_natural": within_natural,
            "between_natural": between_natural,
        },
        out_path=output_dir / f"{encoder}_distributions.png",
    )
    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable console summary matching experiment-setup.md's expected evidence.

    Args:
        summary: Summary dict produced by analyze_encoder.
    """
    d = summary["distributions"]
    print(f"\n=== {summary['encoder']} ===")
    print(
        f"  ElevenLabs voices: {summary['n_elevenlabs_voices']}   "
        f"natural speakers: {summary['n_natural_speakers']}"
    )
    print(
        f"  within-elevenlabs  distance: mean={fmt_stat(d['within_elevenlabs'], 'mean')}  "
        f"median={fmt_stat(d['within_elevenlabs'], 'median')}"
    )
    print(
        f"  between-elevenlabs distance: mean={fmt_stat(d['between_elevenlabs'], 'mean')}  "
        f"median={fmt_stat(d['between_elevenlabs'], 'median')}"
    )
    print(
        f"  within-natural     distance: mean={fmt_stat(d['within_natural'], 'mean')}  "
        f"median={fmt_stat(d['within_natural'], 'median')}"
    )
    print(
        f"  between-natural    distance: mean={fmt_stat(d['between_natural'], 'mean')}  "
        f"median={fmt_stat(d['between_natural'], 'median')}"
    )
    accuracy = summary["identification_accuracy"]["accuracy"]
    print(
        "  nearest-centroid identification accuracy: " f"{accuracy:.3%}"
        if accuracy is not None
        else "  nearest-centroid identification accuracy: n/a"
    )
    print(
        f"  possible voice collisions (NN centroid dist <= natural NN centroid p5="
        f"{summary['collision_threshold_p5_natural_nn']:.3f}): "
        f"{summary['n_possible_collisions']}/{summary['n_elevenlabs_voices']}"
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
        "--elevenlabs-embeddings-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "embeddings",
    )
    parser.add_argument(
        "--natural-embeddings-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "vctk_embeddings",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis"
    )
    parser.add_argument(
        "--encoders", nargs="+", choices=ENCODER_NAMES, default=ENCODER_NAMES
    )
    return parser.parse_args()


def main() -> None:
    """Run the analysis for every requested encoder and print a combined summary."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for encoder in args.encoders:
        summary = analyze_encoder(
            encoder,
            args.elevenlabs_embeddings_dir,
            args.natural_embeddings_dir,
            args.output_dir,
        )
        all_summaries[encoder] = summary
        print_summary(summary)

    with open(args.output_dir / "combined_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(
        f"\nWrote per-encoder summaries, nearest-neighbour tables and plots to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
