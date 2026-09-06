#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Multi-system voice-diversity comparison: ElevenLabs voices vs.
Qwen3-TTS voices cloned from LibriTTS speakers vs. Qwen3-TTS voice-designed
voices (same characters/text as "elevenlabs", but via a style instruction
instead of a persistent voice_id) vs. the VCTK natural reference. Reuses the
exact within/between-group distance, nearest-neighbour collision, and
identification-accuracy definitions from diversity_metrics.py (same as
analyze/voice_diversity.py), so all systems are directly comparable - this
does not replace analyze/voice_diversity.py's 2-way report, it adds the extra
systems alongside it.

The natural corpus's own nearest-neighbour-centroid distribution is used as
the shared collision-threshold calibration for every system (not just the
elevenlabs one), so "possible collision" means the same thing everywhere.

Requires features/extract_embeddings.py already run for all three corpora. See README.md.

Usage (from experiments/tts_voice_analysis/):
    python3 -m analyze.multi_system_diversity
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
    DOUBLE_COL_WIDTH_IN,
    SYSTEM_COLORS,
    system_display_name,
)
from lib.tts_common import DEFAULT_OUTPUT_DIR  # pylint: disable=wrong-import-position

ENCODER_NAMES = ["ecapa", "wavlm_sv"]
DEFAULT_SYSTEMS = {
    "elevenlabs": DEFAULT_OUTPUT_DIR / "embeddings",
    "qwen_cloned": DEFAULT_OUTPUT_DIR / "qwen_embeddings",
    "qwen_designed": DEFAULT_OUTPUT_DIR / "qwen_designed_embeddings",
    "natural": DEFAULT_OUTPUT_DIR / "vctk_embeddings",
}


def analyze_system(
    name: str, embeddings_dir: Path, encoder: str, collision_threshold: float
) -> dict:
    """Compute within/between distances, collisions, and ID accuracy for one system.

    Args:
        name: Name of the system being analyzed (e.g. "elevenlabs", "qwen_cloned").
        embeddings_dir: Directory containing this system's embeddings.
        encoder: Name of the speaker-embedding encoder to use.
        collision_threshold: Distance at or below which a nearest-neighbour pair is
            flagged as a possible voice collision.

    Returns:
        Dict with the system name, group count, within/between distance summaries,
        separation ratio, identification accuracy, collision count, and the raw
        within/between distance arrays (under "_within_raw"/"_between_raw") for
        plotting.
    """
    by_group = load_embeddings(embeddings_dir, encoder)
    centroids = centroids_of(by_group)
    within = within_group_distances(by_group)
    between = between_group_distances(centroids)
    nn_results = nearest_neighbor_analysis(centroids, collision_threshold)
    id_acc = identification_accuracy(by_group, centroids)
    return {
        "system": name,
        "n_groups": len(centroids),
        "within": summarize(within),
        "between": summarize(between),
        "separation_ratio": (
            float(np.mean(between) / np.mean(within))
            if len(within) and len(between)
            else None
        ),
        "identification_accuracy": id_acc["accuracy"],
        "n_possible_collisions": sum(r["possible_collision"] for r in nn_results),
        "_within_raw": within,
        "_between_raw": between,
    }


def plot_comparison(results: list[dict], out_path: Path) -> None:
    """Plot overlaid intra- and inter-speaker distance histograms for all systems.

    Args:
        results: List of per-system result dicts produced by analyze_system.
        out_path: Path to write the PNG (a matching PDF is written alongside it).
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH_IN, 3.2))
    panels = zip(
        ["a", "b"], ["_within_raw", "_between_raw"], ["Intra-Speaker", "Inter-Speaker"]
    )
    for ax, (letter, key, label) in zip(axes, panels):
        for r in results:
            values = r[key]
            if len(values) == 0:
                continue
            ax.hist(
                values,
                bins=40,
                alpha=0.5,
                density=True,
                label=f"{system_display_name(r['system'])} (n = {len(values):,})",
                color=SYSTEM_COLORS.get(r["system"]),
            )
        ax.set_xlabel("Cosine Distance")
        ax.set_ylabel("Density")
        ax.set_ylim(
            top=ax.get_ylim()[1] * 1.35
        )  # headroom so title/legend clear the tallest peak
        ax.set_title(f"({letter}) {label}", loc="left", fontsize=9, fontweight="normal")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def analyze_encoder(
    encoder: str, systems: dict[str, Path], output_dir: Path
) -> list[dict]:
    """Run the multi-system comparison for one encoder and write its outputs.

    Args:
        encoder: Name of the speaker-embedding encoder to analyze.
        systems: Mapping from system name to its embeddings directory.
        output_dir: Directory to write the per-encoder summary JSON and plot to.

    Returns:
        List of per-system summary dicts (with the raw distance arrays stripped
        out), suitable for direct JSON serialization.
    """
    natural_centroids = centroids_of(load_embeddings(systems["natural"], encoder))
    natural_nn = [
        r["distance"]
        for r in nearest_neighbor_analysis(natural_centroids, collision_threshold=0.0)
    ]
    collision_threshold = float(np.percentile(natural_nn, 5)) if natural_nn else 0.0

    results = [
        analyze_system(name, path, encoder, collision_threshold)
        for name, path in systems.items()
    ]

    table = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open(
        output_dir / f"{encoder}_multi_system_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "collision_threshold_p5_natural_nn": collision_threshold,
                "systems": table,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    plot_comparison(
        results,
        out_path=output_dir / f"{encoder}_multi_system_distributions.png",
    )
    return table


def print_table(encoder: str, table: list[dict]) -> None:
    """Print a human-readable console comparison table for one encoder.

    Args:
        encoder: Name of the speaker-embedding encoder the table was computed with.
        table: List of per-system summary dicts, as returned by analyze_encoder.
    """
    print(f"\n=== {encoder} ===")
    header = (
        f"{'system':14s} {'n':>4s} {'within':>8s} {'between':>8s} "
        f"{'sep.ratio':>10s} {'id.acc':>8s} {'collisions':>11s}"
    )
    print(header)
    for r in table:
        sep_ratio = (
            f"{r['separation_ratio']:.2f}"
            if r["separation_ratio"] is not None
            else "n/a"
        )
        id_acc = (
            f"{r['identification_accuracy']:.1%}"
            if r["identification_accuracy"] is not None
            else "n/a"
        )
        print(
            f"{r['system']:14s} {r['n_groups']:>4d} {fmt_stat(r['within'], 'mean', '>8.3f')} "
            f"{fmt_stat(r['between'], 'mean', '>8.3f')} {sep_ratio:>10s} "
            f"{id_acc:>8s} "
            f"{r['n_possible_collisions']:>5d}/{r['n_groups']:<5d}"
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
        "--elevenlabs-dir", type=Path, default=DEFAULT_SYSTEMS["elevenlabs"]
    )
    parser.add_argument(
        "--qwen-cloned-dir", type=Path, default=DEFAULT_SYSTEMS["qwen_cloned"]
    )
    parser.add_argument(
        "--qwen-designed-dir", type=Path, default=DEFAULT_SYSTEMS["qwen_designed"]
    )
    parser.add_argument("--natural-dir", type=Path, default=DEFAULT_SYSTEMS["natural"])
    parser.add_argument(
        "--skip-qwen-designed",
        action="store_true",
        help="Omit the qwen_designed system (e.g. before it has been generated)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis"
    )
    parser.add_argument(
        "--encoders", nargs="+", choices=ENCODER_NAMES, default=ENCODER_NAMES
    )
    return parser.parse_args()


def main() -> None:
    """Run the multi-system comparison for every requested encoder."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems = {
        "elevenlabs": args.elevenlabs_dir,
        "qwen_cloned": args.qwen_cloned_dir,
        "natural": args.natural_dir,
    }
    if not args.skip_qwen_designed:
        systems["qwen_designed"] = args.qwen_designed_dir

    combined = {}
    for encoder in args.encoders:
        table = analyze_encoder(encoder, systems, args.output_dir)
        combined[encoder] = table
        print_table(encoder, table)

    with open(
        args.output_dir / "multi_system_combined.json", "w", encoding="utf-8"
    ) as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nWrote multi-system comparison summaries and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
