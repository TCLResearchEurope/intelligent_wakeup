#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Acoustic characterization + channel-confound check (experiment-setup.md, step 7).

1. Summarizes acoustic feature distributions (ElevenLabs voices vs. natural
   reference), to characterize how the ElevenLabs voices differ acoustically.
2. Confound check: for each channel-related feature (effective bandwidth,
   spectral slope, spectral flatness, estimated SNR, clipping rate), correlates
   pairwise speaker-embedding distance between voice centroids against pairwise
   differences in that feature's per-voice mean. A strong correlation would mean
   embedding distance is tracking recording quality rather than identity; a weak
   one supports the diversity result being about speaker identity.

Requires features/extract_acoustic_features.py already run for both corpora, and
features/extract_embeddings.py already run for the ElevenLabs voices (for the confound
check only). See README.md.

Usage (from experiments/tts_voice_analysis/):
    python3 -m analyze.acoustic_confound
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
from scipy.stats import pearsonr  # pylint: disable=wrong-import-position

from lib.diversity_metrics import (  # pylint: disable=wrong-import-position
    centroids_of,
    cosine_distance,
    load_embeddings,
)
from lib.plot_style import SINGLE_COL_WIDTH_IN  # pylint: disable=wrong-import-position
from lib.tts_common import DEFAULT_OUTPUT_DIR  # pylint: disable=wrong-import-position

CHANNEL_FEATURES = [
    "effective_bandwidth_hz",
    "spectral_slope_db",
    "spectral_flatness",
    "estimated_snr_db",
    "clipping_rate",
]
CHANNEL_FEATURE_DISPLAY_NAMES = {
    "effective_bandwidth_hz": "Bandwidth (Hz)",
    "spectral_slope_db": "Spectral Slope (dB)",
    "spectral_flatness": "Spectral Flatness",
    "estimated_snr_db": "SNR (dB)",
    "clipping_rate": "Clipping Rate",
}
SPEAKER_FEATURES = [
    "median_f0_hz",
    "f0_range_hz",
    "f0_cv",
    "voiced_fraction",
    "hnr_db",
    "jitter_local_pct",
    "shimmer_local_pct",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "speaking_rate_wps",
]


def summarize_features(df: pd.DataFrame) -> dict:
    """Compute mean/std per feature, across all utterances in one corpus.

    Args:
        df: Per-utterance acoustic feature table for one corpus.

    Returns:
        Dict with mean and std per feature, plus n_utterances and n_groups.
    """
    numeric = df[SPEAKER_FEATURES + CHANNEL_FEATURES]
    return {
        "mean": numeric.mean().to_dict(),
        "std": numeric.std().to_dict(),
        "n_utterances": len(df),
        "n_groups": df["group"].nunique(),
    }


def per_voice_feature_means(df: pd.DataFrame) -> dict[str, dict]:
    """Compute the mean of each feature per group (voice/speaker).

    Args:
        df: Per-utterance acoustic feature table.

    Returns:
        Dict mapping each group name to a dict of its per-feature means.
    """
    return (
        df.groupby("group")[SPEAKER_FEATURES + CHANNEL_FEATURES].mean().to_dict("index")
    )


def confound_correlations(
    centroids: dict[str, np.ndarray], voice_features: dict[str, dict]
) -> dict[str, dict]:
    """Compute Pearson correlation between pairwise embedding distance and feature diff.

    For each channel feature, correlates pairwise cosine distance between voice
    embedding centroids against pairwise absolute differences in that feature's
    per-voice mean.

    Args:
        centroids: Mapping from group/voice name to its embedding centroid.
        voice_features: Mapping from group/voice name to its per-feature means.

    Returns:
        Tuple of (per-feature correlation results with r/p/n, the list of pairwise
        embedding distances used, and the list of voice-name pairs used).
    """
    groups = [g for g in centroids if g in voice_features]
    pairs = list(itertools.combinations(groups, 2))
    embed_dist = [cosine_distance(centroids[a], centroids[b]) for a, b in pairs]

    results = {}
    for feature in CHANNEL_FEATURES:
        feature_diff = [
            abs(voice_features[a][feature] - voice_features[b][feature])
            for a, b in pairs
        ]
        if len(set(feature_diff)) < 2:
            results[feature] = {"r": None, "p": None, "n": len(pairs)}
            continue
        r, p = pearsonr(embed_dist, feature_diff)
        results[feature] = {"r": float(r), "p": float(p), "n": len(pairs)}
    return results, embed_dist, pairs


def plot_correlations(correlations: dict[str, dict], out_path: Path) -> None:
    """Plot a bar chart of |r| per channel feature.

    Args:
        correlations: Per-feature correlation results, as returned by
            confound_correlations.
        out_path: Path to save the PNG (a matching PDF is also written alongside it).
    """
    features = list(correlations)
    r_values = [
        abs(correlations[f]["r"]) if correlations[f]["r"] is not None else 0.0
        for f in features
    ]
    labels = [CHANNEL_FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, 2.8))
    ax.bar(labels, r_values, color="#0072B2")
    ax.axhline(
        0.3,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Weak Correlation (|r| = 0.3)",
    )
    ax.set_ylabel("Pearson Correlation (|r|)")
    ax.set_ylim(0, 1)
    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--elevenlabs-features",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "acoustic_features" / "elevenlabs.csv",
    )
    parser.add_argument(
        "--natural-features",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "acoustic_features" / "natural.csv",
    )
    parser.add_argument(
        "--elevenlabs-embeddings-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "embeddings",
    )
    parser.add_argument("--encoder", default="ecapa", choices=["ecapa", "wavlm_sv"])
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis"
    )
    return parser.parse_args()


def main() -> None:
    """Summarize acoustic features and run the confound correlation check."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    el_df = pd.read_csv(args.elevenlabs_features)
    nat_df = pd.read_csv(args.natural_features)

    summary = {
        "elevenlabs": summarize_features(el_df),
        "natural": summarize_features(nat_df),
    }

    el_by_group = load_embeddings(args.elevenlabs_embeddings_dir, args.encoder)
    el_centroids = centroids_of(el_by_group)
    voice_features = per_voice_feature_means(el_df)
    correlations, _embed_dist, pairs = confound_correlations(
        el_centroids, voice_features
    )
    summary["channel_confound_correlations"] = correlations
    summary["encoder_used_for_confound_check"] = args.encoder
    summary["n_voice_pairs"] = len(pairs)

    with open(
        args.output_dir / "acoustic_confound_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_correlations(
        correlations,
        out_path=args.output_dir / "acoustic_confound_correlations.png",
    )

    print("=== acoustic feature means (elevenlabs vs natural) ===")
    for feature in SPEAKER_FEATURES + CHANNEL_FEATURES:
        el, n = (
            summary["elevenlabs"]["mean"][feature],
            summary["natural"]["mean"][feature],
        )
        print(f"  {feature:24s} elevenlabs={el:.3f}  natural={n:.3f}")
    print(
        f"\n=== channel-confound correlations ({args.encoder}, n={len(pairs)} voice pairs) ==="
    )
    for feature, r in correlations.items():
        if r["r"] is None:
            print(f"  {feature:24s} r=n/a (constant feature)")
            continue
        flag = "OK (weak)" if abs(r["r"]) < 0.3 else "WARNING (possible confound)"
        print(f"  {feature:24s} r={r['r']:+.3f}  p={r['p']:.3f}   {flag}")
    print(f"\nWrote acoustic summary and confound plot to {args.output_dir}")


if __name__ == "__main__":
    main()
