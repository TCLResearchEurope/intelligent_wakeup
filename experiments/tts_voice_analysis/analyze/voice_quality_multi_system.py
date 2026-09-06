#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Voice-quality/prosody comparison across all 4 systems (ElevenLabs, Qwen3-TTS
cloned, Qwen3-TTS voice-designed, natural VCTK reference): jitter, shimmer,
formants (F1-F3) and pitch coefficient of variation, plus voiced-frame
fraction. These are not speaker-embedding metrics - they're the underlying
acoustic descriptors that would explain *why* embedding-space diversity or
identification accuracy differs between systems (e.g. unnaturally low
jitter/shimmer is a classic synthetic-speech tell; formants are the main
acoustic correlate of speaker identity independent of pitch; low pitch-CV or
high voiced-fraction flag flat/monotone or pause-free delivery).

Requires features/extract_acoustic_features.py already run for all 4 systems. See
README.md.

Usage (from experiments/tts_voice_analysis/):
    python3 -m analyze.voice_quality_multi_system
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

from lib.plot_style import (  # pylint: disable=wrong-import-position
    DOUBLE_COL_WIDTH_IN,
    SINGLE_COL_WIDTH_IN,
    SYSTEM_COLORS,
    system_display_name,
)
from lib.tts_common import DEFAULT_OUTPUT_DIR  # pylint: disable=wrong-import-position

DEFAULT_FEATURES_DIR = DEFAULT_OUTPUT_DIR / "acoustic_features"
DEFAULT_SYSTEMS = {
    "elevenlabs": DEFAULT_FEATURES_DIR / "elevenlabs.csv",
    "qwen_cloned": DEFAULT_FEATURES_DIR / "qwen_cloned.csv",
    "qwen_designed": DEFAULT_FEATURES_DIR / "qwen_designed.csv",
    "natural": DEFAULT_FEATURES_DIR / "natural.csv",
}
SYSTEM_ORDER = ["elevenlabs", "qwen_cloned", "qwen_designed", "natural"]
FEATURE_GROUPS = {
    "jitter_shimmer": {
        "features": ["jitter_local_pct", "shimmer_local_pct"],
        "ylabels": ["Jitter (%)", "Shimmer (%)"],
    },
    "formants": {
        "features": ["f1_hz", "f2_hz", "f3_hz"],
        "ylabels": ["F1 (Hz)", "F2 (Hz)", "F3 (Hz)"],
    },
    "pitch_cv": {
        "features": ["f0_cv"],
        "ylabels": ["F0 Coefficient of Variation"],
    },
    "voiced_fraction": {
        "features": ["voiced_fraction"],
        "ylabels": ["Voicing Fraction"],
    },
}


def summarize(values: np.ndarray) -> dict:
    """Compute basic distribution summary stats, dropping NaN/None values first.

    Args:
        values: Array of numeric feature values, possibly containing NaN.

    Returns:
        Dict with n, mean, std, median, p5 and p95 (or just {"n": 0} if values is
        empty after dropping NaN).
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def load_systems(systems: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load each system's acoustic-feature CSV into a DataFrame.

    Args:
        systems: Mapping from system name to its acoustic-features CSV path.

    Returns:
        Mapping from system name to the loaded DataFrame.
    """
    return {name: pd.read_csv(path) for name, path in systems.items()}


def plot_group(group: dict, dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Plot boxplot(s) of one feature group's features across all systems.

    Args:
        group: Feature-group config dict with "features" and "ylabels" keys.
        dfs: Mapping from system name to its acoustic-features DataFrame.
        out_path: Path to write the PNG (a matching PDF is written alongside it).
    """
    features = group["features"]
    tick_labels = [system_display_name(name) for name in SYSTEM_ORDER]
    width = DOUBLE_COL_WIDTH_IN if len(features) > 1 else SINGLE_COL_WIDTH_IN
    fig, axes = plt.subplots(1, len(features), figsize=(width, 3.2))
    if len(features) == 1:
        axes = [axes]
    panel_letters = "abc"
    for i, (ax, feature, ylabel) in enumerate(zip(axes, features, group["ylabels"])):
        data = [dfs[name][feature].dropna().values for name in SYSTEM_ORDER]
        box = ax.boxplot(
            data,
            tick_labels=tick_labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black"},
        )
        for patch, name in zip(box["boxes"], SYSTEM_ORDER):
            patch.set_facecolor(SYSTEM_COLORS[name])
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )
        if len(features) > 1:
            ax.set_title(
                f"({panel_letters[i]})", loc="left", fontsize=9, fontweight="normal"
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def print_table(feature: str, summaries: dict[str, dict]) -> None:
    """Print a human-readable console comparison row per system for one feature.

    Args:
        feature: Name of the acoustic feature being reported.
        summaries: Mapping from system name to its summary dict for this feature,
            as produced by summarize.
    """
    print(f"\n=== {feature} ===")
    header = f"{'system':14s} {'n':>5s} {'mean':>10s} {'std':>10s} {'median':>10s}"
    print(header)
    for name in SYSTEM_ORDER:
        s = summaries[name]
        if s["n"] == 0:
            print(f"{name:14s}  n/a")
            continue
        print(
            f"{name:14s} {s['n']:>5d} {s['mean']:>10.3f} {s['std']:>10.3f} {s['median']:>10.3f}"
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
        "--elevenlabs-features", type=Path, default=DEFAULT_SYSTEMS["elevenlabs"]
    )
    parser.add_argument(
        "--qwen-cloned-features", type=Path, default=DEFAULT_SYSTEMS["qwen_cloned"]
    )
    parser.add_argument(
        "--qwen-designed-features", type=Path, default=DEFAULT_SYSTEMS["qwen_designed"]
    )
    parser.add_argument(
        "--natural-features", type=Path, default=DEFAULT_SYSTEMS["natural"]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis"
    )
    return parser.parse_args()


def main() -> None:
    """Compare jitter/shimmer, formants, pitch-CV and voiced-fraction across all systems."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems = {
        "elevenlabs": args.elevenlabs_features,
        "qwen_cloned": args.qwen_cloned_features,
        "qwen_designed": args.qwen_designed_features,
        "natural": args.natural_features,
    }
    dfs = load_systems(systems)

    combined = {}
    for group_key, group in FEATURE_GROUPS.items():
        for feature in group["features"]:
            summaries = {
                name: summarize(dfs[name][feature].values) for name in SYSTEM_ORDER
            }
            combined[feature] = summaries
            print_table(feature, summaries)
        plot_group(group, dfs, args.output_dir / f"voice_quality_{group_key}.png")

    with open(
        args.output_dir / "voice_quality_multi_system.json", "w", encoding="utf-8"
    ) as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nWrote voice-quality comparison summaries and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
