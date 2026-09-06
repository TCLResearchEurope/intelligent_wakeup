#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Extract interpretable acoustic features (acoustic_features.py) for every
utterance in a flat index.json, saving one row per utterance to a CSV.

Works for any corpus produced by this pipeline (ElevenLabs diversity utterances,
VCTK reference, channel variants) since they all share the same flat index
schema (utterance_id, group, audio_path, text, ...).

Usage (from experiments/tts_voice_analysis/):
    python3 -m features.extract_acoustic_features \\
        --index experiments/tts_voice_analysis/output/embeddings/index.json \\
        --audio-root experiments/tts_voice_analysis/output \\
        --output experiments/tts_voice_analysis/output/acoustic_features/elevenlabs.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.acoustic_features import compute_features
from lib.embeddings import load_audio
from lib.tts_common import DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR

DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "acoustic_features"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Compute acoustic features for every utterance in --index and write a CSV."""
    args = parse_arguments()
    records = json.loads(args.index.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    rows = []
    for rec in tqdm(records, desc="Acoustic features"):
        wav = load_audio(args.audio_root / rec["audio_path"])
        features = compute_features(wav, text=rec.get("text"))
        rows.append({"utterance_id": rec["utterance_id"], "group": rec["group"], **features})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
