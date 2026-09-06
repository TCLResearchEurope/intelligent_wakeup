#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Build a natural multi-speaker reference subset from VCTK, for calibrating the
ElevenLabs-voice diversity analysis (experiment-setup.md, step 3).

Streams the HuggingFace mirror `sanchit-gandhi/vctk` (a parquet re-upload of the
official University of Edinburgh VCTK Corpus 0.92, Open Data Commons Attribution
License v1.0) and pulls only as many speakers/utterances as requested, stopping
early rather than downloading the full ~11GB corpus. Only mic1 recordings are
kept and utterances are deduplicated by VCTK text_id, so each kept utterance is
a genuinely different sentence read by that speaker (mic2 is a simultaneous
second recording of the *same* sentence - a real channel-variant pair, useful
for the phase-2 channel-perturbation control, but not for this within/between
-speaker lexical-diversity measurement).

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.vctk_reference
    python3 -m generate.vctk_reference --n-speakers 5 --n-utterances 3
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from lib.provenance import sha256_text
from lib.tts_common import DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR

DATASET_REPO = "sanchit-gandhi/vctk"
DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "vctk"
DEFAULT_N_SPEAKERS = 150  # VCTK only has ~109-110 speakers total; this exhausts the corpus
DEFAULT_N_UTTERANCES = 8
LICENSE_NOTE = (
    "VCTK Corpus 0.92 (CSTR, University of Edinburgh), Open Data Commons "
    "Attribution License (ODC-By) v1.0. Mirror: https://huggingface.co/datasets/" + DATASET_REPO
)


def is_mic1(example: dict) -> bool:
    """Check whether an example is a mic1 recording.

    VCTK provides two simultaneous mic recordings per utterance; only mic1 is kept
    here (see module docstring for why mic2 is excluded).

    Args:
        example: A single streamed dataset example.

    Returns:
        True if example's audio path indicates a mic1 recording.
    """
    path = example["audio"].get("path") or ""
    return "_mic1" in path


def stream_subset(n_speakers: int, n_utterances: int, revision: str | None):
    """Stream VCTK, stopping once n_speakers each have n_utterances distinct sentences.

    Args:
        n_speakers: Number of speakers to collect before stopping.
        n_utterances: Number of distinct-sentence mic1 utterances required per speaker.
        revision: Pinned HuggingFace dataset revision (sha), or None to track main.

    Returns:
        A dict mapping each complete speaker_id to its {text_id: example} utterances.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET_REPO, split="train", streaming=True, revision=revision)
    ds = ds.cast_column("audio", Audio(decode=False))

    by_speaker: dict[str, dict] = defaultdict(dict)  # speaker_id -> {text_id: example}
    complete_speakers = set()

    pbar = tqdm(desc="Streaming VCTK", unit="utterance")
    for example in ds:
        pbar.update(1)
        if not is_mic1(example):
            continue
        speaker = example["speaker_id"]
        if speaker in complete_speakers:
            continue
        text_id = example["text_id"]
        if text_id in by_speaker[speaker]:
            continue
        by_speaker[speaker][text_id] = example
        if len(by_speaker[speaker]) >= n_utterances:
            complete_speakers.add(speaker)
            pbar.set_postfix(speakers=f"{len(complete_speakers)}/{n_speakers}")
        if len(complete_speakers) >= n_speakers:
            break
    pbar.close()
    return {s: by_speaker[s] for s in complete_speakers}


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-speakers", type=int, default=DEFAULT_N_SPEAKERS)
    parser.add_argument("--n-utterances", type=int, default=DEFAULT_N_UTTERANCES)
    parser.add_argument(
        "--revision",
        default="73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de",
        help="Pinned HF dataset revision (sha) for reproducibility; pass '' to track main",
    )
    return parser.parse_args()


def main() -> None:
    """Stream a VCTK subset, write per-utterance flac files + a flat utterance index."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = args.revision or None

    print(
        f"Streaming {DATASET_REPO} (revision={revision}) for "
        f"{args.n_speakers} speakers x {args.n_utterances} utterances..."
    )
    by_speaker = stream_subset(args.n_speakers, args.n_utterances, revision)
    if len(by_speaker) < args.n_speakers:
        print(
            f"WARNING: only found {len(by_speaker)}/{args.n_speakers} speakers with "
            f">= {args.n_utterances} distinct mic1 utterances before the stream ended.",
            file=sys.stderr,
        )

    index = []
    for speaker, utterances in sorted(by_speaker.items()):
        speaker_dir = args.output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for text_id, example in sorted(utterances.items()):
            audio_path = speaker_dir / f"{text_id}.flac"
            audio_path.write_bytes(example["audio"]["bytes"])
            index.append(
                {
                    "utterance_id": f"{speaker}__{text_id}",
                    "group": speaker,
                    "audio_path": str(audio_path.relative_to(args.output_dir)),
                    "text": example["text"],
                    "gender": example.get("gender"),
                    "accent": example.get("accent"),
                    "region": example.get("region"),
                    "source": "natural_vctk",
                }
            )

    with open(args.output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    provenance = {
        "dataset_repo": DATASET_REPO,
        "revision": revision,
        "license": LICENSE_NOTE,
        "n_speakers_requested": args.n_speakers,
        "n_utterances_requested": args.n_utterances,
        "n_speakers_found": len(by_speaker),
        "n_utterances_total": len(index),
        "generated_at": datetime.now().isoformat(),
        "index_sha256": sha256_text(json.dumps(index, sort_keys=True)),
    }
    with open(args.output_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone. {len(by_speaker)} speakers, {len(index)} utterances -> {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        # datasets' streaming HTTP filesystem can segfault on normal interpreter
        # teardown (background thread torn down out of order); the run has
        # already finished and flushed its output by this point, so force a
        # clean process exit instead of letting Python's finalizer crash.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
