#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Pick one reference clip + exact transcript per LibriTTS speaker, for voice
cloning via Qwen3-TTS. Streams the HuggingFace mirror `mythicinfinity/libritts_r`
(LibriTTS-R, CC BY 4.0), train-clean-100 split (247 speakers - comfortably
enough to reach ~100 distinct ones), stopping early rather than pulling the
whole split.

Reference clips are filtered to a plausible duration range (3-10s): too short
gives the cloning model little to go on, too long slows every downstream
generate-audio call for no real benefit.

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.libritts_reference
    python3 -m generate.libritts_reference --n-speakers 3
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from lib.provenance import sha256_text
from lib.tts_common import DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR

DATASET_REPO = "mythicinfinity/libritts_r"
DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "libritts_reference"
DEFAULT_N_SPEAKERS = 100
MIN_DURATION_S = 3.0
MAX_DURATION_S = 10.0
LICENSE_NOTE = (
    "LibriTTS-R (Koizumi et al., 2023), CC BY 4.0. Mirror: "
    f"https://huggingface.co/datasets/{DATASET_REPO}"
)


def clip_duration(audio_bytes: bytes) -> float:
    """Get the duration of an in-memory WAV clip, without a full decode.

    Args:
        audio_bytes: Raw WAV file bytes.

    Returns:
        The clip's duration in seconds.
    """
    info = sf.info(io.BytesIO(audio_bytes))
    return info.frames / info.samplerate


def stream_subset(n_speakers: int, revision: str | None) -> dict[str, dict]:
    """Stream LibriTTS-R, keeping the first suitable-duration clip per new speaker.

    Args:
        n_speakers: Number of distinct speakers to collect before stopping.
        revision: Pinned HuggingFace dataset revision (sha), or None to track main.

    Returns:
        A dict mapping speaker_id to a record with the clip's audio_bytes, text,
        chapter_id, utterance_id, and duration_s.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(
        DATASET_REPO,
        "clean",
        split="train.clean.100",
        streaming=True,
        revision=revision,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    speakers: dict[str, dict] = {}
    pbar = tqdm(desc="Streaming LibriTTS-R", unit="utterance")
    for example in ds:
        pbar.update(1)
        speaker = example["speaker_id"]
        if speaker in speakers:
            continue
        duration = clip_duration(example["audio"]["bytes"])
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            continue
        speakers[speaker] = {
            "audio_bytes": example["audio"]["bytes"],
            "text": example["text_normalized"],
            "chapter_id": example["chapter_id"],
            "utterance_id": example["id"],
            "duration_s": duration,
        }
        pbar.set_postfix(speakers=f"{len(speakers)}/{n_speakers}")
        if len(speakers) >= n_speakers:
            break
    pbar.close()
    return speakers


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
    parser.add_argument("--revision", default=None, help="Pinned HF dataset revision (sha)")
    return parser.parse_args()


def main() -> None:
    """Stream LibriTTS-R, write one reference wav + transcript per speaker."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Streaming {DATASET_REPO} for {args.n_speakers} speakers "
        f"(reference clip {MIN_DURATION_S}-{MAX_DURATION_S}s)..."
    )
    speakers = stream_subset(args.n_speakers, args.revision)
    if len(speakers) < args.n_speakers:
        print(
            f"WARNING: only found {len(speakers)}/{args.n_speakers} speakers with a "
            f"{MIN_DURATION_S}-{MAX_DURATION_S}s clip before the stream ended.",
            file=sys.stderr,
        )

    index = []
    for speaker, rec in sorted(speakers.items()):
        audio_path = args.output_dir / f"{speaker}.wav"
        audio_path.write_bytes(rec["audio_bytes"])
        index.append(
            {
                "speaker_id": speaker,
                "ref_audio_path": audio_path.name,
                "ref_text": rec["text"],
                "duration_s": rec["duration_s"],
                "source_utterance_id": rec["utterance_id"],
            }
        )

    with open(args.output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    provenance = {
        "dataset_repo": DATASET_REPO,
        "revision": args.revision,
        "license": LICENSE_NOTE,
        "n_speakers_requested": args.n_speakers,
        "n_speakers_found": len(index),
        "min_duration_s": MIN_DURATION_S,
        "max_duration_s": MAX_DURATION_S,
        "generated_at": datetime.now().isoformat(),
        "index_sha256": sha256_text(json.dumps(index, sort_keys=True)),
    }
    with open(args.output_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone. {len(index)} speaker(s) -> {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        # Same datasets-streaming teardown segfault as generate/vctk_reference.py.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
