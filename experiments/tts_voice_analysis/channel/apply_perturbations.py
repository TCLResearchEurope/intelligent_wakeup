#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Apply each controlled channel perturbation (channel_perturbations.py) to a subset
of the ElevenLabs diversity utterances (experiment-setup.md, step 5).

Reads diversity_manifest.json (from generate/diversity_utterances.py), picks a
subset of voices x utterances-per-voice, and writes one perturbed .wav per
(utterance, perturbation) plus a flat index.json usable directly by
features/extract_embeddings.py. Each variant's index record carries
`original_utterance_id` so analyze/channel_perturbation.py can pair it back to
the unperturbed embedding from output/embeddings/.

Content-hash-based cache: output-dir/.source_hashes.json records each variant's
source-audio hash, so if the underlying voice audio changes, its variants are
automatically regenerated - no --force needed. A missing hash record
(pre-existing output) is trusted as-is rather than forced to redo.

Usage (from experiments/tts_voice_analysis/):
    python3 -m channel.apply_perturbations
    python3 -m channel.apply_perturbations \\
        --n-voices 5 --n-utterances-per-voice 2
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from lib.channel_perturbations import PERTURBATIONS, SR
from lib.content_cache import hash_file, load_hash_cache, save_hash_cache
from lib.embeddings import load_audio
from lib.tts_common import DEFAULT_OUTPUT_DIR

DEFAULT_N_VOICES = 20
DEFAULT_N_UTTERANCES_PER_VOICE = 3


def build_utterance_subset(
    manifest: list[dict], n_voices: int, n_utterances_per_voice: int
) -> list[dict]:
    """Pick a subset of (character, utterance) pairs to perturb.

    Args:
        manifest: Diversity manifest entries (one per voice), each with audio_paths
            and utterances.
        n_voices: Number of voices (manifest entries) to include.
        n_utterances_per_voice: Number of utterances to include per voice.

    Returns:
        List of utterance records, each with utterance_id, group, audio_path, and
        text.
    """
    subset = []
    for entry in manifest[:n_voices]:
        if "audio_paths" not in entry:
            continue
        for i, (audio_path, text) in enumerate(
            zip(
                entry["audio_paths"][:n_utterances_per_voice],
                entry.get("utterances", []),
            ),
            start=1,
        ):
            subset.append(
                {
                    "utterance_id": f"{entry['name']}__{i:02d}",
                    "group": entry["name"],
                    "audio_path": audio_path,
                    "text": text,
                }
            )
    return subset


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--diversity-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "diversity_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "channel_variants")
    parser.add_argument("--n-voices", type=int, default=DEFAULT_N_VOICES)
    parser.add_argument(
        "--n-utterances-per-voice", type=int, default=DEFAULT_N_UTTERANCES_PER_VOICE
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=list(PERTURBATIONS),
        default=list(PERTURBATIONS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate channel-perturbed variants for a subset of ElevenLabs utterances."""
    args = parse_arguments()
    if not args.diversity_manifest.exists():
        sys.exit(
            f"ERROR: {args.diversity_manifest} not found - "
            "run generate/diversity_utterances.py first"
        )
    manifest = json.loads(args.diversity_manifest.read_text(encoding="utf-8"))
    subset = build_utterance_subset(manifest, args.n_voices, args.n_utterances_per_voice)
    if not subset:
        sys.exit("ERROR: no utterances found in the diversity manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = args.diversity_manifest.parent
    hashes_path = args.output_dir / ".source_hashes.json"
    existing_hashes = {} if args.force else load_hash_cache(hashes_path)

    index = []
    new_hashes = {}
    n_skipped = 0
    print(f"Perturbing {len(subset)} utterance(s) x {len(args.perturbations)} perturbation(s)")
    for rec in tqdm(subset, desc="Voices/utterances"):
        source_hash = hash_file(audio_root / rec["audio_path"])
        wav = None  # only loaded lazily if at least one perturbation actually needs recomputing
        for pert_name in args.perturbations:
            variant_id = f"{rec['utterance_id']}__{pert_name}"
            out_path = (
                args.output_dir / pert_name / f"{rec['group']}" / f"{rec['utterance_id']}.wav"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            recorded_hash = existing_hashes.get(variant_id)
            unchanged = out_path.exists() and (
                recorded_hash is None or recorded_hash == source_hash
            )
            if args.force or not unchanged:
                wav = wav if wav is not None else load_audio(audio_root / rec["audio_path"])
                out_wav = PERTURBATIONS[pert_name](wav)
                sf.write(out_path, out_wav, SR)
            else:
                n_skipped += 1
            new_hashes[variant_id] = source_hash
            index.append(
                {
                    "utterance_id": variant_id,
                    "group": rec["group"],
                    "audio_path": str(out_path.relative_to(args.output_dir)),
                    "text": rec["text"],
                    "source": "channel_variant",
                    "perturbation": pert_name,
                    "original_utterance_id": rec["utterance_id"],
                }
            )

    with open(args.output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")
    save_hash_cache(hashes_path, new_hashes)
    print(
        f"\nDone. {len(index)} variant(s) -> {args.output_dir} "
        f"({n_skipped} unchanged, {len(index) - n_skipped} (re)computed)"
    )


if __name__ == "__main__":
    main()
