#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Extract ECAPA-TDNN and WavLM-SV speaker embeddings for a set of utterances.

Works from a flat "utterance index" - a JSON list of records:
    {"utterance_id": "...", "group": "<voice/speaker identity>", "audio_path": "...", ...}
`audio_path` is resolved relative to --audio-root.

Two convenience input modes build that index automatically:
  --diversity-manifest  output/diversity_manifest.json from generate/diversity_utterances.py
  --index               a pre-built flat index (e.g. from generate/vctk_reference.py)

Embeddings are cached per encoder in output-dir/<encoder>.npz keyed by utterance_id.
Cache validity is content-hash-based (output-dir/<encoder>_hashes.json records each
utterance_id's source audio hash at embedding time): if the audio file's content
changes - e.g. a character's voice_id changed and its audio was regenerated - the
hash no longer matches and that embedding is automatically recomputed, no --force
needed. Utterance_ids no longer present in the index are dropped from the cache.
This lets the same script serve both the ElevenLabs voices and the natural
reference corpus, so distances are computed with identical preprocessing.

Usage (from experiments/tts_voice_analysis/):
    python3 -m features.extract_embeddings \\
        --diversity-manifest experiments/tts_voice_analysis/output/diversity_manifest.json \\
        --output-dir experiments/tts_voice_analysis/output/embeddings
    python3 -m features.extract_embeddings \\
        --index experiments/tts_voice_analysis/output/vctk/index.json \\
        --audio-root experiments/tts_voice_analysis/output/vctk \\
        --output-dir experiments/tts_voice_analysis/output/vctk_embeddings
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lib.content_cache import hash_file, load_hash_cache, save_hash_cache
from lib.embeddings import ENCODERS, load_audio
from lib.tts_common import DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR

DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "embeddings"


def _safe_hash(path: Path) -> str | None:
    """Hash a file's contents, treating a missing file as unhashable rather than an error.

    Returning None routes the utterance to recompute (and a clear error later) instead of
    raising here.

    Args:
        path: Path to the audio file to hash.

    Returns:
        The content hash, or None if the file is missing.
    """
    try:
        return hash_file(path)
    except OSError:
        return None


def build_index_from_diversity_manifest(manifest_path: Path) -> list[dict]:
    """Flatten diversity_manifest.json into a flat utterance index.

    Turns the per-character utterances/audio_paths structure into one flat record
    per utterance, matching the shape expected by the rest of this script.

    Args:
        manifest_path: Path to diversity_manifest.json.

    Returns:
        A flat list of utterance index records (utterance_id, group, voice_id,
        audio_path, text, source).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = []
    for entry in manifest:
        if "audio_paths" not in entry:
            continue
        for i, (audio_path, text) in enumerate(
            zip(entry["audio_paths"], entry.get("utterances", [])), start=1
        ):
            index.append(
                {
                    "utterance_id": f"{entry['name']}__{i:02d}",
                    "group": entry["name"],
                    "voice_id": entry.get("voice_id"),
                    "audio_path": audio_path,
                    "text": text,
                    "source": "elevenlabs_diversity",
                }
            )
    return index


def load_existing_npz(path: Path) -> dict:
    """Load a previously saved embeddings .npz file as a plain dict.

    Args:
        path: Path to the .npz file.

    Returns:
        A dict mapping utterance_id to embedding array, or {} if the file doesn't exist.
    """
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        The parsed arguments, with manifest_path and audio_root resolved to whichever
        of --diversity-manifest/--index was given.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--diversity-manifest", type=Path)
    source_group.add_argument("--index", type=Path)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Base dir audio_path entries are relative to (default: the manifest/index file's dir)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--encoders", nargs="+", choices=list(ENCODERS), default=list(ENCODERS))
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N utterances"
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if already cached")
    args = parser.parse_args()

    if args.diversity_manifest:
        args.manifest_path = args.diversity_manifest
        args.audio_root = args.audio_root or args.diversity_manifest.parent
    else:
        args.manifest_path = args.index
        args.audio_root = args.audio_root or args.index.parent
    return args


def main() -> None:
    """Extract embeddings for every utterance in the index, for each requested encoder."""
    args = parse_arguments()

    if args.diversity_manifest:
        index = build_index_from_diversity_manifest(args.manifest_path)
    else:
        index = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if not index:
        sys.exit(f"ERROR: no utterances found in {args.manifest_path}")
    if args.limit:
        index = index[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for encoder_name in args.encoders:
        npz_path = args.output_dir / f"{encoder_name}.npz"
        hashes_path = args.output_dir / f"{encoder_name}_hashes.json"
        existing_embeddings = {} if args.force else load_existing_npz(npz_path)
        existing_hashes = {} if args.force else load_hash_cache(hashes_path)

        current_hashes = {
            rec["utterance_id"]: _safe_hash(args.audio_root / rec["audio_path"]) for rec in index
        }
        # An embedding with no recorded hash yet (first run after adding this cache,
        # or any pre-existing .npz) is trusted as-is rather than treated as stale -
        # otherwise turning this cache on would force a full, pointless recompute of
        # everything already on disk. Real changes are only detectable once a hash
        # has been recorded at least once, from here on.
        valid_embeddings = {
            uid: existing_embeddings[uid]
            for uid, h in current_hashes.items()
            if uid in existing_embeddings
            and (uid not in existing_hashes or existing_hashes[uid] == h)
        }
        todo = [rec for rec in index if rec["utterance_id"] not in valid_embeddings]
        n_stale = sum(1 for uid in current_hashes if uid in existing_embeddings) - len(
            valid_embeddings
        )
        print(
            f"[{encoder_name}] {len(valid_embeddings)} cached and unchanged, "
            f"{n_stale} stale (audio changed), {len(todo) - n_stale} new -> {len(todo)} to compute"
        )

        if todo:
            encoder = ENCODERS[encoder_name]()
            for rec in tqdm(todo, desc=f"Embedding ({encoder_name})"):
                uid = rec["utterance_id"]
                wav_path = args.audio_root / rec["audio_path"]
                try:
                    wav = load_audio(wav_path)
                    valid_embeddings[uid] = encoder.embed(wav)
                except Exception as e:  # noqa: BLE001 - report and continue with other utterances
                    print(f"  ERROR embedding {uid} ({wav_path}): {e}", file=sys.stderr)

        # Only keep entries for utterances still in the current index, dropping anything
        # removed/renamed upstream, and only hash what we actually have an embedding for.
        final_hashes = {uid: current_hashes[uid] for uid in valid_embeddings}
        np.savez(npz_path, **valid_embeddings)
        save_hash_cache(hashes_path, final_hashes)

        n_missing = len(index) - len(valid_embeddings)
        print(
            f"[{encoder_name}] {len(valid_embeddings)} embeddings saved to {npz_path} "
            f"({n_missing} missing)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
