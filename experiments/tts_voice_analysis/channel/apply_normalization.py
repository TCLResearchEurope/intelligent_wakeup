#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Apply conservative channel normalization (channel_normalization.py) to every
utterance in a flat index.json, writing normalized .wav files + a matching index
usable directly by features/extract_embeddings.py.

Works for either corpus - point it at the ElevenLabs voices' embeddings index or
the VCTK reference's index (both already flat-index shaped).

Content-hash-based cache: output-dir/.source_hashes.json records each output's
source-audio hash, so if upstream audio changes (e.g. a voice was regenerated),
the stale normalized file is automatically redone - no --force needed. A
missing hash record (pre-existing output) is trusted as-is rather than forced
to redo.

Usage (from experiments/tts_voice_analysis/):
    python3 -m channel.apply_normalization \\
        --index experiments/tts_voice_analysis/output/embeddings/index.json \\
        --audio-root experiments/tts_voice_analysis/output \\
        --output-dir experiments/tts_voice_analysis/output/diversity_normalized

    python3 -m channel.apply_normalization \\
        --index experiments/tts_voice_analysis/output/vctk/index.json \\
        --audio-root experiments/tts_voice_analysis/output/vctk \\
        --output-dir experiments/tts_voice_analysis/output/vctk_normalized
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from lib.channel_normalization import SR, normalize_channel
from lib.content_cache import hash_file, load_hash_cache, save_hash_cache
from lib.embeddings import load_audio


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Normalize every utterance referenced by --index and write a new flat index."""
    args = parse_arguments()
    records = json.loads(args.index.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes_path = args.output_dir / ".source_hashes.json"
    existing_hashes = {} if args.force else load_hash_cache(hashes_path)

    new_index = []
    new_hashes = {}
    n_skipped = 0
    for rec in tqdm(records, desc="Normalizing"):
        uid = rec["utterance_id"]
        out_path = args.output_dir / rec["group"] / f"{uid}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        source_hash = hash_file(args.audio_root / rec["audio_path"])
        recorded_hash = existing_hashes.get(uid)
        unchanged = out_path.exists() and (recorded_hash is None or recorded_hash == source_hash)
        if args.force or not unchanged:
            wav = load_audio(args.audio_root / rec["audio_path"])
            sf.write(out_path, normalize_channel(wav), SR)
        else:
            n_skipped += 1
        new_hashes[uid] = source_hash
        new_rec = dict(rec)
        new_rec["audio_path"] = str(out_path.relative_to(args.output_dir))
        new_rec["source"] = new_rec.get("source", "") + "_normalized"
        new_index.append(new_rec)

    with open(args.output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)
        f.write("\n")
    save_hash_cache(hashes_path, new_hashes)
    print(
        f"Done. {len(new_index)} normalized utterance(s) -> {args.output_dir} "
        f"({n_skipped} unchanged, {len(new_index) - n_skipped} (re)computed)"
    )


if __name__ == "__main__":
    main()
