#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Synthesize each character's existing diversity utterances (the same 8 texts
used for the "elevenlabs" system, from diversity_manifest.json) via
Qwen3-TTS's /generate-voice-design endpoint, using the short voice-design
instruction produced by generate/qwen_design_instructions.py.

Reusing the same text as the ElevenLabs system (rather than writing yet another
fresh set) isolates the actual variable of interest: same characters, same
content, different voice-generation approach (ElevenLabs full-scenario voice_id
vs. Qwen instruction-based design) - a 4th system alongside elevenlabs/
qwen_cloned/natural in analyze/multi_system_diversity.py.

Qwen3-TTS calls run strictly sequentially (same conservative approach as
generate/qwen_speakers.py) against the shared internal GPU server. Content-hash
caching: an utterance is only resynthesized if (instruction, text) changed.

Requires a reachable Qwen3-TTS server (see qwen_tts_client.py) and that
generate/qwen_design_instructions.py and generate/diversity_utterances.py have
already been run.

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.qwen_designed_speakers
    python3 -m generate.qwen_designed_speakers --character Marta
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from lib import qwen_tts_client
from lib.content_cache import hash_text, load_hash_cache, save_hash_cache
from lib.provenance import git_head_commit, git_repo_dirty
from lib.tts_common import DEFAULT_OUTPUT_DIR, REPO_ROOT, sanitize

DEFAULT_INSTRUCTIONS_DIR = DEFAULT_OUTPUT_DIR / "qwen_designed"
DEFAULT_OUTPUT_DIR_ = DEFAULT_OUTPUT_DIR / "qwen_designed"


def load_inputs(instructions_dir: Path, diversity_manifest_path: Path) -> list[dict]:
    """Join each character's voice-design instruction with its diversity utterances.

    Args:
        instructions_dir: Directory containing instructions_manifest.json, as produced
            by generate/qwen_design_instructions.py.
        diversity_manifest_path: Path to diversity_manifest.json, as produced by
            generate/diversity_utterances.py.

    Returns:
        A list of dicts, one per character with both an instruction and diversity
        utterances available: name, instruction, utterances.
    """
    instructions = {
        e["name"]: e["instruction"]
        for e in json.loads((instructions_dir / "instructions_manifest.json").read_text())
    }
    diversity = {
        e["name"]: e for e in json.loads(diversity_manifest_path.read_text()) if "utterances" in e
    }
    joined = []
    for name, instruction in instructions.items():
        if name not in diversity:
            continue
        joined.append(
            {
                "name": name,
                "instruction": instruction,
                "utterances": diversity[name]["utterances"],
            }
        )
    return joined


def clone_speaker(
    speaker: dict, args: argparse.Namespace, existing_hashes: dict, new_hashes: dict
) -> dict:
    """Synthesize every utterance for one character via Qwen3-TTS voice design.

    Args:
        speaker: Dict with this character's name, voice-design instruction, and
            diversity utterances (as produced by load_inputs).
        args: Parsed CLI arguments.
        existing_hashes: Previously recorded (instruction, text) input hashes,
            keyed by utterance_id.
        new_hashes: Dict this call writes the current run's input hashes into,
            keyed by utterance_id.

    Returns:
        A manifest entry dict for this character: name, instruction, utterances,
        audio_paths.
    """
    name = speaker["name"]
    safe_name = sanitize(name)
    audio_dir = args.output_dir / "audio" / safe_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = []
    for i, text in enumerate(speaker["utterances"], start=1):
        utterance_id = f"{name}__{i:02d}"
        out_path = audio_dir / f"{i:02d}.wav"
        audio_paths.append(str(out_path.relative_to(args.output_dir)))
        input_hash = hash_text(speaker["instruction"], text)
        recorded_hash = existing_hashes.get(utterance_id)
        unchanged = out_path.exists() and (recorded_hash is None or recorded_hash == input_hash)
        if args.dry_run:
            new_hashes[utterance_id] = input_hash
            continue
        if args.force or not unchanged:
            wav_bytes = qwen_tts_client.generate_voice_design(
                text=text,
                instruct=speaker["instruction"],
                language="English",
                base_url=args.qwen_base_url,
            )
            out_path.write_bytes(wav_bytes)
        new_hashes[utterance_id] = input_hash

    return {
        "name": name,
        "instruction": speaker["instruction"],
        "utterances": speaker["utterances"],
        "audio_paths": audio_paths,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--instructions-dir", type=Path, default=DEFAULT_INSTRUCTIONS_DIR)
    parser.add_argument(
        "--diversity-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "diversity_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_)
    parser.add_argument("--qwen-base-url", default=qwen_tts_client.DEFAULT_BASE_URL)
    parser.add_argument("--character", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Synthesize every character's diversity utterances via Qwen3-TTS voice design."""
    args = parse_arguments()
    speakers = load_inputs(args.instructions_dir, args.diversity_manifest)
    if args.character:
        missing = [c for c in args.character if c not in {s["name"] for s in speakers}]
        if missing:
            sys.exit(f"ERROR: character(s) not found in instructions/diversity manifest: {missing}")
        speakers = [s for s in speakers if s["name"] in args.character]
    if args.limit:
        speakers = speakers[: args.limit]
    if not speakers:
        sys.exit("ERROR: no characters found - run generate/qwen_design_instructions.py first")

    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)
    hashes_path = args.output_dir / "audio_hashes.json"
    existing_hashes = {} if args.force else load_hash_cache(hashes_path)
    new_hashes: dict = {}

    if not args.dry_run:
        version = qwen_tts_client.get_version(args.qwen_base_url)
        provenance = {
            "generated_at": datetime.now().isoformat(),
            "git_commit": git_head_commit(REPO_ROOT),
            "git_dirty": git_repo_dirty(REPO_ROOT),
            "qwen_base_url": args.qwen_base_url,
            "qwen_server_version": version,
            "mode": "voice-design (/generate-voice-design)",
            "language": "English",
            "text_source": ("diversity_manifest.json (same texts as the elevenlabs system)"),
        }
        with open(args.output_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(
        f"Designing {len(speakers)} voice(s) x {len(speakers[0]['utterances'])} utterances "
        f"via Qwen3-TTS voice design (sequential)..."
    )
    manifest = []
    for speaker in tqdm(speakers, desc="Designing"):
        try:
            manifest.append(clone_speaker(speaker, args, existing_hashes, new_hashes))
        except Exception as e:  # noqa: BLE001 - report and continue with other characters
            print(f"  ERROR processing {speaker['name']}: {e}", file=sys.stderr)
            manifest.append({"name": speaker["name"], "error": str(e)})

    if not args.dry_run:
        save_hash_cache(hashes_path, {**existing_hashes, **new_hashes})

    n_errors = sum(1 for e in manifest if "error" in e)
    print(f"\nDone. {len(manifest)} character(s) processed, {n_errors} error(s).")

    manifest_path = args.output_dir / "manifest.json"
    if not args.dry_run:
        # Merge into the existing manifest (additive, keyed by name) - a
        # --character-filtered run must not drop every other character's entry.
        existing_manifest = {}
        if manifest_path.exists():
            existing_manifest = {e["name"]: e for e in json.loads(manifest_path.read_text())}
        for entry in manifest:
            existing_manifest[entry["name"]] = entry
        full_manifest = sorted(existing_manifest.values(), key=lambda e: e["name"])

        flat_index = [
            {
                "utterance_id": f"{e['name']}__{i:02d}",
                "group": e["name"],
                "audio_path": audio_path,
                "text": text,
                "source": "qwen_voice_design",
            }
            for e in full_manifest
            if "audio_paths" in e
            for i, (audio_path, text) in enumerate(zip(e["audio_paths"], e["utterances"]), start=1)
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(args.output_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(flat_index, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(
            f"Manifest + flat index written to {args.output_dir} "
            f"({len(full_manifest)} character(s) total)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
