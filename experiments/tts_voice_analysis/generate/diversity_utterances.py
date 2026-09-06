#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Generate N (default 8) lexically distinct, in-character utterances per voice in
dataset/config/voice_mapping.json, and synthesize each with its ElevenLabs voice_id.

This is the input corpus for the voice-diversity / channel-confound analysis
described in experiment-setup.md: it needs several utterances per voice with
different content so within-voice and between-voice speaker-embedding distances
can be measured (a single shared sentence per voice, as used for the phoneme
sample in generate/voice_samples.py, cannot support that analysis).

All N utterances for a character are requested in a single LLM call (JSON mode)
so the model naturally spreads them across different topics instead of drifting
toward repetitive phrasing across separate calls.

Both text and audio caching are content-hash-based: text is keyed by a hash of
(description, backstory, n, llm_model), audio per-utterance by a hash of
(voice_id, tts_model_id, text). Editing a character's description/backstory or
changing its voice_id in voice_mapping.json and rerunning this script - no
flags needed - regenerates exactly what changed and nothing else.

Requires ELEVENLABS_API_KEY and OPENAI_API_KEY environment variables.

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.diversity_utterances
    python3 -m generate.diversity_utterances \\
        --character Aisha --limit 3
    python3 -m generate.diversity_utterances --dry-run --limit 5
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from lib.content_cache import hash_text, load_hash_cache, save_hash_cache
from lib.provenance import build_run_provenance, file_provenance, sha256_text
from lib.tts_common import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TTS_MODEL_ID,
    REPO_ROOT,
    elevenlabs_tts,
    load_backstory,
    load_voice_mapping,
    sanitize,
)

DEFAULT_N_UTTERANCES = 8

DIVERSITY_SYSTEM_PROMPT = (
    "You write short, natural spoken lines for a synthetic character in a voice dataset. "
    "Given a character's name and description, write {n} DIFFERENT short utterances "
    "(1 sentence each, 8-20 words) that this character might plausibly say out loud in "
    "everyday conversation. Each utterance must be about a different topic/situation "
    "(e.g. one about work, one about food, one giving an opinion, one asking a question, "
    "one reacting to news, one making small talk, etc.) so the set is lexically varied - "
    "do not write variations of the same sentence. Casual spoken style matching the "
    "character's personality, no quotation marks, no stage directions, no markdown, no "
    "emoji, no numbering - plain text only, ready to be read aloud by a text-to-speech "
    'system. Respond with a JSON object: {{"utterances": ["...", "...", ...]}} '
    "containing exactly {n} strings."
)


def generate_diversity_texts(
    client: OpenAI,
    model: str,
    name: str,
    description: str,
    backstory: Optional[str],
    n: int,
) -> list[str]:
    """Ask the LLM for n lexically distinct in-character utterances, in one call.

    Args:
        client: OpenAI client used to make the chat completion request.
        model: OpenAI chat model to use.
        name: Character's name.
        description: Character's short description.
        backstory: Character's full backstory text, if available (truncated to 2000 chars).
        n: Number of distinct utterances to request.

    Returns:
        A list of n lexically distinct in-character utterance strings.
    """
    context = f"Name: {name}\nShort description: {description}"
    if backstory:
        context += f"\n\nFull backstory:\n{backstory[:2000]}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DIVERSITY_SYSTEM_PROMPT.format(n=n)},
            {"role": "user", "content": context},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    utterances = [u.strip() for u in data.get("utterances", []) if u and u.strip()]
    if len(utterances) < n:
        raise ValueError(f"LLM returned {len(utterances)} utterances for {name}, expected {n}")
    return utterances[:n]


def process_character(
    name: str,
    char: dict,
    args: argparse.Namespace,
    openai_client: Optional[OpenAI],
    existing_text_hashes: dict,
    existing_audio_hashes: dict,
) -> dict:
    """Generate (or reuse cached) diversity utterances + audio for one character.

    Text and audio are only regenerated if their content-hash inputs changed
    since the last run (existing_text_hashes / existing_audio_hashes) - not
    merely if the cached file happens to be missing.

    Args:
        name: Character's name.
        char: Character's voice_mapping.json entry (description, voices, etc.).
        args: Parsed CLI arguments.
        openai_client: OpenAI client used for text generation, or None in --dry-run mode.
        existing_text_hashes: Previously recorded text input hashes, keyed by character name.
        existing_audio_hashes: Previously recorded audio input hashes, keyed by utterance_id.

    Returns:
        A manifest entry dict for this character: name, voice_id, description, utterances,
        audio_paths, and (unless skipped) the new hashes to merge into the persistent
        caches. If the character has no ElevenLabs voice_id, returns a dict with just
        name and skipped_reason.
    """
    voice_id = char.get("voices", {}).get("ElevenLabs", {}).get("voice_id")
    if not voice_id:
        return {"name": name, "skipped_reason": "no ElevenLabs voice_id"}

    description = char.get("description", "")
    safe_name = sanitize(name)
    entry = {"name": name, "voice_id": voice_id, "description": description}

    backstory_path = args.config_dir / "characters" / f"{name.lower()}.txt"
    entry["description_source"] = file_provenance(REPO_ROOT, backstory_path)
    backstory = load_backstory(args.config_dir, name)

    text_hash = hash_text(description, backstory or "", str(args.n_utterances), args.llm_model)
    text_path = args.output_dir / "diversity_texts" / f"{safe_name}.json"
    cached = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else None
    # No recorded hash yet (first run after adding this cache) means trust the
    # existing cached text rather than re-calling the LLM for everything - only a
    # recorded-and-mismatching hash means the inputs actually changed.
    recorded_text_hash = existing_text_hashes.get(name)
    text_unchanged = (
        cached is not None
        and len(cached) == args.n_utterances
        and (recorded_text_hash is None or recorded_text_hash == text_hash)
    )
    if text_unchanged and not args.force_text:
        utterances = cached
    elif args.dry_run:
        utterances = [
            f"<dry-run: utterance {i + 1} not generated>" for i in range(args.n_utterances)
        ]
    else:
        utterances = generate_diversity_texts(
            openai_client,
            args.llm_model,
            name,
            description,
            backstory,
            args.n_utterances,
        )
        text_path.write_text(
            json.dumps(utterances, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    entry["utterances"] = utterances
    entry["_new_text_hash"] = (name, text_hash)

    char_dir = args.output_dir / "diversity" / safe_name
    char_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []
    new_audio_hashes = {}
    for i, text in enumerate(utterances, start=1):
        utterance_id = f"{name}__{i:02d}"
        audio_path = char_dir / f"{i:02d}.mp3"
        audio_paths.append(str(audio_path.relative_to(args.output_dir)))
        audio_hash = hash_text(voice_id, args.tts_model_id, text)
        recorded_audio_hash = existing_audio_hashes.get(utterance_id)
        audio_unchanged = audio_path.exists() and (
            recorded_audio_hash is None or recorded_audio_hash == audio_hash
        )
        if not args.dry_run and (args.force or not audio_unchanged):
            audio = elevenlabs_tts(text, voice_id, args.elevenlabs_api_key, args.tts_model_id)
            audio_path.write_bytes(audio)
        new_audio_hashes[utterance_id] = audio_hash
    entry["audio_paths"] = audio_paths
    entry["_new_audio_hashes"] = new_audio_hashes

    return entry


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        The parsed arguments, with elevenlabs_api_key resolved from the environment.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--character", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-utterances", type=int, default=DEFAULT_N_UTTERANCES)
    parser.add_argument("--tts-model-id", default=DEFAULT_TTS_MODEL_ID)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate audio even if it already exists",
    )
    parser.add_argument(
        "--force-text",
        action="store_true",
        help="Regenerate utterance text even if cached",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not args.dry_run and not args.elevenlabs_api_key:
        parser.error("ELEVENLABS_API_KEY environment variable not set")
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY environment variable not set")
    return args


def main() -> None:
    """Generate diversity utterances + audio for every requested character."""
    args = parse_arguments()
    characters = load_voice_mapping(args.config_dir)

    if args.character:
        missing = [c for c in args.character if c not in characters]
        if missing:
            sys.exit(f"ERROR: character(s) not found in voice_mapping.json: {missing}")
        names = args.character
    else:
        names = list(characters.keys())
    if args.limit:
        names = names[: args.limit]

    for sub in ("diversity", "diversity_texts"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    run_provenance = build_run_provenance(REPO_ROOT, args.config_dir)
    run_provenance["generated_at"] = datetime.now().isoformat()
    run_provenance["generation_settings"] = {
        "tts_model_id": args.tts_model_id,
        "llm_model": args.llm_model,
        "n_utterances": args.n_utterances,
        "diversity_system_prompt_template": DIVERSITY_SYSTEM_PROMPT,
        "diversity_system_prompt_sha256": sha256_text(DIVERSITY_SYSTEM_PROMPT),
    }
    if run_provenance["git_dirty"]:
        print(
            "WARNING: repo has uncommitted changes; provenance records content hashes "
            "but the git commit may not reflect the exact state used."
        )
    if not args.dry_run:
        with open(args.output_dir / "diversity_provenance.json", "w", encoding="utf-8") as f:
            json.dump(run_provenance, f, indent=2, ensure_ascii=False)
            f.write("\n")

    openai_client = None if args.dry_run else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    text_hashes_path = args.output_dir / "diversity_texts_hashes.json"
    audio_hashes_path = args.output_dir / "diversity_audio_hashes.json"
    existing_text_hashes = load_hash_cache(text_hashes_path)
    existing_audio_hashes = load_hash_cache(audio_hashes_path)

    print(
        f"Processing {len(names)} character(s) x {args.n_utterances} utterances "
        f"-> {args.output_dir}"
    )
    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_character,
                name,
                characters[name],
                args,
                openai_client,
                existing_text_hashes,
                existing_audio_hashes,
            ): name
            for name in names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            name = futures[future]
            try:
                manifest.append(future.result())
            except Exception as e:  # noqa: BLE001 - report and continue with other characters
                print(f"  ERROR processing {name}: {e}", file=sys.stderr)
                manifest.append({"name": name, "error": str(e)})

    # Merge this run's hash updates into the persistent caches (additive - characters
    # not processed this run, e.g. via --character, keep their existing cache entries).
    if not args.dry_run:
        for entry in manifest:
            if "_new_text_hash" in entry:
                key, text_hash = entry.pop("_new_text_hash")
                existing_text_hashes[key] = text_hash
            if "_new_audio_hashes" in entry:
                existing_audio_hashes.update(entry.pop("_new_audio_hashes"))
        save_hash_cache(text_hashes_path, existing_text_hashes)
        save_hash_cache(audio_hashes_path, existing_audio_hashes)
    else:
        for entry in manifest:
            entry.pop("_new_text_hash", None)
            entry.pop("_new_audio_hashes", None)

    manifest.sort(key=lambda e: e["name"])
    n_errors = sum(1 for e in manifest if "error" in e)
    n_skipped = sum(1 for e in manifest if "skipped_reason" in e)
    print(
        f"\nDone. {len(manifest)} character(s) processed, {n_skipped} skipped, {n_errors} error(s)."
    )

    manifest_path = args.output_dir / "diversity_manifest.json"
    if not args.dry_run:
        # Merge into the existing manifest (additive, keyed by name) rather than
        # overwriting it - a --character-filtered run must not drop every other
        # character's entry from the file.
        existing_manifest = {}
        if manifest_path.exists():
            existing_manifest = {e["name"]: e for e in json.loads(manifest_path.read_text())}
        for entry in manifest:
            existing_manifest[entry["name"]] = entry
        full_manifest = sorted(existing_manifest.values(), key=lambda e: e["name"])
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Manifest written to {manifest_path} ({len(full_manifest)} character(s) total)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
