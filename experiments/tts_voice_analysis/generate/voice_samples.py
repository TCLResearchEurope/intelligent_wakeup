#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Generate two ElevenLabs audio samples per character voice in dataset/config/voice_mapping.json:

  1. "phoneme"  - the same phonetically rich passage read by every voice, for acoustic /
                  ML feature comparison and clustering across voices.
  2. "intro"    - a short first-person character introduction ("Hi, I'm ... and I ..."),
                  written per character by an LLM from the character's description and
                  (when available) their backstory file under dataset/config/characters/.

Only the ElevenLabs backend is used, since it is the only backend where every character
has a genuinely distinct designed voice_id (Kokoro/OpenAI backends reuse a small pool of
stock voices across many characters).

The phoneme sample is generated once per unique ElevenLabs voice_id (not once per
character) since it exists purely to compare voices acoustically; duplicate characters
sharing a voice_id are recorded in manifest.json instead of re-synthesizing identical audio.
The intro sample is generated per character, since the text itself differs per character.

Requires ELEVENLABS_API_KEY and OPENAI_API_KEY environment variables.

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.voice_samples
    python3 -m generate.voice_samples \\
        --character Aisha --character Timo
    python3 -m generate.voice_samples --limit 5 --dry-run
    python3 -m generate.voice_samples --force-text --force
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

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

# Same text is read by every voice so downstream ML/acoustic features are comparable
# across voices rather than confounded by content. Strings together several classic
# pangrams plus sibilants, fricatives, nasals, diphthongs and digits for broad phoneme
# coverage while staying short enough to keep per-voice generation cost low.
PHONEME_TEXT = (
    "The quick brown fox jumps over the lazy dog, while five boxing wizards jump quickly. "
    "Pack my box with a dozen liquor jugs, then judge the vexed sphinx of black quartz. "
    "She sells seashells by the seashore, though this thin, healthy youth thoroughly enjoys "
    "a smooth, rough journey through Zurich and Zagreb. Voice quality varies with pitch, "
    "rhythm, and the subtle texture of vowels like 'oo', 'ee', and 'ah'. Numbers such as "
    "one, two, three, seven, and eleven roll easily off the tongue."
)

INTRO_SYSTEM_PROMPT = (
    "You write extremely short, natural, first-person spoken introductions for voice-over "
    "character samples used in a synthetic conversation dataset. Given a character's name "
    "and description, write 1-2 sentences (25-40 words total) in their voice, starting with "
    'a greeting and their name, e.g. "Hi, I\'m ... and I ...". Casual spoken style, no '
    "quotation marks, no stage directions, no markdown, no emoji - plain text only, ready "
    "to be read aloud by a text-to-speech system."
)


def generate_intro_text(
    client: OpenAI, model: str, name: str, description: str, backstory: Optional[str]
) -> str:
    """Ask the LLM to write a short first-person introduction for this character.

    Args:
        client: OpenAI client used to make the chat completion request.
        model: OpenAI chat model to use.
        name: Character's name.
        description: Character's short description.
        backstory: Character's full backstory text, if available (truncated to 2000 chars).

    Returns:
        A short (1-2 sentence) first-person introduction, spoken in the character's voice.
    """
    context = f"Name: {name}\nShort description: {description}"
    if backstory:
        context += f"\n\nFull backstory:\n{backstory[:2000]}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INTRO_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def process_character(
    name: str,
    char: dict,
    args: argparse.Namespace,
    openai_client: Optional[OpenAI],
    phoneme_owners: dict,
    phoneme_lock: threading.Lock,
) -> dict:
    """Generate (or skip existing) samples for one character.

    Args:
        name: Character's name.
        char: Character's voice_mapping.json entry (description, voices, etc.).
        args: Parsed CLI arguments.
        openai_client: OpenAI client used for intro text generation, or None in
            --dry-run mode.
        phoneme_owners: Shared dict mapping voice_id to the name of the character that
            "owns" (first generated) the phoneme sample for that voice, so duplicate
            voice_ids reuse the same audio instead of resynthesizing it.
        phoneme_lock: Lock guarding concurrent access to phoneme_owners across threads.

    Returns:
        A manifest entry dict for this character: name, voice_id, description,
        phoneme_sample, phoneme_sample_owner, intro_sample, intro_text, and related
        provenance fields. If the character has no ElevenLabs voice_id, returns a dict
        with just name and skipped_reason.
    """
    voice_id = char.get("voices", {}).get("ElevenLabs", {}).get("voice_id")
    if not voice_id:
        return {"name": name, "skipped_reason": "no ElevenLabs voice_id"}

    description = char.get("description", "")
    safe_name = sanitize(name)
    entry = {"name": name, "voice_id": voice_id, "description": description}

    # --- Sample 1: phoneme-rich passage, deduplicated by voice_id ---
    phoneme_dir = args.output_dir / "phoneme"
    with phoneme_lock:
        is_first_for_voice = voice_id not in phoneme_owners
        if is_first_for_voice:
            phoneme_owners[voice_id] = name
        owner = phoneme_owners[voice_id]

    phoneme_path = phoneme_dir / f"{voice_id}__{sanitize(owner)}.mp3"
    entry["phoneme_sample"] = str(phoneme_path.relative_to(args.output_dir))
    entry["phoneme_sample_owner"] = owner
    if owner != name:
        entry["phoneme_sample_note"] = f"reused from '{owner}' (same voice_id)"

    if is_first_for_voice and not args.dry_run and (args.force or not phoneme_path.exists()):
        audio = elevenlabs_tts(PHONEME_TEXT, voice_id, args.elevenlabs_api_key, args.tts_model_id)
        phoneme_path.write_bytes(audio)

    # --- Sample 2: character introduction (per-character text) ---
    text_path = args.output_dir / "intro_texts" / f"{safe_name}.txt"
    audio_path = args.output_dir / "intro" / f"{safe_name}__{voice_id}.mp3"
    entry["intro_sample"] = str(audio_path.relative_to(args.output_dir))

    backstory_path = args.config_dir / "characters" / f"{name.lower()}.txt"
    entry["description_source"] = file_provenance(REPO_ROOT, backstory_path)

    if text_path.exists() and not args.force_text:
        intro_text = text_path.read_text(encoding="utf-8").strip()
    elif args.dry_run:
        intro_text = "<dry-run: text not generated>"
    else:
        backstory = load_backstory(args.config_dir, name)
        intro_text = generate_intro_text(
            openai_client, args.llm_model, name, description, backstory
        )
        text_path.write_text(intro_text + "\n", encoding="utf-8")
    entry["intro_text"] = intro_text

    if not args.dry_run and (args.force or not audio_path.exists()):
        audio = elevenlabs_tts(intro_text, voice_id, args.elevenlabs_api_key, args.tts_model_id)
        audio_path.write_bytes(audio)

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
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing voice_mapping.json and characters/ (default: dataset/config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write audio/text/manifest into "
        "(default: experiments/tts_voice_analysis/output)",
    )
    parser.add_argument(
        "--character",
        action="append",
        default=None,
        help="Process only this character (repeatable). "
        "Default: every character in voice_mapping.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N characters (for smoke testing)",
    )
    parser.add_argument(
        "--tts-model-id",
        default=DEFAULT_TTS_MODEL_ID,
        help=f"ElevenLabs TTS model_id (default: {DEFAULT_TTS_MODEL_ID})",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"OpenAI chat model used to write intro text (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent ElevenLabs requests (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate audio even if the output file already exists",
    )
    parser.add_argument(
        "--force-text",
        action="store_true",
        help="Regenerate intro text via the LLM even if a cached .txt already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without calling any API",
    )
    args = parser.parse_args()
    args.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not args.dry_run and not args.elevenlabs_api_key:
        parser.error("ELEVENLABS_API_KEY environment variable not set")
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY environment variable not set")
    return args


def main() -> None:
    """Generate phoneme and intro samples for every requested character."""
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

    for sub in ("phoneme", "intro", "intro_texts"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    run_provenance = build_run_provenance(REPO_ROOT, args.config_dir)
    run_provenance["generated_at"] = datetime.now().isoformat()
    run_provenance["generation_settings"] = {
        "tts_model_id": args.tts_model_id,
        "llm_model": args.llm_model,
        "phoneme_text": PHONEME_TEXT,
        "phoneme_text_sha256": sha256_text(PHONEME_TEXT),
        "intro_system_prompt": INTRO_SYSTEM_PROMPT,
        "intro_system_prompt_sha256": sha256_text(INTRO_SYSTEM_PROMPT),
    }
    if run_provenance["git_dirty"]:
        print(
            "WARNING: repo has uncommitted changes; provenance.json records content "
            "hashes but the git commit may not reflect the exact state used."
        )
    if run_provenance["voice_mapping"] and run_provenance["voice_mapping"]["git_dirty"]:
        print("WARNING: voice_mapping.json has uncommitted changes vs its last commit.")
    if run_provenance["characters_dir"]["git_dirty"]:
        print("WARNING: dataset/config/characters/ has uncommitted changes vs its last commit.")
    if not args.dry_run:
        with open(args.output_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(run_provenance, f, indent=2, ensure_ascii=False)
            f.write("\n")

    openai_client = None if args.dry_run else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    phoneme_owners: dict = {}
    phoneme_lock = threading.Lock()

    print(f"Processing {len(names)} character(s) -> {args.output_dir}")
    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_character,
                name,
                characters[name],
                args,
                openai_client,
                phoneme_owners,
                phoneme_lock,
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

    manifest.sort(key=lambda e: e["name"])
    manifest_path = args.output_dir / "manifest.json"
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")

    n_errors = sum(1 for e in manifest if "error" in e)
    n_skipped = sum(1 for e in manifest if "skipped_reason" in e)
    unique_voices = len(phoneme_owners)
    print(
        f"\nDone. {len(manifest)} character(s) processed, {unique_voices} unique voice(s), "
        f"{n_skipped} skipped, {n_errors} error(s)."
    )
    if not args.dry_run:
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
