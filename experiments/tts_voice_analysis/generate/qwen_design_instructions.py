#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Convert each character's voice_mapping.json `description` into a short
voice-design instruction for Qwen3-TTS's /generate-voice-design endpoint.

Investigation behind this step: the endpoint's own schema describes `instruct`
as a natural-language voice/style instruction with a one-sentence example
("Speak in a calm, warm, elderly female voice") - not a biography. Empirically
it doesn't error on a long instruct (tested with voice_mapping.json's longest
description, 437 chars), but voice_mapping descriptions mix real vocal cues
(accent, pitch, pace - already present for some characters, e.g. Marta) with
irrelevant biographical facts (job, city of residence) that dilute a style
instruction and, since this endpoint has no persistent voice concept, plausibly
hurt how consistently repeated calls land on the same-sounding voice. So: use
an LLM to distill (description, gender) down to one short (~12-25 word) sentence
naming only what would actually change how the voice sounds - gender, an
age register, accent/nationality only if it affects pronunciation, and a couple
of vocal/delivery qualities - matching the API's own example's register.

This mirrors design_voices.py's ElevenLabs flow, which used voice_mapping's
`description` directly as the voice-design prompt - so this is the same source
of truth, just distilled for a model tuned on shorter style instructions.

Content-hash-based caching (see content_cache.py): an instruction is only
regenerated if (description, gender, llm_model) changed since the last run.

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.qwen_design_instructions
    python3 -m generate.qwen_design_instructions --character Marta
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from lib.content_cache import hash_text, load_hash_cache, save_hash_cache
from lib.tts_common import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_LLM_MODEL,
    load_voice_mapping,
    sanitize,
)
from lib.tts_common import DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR

DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "qwen_designed"

INSTRUCTION_SYSTEM_PROMPT = (
    "You convert a character's biographical description into a short voice-design "
    "instruction for a text-to-speech system whose prompts look like 'Speak in a "
    "calm, warm, elderly female voice.' Given a character's name, gender, and "
    "description, write ONE short sentence (12-25 words) describing ONLY how the "
    "voice should sound: gender, an approximate age register (young adult / "
    "middle-aged / older adult), an accent or nationality ONLY if it plausibly "
    "affects pronunciation, and 1-3 vocal/delivery qualities (pace, pitch, warmth, "
    "energy, tone). Omit biographical facts that don't affect how the voice sounds "
    "(job title, city of residence, family details, hobbies) - unless the "
    "description gives no vocal information at all, in which case infer a "
    "plausible tone from the character's personality instead. Output ONLY the "
    "instruction sentence: no preamble, no quotes, no markdown."
)


def generate_instruction(
    client: OpenAI, model: str, name: str, description: str, gender: str
) -> str:
    """Ask the LLM to distill one character's description into a voice-design instruction.

    Args:
        client: OpenAI client used to make the chat completion request.
        model: OpenAI chat model to use.
        name: Character's name.
        description: Character's biographical description from voice_mapping.json.
        gender: Character's gender.

    Returns:
        A single short (~12-25 word) voice-design instruction sentence.
    """
    context = f"Name: {name}\nGender: {gender}\nDescription: {description}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INSTRUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip().strip('"')


def process_character(
    name: str,
    char: dict,
    args: argparse.Namespace,
    openai_client: Optional[OpenAI],
    existing_hashes: dict,
) -> dict:
    """Generate (or reuse cached) voice-design instruction for one character.

    Args:
        name: Character's name.
        char: Character's voice_mapping.json entry (description, gender, etc.).
        args: Parsed CLI arguments.
        openai_client: OpenAI client used for instruction generation, or None in
            --dry-run mode.
        existing_hashes: Previously recorded input hashes, keyed by character name.

    Returns:
        A manifest entry dict for this character: name, description, gender,
        instruction, and the new hash to merge into the persistent cache.
    """
    description = char.get("description", "")
    gender = char.get("gender", "neutral")
    safe_name = sanitize(name)
    input_hash = hash_text(description, gender, args.llm_model)

    instruction_path = args.output_dir / "instructions" / f"{safe_name}.txt"
    cached = (
        instruction_path.read_text(encoding="utf-8").strip() if instruction_path.exists() else None
    )
    recorded_hash = existing_hashes.get(name)
    unchanged = cached and (recorded_hash is None or recorded_hash == input_hash)

    if unchanged and not args.force:
        instruction = cached
    elif args.dry_run:
        instruction = "<dry-run: instruction not generated>"
    else:
        instruction = generate_instruction(openai_client, args.llm_model, name, description, gender)
        instruction_path.parent.mkdir(parents=True, exist_ok=True)
        instruction_path.write_text(instruction + "\n", encoding="utf-8")

    return {
        "name": name,
        "description": description,
        "gender": gender,
        "instruction": instruction,
        "_new_hash": (name, input_hash),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--character", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY environment variable not set")
    return args


def main() -> None:
    """Generate voice-design instructions for every requested character."""
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

    (args.output_dir / "instructions").mkdir(parents=True, exist_ok=True)
    hashes_path = args.output_dir / "instructions_hashes.json"
    existing_hashes = load_hash_cache(hashes_path)
    openai_client = None if args.dry_run else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Processing {len(names)} character(s) -> {args.output_dir}")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_character,
                name,
                characters[name],
                args,
                openai_client,
                existing_hashes,
            ): name
            for name in names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Instructions"):
            name = futures[future]
            try:
                results.append(future.result())
            except Exception as e:  # noqa: BLE001 - report and continue with other characters
                print(f"  ERROR processing {name}: {e}", file=sys.stderr)
                results.append({"name": name, "error": str(e)})

    if not args.dry_run:
        for entry in results:
            if "_new_hash" in entry:
                key, h = entry.pop("_new_hash")
                existing_hashes[key] = h
        save_hash_cache(hashes_path, existing_hashes)
    else:
        for entry in results:
            entry.pop("_new_hash", None)

    manifest_path = args.output_dir / "instructions_manifest.json"
    if not args.dry_run:
        existing_manifest = {}
        if manifest_path.exists():
            existing_manifest = {e["name"]: e for e in json.loads(manifest_path.read_text())}
        for entry in results:
            existing_manifest[entry["name"]] = entry
        full_manifest = sorted(existing_manifest.values(), key=lambda e: e["name"])
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Manifest written to {manifest_path} ({len(full_manifest)} character(s) total)")

    n_errors = sum(1 for e in results if "error" in e)
    print(f"\nDone. {len(results)} character(s) processed, {n_errors} error(s).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
