#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Clone ~100 LibriTTS speakers via Qwen3-TTS (ICL mode) on a fresh set of
LLM-written utterances, as the 3rd system in the voice-diversity comparison
(ElevenLabs-generated vs. Qwen3-TTS-cloned vs. VCTK-natural). See RESULTS.md.

Text generation (OpenAI) runs with normal concurrency since it doesn't touch
the Qwen server; Qwen3-TTS calls run strictly sequentially, one at a time, to
stay conservative with a shared internal GPU box (each call takes ~10s).

Requires OPENAI_API_KEY and a reachable Qwen3-TTS server (see qwen_tts_client.py).

Usage (from experiments/tts_voice_analysis/):
    python3 -m generate.qwen_speakers
    python3 -m generate.qwen_speakers --limit 3
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

from lib import qwen_tts_client
from lib.provenance import git_head_commit, git_repo_dirty, sha256_text
from lib.tts_common import DEFAULT_LLM_MODEL, REPO_ROOT, sanitize
from lib.tts_common import DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR

DEFAULT_REFERENCE_DIR = BASE_OUTPUT_DIR / "libritts_reference"
DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "qwen"
DEFAULT_N_UTTERANCES = 8

DIVERSITY_SYSTEM_PROMPT = (
    "You write short, natural spoken sentences for a voice-cloning benchmark. Write {n} "
    "DIFFERENT short sentences (1 sentence each, 8-20 words) covering different everyday "
    "topics (e.g. work, weather, food, an opinion, a question, reacting to news, small "
    "talk) so the set is lexically varied - do not write variations of the same sentence. "
    "The sentences are not tied to any character or persona - neutral, generic, casual "
    "spoken style. No quotation marks, no stage directions, no markdown, no emoji, no "
    "numbering - plain text only, ready to be read aloud by a text-to-speech system. "
    'Respond with a JSON object: {{"utterances": ["...", "...", ...]}} containing '
    "exactly {n} strings."
)


def generate_texts(client: OpenAI, model: str, n: int) -> list[str]:
    """Ask the LLM for n lexically distinct, persona-free utterances, in one call.

    Args:
        client: OpenAI client used to make the chat completion request.
        model: OpenAI chat model to use.
        n: Number of distinct utterances to request.

    Returns:
        A list of n lexically distinct, persona-free utterance strings.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": DIVERSITY_SYSTEM_PROMPT.format(n=n)}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    utterances = [u.strip() for u in data.get("utterances", []) if u and u.strip()]
    if len(utterances) < n:
        raise ValueError(f"LLM returned {len(utterances)} utterances, expected {n}")
    return utterances[:n]


def get_or_generate_texts(
    speaker_id: str, args: argparse.Namespace, openai_client: Optional[OpenAI]
) -> list[str]:
    """Load cached utterance text for a speaker, or generate + cache it.

    Args:
        speaker_id: Identifier of the LibriTTS speaker these utterances are for.
        args: Parsed CLI arguments.
        openai_client: OpenAI client used for text generation, or None in --dry-run mode.

    Returns:
        A list of n_utterances utterance strings for this speaker.
    """
    text_path = args.output_dir / "texts" / f"{speaker_id}.json"
    cached = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else None
    if cached is not None and len(cached) == args.n_utterances and not args.force_text:
        return cached
    if args.dry_run:
        return [f"<dry-run: utterance {i + 1} not generated>" for i in range(args.n_utterances)]
    utterances = generate_texts(openai_client, args.llm_model, args.n_utterances)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        json.dumps(utterances, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return utterances


def clone_speaker(speaker: dict, utterances: list[str], args: argparse.Namespace) -> dict:
    """Synthesize every utterance for one speaker via Qwen3-TTS, sequentially.

    Args:
        speaker: Speaker's reference-index entry (speaker_id, ref_audio_path, ref_text).
        utterances: The texts to synthesize with this speaker's cloned voice.
        args: Parsed CLI arguments.

    Returns:
        A manifest entry dict for this speaker: speaker_id, utterances, audio_paths,
        ref_audio_path, ref_text.
    """
    speaker_id = speaker["speaker_id"]
    safe_id = sanitize(speaker_id)
    ref_audio_path = args.reference_dir / speaker["ref_audio_path"]

    audio_dir = args.output_dir / "audio" / safe_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []
    for i, text in enumerate(utterances, start=1):
        out_path = audio_dir / f"{i:02d}.wav"
        audio_paths.append(str(out_path.relative_to(args.output_dir)))
        if args.dry_run or (out_path.exists() and not args.force):
            continue
        wav_bytes = qwen_tts_client.generate_audio(
            ref_audio_path=ref_audio_path,
            text=text,
            ref_text=speaker["ref_text"],
            language="English",
            x_vector_only_mode=False,
            base_url=args.qwen_base_url,
        )
        out_path.write_bytes(wav_bytes)
    return {
        "speaker_id": speaker_id,
        "utterances": utterances,
        "audio_paths": audio_paths,
        "ref_audio_path": speaker["ref_audio_path"],
        "ref_text": speaker["ref_text"],
    }


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-utterances", type=int, default=DEFAULT_N_UTTERANCES)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--qwen-base-url", default=qwen_tts_client.DEFAULT_BASE_URL)
    parser.add_argument(
        "--text-concurrency",
        type=int,
        default=8,
        help="LLM call concurrency (OpenAI, not Qwen)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N speakers")
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
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY environment variable not set")
    return args


def main() -> None:
    """Generate texts (concurrent) then clone each speaker via Qwen3-TTS (sequential)."""
    args = parse_arguments()
    speakers = json.loads((args.reference_dir / "index.json").read_text(encoding="utf-8"))
    if args.limit:
        speakers = speakers[: args.limit]

    for sub in ("audio", "texts"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        version = qwen_tts_client.get_version(args.qwen_base_url)
        run_provenance = {
            "generated_at": datetime.now().isoformat(),
            "git_commit": git_head_commit(REPO_ROOT),
            "git_dirty": git_repo_dirty(REPO_ROOT),
            "qwen_base_url": args.qwen_base_url,
            "qwen_server_version": version,
            "cloning_mode": "ICL (ref_text)",
            "language": "English",
            "llm_model": args.llm_model,
            "n_utterances": args.n_utterances,
            "diversity_system_prompt_template": DIVERSITY_SYSTEM_PROMPT,
            "diversity_system_prompt_sha256": sha256_text(DIVERSITY_SYSTEM_PROMPT),
        }
        with open(args.output_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(run_provenance, f, indent=2, ensure_ascii=False)
            f.write("\n")

    openai_client = None if args.dry_run else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(
        f"Generating text for {len(speakers)} speaker(s) "
        f"({args.text_concurrency}-way concurrent)..."
    )
    with ThreadPoolExecutor(max_workers=args.text_concurrency) as pool:
        futures = {
            pool.submit(get_or_generate_texts, s["speaker_id"], args, openai_client): s
            for s in speakers
        }
        texts_by_speaker = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Text"):
            s = futures[future]
            texts_by_speaker[s["speaker_id"]] = future.result()

    print(
        f"Cloning {len(speakers)} speaker(s) x {args.n_utterances} utterances via Qwen3-TTS "
        f"(sequential, ~10s/call)..."
    )
    manifest = []
    for s in tqdm(speakers, desc="Cloning"):
        try:
            manifest.append(clone_speaker(s, texts_by_speaker[s["speaker_id"]], args))
        except Exception as e:  # noqa: BLE001 - report and continue with other speakers
            print(f"  ERROR cloning {s['speaker_id']}: {e}", file=sys.stderr)
            manifest.append({"speaker_id": s["speaker_id"], "error": str(e)})

    n_errors = sum(1 for e in manifest if "error" in e)
    print(f"\nDone. {len(manifest)} speaker(s) processed, {n_errors} error(s).")

    manifest_path = args.output_dir / "manifest.json"
    if not args.dry_run:
        # Merge into the existing manifest (additive, keyed by speaker_id) - a
        # --limit/--character-filtered run must not drop every other speaker's entry.
        existing_manifest = {}
        if manifest_path.exists():
            existing_manifest = {e["speaker_id"]: e for e in json.loads(manifest_path.read_text())}
        for entry in manifest:
            existing_manifest[entry["speaker_id"]] = entry
        full_manifest = sorted(existing_manifest.values(), key=lambda e: e["speaker_id"])

        flat_index = [
            {
                "utterance_id": f"{e['speaker_id']}__{i:02d}",
                "group": e["speaker_id"],
                "audio_path": audio_path,
                "text": text,
                "source": "qwen_tts_cloned",
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
            f"({len(full_manifest)} speaker(s) total)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
