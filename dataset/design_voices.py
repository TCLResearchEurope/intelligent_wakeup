#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe.

Interactive ElevenLabs voice setup for voice_mapping.json.

For each character with a missing ElevenLabs voice_id:
  1. Check if a voice named 'int_wakeup_<name>' already exists in your EL account.
  2. If yes  → use it automatically.
  3. If no   → ask whether to design a new one, play all 3 preview samples,
               let you pick the best, save it as a permanent voice, and update
               voice_mapping.json.

This script is designed to be shared: anyone replicating the dataset can run it
to populate their own voices without needing the original author's voice IDs.

Usage:
    python3 dataset/design_voices.py
    python3 dataset/design_voices.py --character Aisha        # single character
    python3 dataset/design_voices.py --new-characters-only     # everyone missing an ElevenLabs voice_id
    python3 dataset/design_voices.py --guidance-scale 3         # more varied/creative previews
    python3 dataset/design_voices.py --seed 42                  # reproducible generation
"""

import argparse
import base64
import json
import os
import random
import select
import subprocess
import sys
import tempfile
import termios
import time
import tty
from pathlib import Path

import requests

VOICE_MAPPING = Path(__file__).parent / "config/voice_mapping.json"
API_BASE = "https://api.elevenlabs.io/v1"
MODEL_ID = "eleven_multilingual_ttv_v2"
# ElevenLabs docs: "use longer, more detailed prompts at lower Guidance Scale" —
# our voice descriptions are long/detailed, and the API default (5) also
# narrows the 3 previews toward near-identical samples. Lower value trades
# strict prompt adherence for more natural, more varied previews.
# Override per run with --guidance-scale.
DEFAULT_GUIDANCE_SCALE = 5
# ElevenLabs' own docs claim the seed range is 0-4294967295 (unsigned 32-bit),
# but the live API actually validates it as a signed 32-bit int and rejects
# anything above this with a 422 ("Input should be less than or equal to
# 2147483647") — confirmed directly against the endpoint. Trust this bound,
# not the documented one.
MAX_SEED = 2147483647


# ---------------------------------------------------------------------------
# ElevenLabs helpers
# ---------------------------------------------------------------------------


def get_headers(api_key: str) -> dict:
    """Return auth headers for ElevenLabs API requests."""
    return {"xi-api-key": api_key, "Content-Type": "application/json"}


def list_existing_voices(api_key: str) -> dict[str, str]:
    """Return {voice_name: voice_id} for all voices in the account."""
    r = requests.get(f"{API_BASE}/voices", headers={"xi-api-key": api_key}, timeout=30)
    r.raise_for_status()
    return {v["name"]: v["voice_id"] for v in r.json().get("voices", [])}


def design_previews(
    prompt: str,
    api_key: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
) -> list[dict]:
    """Call the design endpoint and return list of preview dicts.

    Args:
        prompt: Voice description sent to the design endpoint.
        api_key: ElevenLabs API key.
        guidance_scale: How closely the model follows the prompt. Lower
            values give more creative freedom and more varied previews;
            higher values stick closer to the prompt but can sound robotic
            and collapse the 3 previews toward near-identical samples.
        seed: Optional integer for reproducible generation. Same seed with
            the same inputs should return the same voice (ElevenLabs
            documents this as best-effort, not guaranteed). Leave unset for
            fresh randomization each run.
    """
    body = {
        "voice_description": prompt,
        "model_id": MODEL_ID,
        "auto_generate_text": True,
        "guidance_scale": guidance_scale,
    }
    if seed is not None:
        body["seed"] = seed

    r = requests.post(
        f"{API_BASE}/text-to-voice/design",
        headers=get_headers(api_key),
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("previews", [])


def save_permanent_voice(
    name: str, prompt: str, generated_voice_id: str, api_key: str
) -> str:
    """Save a preview as a permanent voice and return its voice_id."""
    r = requests.post(
        f"{API_BASE}/text-to-voice",
        headers=get_headers(api_key),
        json={
            "voice_name": f"int_wakeup_{name}",
            "voice_description": prompt,
            "generated_voice_id": generated_voice_id,
        },
        timeout=30,
    )
    r.raise_for_status()
    voice_id = r.json().get("voice_id")
    if not voice_id:
        raise ValueError("No voice_id in response")
    return voice_id


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------


def _wait_for_proc_or_skip(proc: subprocess.Popen) -> bool:
    """Wait for proc to finish, or kill it when user presses 'c'. Returns True if skipped."""
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (termios.error, AttributeError):
        proc.wait()
        return False

    skipped = False
    try:
        tty.setraw(fd)
        while proc.poll() is None:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "c":
                    proc.terminate()
                    skipped = True
                    break
                elif ch == "\x03":  # Ctrl+C
                    proc.terminate()
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    proc.wait()
    return skipped


def play_audio(audio_bytes: bytes) -> bool:
    """Play MP3 bytes using ffplay, mpg123, or mplayer. Returns True if user skipped."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        for cmd in [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            ["mpg123", "-q", tmp_path],
            ["mplayer", "-really-quiet", tmp_path],
        ]:
            try:
                proc = subprocess.Popen(cmd)
                return _wait_for_proc_or_skip(proc)
            except FileNotFoundError:
                continue
        print("    [could not play audio — install ffplay or mpg123 to hear previews]")
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# voice_mapping.json I/O
# ---------------------------------------------------------------------------


def load_vm() -> dict:
    """Load voice_mapping.json and return its contents."""
    with open(VOICE_MAPPING, encoding="utf-8") as f:
        return json.load(f)


def save_vm(vm: dict) -> None:
    """Write vm dict back to voice_mapping.json."""
    with open(VOICE_MAPPING, "w", encoding="utf-8") as f:
        json.dump(vm, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Interactive preview selection
# ---------------------------------------------------------------------------


def _play_all_previews(previews: list[dict]) -> None:
    """Play each preview sample once in order. Press 'c' to skip to the next."""
    for i, preview in enumerate(previews, 1):
        print(f"\n  ► Playing sample {i}/{len(previews)}... (press 'c' to skip)")
        skipped = play_audio(base64.b64decode(preview["audio_base_64"]))
        if skipped:
            print("  [skipped]")
        time.sleep(0.3)


def _pick_preview(previews: list[dict]) -> int | str:
    """Prompt the user to pick a preview, replay one, or request a new batch.

    Returns a 0-based index to select that preview, or the string "new" to
    request a fresh batch of previews generated with a new random seed.
    """
    n = len(previews)
    while True:
        choice = (
            input(
                f"\n  Select sample [1-{n}], r<n> to replay (e.g. r2), "
                "or n for new samples: "
            )
            .strip()
            .lower()
        )
        if choice == "n":
            return "new"
        if choice.startswith("r"):
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < n:
                    print(f"  ► Replaying sample {idx + 1}...")
                    play_audio(base64.b64decode(previews[idx]["audio_base_64"]))
                else:
                    print(f"  Invalid index, choose 1-{n}")
            except ValueError:
                print("  Enter e.g. r2 to replay sample 2")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < n:
                    return idx
                print(f"  Choose between 1 and {n}")
            except ValueError:
                print(f"  Enter a number between 1 and {n}")


def _design_and_save(
    name: str,
    prompt: str,
    existing_voices: dict,
    api_key: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
) -> str | None:
    """Fetch previews, let user pick one (or regenerate), save as permanent voice.

    The user can request a fresh batch of 3 previews at the picker prompt
    ('n' for new samples), which re-rolls with a new random seed each time.
    Returns voice_id or None.
    """
    current_seed = seed
    while True:
        seed_note = f" (seed: {current_seed})" if current_seed is not None else ""
        print(f"  Fetching 3 voice previews from ElevenLabs...{seed_note}", flush=True)
        try:
            previews = design_previews(
                prompt, api_key, guidance_scale=guidance_scale, seed=current_seed
            )
        except requests.RequestException as e:
            print(f"  ERROR fetching previews: {e}")
            return None

        if not previews:
            print("  No previews returned.")
            return None

        _play_all_previews(previews)
        choice = _pick_preview(previews)
        if choice == "new":
            current_seed = random.randint(0, MAX_SEED)
            print(f"  Generating a new batch with seed {current_seed}...")
            continue
        idx = choice
        selected = previews[idx]
        break

    el_name = f"int_wakeup_{name}"
    print(f"  Saving sample {idx + 1} as '{el_name}'...", flush=True)
    try:
        voice_id = save_permanent_voice(
            name, prompt, selected["generated_voice_id"], api_key
        )
    except (requests.RequestException, ValueError) as e:
        print(f"  ERROR saving voice: {e}")
        return None

    existing_voices[el_name] = voice_id
    return voice_id


# ---------------------------------------------------------------------------
# Main per-character flow
# ---------------------------------------------------------------------------


def process_character(
    name: str,
    vm: dict,
    existing_voices: dict,
    api_key: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
) -> bool:
    """Handle one character. Returns True if voice_mapping was updated."""
    el_entry = vm["characters"][name]["voices"].setdefault("ElevenLabs", {})
    current_id = el_entry.get("voice_id", "")
    el_name = f"int_wakeup_{name}"

    if current_id:
        print(f"  {name}: already set ({current_id}) — skipping")
        return False

    if el_name in existing_voices:
        found_id = existing_voices[el_name]
        print(f"  {name}: found existing EL voice '{el_name}' ({found_id}) — using it")
        el_entry["voice_id"] = found_id
        save_vm(vm)
        return True

    print(f"\n{'─'*60}")
    print(f"  Character  : {name}")
    print(f"  Description: {vm['characters'][name].get('description', '')}")
    prompt = vm["characters"][name].get("description", name)
    print(f"  Voice prompt: {prompt}")

    answer = input("  Design a new voice for this character? [Y/n] ").strip().lower()
    if answer in ("n", "no"):
        print("  Skipped.")
        return False

    voice_id = _design_and_save(
        name, prompt, existing_voices, api_key, guidance_scale=guidance_scale, seed=seed
    )
    if not voice_id:
        return False

    print(f"  ✓ Created: {voice_id}")
    el_entry["voice_id"] = voice_id
    save_vm(vm)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive ElevenLabs voice setup for voice_mapping.json."
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--character",
        default=None,
        help="Process a single named character only.",
    )
    target_group.add_argument(
        "--new-characters-only",
        action="store_true",
        help="Process every character whose ElevenLabs voice_id is missing/empty, "
        "instead of walking the entire roster.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=f"How closely previews follow the voice_description (default: "
        f"{DEFAULT_GUIDANCE_SCALE}). Lower values (e.g. 2-3) give more creative, "
        "more varied previews; higher values stick closer to the prompt but can "
        "sound robotic and make the 3 previews collapse toward each other.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Optional integer seed (0-{MAX_SEED}) for reproducible preview "
        "generation. Same seed with the same inputs should return the same "
        "voice (best-effort per ElevenLabs, not guaranteed). Omit for fresh "
        "randomization.",
    )
    args = parser.parse_args()
    if args.seed is not None and not (0 <= args.seed <= MAX_SEED):
        parser.error(f"--seed must be between 0 and {MAX_SEED}, got {args.seed}")
    return args


def _characters_missing_voice(vm: dict) -> list[str]:
    """Return character names with no ElevenLabs voice_id set yet."""
    return [
        name
        for name, char in vm["characters"].items()
        if not char.get("voices", {}).get("ElevenLabs", {}).get("voice_id")
    ]


def main():
    """Run the interactive voice setup loop."""
    args = _parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY environment variable not set.")
        sys.exit(1)

    vm = load_vm()

    if args.character:
        if args.character not in vm["characters"]:
            print(f"ERROR: '{args.character}' not found in voice_mapping.json")
            sys.exit(1)
        characters = [args.character]
    elif args.new_characters_only:
        characters = _characters_missing_voice(vm)
        print(f"Found {len(characters)} character(s) missing an ElevenLabs voice_id.\n")
    else:
        characters = list(vm["characters"].keys())

    print("Fetching your ElevenLabs voice library...")
    try:
        existing_voices = list_existing_voices(api_key)
    except requests.RequestException as e:
        print(f"ERROR fetching voices: {e}")
        sys.exit(1)
    print(f"Found {len(existing_voices)} voices in your account.\n")

    updated = sum(
        process_character(
            name,
            vm,
            existing_voices,
            api_key,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        for name in characters
    )

    print(f"\n{'─'*60}")
    print(f"Done. Updated {updated} character(s) in voice_mapping.json.")


if __name__ == "__main__":
    main()
