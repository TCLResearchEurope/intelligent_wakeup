#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe.

Build the audio and transcript assets for the docs/ demo page.

Source material lives outside the repository (dataset releases and demo bundles
under ~/Databases/intelligent_wakeup), so this script is the record of which
scenes the page shows and where each one came from. Re-run it to refresh
docs/assets and docs/data after picking different scenes.

Runs on Python 3.9 and later, so any interpreter to hand can check the sources.
Turn boundaries are only recovered from the audio when the interpreter also has
faster-whisper installed; see align_turn_starts.

Usage:
    python3 docs/build_assets.py                # encode everything
    python3 docs/build_assets.py --check        # report sources, encode nothing
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

DOCS = Path(__file__).resolve().parent
DB = Path.home() / "Databases" / "intelligent_wakeup"

RELEASE = DB / "intelligent_wakeup-dataset-v0.4.0-20260717" / "intelligent_wakeup-dataset-v0.4.0"
CHARACTERS = DB / "int_wakeup-characters-20260826" / "int_wakeup-characters"
NATURALNESS = (
    DB
    / "int_wakeup-naturalness_eval-input_data-v0.1.0-20280824"
    / "int_wakeup-naturalness_eval-input_data-v0.1.0"
)
VOICE_MAPPING = DOCS.parent / "dataset" / "config" / "voice_mapping.json"

# Bitrate for scene audio. These are 2-3 minute mono conversations where the
# point is naturalness of speech, not fidelity of music, so 96k mono is
# transparent enough while keeping the whole page well under 40 MB.
SCENE_BITRATE = "96k"
# Voice cards are the one place a listener judges timbre directly, and they are
# only ~10 s each, so they keep their original 128k.
VOICE_BITRATE = "128k"

# Full conversations shown with a wake-up timeline and transcript.
# Chosen from the v0.4.0 release for: an explicit wake-up followed by a
# contextual follow-up, no placeholder speaker names, no leaked nested
# exchanges, and low filler density.
FEATURED = [
    {
        "id": "reel-filming",
        "title": "Planning a travel reel",
        "blurb": (
            "Timo and Anja are unsure how to shoot food. Anja names the assistant once; "
            "her next question — about lighting — never names it again, and it still answers."
        ),
        "scenario": "ReelFilming",
    },
    {
        "id": "books-to-read",
        "title": "Building a reading list",
        "blurb": (
            "A reading list assembled out loud. The assistant is addressed repeatedly here, "
            "and holds one running list across both speakers."
        ),
        "scenario": "BooksToRead",
    },
]

# The with/without pair. Same characters, same evening, same kind of talk; the
# no_va twin is a negative example in which the assistant is never addressed.
#
# Chosen because its two speakers read as clearly distinct voices, which not
# every pair in the release does. Watch the build warnings if you swap it.
PAIR = {
    "id": "weeknight-in",
    "title": "A weeknight after work",
    "with_scenario": "SliceOfLifeWeeknightIn",
    "without_scenario": "SliceOfLifeWeeknightIn_no_va",
    # Both sides of this pair stay substantive for about twenty turns and then
    # circle the same goodnight ("let's see where the night takes us") to the
    # end. Keeping the turns up to the drift and cutting the audio there gives
    # a sample that holds attention; raise these to show the full scene.
    "with_turns": 22,
    "without_turns": 21,
}

# Naturalness comparison: our synthetic meetings against real recorded
# meetings from NOTSOFAR, both already normalised to mono 44.1 kHz.
# Paired shortest-first so each side plays at a comparable length.
NATURALNESS_PAIRS = [
    ("no_va_engineering_standup", "close_talk_mixed_MTG_32091"),
    ("no_va_team_social_friday", "close_talk_mixed_MTG_32029"),
    ("no_va_sprint_retrospective", "close_talk_mixed_MTG_32089"),
    ("no_va_incident_postmortem", "close_talk_mixed_MTG_32087"),
    ("no_va_campaign_kickoff", "close_talk_mixed_MTG_32108"),
    ("no_va_design_review", "close_talk_mixed_MTG_32026"),
]


def die(message: str) -> None:
    """Abort with a message naming the missing source."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def find_one(pattern_root: Path, scenario: str, suffix: str) -> Path:
    """Locate the single corpus file for a scenario, or abort."""
    hits = sorted(pattern_root.glob(f"{scenario}/**/*{suffix}"))
    hits = [h for h in hits if "_background_debug" not in h.parts]
    if not hits:
        die(f"no {suffix} found for scenario {scenario} under {pattern_root}")
    return hits[0]


def wav_duration(path: Path) -> float:
    """Duration of a wav file in seconds."""
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def encode(
    src: Path,
    dst: Path,
    bitrate: str,
    check_only: bool,
    duration: Optional[float] = None,
) -> None:
    """Transcode one audio file to mono mp3, skipping work already done.

    `duration` cuts the output short, fading the last second so an early stop
    does not end on a hard edge.
    """
    if check_only:
        cut = f", first {duration:.0f}s" if duration else ""
        print(f"  would encode {src.name} -> {dst.relative_to(DOCS)} @ {bitrate}{cut}")
        return
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"  up to date  {dst.relative_to(DOCS)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if duration:
        command += ["-t", f"{duration:.3f}", "-af", f"afade=out:st={max(duration - 1, 0):.3f}:d=1"]
    command += ["-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    subprocess.run(command, check=True)
    size = dst.stat().st_size / 1e6
    print(f"  encoded     {dst.relative_to(DOCS)}  ({size:.1f} MB)")


# Mean of the generator's random inter-utterance gap, random.uniform(0.3, 1.2).
# The drawn values are not recorded anywhere, so the mean is the fallback when
# the audio cannot be transcribed.
MEAN_TURN_GAP = 0.75

# Whisper model used to recover turn boundaries. "base.en" transcribes a
# two-minute scene in a few seconds on CPU and locates words far more
# accurately than any estimate from text length; the corpus is English-only.
ASR_MODEL = "base.en"

_asr_model = None
_asr_warned = False


def _words_from_audio(wav: Path) -> list:
    """Transcribe a scene into (word, start_time) pairs.

    Returns an empty list when faster-whisper is unavailable, which sends the
    caller to the proportional fallback.
    """
    global _asr_model, _asr_warned  # noqa: PLW0603 - one model reused across scenes
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        if not _asr_warned:
            print(
                "  NOTE no faster-whisper here, so turn times are estimated from "
                "text length — run with an interpreter that has it for "
                "audio-derived times"
            )
            _asr_warned = True
        return []

    if _asr_model is None:
        print(f"  loading ASR model {ASR_MODEL} (first scene only)")
        _asr_model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")

    segments, _ = _asr_model.transcribe(str(wav), word_timestamps=True, vad_filter=False)
    words = []
    for segment in segments:
        for word in segment.words or []:
            token = _normalise(word.word)
            if token:
                words.append((token[0], word.start))
    return words


def _normalise(text: str) -> list:
    """Lowercase alphanumeric words, for matching transcript against ASR.

    Contractions are folded rather than split: replacing the apostrophe with a
    space would turn "That's" into "that" plus a stray "s" that the recogniser
    never emits, which costs the true position a perfect score and lets the
    previous turn's tail tie with it.
    """
    folded = re.sub(r"['‘’ʼ]", "", text.lower())
    return re.sub(r"[^a-z0-9 ]", " ", folded).split()


def align_turn_starts(conversation: list, wav: Path, duration: float) -> tuple:
    """Find when each turn starts in the mixed scene, in seconds.

    The corpus `time` field cannot be used: it is planned during text
    generation from an assumed 11 characters per second, while the synthesised
    speech actually runs near 19, so those values drift far past the end of the
    audio — more than half the turns in a typical scene.

    Since the transcript of every turn is known, the reliable way to place it
    is to transcribe the scene and match each turn's opening words against the
    recognised word stream. That gives boundaries drawn from the audio itself
    rather than guessed from text length.

    Returns (starts, method) so the caller can record which one was used.
    """
    words = _words_from_audio(wav)
    if not words:
        return estimate_turn_starts(conversation, duration), "estimated"

    starts = []
    cursor = 0
    matched = 0
    for index, turn in enumerate(conversation):
        tokens = _normalise(turn.get("content") or "")
        if not tokens:
            starts.append(None)
            continue

        # Match the turn's opening words, searching forward from where the
        # previous turn ended. The probe is short so that a filler word the
        # recogniser dropped cannot break the match.
        # Six words, keeping the earliest of equally good positions. Grid-tested
        # over the four scenes on the page (74 turns): a shorter probe misses
        # turns whose leading filler the recogniser dropped, and preferring the
        # latest tie instead of the earliest loses matches. This setting places
        # every turn, each on a real word onset.
        probe = tokens[:6]
        best_key, best_at = (0.0, 0), None
        for at in range(cursor, min(cursor + 90, len(words))):
            candidate = [word for word, _ in words[at:at + len(probe)]]
            ratio = SequenceMatcher(None, probe, candidate).ratio()
            # Two positions one word apart can score the same, one of them
            # starting on the previous turn's last word. Break that tie on
            # whether the opening word itself lines up.
            key = (ratio, 1 if candidate and candidate[0] == probe[0] else 0)
            if key > best_key:
                best_key, best_at = key, at
        best_ratio = best_key[0]

        if best_at is not None and best_ratio > 0.45:
            starts.append(round(words[best_at][1], 2))
            cursor = min(best_at + max(len(tokens) - 2, 1), len(words) - 1)
            matched += 1
        else:
            starts.append(None)

    # The opening turn begins the scene; nothing can precede it.
    if starts and starts[0] is None:
        starts[0] = 0.0
        matched += 1

    # A scene the recogniser largely disagreed with is not worth trusting
    # turn by turn; fall back rather than publish scattered markers.
    if matched < 0.8 * len(conversation):
        print(f"  ASR matched only {matched}/{len(conversation)} turns, estimating instead")
        return estimate_turn_starts(conversation, duration), "estimated"

    starts = _fill_and_sort(starts, duration)
    print(f"  aligned {matched}/{len(conversation)} turns to the audio")
    return starts, "aligned"


def _fill_and_sort(starts: list, duration: float) -> list:
    """Interpolate any unmatched turn and keep the sequence increasing."""
    filled = list(starts)
    for i, value in enumerate(filled):
        if value is not None:
            continue
        before = next((filled[j] for j in range(i - 1, -1, -1) if filled[j] is not None), 0.0)
        after = next(
            (filled[j] for j in range(i + 1, len(filled)) if filled[j] is not None),
            duration,
        )
        filled[i] = round((before + after) / 2, 2)

    for i in range(1, len(filled)):
        if filled[i] < filled[i - 1]:
            filled[i] = filled[i - 1]
    return [min(value, duration) for value in filled]


def estimate_turn_starts(conversation: list, duration: float) -> list:
    """Split the scene across turns by text length, as a fallback.

    Used when the audio cannot be transcribed. Speaking time is roughly
    proportional to text length, so this divides the real duration by
    character count with a constant gap between turns. Measured against ASR
    alignment on one scene, it averages 0.6 s of error and peaks near 1.4 s.
    """
    counts = [max(len((turn.get("content") or "")), 1) for turn in conversation]
    gaps = MEAN_TURN_GAP * (len(conversation) - 1)
    # Guard a pathological case: never let gaps claim the whole scene.
    speech = max(duration - gaps, duration * 0.5)
    per_char = speech / sum(counts)

    starts = []
    at = 0.0
    for i, count in enumerate(counts):
        starts.append(round(at, 2))
        at += count * per_char
        if i < len(counts) - 1:
            at += MEAN_TURN_GAP
    return starts


def turns_for_page(conversation: list, starts: list, assistant: str = "Sigma") -> list:
    """Reduce corpus turns to what the page renders: who, when, what."""
    out = []
    for turn, start in zip(conversation, starts):
        speaker = turn["speaker"]
        out.append(
            {
                "t": start,
                "speaker": speaker,
                "text": turn.get("content") or "",
                "role": (
                    "assistant"
                    if speaker == assistant
                    else "ambient"
                    if speaker == "background_noise"
                    else "user"
                ),
            }
        )
    return out


def wake_index(turns: list) -> Optional[int]:
    """Index of the turn that wakes the assistant, if any.

    The wake turn is the last user turn at or before the first assistant turn
    that names the assistant — the same "named within 1-2 turns" rule the
    generator's dialogue review enforces.
    """
    first = next((i for i, t in enumerate(turns) if t["role"] == "assistant"), None)
    if first is None:
        return None
    for i in range(first - 1, max(-1, first - 4), -1):
        if i >= 0 and "sigma" in turns[i]["text"].lower():
            return i
    return None


def warn_on_voice_mixup(wav: Path, turns: list, scenario: str) -> None:
    """Warn when a scene's speakers do not sound like different people.

    Some scenes in the release hand a male character a female voice, and only
    in the variant that has an assistant — the twin is fine. The result is two
    people who sound identical, which defeats a multi-speaker sample. Pitch
    catches it: a male voice sits near 130 Hz and a female one near 250 Hz.
    """
    try:
        import numpy as np
        import soundfile as sound
    except ImportError:
        return

    mapping = json.loads(VOICE_MAPPING.read_text(encoding="utf-8"))["characters"]
    audio, rate = sound.read(str(wav))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    def median_pitch(segment) -> Optional[float]:
        """Autocorrelation pitch over the voiced frames of a segment."""
        hop = int(0.03 * rate)
        found = []
        for at in range(0, max(len(segment) - hop, 0), hop):
            frame = segment[at:at + hop]
            if np.sqrt((frame ** 2).mean()) < 0.02:
                continue
            frame = frame - frame.mean()
            corr = np.correlate(frame, frame, "full")[len(frame) - 1:]
            low, high = int(rate / 300), int(rate / 70)
            if high >= len(corr):
                continue
            peak = int(np.argmax(corr[low:high])) + low
            if corr[peak] > 0.3 * corr[0]:
                found.append(rate / peak)
        return float(np.median(found)) if found else None

    by_speaker: dict = {}
    for i, turn in enumerate(turns):
        start = turn["t"]
        end = turns[i + 1]["t"] if i + 1 < len(turns) else len(audio) / rate
        pitch = median_pitch(audio[int(start * rate):int(min(end, start + 5) * rate)])
        if pitch is not None:
            by_speaker.setdefault(turn["speaker"], []).append(pitch)

    humans = {}
    for name, values in by_speaker.items():
        pitch = float(np.median(values))
        gender = mapping.get(name, {}).get("gender")
        if gender == "male" and pitch > 165:
            print(f"  WARNING {scenario}: {name} is male but sounds like {pitch:.0f} Hz")
        if gender == "female" and pitch < 150:
            print(f"  WARNING {scenario}: {name} is female but sounds like {pitch:.0f} Hz")
        if name != "Sigma":
            humans[name] = pitch

    if len(humans) > 1:
        spread = max(humans.values()) - min(humans.values())
        if spread < 40:
            print(
                f"  WARNING {scenario}: speakers are only {spread:.0f} Hz apart, "
                "they will sound like one person"
            )


def build_scene(scenario: str, out_id: str, check_only: bool, keep_turns: int = 0) -> dict:
    """Encode one scene's audio and collect its transcript.

    `keep_turns` cuts the sample after that many turns, audio and transcript
    together, for scenes whose tail repeats itself. Zero keeps the whole scene.
    """
    wav = find_one(RELEASE / "speech_corpora", scenario, ".wav")
    js = find_one(RELEASE / "text_corpora", scenario, ".json")
    data = json.loads(js.read_text(encoding="utf-8"))
    duration = wav_duration(wav)

    if check_only:
        # --check exists to prove the sources resolve. Transcribing to find turn
        # boundaries would take minutes and produce a trim length that the real
        # run then recomputes, so report what was found and stop.
        turns_found = len(data["conversation"])
        trim = f", trimming to {keep_turns} of {turns_found} turns" if keep_turns else ""
        print(f"  found {scenario}: {duration:.0f}s, {turns_found} turns{trim}")
        encode(wav, DOCS / "assets" / "scenes" / f"{out_id}.mp3", SCENE_BITRATE, True)
        return {"audio": f"assets/scenes/{out_id}.mp3", "duration": round(duration, 1)}

    starts, timing = align_turn_starts(data["conversation"], wav, duration)
    turns = turns_for_page(data["conversation"], starts)

    if keep_turns and keep_turns < len(turns):
        # Cut a moment after the last kept turn begins so its words finish,
        # bounded by where the next turn starts.
        last = turns[keep_turns - 1]["t"]
        following = turns[keep_turns]["t"]
        duration = round(min(following, last + 6.0), 2)
        turns = turns[:keep_turns]
        print(f"  trimmed {scenario} to {keep_turns} turns ({duration:.0f}s)")

    encode(
        wav,
        DOCS / "assets" / "scenes" / f"{out_id}.mp3",
        SCENE_BITRATE,
        check_only,
        duration=duration if keep_turns else None,
    )
    if not check_only:
        warn_on_voice_mixup(DOCS / "assets" / "scenes" / f"{out_id}.mp3", turns, scenario)
    return {
        "audio": f"assets/scenes/{out_id}.mp3",
        "duration": round(duration, 1),
        "context": data.get("context", ""),
        "variant": data.get("variant_type", ""),
        "speakers": sorted({t["speaker"] for t in turns if t["role"] == "user"}),
        "assistantTurns": sum(1 for t in turns if t["role"] == "assistant"),
        "wakeIndex": wake_index(turns),
        "timing": timing,
        "turns": turns,
        "source": str(js.relative_to(RELEASE)),
    }


def build_voices(check_only: bool) -> list:
    """Encode the character voice cards and pair them with their descriptions."""
    mapping = json.loads(VOICE_MAPPING.read_text(encoding="utf-8"))["characters"]
    cards = []
    for mp3 in sorted(CHARACTERS.glob("intro_selected/*.mp3")):
        name, voice_id = mp3.stem.split("__")
        intro = CHARACTERS / "intro_texts" / f"{name}.txt"
        if not intro.exists():
            die(f"no intro text for {name}")
        entry = mapping.get(name, {})
        listed = entry.get("voices", {}).get("ElevenLabs", {}).get("voice_id")
        if listed != voice_id:
            die(
                f"{name}: sample voice id {voice_id} does not match "
                f"voice_mapping.json ({listed})"
            )
        dst = DOCS / "assets" / "voices" / f"{name}.mp3"
        if check_only:
            print(f"  would copy {mp3.name} -> {dst.relative_to(DOCS)}")
        elif not dst.exists() or dst.stat().st_mtime < mp3.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mp3, dst)
            print(f"  copied      {dst.relative_to(DOCS)}")
        else:
            print(f"  up to date  {dst.relative_to(DOCS)}")
        cards.append(
            {
                "name": name,
                "audio": f"assets/voices/{name}.mp3",
                "intro": intro.read_text(encoding="utf-8").strip(),
                "description": entry.get("description", ""),
                "gender": entry.get("gender", ""),
                "voiceId": voice_id,
            }
        )
    return cards


def build_naturalness(check_only: bool) -> list:
    """Encode the synthetic/real comparison pairs."""
    rows = []
    for ours_name, real_name in NATURALNESS_PAIRS:
        ours = NATURALNESS / "int_wakeup" / f"{ours_name}.wav"
        real = NATURALNESS / "notsofar" / f"{real_name}.wav"
        for path in (ours, real):
            if not path.exists():
                die(f"missing naturalness source {path}")
        out = DOCS / "assets" / "naturalness"
        encode(ours, out / f"{ours_name}.mp3", SCENE_BITRATE, check_only)
        encode(real, out / f"{real_name}.mp3", SCENE_BITRATE, check_only)
        rows.append(
            {
                "id": ours_name,
                "label": ours_name.removeprefix("no_va_").replace("_", " "),
                "ours": {
                    "audio": f"assets/naturalness/{ours_name}.mp3",
                    "duration": round(wav_duration(ours), 1),
                },
                "real": {
                    "audio": f"assets/naturalness/{real_name}.mp3",
                    "duration": round(wav_duration(real), 1),
                    "id": real_name,
                },
            }
        )
    return rows


def main() -> None:
    """Encode every asset the page needs and write docs/data/demo.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify sources and report what would be built, without encoding",
    )
    args = parser.parse_args()

    for root in (RELEASE, CHARACTERS, NATURALNESS, VOICE_MAPPING):
        if not root.exists():
            die(f"source not found: {root}")

    print("featured scenes")
    featured = []
    for spec in FEATURED:
        scene = build_scene(spec["scenario"], spec["id"], args.check)
        scene.update(id=spec["id"], title=spec["title"], blurb=spec["blurb"])
        featured.append(scene)

    print("with / without pair")
    pair = {
        "id": PAIR["id"],
        "title": PAIR["title"],
        "with": build_scene(
            PAIR["with_scenario"],
            PAIR["id"] + "-with",
            args.check,
            PAIR.get("with_turns", 0),
        ),
        "without": build_scene(
            PAIR["without_scenario"],
            PAIR["id"] + "-without",
            args.check,
            PAIR.get("without_turns", 0),
        ),
    }

    print("voice cards")
    voices = build_voices(args.check)

    print("naturalness pairs")
    naturalness = build_naturalness(args.check)

    payload = {
        "featured": featured,
        "pair": pair,
        "voices": voices,
        "naturalness": naturalness,
        "provenance": {
            "release": RELEASE.name,
            "characters": CHARACTERS.parent.name,
            "naturalness": NATURALNESS.parent.name,
        },
    }
    if args.check:
        print("\n--check: docs/data/demo.json not written")
        return

    out = DOCS / "data" / "demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out.relative_to(DOCS)}")

    total = sum(f.stat().st_size for f in (DOCS / "assets").rglob("*.mp3"))
    print(f"assets total: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
