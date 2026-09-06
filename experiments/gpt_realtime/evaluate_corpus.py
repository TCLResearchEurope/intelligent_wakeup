"""
Stage 2: Evaluation runner.

Iterates text_corpora JSONs, reads per-turn utterance WAVs (audio_path field),
sends each to the OpenAI Realtime API, and records whether the model triggered
(non-[SILENCE] response) or stayed silent.

Expected-trigger logic
----------------------
A turn is expected to trigger if the *next* turn in the conversation is
spoken by the "Sigma" VA — i.e. the corpus shows the assistant responding.
This covers both cases:
  - Direct trigger: current turn explicitly contains the word "sigma"
  - Contextual trigger: current turn is a follow-up in an ongoing sigma
    exchange (no "sigma" keyword, but the assistant is expected to respond)

Results are written to results.jsonl (one JSON object per line, safe to
resume — already-evaluated turns are skipped with --resume).

Usage:
    python evaluate_corpus.py \\
        --text-corpora /path/to/text_corpora \\
        --output results.jsonl \\
        [--scenario ArtCraftGuidance] \\
        [--model gpt-realtime-mini] \\
        [--resume]
"""

import argparse
import array
import asyncio
import datetime
import json
import logging
import os
import sys
import wave
from pathlib import Path

from realtime_va.core import RealtimeVACore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# websockets protocol messages (frame-level send/receive) are never useful here
logging.getLogger("websockets").setLevel(logging.WARNING)

TARGET_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHUNK_SIZE = 4800  # 200 ms at 24 kHz


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def read_and_resample(wav_path: Path) -> bytes:
    """Read an entire WAV file and resample to TARGET_RATE mono PCM.

    Returns empty bytes if the file does not exist or contains no frames.
    """
    if not wav_path.exists():
        return b""

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()

        if total_frames <= 0:
            return b""

        raw = wf.readframes(total_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width {sample_width} in {wav_path}")

    samples = array.array("h", raw)

    # Mix down to mono if needed
    if n_channels > 1:
        mono = array.array("h", [0] * (len(samples) // n_channels))
        for i in range(len(mono)):
            mono[i] = (
                sum(samples[i * n_channels + c] for c in range(n_channels))
                // n_channels
            )
        samples = mono

    # Resample if needed (nearest-neighbour, same as audio_io.py)
    if sample_rate != TARGET_RATE:
        ratio = sample_rate / TARGET_RATE
        new_length = int(len(samples) / ratio)
        resampled = array.array("h", [0] * new_length)
        for i in range(new_length):
            src_idx = min(int(i * ratio), len(samples) - 1)
            resampled[i] = samples[src_idx]
        samples = resampled

    return samples.tobytes()


def slice_and_resample(wav_path: Path, start_sec: float, end_sec: float) -> bytes:
    """Read a time slice from a WAV file and resample to TARGET_RATE mono PCM.

    Clamps start/end to the actual file duration so turns whose timestamps
    exceed the WAV length return empty bytes rather than raising an error.
    """
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()

        start_frame = max(0, min(int(start_sec * sample_rate), total_frames))
        end_frame = max(0, min(int(end_sec * sample_rate), total_frames))
        n_frames = end_frame - start_frame

        if n_frames <= 0:
            return b""

        wf.setpos(start_frame)
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width {sample_width} in {wav_path}")

    samples = array.array("h", raw)

    # Mix down to mono if needed
    if n_channels > 1:
        mono = array.array("h", [0] * (len(samples) // n_channels))
        for i in range(len(mono)):
            mono[i] = (
                sum(samples[i * n_channels + c] for c in range(n_channels))
                // n_channels
            )
        samples = mono

    # Resample if needed (nearest-neighbour, same as audio_io.py)
    if sample_rate != TARGET_RATE:
        ratio = sample_rate / TARGET_RATE
        new_length = int(len(samples) / ratio)
        resampled = array.array("h", [0] * new_length)
        for i in range(new_length):
            src_idx = min(int(i * ratio), len(samples) - 1)
            resampled[i] = samples[src_idx]
        samples = resampled

    return samples.tobytes()


# ---------------------------------------------------------------------------
# Turn labelling helpers
# ---------------------------------------------------------------------------


def label_turns(turns: list[dict]) -> list[dict]:
    """Annotate each turn with evaluation metadata.

    Adds:
      expected        – True if the next turn is spoken by the Sigma VA
      trigger_type    – "direct" | "contextual" | "none"
      context_turns   – for contextual turns, list of turn indices to prepend
                        as audio context ([sigma-invoke turn, VA response turn])
    """
    labelled = []
    for idx, turn in enumerate(turns):
        next_turn = turns[idx + 1] if idx + 1 < len(turns) else None
        expected = next_turn is not None and next_turn["speaker"] == "Sigma"

        has_sigma_word = turn.get("contains_sigma", False)

        if not expected:
            trigger_type = "non-assistance"
            context_turns = []
        elif has_sigma_word:
            trigger_type = "direct"
            context_turns = []
        else:
            # Contextual: walk back to find the most recent sigma invocation
            # and the following VA response
            trigger_type = "contextual"
            context_turns = _find_context_turns(turns, idx)

        labelled.append(
            {
                **turn,
                "expected": expected,
                "trigger_type": trigger_type,
                "context_turns": context_turns,
            }
        )
    return labelled


def _find_context_turns(turns: list[dict], current_idx: int) -> list[int]:
    """Return [sigma_invoke_idx, va_response_idx] for the most recent exchange.

    Walks backwards from current_idx to find the last turn where a human
    said "sigma" (sigma_invoke), then the Sigma VA turn that followed it.
    Returns empty list if no prior context is found.
    """
    for i in range(current_idx - 1, -1, -1):
        t = turns[i]
        if t["speaker"] == "Sigma":
            # This is a VA response — check if the turn before it invoked sigma
            if i > 0 and turns[i - 1].get("contains_sigma", False):
                return [i - 1, i]
            # VA response found but no explicit invocation before it
            # (e.g. first VA turn in conversation); still useful as context
            return [i]
    return []


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict]) -> dict:
    """Compute TP/FP/TN/FN and derived metrics, broken down by trigger type."""
    categories = {
        "overall": [],
        "direct": [],
        "contextual": [],
        "non-assistance": [],
    }
    for r in results:
        categories["overall"].append(r)
        categories[r.get("trigger_type", "non-assistance")].append(r)

    out = {}
    for name, group in categories.items():
        if not group:
            continue
        tp = sum(1 for r in group if r["expected"] and r["triggered"])
        fp = sum(1 for r in group if not r["expected"] and r["triggered"])
        tn = sum(1 for r in group if not r["expected"] and not r["triggered"])
        fn = sum(1 for r in group if r["expected"] and not r["triggered"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out[name] = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "F1": round(f1, 3),
            "n": len(group),
        }
    return out


def log_metrics(metrics: dict, header: str):
    """Print a metrics table to the logger."""
    logger.info("%s", header)
    logger.info(
        "  %-16s  %4s %4s %4s %4s %4s   prec   rec    F1",
        "category",
        "TC",
        "TP",
        "FP",
        "TN",
        "FN",
    )
    logger.info("  %s", "-" * 66)
    for name in ("overall", "direct", "contextual", "non-assistance"):
        if name not in metrics:
            if name == "contextual":
                logger.info(
                    "  %-16s  %4d %4d %4d %4d %4d   %.3f  %.3f  %.3f",
                    name,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                )
            continue
        m = metrics[name]
        logger.info(
            "  %-16s  %4d %4d %4d %4d %4d   %.3f  %.3f  %.3f",
            name,
            m["n"],
            m["TP"],
            m["FP"],
            m["TN"],
            m["FN"],
            m["precision"],
            m["recall"],
            m["F1"],
        )


# ---------------------------------------------------------------------------
# Per-turn evaluation
# ---------------------------------------------------------------------------


async def evaluate_turn(
    turn: dict,
    file_key: str,
    scenario: str,
    conversation_type: str,
    va_core: RealtimeVACore,
) -> dict | None:
    """Evaluate a single turn using a persistent WebSocket session.

    The caller is responsible for creating, connecting, and closing va_core.
    Callbacks are set per-turn and cleared afterwards.
    """
    audio_path = Path(turn["audio_path"])
    audio_data = read_and_resample(audio_path)
    if not audio_data:
        logger.warning(
            "Utterance WAV not found or empty for turn %d: %s — skipping",
            turn["turn_index"],
            audio_path,
        )
        return None

    response_transcript: list[str] = [""]
    response_done_event = asyncio.Event()

    async def on_transcript_done(transcript: str):
        response_transcript[0] = transcript

    async def on_response_done():
        response_done_event.set()

    va_core.on_transcript_done = on_transcript_done
    va_core.on_response_done = on_response_done

    try:
        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset : offset + CHUNK_SIZE * SAMPLE_WIDTH]
            await va_core.send_audio_chunk(chunk)
            offset += CHUNK_SIZE * SAMPLE_WIDTH

        await va_core.commit_audio()

        try:
            await asyncio.wait_for(response_done_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout on turn %d of %s", turn["turn_index"], file_key)
            response_transcript[0] = "[TIMEOUT]"
    finally:
        va_core.on_transcript_done = None
        va_core.on_response_done = None

    transcript = response_transcript[0]
    triggered = bool(transcript) and "[SILENCE]" not in transcript.upper()
    expected = turn["expected"]

    outcome = (
        "TP"
        if expected and triggered
        else (
            "TN"
            if not expected and not triggered
            else "FP" if not expected and triggered else "FN"
        )
    )
    ttype = turn.get("trigger_type", "non-assistance")
    ttype_label = ttype.upper().replace("-", "_")
    pass_fail = "PASS" if outcome in ("TP", "TN") else "FAIL"

    logger.info(
        "\nturn %02d  %s  %s [%s]  %s",
        turn["turn_index"],
        ttype_label,
        pass_fail,
        outcome,
        turn["speaker"],
    )
    logger.info('  corpus: "%s"', turn["text"])
    logger.info('  model:  "%s"', transcript or "[no transcript]")

    return {
        "file": file_key,
        "scenario": scenario,
        "conversation_type": conversation_type,
        "turn_index": turn["turn_index"],
        "speaker": turn["speaker"],
        "text": turn["text"],
        "expected": expected,
        "trigger_type": ttype,
        "triggered": triggered,
        "response": transcript,
        "audio_path": turn["audio_path"],
    }


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------


async def evaluate_file(
    corpus_json: Path,
    model: str,
    prompt_file: Path | None,
) -> list[dict]:
    """Evaluate all turns in one corpus JSON using a single persistent session.

    One WebSocket connection is kept open for the entire conversation so the
    model accumulates context naturally across turns.  After each evaluated turn,
    the corpus Sigma response (if any) is injected as an assistant message so the
    model knows what the VA "said" when deciding how to handle follow-up turns.
    """
    with open(corpus_json, encoding="utf-8") as f:
        corpus = json.load(f)

    scenario = corpus["scenario_type"]
    conversation_type = corpus["variant_type"]
    file_key = f"{scenario}/{conversation_type}/{corpus['variant_name']}"

    # Normalise conversation entries into the turn format used by label_turns
    raw_turns = []
    for idx, entry in enumerate(corpus["conversation"]):
        audio_path = Path(entry["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (corpus_json.parent / audio_path).resolve()
        raw_turns.append(
            {
                "turn_index": idx,
                "speaker": entry["speaker"],
                "text": entry["content"],
                "audio_path": str(audio_path),
                "contains_sigma": "sigma" in entry["content"].lower(),
            }
        )

    turns = label_turns(raw_turns)
    n_sigma = sum(1 for t in turns if t["speaker"] == "Sigma")

    logger.info(
        "\nEvaluating %s  (%d turns, %d VA turns)",
        file_key,
        len(turns) - n_sigma,
        n_sigma,
    )

    va_core = RealtimeVACore(
        model=model,
        prompt_file=str(prompt_file) if prompt_file else None,
        mode="batch",
    )
    await va_core.connect()
    listener = asyncio.create_task(va_core.handle_server_messages())

    results = []
    try:
        for turn in turns:
            if turn["speaker"] == "Sigma":
                logger.info(
                    "\nturn %02d  [SIGMA — injecting as context]  %s",
                    turn["turn_index"],
                    turn["speaker"],
                )
                logger.info('  corpus: "%s"', turn["text"])
                await va_core.inject_assistant_message(turn["text"])
                continue

            try:
                result = await evaluate_turn(
                    turn,
                    file_key,
                    scenario,
                    conversation_type,
                    va_core,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Error on turn %d of %s: %s", turn["turn_index"], file_key, exc
                )
                continue

            if result is not None:
                results.append(result)
    finally:
        listener.cancel()
        await va_core.close()

    if results:
        metrics = compute_metrics(results)
        log_metrics(metrics, f"--- {file_key} ---")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_existing_results(output_path: Path) -> list[dict]:
    """Load turn results from an existing output JSON file (for --resume)."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", [])
    except (json.JSONDecodeError, KeyError):
        return []


def load_dataset_versions(speech_corpora: Path, scenarios: list[str]) -> dict:
    """Load speech_version.json for each scenario that was evaluated."""
    versions = {}
    for scenario in sorted(set(scenarios)):
        version_file = speech_corpora / scenario / "speech_version.json"
        if version_file.exists():
            with open(version_file, encoding="utf-8") as f:
                versions[scenario] = json.load(f)
        else:
            versions[scenario] = None
    return versions


def iter_corpus_files(text_corpora: Path, scenario_filter: str | None) -> list[Path]:
    """Return all text corpus JSON paths, optionally filtered to one scenario."""
    files = sorted(
        p for p in text_corpora.rglob("*.json") if p.name != "corpora_version.json"
    )
    if scenario_filter:
        files = [p for p in files if scenario_filter in p.parts]
    return files


def resolve_file_arg(text_corpora: Path, file_arg: str) -> Path:
    """Resolve --file to an absolute corpus JSON path.

    Accepts:
      - Absolute path:               /path/to/text_corpora/EducationLearning/couple/homework_help.json
      - Relative key (no .json):     EducationLearning/couple/homework_help
      - Relative key (with .json):   EducationLearning/couple/homework_help.json
    """
    p = Path(file_arg)
    if p.is_absolute():
        return p.with_suffix(".json") if p.suffix != ".json" else p
    # treat as scenario/conv_type/file_stem relative to text_corpora
    return (text_corpora / p).with_suffix(".json")


async def run(args):
    """Main async entry point."""
    if args.file:
        corpus_files = [resolve_file_arg(args.text_corpora, args.file)]
    else:
        corpus_files = iter_corpus_files(args.text_corpora, args.scenario)
    logger.info("Found %d conversation file(s) to evaluate", len(corpus_files))
    logger.info(
        "Model: %s | each turn is a separate run (fresh WebSocket connection, no shared state)",
        args.model,
    )
    logger.info(
        "Expected to trigger: turn precedes a Sigma VA response (direct = 'sigma' in text, "
        "contextual = follow-up)"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[tuple[str, int]] = set()
    all_results: list[dict] = []

    if args.resume:
        all_results = load_existing_results(output_path)
        already_done = {(r["file"], r["turn_index"]) for r in all_results}
        logger.info("Resuming: %d turns already evaluated", len(already_done))

    for corpus_json in corpus_files:
        if args.resume:
            with open(corpus_json, encoding="utf-8") as f:
                corpus = json.load(f)
            scenario = corpus["scenario_type"]
            conv_type = corpus["variant_type"]
            file_key = f"{scenario}/{conv_type}/{corpus['variant_name']}"
            n_eval = sum(
                1 for e in corpus["conversation"] if e.get("speaker") != "Sigma"
            )
            if all((file_key, i) in already_done for i in range(n_eval)):
                logger.info("Skipping (already done): %s", file_key)
                continue

        try:
            results = await evaluate_file(
                corpus_json,
                args.model,
                Path(args.prompt) if args.prompt else None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to evaluate %s: %s", corpus_json, exc)
            continue

        for result in results:
            if (result["file"], result["turn_index"]) not in already_done:
                all_results.append(result)
                already_done.add((result["file"], result["turn_index"]))

    global_metrics: dict = {}
    if all_results:
        global_metrics = compute_metrics(all_results)
        log_metrics(global_metrics, "=== GLOBAL SUMMARY ===")

    if "contextual" not in global_metrics:
        global_metrics["contextual"] = {
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
            "precision": 0.0,
            "recall": 0.0,
            "F1": 0.0,
            "n": 0,
        }

    speech_corpora = args.text_corpora.parent / "speech_corpora"
    output = {
        "evaluated_at": datetime.datetime.now().isoformat(),
        "model": args.model,
        "dataset_versions": load_dataset_versions(
            speech_corpora, [r["scenario"] for r in all_results]
        ),
        "metrics": global_metrics,
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Done. Results written to %s", output_path)


def main():
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description="Corpus evaluation runner")
    parser.add_argument(
        "--text-corpora",
        default=(
            Path(os.environ["TEXT_CORPORA_DIR"])
            if os.environ.get("TEXT_CORPORA_DIR")
            else None
        ),
        type=Path,
        help=(
            "Root of the text corpora tree. Defaults to $TEXT_CORPORA_DIR; "
            "required when that is unset, since the corpora live outside this repo."
        ),
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output file path (JSON with metadata, metrics, and per-turn results)",
    )
    parser.add_argument(
        "--model",
        default="gpt-realtime-mini",
        help="OpenAI Realtime model (default: gpt-realtime-mini)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Path to system prompt file (default: prompts/sigma_wakeup.txt)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Evaluate only this scenario (e.g. ArtCraftGuidance)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help=(
            "Evaluate a single file. Accepts a relative key "
            "(e.g. EducationLearning/couple/homework_help) or an absolute path."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip turns already present in the output file",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.text_corpora is None:
        parser.error(
            "--text-corpora is required (or set TEXT_CORPORA_DIR). The text corpora "
            "are not part of this repository, so no default path is shipped."
        )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
