"""
Corpus evaluation runner.

Streams conversations from the published Hugging Face dataset
(``TCLResearchEurope/intelligent_wakeup``), cuts each one into turns at the
recorded onsets, sends every non-assistant turn to the OpenAI Realtime API and
records whether the model triggered (a non-``[SILENCE]`` response) or stayed
silent.

The dataset holds one row per conversation: a whole session as a single mixed
render, with per-turn text, speaker and onset travelling alongside it in
``turns``. Turn ``i`` is therefore the audio between its own onset and the next
one, and the last turn runs to the end of the session. That slice is the mixed
scene, so it carries room tone, background and — where the scenario enabled
overlapping speech — a few hundred milliseconds of a neighbouring speaker.
Earlier revisions of this script read isolated, clean per-utterance wavs from
the generating machine, which were never published; expect results to be a
little worse here, and a little closer to what a device actually hears.

Requires dataset revision v1.0.1 or later. v1.0.0 published turn times that
were planned during text generation rather than measured from the audio, and
they drift past the end of the session for most turns — see
``dataset/scripts/recover_turn_onsets.py``.

Expected-trigger logic
----------------------
A turn is expected to trigger if the *next* turn is spoken by the assistant,
which covers both:
  - Direct trigger: the turn explicitly contains the word "sigma"
  - Contextual trigger: a follow-up in an ongoing exchange, no keyword

Usage:
    python evaluate_corpus.py --split test --output results.json
    python evaluate_corpus.py --split test --category KitchenConversations
    python evaluate_corpus.py --id "EducationLearning/couple/homework_help"
"""

import argparse
import asyncio
import datetime
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from realtime_va.core import RealtimeVACore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# websockets protocol messages (frame-level send/receive) are never useful here
logging.getLogger("websockets").setLevel(logging.WARNING)

DEFAULT_REPO_ID = "TCLResearchEurope/intelligent_wakeup"
DEFAULT_REVISION = "v1.0.1"

TARGET_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHUNK_SIZE = 4800  # 200 ms at 24 kHz

ASSISTANT_SPEAKER = "Sigma"

# A slice shorter than this holds no utterance worth sending; it means two
# onsets landed on top of each other.
MIN_SLICE_SECONDS = 0.2

# Stop after this many conversations fail back to back. One failure is a bad
# session; this many in a row means the API is unreachable, out of credit, or
# refusing the key, and continuing only wastes time.
MAX_CONSECUTIVE_FAILURES = 3


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def load_rows(args):
    """Open the dataset and return the conversations to evaluate.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (iterable of rows, description dict recorded in the output).

    Raises:
        SystemExit: If ``datasets`` is missing or too old to decode audio.
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit(
            "datasets is required: pip install 'datasets>=4.0' torchcodec soundfile"
        )

    logger.info(
        "Loading %s (revision %s, split %s)%s",
        args.repo_id,
        args.revision,
        args.split,
        " streaming" if args.streaming else "",
    )
    dataset = load_dataset(
        args.repo_id,
        revision=args.revision,
        split=args.split,
        streaming=args.streaming,
    )
    # Ask for the rate the Realtime API expects, so the decoder resamples once
    # and the script never has to. Channel count is left alone deliberately:
    # the keyword for it was renamed between datasets 4.x and 5.x, the corpus
    # is mono anyway, and decode_audio mixes down whatever it is handed.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_RATE))

    # Every filter names its column: without input_columns the audio is decoded
    # for each row just to look at a string, which for this dataset means
    # decoding gigabytes to answer a question about metadata.
    if args.category:
        dataset = dataset.filter(
            lambda category: category == args.category, input_columns=["category"]
        )
    if args.id:
        wanted = set(args.id)
        dataset = dataset.filter(
            lambda row_id: row_id in wanted, input_columns=["id"]
        )
    if args.has_va is not None:
        dataset = dataset.filter(
            lambda has_va: has_va == args.has_va, input_columns=["has_va"]
        )
    if args.limit:
        dataset = (
            dataset.take(args.limit)
            if args.streaming
            else dataset.select(range(min(args.limit, len(dataset))))
        )

    description = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "split": args.split,
        "category": args.category,
        "ids": args.id,
        "has_va": args.has_va,
        "limit": args.limit,
    }
    return dataset, description


def decode_audio(value):
    """Return one row's audio as mono float samples plus its rate.

    ``datasets`` 4.0 hands back a torchcodec ``AudioDecoder``; older versions
    hand back a dict. Both appear in the wild depending on what is installed,
    so both are accepted.

    Args:
        value: The row's ``audio`` field.

    Returns:
        Tuple of (samples as float32 in [-1, 1], sample rate).
    """
    if isinstance(value, dict):
        return np.asarray(value["array"], dtype=np.float32), int(
            value["sampling_rate"]
        )

    samples = value.get_all_samples()
    data = samples.data
    if hasattr(data, "numpy"):
        data = data.numpy()
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:  # (channels, samples)
        data = data.mean(axis=0)
    return data, int(samples.sample_rate)


def to_pcm16(samples):
    """Convert float samples in [-1, 1] to little-endian 16-bit PCM bytes.

    Args:
        samples: Float sample array.

    Returns:
        Raw PCM bytes.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample(samples, source_rate, target_rate):
    """Resample audio, preferring a polyphase filter when SciPy is available.

    Args:
        samples: Float sample array.
        source_rate: Rate the samples are at.
        target_rate: Rate wanted.

    Returns:
        Resampled float array.
    """
    if source_rate == target_rate or len(samples) == 0:
        return samples
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), int(target_rate))
        return resample_poly(
            samples, int(target_rate) // divisor, int(source_rate) // divisor
        )
    except ImportError:
        count = int(round(len(samples) * target_rate / source_rate))
        return np.interp(
            np.linspace(0.0, len(samples), count, endpoint=False),
            np.arange(len(samples)),
            samples,
        )


# ---------------------------------------------------------------------------
# Turn labelling
# ---------------------------------------------------------------------------


def label_turns(turns, duration):
    """Annotate each turn with its audio span and evaluation metadata.

    Adds:
      start, end     – seconds, the turn's span in the session render
      expected       – True if the next turn is spoken by the assistant
      trigger_type   – "direct" | "contextual" | "non-assistance"

    Args:
        turns: The row's ``turns`` list, in order.
        duration: Session duration in seconds, ending the last turn.

    Returns:
        List of annotated turn dicts.
    """
    labelled = []
    for index, turn in enumerate(turns):
        next_turn = turns[index + 1] if index + 1 < len(turns) else None
        expected = next_turn is not None and next_turn["speaker"] == ASSISTANT_SPEAKER
        content = turn.get("content") or ""

        if not expected:
            trigger_type = "non-assistance"
        elif ASSISTANT_SPEAKER.lower() in content.lower():
            trigger_type = "direct"
        else:
            trigger_type = "contextual"

        labelled.append(
            {
                "turn_index": index,
                "speaker": turn["speaker"],
                "text": content,
                "start": float(turn["time"]),
                # A turn runs until the next one starts; the last runs to the
                # end of the session, trailing ambience included.
                "end": float(next_turn["time"]) if next_turn else float(duration),
                "expected": expected,
                "trigger_type": trigger_type,
            }
        )
    return labelled


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results):
    """Compute TP/FP/TN/FN and derived metrics, broken down by trigger type.

    Args:
        results: Per-turn result dicts.

    Returns:
        Mapping of category name to its metrics, plus a ``false_accepts``
        entry measuring triggers per hour of audio the model should have
        ignored.
    """
    categories = {
        "overall": [],
        "direct": [],
        "contextual": [],
        "non-assistance": [],
    }
    for result in results:
        categories["overall"].append(result)
        categories[result.get("trigger_type", "non-assistance")].append(result)

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

    # The headline number for an always-on assistant: how often it speaks up
    # per hour of speech that was never addressed to it. Counting turns alone
    # hides this, since sessions differ in how much audio they carry.
    negative = [r for r in results if not r["expected"]]
    negative_hours = sum(r.get("audio_seconds", 0.0) for r in negative) / 3600
    false_accepts = sum(1 for r in negative if r["triggered"])
    out["false_accepts"] = {
        "count": false_accepts,
        "hours": round(negative_hours, 3),
        "per_hour": round(false_accepts / negative_hours, 2) if negative_hours else 0.0,
    }
    return out


def log_metrics(metrics, header):
    """Print a metrics table.

    Args:
        metrics: Output of :func:`compute_metrics`.
        header: Line printed above the table.

    Returns:
        None.
    """
    logger.info("%s", header)
    logger.info(
        "  %-16s  %4s %4s %4s %4s %4s   prec   rec    F1",
        "category",
        "n",
        "TP",
        "FP",
        "TN",
        "FN",
    )
    logger.info("  %s", "-" * 66)
    for name in ("overall", "direct", "contextual", "non-assistance"):
        metric = metrics.get(name)
        if metric is None:
            continue
        logger.info(
            "  %-16s  %4d %4d %4d %4d %4d   %.3f  %.3f  %.3f",
            name,
            metric["n"],
            metric["TP"],
            metric["FP"],
            metric["TN"],
            metric["FN"],
            metric["precision"],
            metric["recall"],
            metric["F1"],
        )
    fa = metrics.get("false_accepts")
    if fa:
        logger.info(
            "  %-16s  %d in %.2f h = %.2f/hour",
            "false accepts",
            fa["count"],
            fa["hours"],
            fa["per_hour"],
        )


# ---------------------------------------------------------------------------
# Per-turn evaluation
# ---------------------------------------------------------------------------


async def evaluate_turn(turn, audio_data, row_id, category, variant_type, va_core):
    """Evaluate a single turn over an already-open session.

    The caller creates, connects and closes ``va_core``; callbacks are set for
    the turn and cleared afterwards.

    Args:
        turn: Annotated turn dict.
        audio_data: The turn's PCM16 audio at TARGET_RATE.
        row_id: The conversation's dataset id.
        category: Scenario category.
        variant_type: ``couple`` or ``single``.
        va_core: Connected :class:`RealtimeVACore`.

    Returns:
        A result dict.
    """
    response_transcript = [""]
    response_done_event = asyncio.Event()

    async def on_transcript_done(transcript):
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
            logger.warning("Timeout on turn %d of %s", turn["turn_index"], row_id)
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
    pass_fail = "PASS" if outcome in ("TP", "TN") else "FAIL"

    logger.info(
        "\nturn %02d  %s  %s [%s]  %s",
        turn["turn_index"],
        turn["trigger_type"].upper().replace("-", "_"),
        pass_fail,
        outcome,
        turn["speaker"],
    )
    logger.info('  corpus: "%s"', turn["text"])
    logger.info('  model:  "%s"', transcript or "[no transcript]")

    return {
        "id": row_id,
        "category": category,
        "variant_type": variant_type,
        "turn_index": turn["turn_index"],
        "speaker": turn["speaker"],
        "text": turn["text"],
        "start": round(turn["start"], 3),
        "end": round(turn["end"], 3),
        "audio_seconds": round(len(audio_data) / (TARGET_RATE * SAMPLE_WIDTH), 3),
        "expected": expected,
        "trigger_type": turn["trigger_type"],
        "triggered": triggered,
        "response": transcript,
    }


# ---------------------------------------------------------------------------
# Per-conversation evaluation
# ---------------------------------------------------------------------------


async def evaluate_row(row, model, prompt_file, already_done):
    """Evaluate one conversation over a single persistent session.

    One WebSocket connection stays open for the whole conversation so the model
    accumulates context across turns. After each assistant turn in the corpus,
    its text is injected as an assistant message, so the model knows what the
    VA "said" when it decides how to handle the follow-up.

    Args:
        row: A dataset row.
        model: Realtime model name.
        prompt_file: System prompt path, or None for the default.
        already_done: Set of (id, turn_index) pairs to skip.

    Returns:
        List of result dicts.
    """
    row_id = row["id"]

    # Checked before the audio is decoded: on a resumed run most rows are
    # already complete, and decoding a session costs more than the check.
    pending = [
        index
        for index, turn in enumerate(row["turns"])
        if turn["speaker"] != ASSISTANT_SPEAKER and (row_id, index) not in already_done
    ]
    if not pending:
        logger.info("Skipping (already done): %s", row_id)
        return []

    samples, rate = decode_audio(row["audio"])
    if rate != TARGET_RATE:
        logger.warning(
            "%s decoded at %d Hz, resampling to %d Hz", row_id, rate, TARGET_RATE
        )
        samples = resample(samples, rate, TARGET_RATE)

    duration = len(samples) / TARGET_RATE
    turns = label_turns(row["turns"], duration)
    n_assistant = sum(1 for t in turns if t["speaker"] == ASSISTANT_SPEAKER)

    logger.info(
        "\nEvaluating %s  (%d turns, %d VA turns, %.1fs)",
        row_id,
        len(turns) - n_assistant,
        n_assistant,
        duration,
    )

    if turns and turns[-1]["start"] > duration:
        logger.warning(
            "%s: last onset %.1fs is past the %.1fs of audio — is this revision "
            "older than v1.0.1?",
            row_id,
            turns[-1]["start"],
            duration,
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
            if turn["speaker"] == ASSISTANT_SPEAKER:
                logger.info(
                    "\nturn %02d  [SIGMA — injecting as context]  %s",
                    turn["turn_index"],
                    turn["speaker"],
                )
                logger.info('  corpus: "%s"', turn["text"])
                await va_core.inject_assistant_message(turn["text"])
                continue

            if (row_id, turn["turn_index"]) in already_done:
                continue

            start = max(0, int(turn["start"] * TARGET_RATE))
            end = min(len(samples), int(turn["end"] * TARGET_RATE))
            if (end - start) < MIN_SLICE_SECONDS * TARGET_RATE:
                logger.warning(
                    "Turn %d of %s spans %.2fs — skipping",
                    turn["turn_index"],
                    row_id,
                    max(0, (end - start) / TARGET_RATE),
                )
                continue

            try:
                result = await evaluate_turn(
                    turn,
                    to_pcm16(samples[start:end]),
                    row_id,
                    row["category"],
                    row["variant_type"],
                    va_core,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Error on turn %d of %s: %s", turn["turn_index"], row_id, exc
                )
                continue

            results.append(result)
    finally:
        listener.cancel()
        await va_core.close()

    if results:
        log_metrics(compute_metrics(results), f"--- {row_id} ---")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_existing_results(output_path):
    """Load turn results from an existing output file, for ``--resume``.

    Args:
        output_path: Path to a previous output JSON.

    Returns:
        List of result dicts, empty when there is nothing usable.
    """
    if not output_path.exists():
        return []
    try:
        results = json.loads(output_path.read_text(encoding="utf-8")).get("results", [])
    except (json.JSONDecodeError, KeyError, OSError):
        return []

    # Results written before this script moved to the Hub key turns by a
    # corpus file path rather than a dataset id, and carry no slice duration.
    # They cannot be merged with new ones, so the run restarts instead of
    # failing halfway through on a missing key.
    if any("id" not in result for result in results):
        logger.warning(
            "%s is in an older format and cannot be resumed; starting fresh",
            output_path,
        )
        return []
    return results


async def run(args):
    """Evaluate every selected conversation and write the results.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    rows, description = load_rows(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    already_done = set()
    if args.resume:
        all_results = load_existing_results(output_path)
        already_done = {(r["id"], r["turn_index"]) for r in all_results}
        logger.info("Resuming: %d turns already evaluated", len(already_done))

    logger.info("Model: %s", args.model)
    logger.info(
        "Expected to trigger: turn precedes an assistant response "
        "(direct = 'sigma' in text, contextual = follow-up)"
    )

    # A run that cannot reach the API at all should say so and stop, not work
    # through hundreds of conversations producing nothing. Exhausted credit and
    # a bad key both look like this, and both waste a long run before the
    # summary reveals an empty result set.
    consecutive_failures = 0

    for row in rows:
        try:
            results = await evaluate_row(
                row,
                args.model,
                Path(args.prompt) if args.prompt else None,
                already_done,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to evaluate %s: %s", row.get("id", "?"), exc)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "\nGiving up: %d conversations in a row failed. The last error "
                    "was:\n  %s",
                    consecutive_failures,
                    exc,
                )
                break
            continue

        consecutive_failures = 0
        for result in results:
            all_results.append(result)
            already_done.add((result["id"], result["turn_index"]))

    metrics = compute_metrics(all_results) if all_results else {}
    if metrics:
        log_metrics(metrics, "=== GLOBAL SUMMARY ===")

    output_path.write_text(
        json.dumps(
            {
                "evaluated_at": datetime.datetime.now().isoformat(),
                "model": args.model,
                "prompt": args.prompt,
                "dataset": description,
                "conversations": len({r["id"] for r in all_results}),
                "metrics": metrics,
                "results": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Done. Results written to %s", output_path)


def main():
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Realtime model against the intelligent_wakeup corpus"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hub dataset id")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Dataset revision (default: {DEFAULT_REVISION}; v1.0.0 turn times "
        "do not match its audio)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Split to evaluate (default: test)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream rows instead of downloading the split first",
    )
    parser.add_argument(
        "--category", default=None, help="Evaluate only this scenario category"
    )
    parser.add_argument(
        "--id",
        action="append",
        default=None,
        help="Evaluate only this conversation id, repeatable "
        "(e.g. EducationLearning/couple/homework_help)",
    )
    parser.add_argument(
        "--has-va",
        dest="has_va",
        action="store_true",
        default=None,
        help="Only sessions that address the assistant",
    )
    parser.add_argument(
        "--no-va",
        dest="has_va",
        action="store_false",
        help="Only sessions that never address the assistant",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate at most N conversations"
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output path (JSON with metadata, metrics and per-turn results)",
    )
    parser.add_argument(
        "--model",
        default="gpt-realtime-mini",
        help="OpenAI Realtime model (default: gpt-realtime-mini)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="System prompt file (default: prompts/sigma_wakeup.txt)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip turns already present in the output file",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
