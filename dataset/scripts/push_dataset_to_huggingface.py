#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Hugging Face Dataset Exporter

Publishes the corpus to the Hub as Parquet with embedded audio, one row per
conversation. Dataset loading scripts were removed in ``datasets`` 4.0, so the
old script-based approach no longer loads at all; Parquet is the supported
route and it gives a working Dataset Viewer and streaming for free.

One row per conversation rather than per turn: an always-on wakeup model is
judged on false accepts per hour of continuous audio, which is not recoverable
once sessions are chopped into utterances. Per-turn text, timings and speakers
travel with the row in the ``turns`` column, so consumers can explode to
utterances themselves.

Splits are assigned by GROUP, never by file. A scenario's short variation, its
``_long`` sibling, its ``_variant2``/``_variant3`` re-takes and its ``no_va_``
twin all play the same situation in the same room; splitting them apart would
put effectively-seen material in the test set. The grouping rule lives in
``split_corpus.group_key`` and is imported here so the two can never disagree.

Groups are hashed individually, not stratified by category, so a category with
few groups can be absent from a split entirely. Check the splitter's report
before publishing if per-category evaluation matters.

Usage:
    # inspect the plan, nothing uploaded
    python -m dataset.scripts.push_dataset_to_huggingface \
        --data-dir dataset/data/release_v1.0.0

    # publish and tag
    python -m dataset.scripts.push_dataset_to_huggingface \
        --data-dir dataset/data/release_v1.0.0 \
        --repo-id TCLResearchEurope/intelligent_wakeup --version v1.0.0 --push
"""

import sys
import json
import argparse
import contextlib
import wave
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Single definition of the split unit, shared with the splitter so the two can
# never disagree about which conversations must stay together.
from dataset.scripts.split_corpus import assign as assign_group  # noqa: E402
from dataset.scripts.split_corpus import group_key  # noqa: E402

# Turn fields worth publishing. `audio_path` is deliberately excluded: it holds
# an absolute path from the generating machine.
TURN_FIELDS = ("speaker", "content", "time")


def build_config_index(config_dir):
    """Index every way a generated file can be traced back to its config.

    Three keys are needed. Variation names are the authoritative link, since a
    config can declare several (``x`` and ``x_long``, or ``Name_0``/``Name_1``
    from the variation generator) and none of them need match the filename. The
    file stem covers configs whose variations are unnamed, and (scenario name,
    variant type) is the fallback for output generated before a config was
    renamed.

    Args:
        config_dir: Config root containing ``scenarios/``.

    Returns:
        Tuple of (by_variant, by_name) dicts, each mapping to
        (category, variant_type, path).
    """
    by_variant, by_name = {}, {}
    for path in sorted((config_dir / "scenarios").rglob("*.json")):
        rel = path.relative_to(config_dir / "scenarios")
        if len(rel.parts) < 3:
            continue
        entry = (rel.parts[0], rel.parts[1], path)
        by_variant.setdefault(path.stem, entry)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for variation in data.get("variations") or []:
            name = variation.get("variant_name")
            if name:
                by_variant.setdefault(str(name), entry)
        scenario_name = data.get("name")
        if scenario_name:
            by_name.setdefault((str(scenario_name), rel.parts[1]), entry)
    return by_variant, by_name


def resolve_config(
    variant_name, stem, scenario_name, variant_type, by_variant, by_name
):
    """Find the config a generated conversation came from.

    Args:
        variant_name: Variation name recorded in the transcript.
        stem: Transcript filename stem.
        scenario_name: Output directory name, i.e. the config's ``name``.
        variant_type: ``couple`` or ``single``.
        by_variant: Variant-name index.
        by_name: (scenario name, variant type) index.

    Returns:
        (category, variant_type, path), or (None, None, None) when unresolvable.
    """
    for key in (variant_name, stem):
        if key and key in by_variant:
            return by_variant[key]
    # Trailing "_long" or "_0" style suffixes added to a variation name.
    for key in (variant_name, stem):
        if not key:
            continue
        for suffix in ("_long",):
            if key.endswith(suffix) and key[: -len(suffix)] in by_variant:
                return by_variant[key[: -len(suffix)]]
        base = key.rsplit("_", 1)
        if len(base) == 2 and base[1].isdigit() and base[0] in by_variant:
            return by_variant[base[0]]
    return by_name.get((scenario_name, variant_type), (None, None, None))


def wav_info(path):
    """Read frame count, rate and channel count from a wav header.

    Args:
        path: Path to a wav file.

    Returns:
        Tuple of (duration_seconds, sample_rate, channels), or (None, None,
        None) if the header cannot be read.
    """
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            return (frames / rate if rate else None, rate, handle.getnchannels())
    except (wave.Error, OSError):
        return (None, None, None)


def collect(data_dir, indexes, assistant_names):
    """Pair every generated transcript with its audio and build export rows.

    Args:
        data_dir: Release directory holding ``text_corpora/`` and
            ``speech_corpora/``.
        indexes: The two indexes from :func:`build_config_index`.
        assistant_names: Speaker names treated as the assistant.

    Returns:
        Tuple of (rows, problems) where problems lists skipped files and why.
    """
    # Two layouts exist in the wild: the packaged releases use
    # text_corpora/speech_corpora, a fresh generation run writes text/audio.
    layouts = [("text_corpora", "speech_corpora"), ("text", "audio")]
    for t, a in layouts:
        if (data_dir / t).is_dir() and (data_dir / a).is_dir():
            text_root, speech_root = data_dir / t, data_dir / a
            break
    else:
        sys.exit(
            f"found neither text_corpora/speech_corpora nor text/audio under {data_dir}"
        )
    rows, problems = [], []

    for tpath in sorted(text_root.rglob("*.json")):
        rel = tpath.relative_to(text_root)
        if len(rel.parts) < 3:
            continue
        if tpath.stem.endswith(tuple(f"-take{n}" for n in range(1, 10))) or (
            "director_notes" in tpath.stem
        ):
            continue  # intermediates, not part of the release

        wpath = (speech_root / rel).with_suffix(".wav")  # scene-level render
        if not wpath.exists():
            problems.append((str(rel), "no matching wav"))
            continue
        try:
            data = json.loads(tpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append((str(rel), f"bad json: {exc}"))
            continue

        turns = data.get("conversation")
        if not isinstance(turns, list) or not turns:
            problems.append((str(rel), "no conversation turns"))
            continue

        variant_name = data.get("variant_name") or tpath.stem
        variant_type = data.get("variant_type") or rel.parts[1]
        by_variant, by_name = indexes
        category, _, _cfg = resolve_config(
            variant_name, tpath.stem, rel.parts[0], variant_type, by_variant, by_name
        )
        if category is None:
            problems.append((str(rel), "no matching config; cannot group safely"))
            continue

        speakers = sorted({t.get("speaker") for t in turns if t.get("speaker")})
        has_va = any(s in assistant_names for s in speakers)
        duration, rate, channels = wav_info(wpath)

        rows.append(
            {
                "audio": str(wpath),
                "id": f"{category}/{variant_type}/{variant_name}",
                "scenario_name": rel.parts[0],
                "category": category,
                "variant_type": variant_type,
                "variant_name": variant_name,
                "conversation_type": data.get("conversation_type"),
                "context": data.get("context"),
                "has_va": has_va,
                "num_turns": len(turns),
                "duration_seconds": duration,
                "sampling_rate": rate,
                "channels": channels,
                "speakers": speakers,
                "turns": [{k: t.get(k) for k in TURN_FIELDS} for t in turns],
                "group_key": group_key(category, variant_type, variant_name),
            }
        )
    return rows, problems


def load_manifest(path):
    """Read a split manifest produced by ``split_corpus.py``.

    Args:
        path: Manifest path, or None.

    Returns:
        Tuple of (group to split mapping, seed, ratios); all None when no
        manifest was given.
    """
    if not path:
        return None, None, None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["groups"], data.get("seed"), tuple(data.get("ratios") or ())


def assign_splits(rows, manifest, seed, ratios):
    """Attach a split to every row via its group.

    The manifest is authoritative: it is computed from the configs, so it
    covers scenarios that have not been generated yet and keeps assignments
    identical between partial and complete exports. A group missing from the
    manifest (a scenario added after it was written) is assigned by the same
    hash rule, so the answer matches what regenerating the manifest would give.

    Args:
        rows: Export rows.
        manifest: Group to split mapping, or None to hash everything.
        seed: Hash salt, used for groups absent from the manifest.
        ratios: Split fractions, used for groups absent from the manifest.

    Returns:
        Tuple of (assignment, unlisted) where unlisted are groups that were not
        in the manifest.
    """
    assignment, unlisted = {}, []
    for row in rows:
        group = row["group_key"]
        if group in assignment:
            continue
        if manifest and group in manifest:
            assignment[group] = manifest[group]
        else:
            assignment[group] = assign_group(group, seed, ratios)
            if manifest is not None:
                unlisted.append(group)
    return assignment, unlisted


def report(rows, assignment, problems):
    """Print the export plan.

    Args:
        rows: Export rows.
        assignment: Group to split mapping.
        problems: Skipped files with reasons.

    Returns:
        None.
    """
    per = defaultdict(lambda: {"n": 0, "sec": 0.0, "va": 0, "groups": set()})
    for row in rows:
        bucket = per[assignment[row["group_key"]]]
        bucket["n"] += 1
        bucket["sec"] += row["duration_seconds"] or 0
        bucket["va"] += bool(row["has_va"])
        bucket["groups"].add(row["group_key"])

    print(f"\n{'split':12}{'rows':>7}{'groups':>8}{'hours':>8}{'VA':>7}{'no_VA':>7}")
    print("-" * 49)
    for split in ("train", "validation", "test"):
        b = per[split]
        print(
            f"{split:12}{b['n']:>7}{len(b['groups']):>8}{b['sec']/3600:>8.1f}"
            f"{b['va']:>7}{b['n']-b['va']:>7}"
        )
    total_sec = sum(r["duration_seconds"] or 0 for r in rows)
    print("-" * 49)
    print(f"{'TOTAL':12}{len(rows):>7}{len(assignment):>8}{total_sec/3600:>8.1f}")

    rates = Counter(r["sampling_rate"] for r in rows)
    chans = Counter(r["channels"] for r in rows)
    print(f"\naudio: sampling rates {dict(rates)}  channels {dict(chans)}")
    est = sum(
        (r["duration_seconds"] or 0) * (r["sampling_rate"] or 0) * 2 for r in rows
    )
    print(f"raw pcm size ~{est/1e9:.1f} GB (before parquet compression)")

    # No group may appear in two splits: that is the whole point of grouping.
    seen = defaultdict(set)
    for row in rows:
        seen[row["group_key"]].add(assignment[row["group_key"]])
    straddling = [g for g, s in seen.items() if len(s) > 1]
    print(f"groups straddling splits: {len(straddling)}")

    if problems:
        print(f"\nskipped {len(problems)} files:")
        for rel, why in problems[:10]:
            print(f"   {rel}: {why}")
        if len(problems) > 10:
            print(f"   ...and {len(problems) - 10} more")


def build_and_push(rows, assignment, args):
    """Build the DatasetDict and upload it.

    Args:
        rows: Export rows.
        assignment: Group to split mapping.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    from datasets import Audio, Dataset, DatasetDict

    by_split = defaultdict(list)
    for row in rows:
        row = dict(row)
        row["split"] = assignment[row["group_key"]]
        by_split[row["split"]].append(row)

    splits = {}
    for split, items in by_split.items():
        ds = Dataset.from_list(items)
        ds = ds.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
        splits[split] = ds
    dsd = DatasetDict(splits)

    print(f"\npushing to {args.repo_id} (private={args.private}) ...")
    dsd.push_to_hub(
        args.repo_id,
        private=args.private,
        max_shard_size=args.max_shard_size,
        revision=args.branch,
        # Explicit although it is the default: without embedding, rows keep only
        # the local wav path and the published dataset has no audio at all.
        # (Dataset.to_parquet has no equivalent option, so a local parquet dump
        # is NOT a valid rehearsal of this step.)
        embed_external_files=True,
        commit_message=f"Release {args.version or 'update'}",
    )
    print("upload complete")

    if args.version:
        from huggingface_hub import create_tag

        create_tag(
            args.repo_id,
            repo_type="dataset",
            revision=args.branch,
            tag=args.version,
            tag_message=f"Release {args.version}",
        )
        print(
            f"tagged {args.version} — consumers pin with "
            f'load_dataset("{args.repo_id}", revision="{args.version}")'
        )


def main():
    """Parse arguments, plan the export and optionally push it."""
    ap = argparse.ArgumentParser(
        description="Publish the corpus to the Hub, one row per conversation."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Release dir holding text_corpora/ and speech_corpora/",
    )
    ap.add_argument("--config-dir", type=Path, default=Path("dataset/config"))
    ap.add_argument("--repo-id", default="TCLResearchEurope/intelligent_wakeup")
    ap.add_argument(
        "--version", default=None, help="Git tag to create after upload, e.g. v1.0.0"
    )
    ap.add_argument(
        "--splits",
        type=Path,
        default=Path("dataset/config/splits.json"),
        help="Split manifest from split_corpus.py. Authoritative: it "
        "is computed from the configs, so assignments do not "
        "shift between a partial and a complete export.",
    )
    ap.add_argument(
        "--split-ratios",
        default="0.8,0.1,0.1",
        help="Only used for groups missing from the manifest",
    )
    ap.add_argument("--seed", default="split-v1")
    ap.add_argument(
        "--sampling-rate",
        type=int,
        default=44100,
        help="Audio is published at this rate. 16000 cuts the "
        "upload roughly 3x and is usually enough for wakeup "
        "models; 44100 keeps the source untouched.",
    )
    ap.add_argument(
        "--assistant-names",
        default="Sigma",
        help="Comma-separated speaker names counted as the assistant",
    )
    ap.add_argument("--max-shard-size", default="500MB")
    ap.add_argument(
        "--branch",
        default=None,
        help="Push to this branch instead of main, so a release can "
        "be verified before it becomes the default revision",
    )
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--push",
        action="store_true",
        help="Actually upload. Without it the plan is only printed.",
    )
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.split_ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        ap.error("--split-ratios needs three fractions summing to 1")

    indexes = build_config_index(args.config_dir)
    assistant_names = {n.strip() for n in args.assistant_names.split(",") if n.strip()}
    rows, problems = collect(args.data_dir, indexes, assistant_names)
    if not rows:
        sys.exit("nothing to export")
    manifest, m_seed, m_ratios = load_manifest(args.splits)
    if manifest:
        print(
            f"\nsplit manifest: {args.splits}  "
            f"({len(manifest)} groups, seed={m_seed}, ratios={list(m_ratios)})"
        )
        seed, ratios = m_seed or args.seed, m_ratios or ratios
    else:
        seed = args.seed
        print("\nno --splits manifest given; hashing groups on the fly")
    assignment, unlisted = assign_splits(rows, manifest, seed, ratios)
    report(rows, assignment, problems)
    if unlisted:
        print(
            f"\n{len(unlisted)} groups are NOT in the manifest and were hashed "
            f"instead — regenerate it to keep the record complete:"
        )
        for g in unlisted[:5]:
            print(f"   {g} -> {assignment[g]}")

    if not args.push:
        print("\nDRY RUN — nothing uploaded. Re-run with --push.\n")
        return
    build_and_push(rows, assignment, args)


if __name__ == "__main__":
    main()
