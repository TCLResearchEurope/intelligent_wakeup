#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Corpus Splitter

Assigns every scenario to train/validation/test and writes a manifest the
exporter reads. The split is a property of the scenario config, not of whatever
happens to be generated, so it can be computed before any audio exists and
stays meaningful as generation progresses.

Two properties matter and neither is free:

**Stable under change.** Assignment is a hash of the group id thresholded
against the ratios, never a sort-and-cut. Cutting a sorted list at 80% moves
every item past the cut as soon as one scenario is added or removed; hashing
leaves existing scenarios exactly where they were and places only the new ones.

**Grouped.** A scenario's short variation, its ``_long`` sibling, its
``_variant2``/``_variant3`` re-takes and its ``no_va_`` twin all play the same
situation in the same room. They are assigned as one unit, because splitting
them apart puts effectively-seen material in the test set: the variants are
re-cast, but the scene, the phase structure and the trigger are the same.

Hashing is applied per category so every split covers the whole domain. The cost
is that realised proportions drift a little from the requested ratios -- that is
the price of stability, and the report shows the actual figures.

Usage:
    python -m dataset.scripts.split_corpus --ratios 0.8,0.1,0.1
    python -m dataset.scripts.split_corpus --write dataset/config/splits.json
    python -m dataset.scripts.split_corpus --check dataset/config/splits.json
"""

import re
import sys
import json
import argparse
import hashlib
from pathlib import Path

SPLITS = ("train", "validation", "test")


def group_key(category, variant_type, variant_name):
    """Compute the unit a scenario is split as.

    Strips the ``no_va_`` prefix and the ``_long`` / ``_variantN`` suffixes so a
    scenario's twin, its long sibling and its re-cast variants all join the same
    group as the original. Suffixes are removed repeatedly so any combination
    ("x_variant2_long") collapses to the same base.

    Args:
        category: Scenario category directory name.
        variant_type: ``couple`` or ``single``.
        variant_name: The variation's name.

    Returns:
        A stable group identifier.
    """
    base = variant_name
    if base.startswith("no_va_"):
        base = base[len("no_va_") :]
    while True:
        stripped = re.sub(r"_(?:long|variant\d+)$", "", base)
        if stripped == base:
            break
        base = stripped
    return f"{category}/{variant_type}/{base}"


def bucket(key, seed):
    """Map a group id to a uniform position in [0, 1).

    Args:
        key: Group identifier.
        seed: Salt. Changing it reshuffles the whole corpus, so treat it as
            part of the release definition.

    Returns:
        Float in [0, 1), stable across runs, machines and Python versions.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(group, seed, ratios):
    """Choose a split for one group.

    Args:
        group: Group identifier.
        seed: Hash salt.
        ratios: (train, validation, test) fractions summing to 1.

    Returns:
        One of ``train``, ``validation``, ``test``.
    """
    position = bucket(group, seed)
    train_r, val_r, _ = ratios
    if position < train_r:
        return "train"
    if position < train_r + val_r:
        return "validation"
    return "test"


def scan(config_dir):
    """Collect every variation in the corpus with its group and metadata.

    Args:
        config_dir: Config root containing ``scenarios/``.

    Returns:
        List of per-variation records.
    """
    root = config_dir / "scenarios"
    if not root.is_dir():
        sys.exit(f"no scenarios under {config_dir}")
    records = []
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        category, variant_type = rel.parts[0], rel.parts[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! unparseable {rel}: {exc}", file=sys.stderr)
            continue
        assistant = (data.get("agent_configs") or {}).get("assistant") or {}
        is_no_va = assistant.get("max_turns") == 0 or path.name.startswith("no_va_")
        for idx, variation in enumerate(data.get("variations") or []):
            name = variation.get("variant_name") or path.stem
            turns = sum(
                len(p.get("speakers") or [])
                for p in variation.get("scenario_template") or []
            )
            records.append(
                dict(
                    category=category,
                    variant_type=variant_type,
                    variant_name=name,
                    group=group_key(category, variant_type, name),
                    no_va=is_no_va,
                    turns=turns or variation.get("max_turns", 12),
                    config=str(rel),
                    index=idx,
                )
            )
    return records


def build(records, seed, ratios):
    """Assign a split to every group.

    Args:
        records: Variation records from :func:`scan`.
        seed: Hash salt.
        ratios: (train, validation, test) fractions.

    Returns:
        Dict mapping group id to split name.
    """
    return {r["group"]: assign(r["group"], seed, ratios) for r in records}


def report(records, mapping, seconds_per_turn):
    """Print realised split sizes against the request.

    Args:
        records: Variation records.
        mapping: Group to split assignment.
        seconds_per_turn: Measured audio seconds per turn, for hour estimates.

    Returns:
        None.
    """
    per = {s: dict(convs=0, turns=0, groups=set(), no_va=0) for s in SPLITS}
    for r in records:
        b = per[mapping[r["group"]]]
        b["convs"] += 1
        b["turns"] += r["turns"]
        b["groups"].add(r["group"])
        b["no_va"] += bool(r["no_va"])
    total = len(records)

    print(
        f"\n{'split':12}{'convs':>7}{'share':>8}{'groups':>8}{'turns':>8}"
        f"{'hours':>8}{'no_VA':>8}"
    )
    print("-" * 59)
    for s in SPLITS:
        b = per[s]
        print(
            f"{s:12}{b['convs']:>7}{b['convs']/total:>7.1%}{len(b['groups']):>8}"
            f"{b['turns']:>8}{b['turns']*seconds_per_turn/3600:>8.1f}"
            f"{b['no_va']/max(1,b['convs']):>7.0%}"
        )
    print("-" * 59)
    print(
        f"{'TOTAL':12}{total:>7}{'':>8}{len(mapping):>8}"
        f"{sum(r['turns'] for r in records):>8}"
        f"{sum(r['turns'] for r in records)*seconds_per_turn/3600:>8.1f}"
    )

    missing = [
        c
        for c in {r["category"] for r in records}
        if len({mapping[r["group"]] for r in records if r["category"] == c}) < 3
    ]
    if missing:
        print(
            f"\ncategories not present in all three splits ({len(missing)}): "
            f"{sorted(missing)[:6]}"
        )


def check(records, mapping, manifest_path):
    """Compare a fresh assignment against a stored manifest.

    Args:
        records: Variation records.
        mapping: Freshly computed group to split assignment.
        manifest_path: Manifest to compare against.

    Returns:
        Exit status: 0 when nothing moved, 1 when a group changed split.
    """
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    old = stored["groups"]
    moved = {
        g: (old[g], mapping[g]) for g in set(old) & set(mapping) if old[g] != mapping[g]
    }
    added = sorted(set(mapping) - set(old))
    removed = sorted(set(old) - set(mapping))
    print(
        f"\nunchanged {len(set(old) & set(mapping)) - len(moved)}"
        f" | moved {len(moved)} | added {len(added)} | removed {len(removed)}"
    )
    for g, (a, b) in list(moved.items())[:10]:
        print(f"   MOVED {g}: {a} -> {b}")
    for g in added[:10]:
        print(f"   added {g} -> {mapping[g]}")
    for g in removed[:10]:
        print(f"   removed {g} (was {old[g]})")
    if moved:
        print(
            "\nFAIL: existing groups changed split. Seed or ratios must have changed;"
            " published results are no longer comparable."
        )
        return 1
    print("\nOK: no existing group changed split.")
    return 0


def main():
    """Compute the split, then report, write or verify it."""
    ap = argparse.ArgumentParser(description="Assign scenarios to train/val/test.")
    ap.add_argument("--config-dir", type=Path, default=Path("dataset/config"))
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument(
        "--seed",
        default="intelligent-wakeup-v1",
        help="Part of the release definition: changing it reshuffles "
        "every scenario and invalidates published results",
    )
    ap.add_argument(
        "--seconds-per-turn",
        type=float,
        default=10.0,
        help="Measured from generated audio; used for hour estimates",
    )
    ap.add_argument("--write", type=Path, help="Write the manifest here")
    ap.add_argument("--check", type=Path, help="Verify nothing moved vs this manifest")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        ap.error("--ratios needs three fractions summing to 1")

    records = scan(args.config_dir)
    mapping = build(records, args.seed, ratios)
    report(records, mapping, args.seconds_per_turn)

    if args.check:
        sys.exit(check(records, mapping, args.check))

    if args.write:
        payload = {
            "seed": args.seed,
            "ratios": list(ratios),
            "groups": dict(sorted(mapping.items())),
            "variations": {
                f"{r['category']}/{r['variant_type']}/{r['variant_name']}": mapping[
                    r["group"]
                ]
                for r in sorted(
                    records, key=lambda r: (r["category"], r["variant_name"])
                )
            },
        }
        args.write.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"\nwrote {args.write} "
            f"({len(payload['groups'])} groups, {len(payload['variations'])} variations)"
        )


if __name__ == "__main__":
    main()
