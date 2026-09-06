#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Character Rebalancer

Flattens character usage across scenario variant configs. A small number of
characters currently dominate the corpus (Marcus appears in 39.6% of multi_user
configs, Elena in 37.3%), which means a handful of TTS voices carry a large
share of the generated audio and cut across any train/test split.

This script caps how many configs any single character may appear in and
reassigns the excess to under-used characters, preserving gender so that
pronouns and voice casting stay coherent.

Character names appear throughout a config -- agent_configs, initial_prompt,
system_prompt, variation contexts and phase descriptions -- not just in the
default_characters block, so renames are applied to the whole file text with
word boundaries. Substitutions are applied in a single pass so that a rename
can never cascade into another rename.

Usage:
    # report only, touches nothing (default)
    python -m dataset.scripts.rebalance_characters --config-dir dataset/config

    # write the changes
    python -m dataset.scripts.rebalance_characters --config-dir dataset/config --apply
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Names that are also ordinary English words. Renaming these by word boundary can
# hit prose ("...with grace..."), so occurrences are reported for eyeballing.
RISKY_NAMES = {
    "Grace",
    "Hope",
    "Faith",
    "Rose",
    "Will",
    "Mark",
    "Art",
    "May",
    "June",
    "Iris",
    "Sunny",
    "Joy",
    "Dawn",
    "Summer",
    "Autumn",
    "Melody",
    "Angel",
    "Sky",
    "River",
    "Bill",
    "Frank",
    "Rich",
    "Chase",
    "Drew",
    "Wade",
}

# Never reassigned: the assistant is not a cast member.
ASSISTANT_NAMES = {"Sigma"}


def load_pool(config_dir):
    """Build the pool of characters usable as replacements.

    A character only qualifies if it has both a voice entry and a backstory file;
    a voice with no backstory cannot be cast, and a backstory with no voice cannot
    be synthesised.

    Args:
        config_dir: Config root holding ``voice_mapping.json`` and ``characters/``.

    Returns:
        Dict keyed by lowercase character key, each value holding ``name``,
        ``gender`` and ``backstory_file``.
    """
    vm_path = config_dir / "voice_mapping.json"
    vm = json.loads(vm_path.read_text(encoding="utf-8"))["characters"]
    backs = {p.stem.lower() for p in (config_dir / "characters").glob("*.txt")}

    pool = {}
    for name, entry in vm.items():
        key = name.lower()
        if key not in backs:
            continue  # voice but no backstory -- not usable as a replacement
        pool[key] = {
            "name": name,
            "gender": (entry.get("gender") or "neutral").lower(),
            "backstory_file": f"{key}.txt",
        }
    return pool


def iter_slots(block):
    """Iterate the real character slots of a ``default_characters`` block.

    Slots carrying no ``backstory_file`` are placeholders (for example the
    ``User1``/``User2`` entries in single_user blocks) and are skipped.

    Args:
        block: One conversation-type block, or any non-dict value.

    Yields:
        Tuples of (role, cfg) for each slot backed by a backstory file.
    """
    if not isinstance(block, dict):
        return
    for role, cfg in block.items():
        if isinstance(cfg, dict) and cfg.get("backstory_file"):
            yield role, cfg


def scan(scenarios_dir, block_name):
    """Collect the character slots of every config defining a given block.

    Configs that fail to parse are reported on stderr and skipped rather than
    aborting the run.

    Args:
        scenarios_dir: Root of the scenario tree to walk.
        block_name: Conversation-type block to read, ``multi_user`` or
            ``single_user``.

    Returns:
        List of records, each with ``path``, ``rel``, ``category`` and ``slots``
        (a list of (role, character_key, name) tuples). Configs that do not
        define the block are omitted.
    """
    records = []
    for path in sorted(scenarios_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! skipping unparseable {path}: {exc}", file=sys.stderr)
            continue
        block = (data.get("default_characters") or {}).get(block_name)
        slots = list(iter_slots(block))
        if not slots:
            continue
        rel = path.relative_to(scenarios_dir)
        records.append(
            {
                "path": path,
                "rel": rel.as_posix(),
                "category": rel.parts[0],
                "slots": [
                    (
                        role,
                        cfg["backstory_file"].replace(".txt", "").lower(),
                        cfg.get("name"),
                    )
                    for role, cfg in slots
                ],
            }
        )
    return records


def stable_order(records, key):
    """Order records deterministically, independent of filesystem order.

    Hashing uses the path relative to the scenarios dir, so the same plan is
    produced no matter where the config tree is checked out.

    Args:
        records: Records to order, as returned by :func:`scan`.
        key: Salt mixed into the hash; vary it to draw a different ordering.

    Returns:
        The records sorted by their salted hash.
    """
    return sorted(
        records,
        key=lambda r: hashlib.sha1(f"{key}:{r['rel']}".encode("utf-8")).hexdigest(),
    )


def plan(records, pool, max_share, seed, protect):
    """Decide which character slots to reassign and to whom.

    Characters appearing in more than ``max_share`` of the configs are capped;
    their excess slots are handed to the least-used same-gender character not
    already cast in that file. The most over-represented characters are processed
    first so they get first pick of the pool.

    Args:
        records: Records to plan over, as returned by :func:`scan`.
        pool: Replacement pool from :func:`load_pool`.
        max_share: Maximum fraction of configs one character may appear in.
        seed: Salt deciding which configs keep their original character.
        protect: Character keys never to reassign or assign.

    Returns:
        Tuple of (replacements, new_backstory, before, projected, cap, warnings):
        ``replacements`` maps path to {old_name: new_name}, ``new_backstory`` maps
        path to {role: new_key}, ``before`` and ``projected`` are per-character
        config counts before and after, ``cap`` is the per-character file limit,
        and ``warnings`` lists slots left unassigned because the pool ran dry.
    """
    n_files = len(records)
    cap = max(1, int(max_share * n_files))
    warnings = []

    files_using = defaultdict(list)
    for rec in records:
        for role, key, name in rec["slots"]:
            files_using[key].append((rec, role, name))

    before = Counter({k: len({id(r) for r, _, _ in v}) for k, v in files_using.items()})
    projected = Counter(before)
    replacements = defaultdict(dict)  # path -> {OldName: NewName}
    new_backstory = defaultdict(dict)  # path -> {role: new_key}

    # Highest offenders first so they get first pick of the pool.
    for key in sorted(before, key=lambda k: -before[k]):
        if key in protect or before[key] <= cap:
            continue
        gender = pool.get(key, {}).get("gender", "neutral")
        entries = stable_order([r for r, _, _ in files_using[key]], f"{seed}:{key}")
        keep = {id(r) for r in entries[:cap]}

        for rec, role, name in files_using[key]:
            if id(rec) in keep or not name:
                continue
            present = {n for _, _, n in rec["slots"] if n}
            present |= set(replacements[rec["path"]].values()) | ASSISTANT_NAMES

            candidate = pick(pool, gender, projected, present, key, protect)
            if candidate is None:
                warnings.append(
                    f"no {gender} candidate left for {name} in {rec['path']}"
                )
                continue

            replacements[rec["path"]][name] = pool[candidate]["name"]
            new_backstory[rec["path"]][role] = candidate
            projected[key] -= 1
            projected[candidate] += 1

    return replacements, new_backstory, before, projected, cap, warnings


def pick(pool, gender, projected, present, exclude_key, protect):
    """Choose the least-used eligible replacement for one slot.

    Gender is matched so that pronouns in the surrounding prose stay correct;
    ``neutral`` and ``non-binary`` slots accept any gender.

    Args:
        pool: Replacement pool from :func:`load_pool`.
        gender: Gender to match.
        projected: Running per-character config counts, including reassignments
            already planned.
        present: Names already cast in this file, which must not be reused.
        exclude_key: The character being replaced.
        protect: Character keys never to assign.

    Returns:
        The chosen character key, or None if no eligible candidate remains.
    """
    best, best_n = None, None
    for key, info in pool.items():
        if key == exclude_key or key in protect:
            continue
        if info["name"] in present:
            continue
        if gender in ("male", "female") and info["gender"] != gender:
            continue
        n = projected.get(key, 0)
        if best_n is None or n < best_n:
            best, best_n = key, n
    return best


def apply_to_text(text, name_map, backstory_map, block_name):
    """Rewrite one config's text with the planned renames.

    Character names occur throughout a config -- ``agent_configs``, prompts,
    variation contexts and phase descriptions -- so renaming is applied to the
    whole file on word boundaries. All names are substituted in a single pass so
    that one rename can never cascade into another.

    Args:
        text: Raw config text.
        name_map: Mapping of old display name to new display name.
        backstory_map: Mapping of role to new character key.
        block_name: Block whose ``backstory_file`` entries are retargeted, so
            the other conversation-type block is left untouched.

    Returns:
        The rewritten text. Formatting outside the substitutions is preserved.
    """
    if name_map:
        pattern = re.compile(
            r"\b("
            + "|".join(re.escape(n) for n in sorted(name_map, key=len, reverse=True))
            + r")\b"
        )
        text = pattern.sub(lambda m: name_map[m.group(1)], text)

    for role, new_key in backstory_map.items():
        # Retarget the backstory file for this role inside the right block only.
        text = re.sub(
            r'("'
            + re.escape(block_name)
            + r'"\s*:\s*\{(?:[^{}]|\{[^{}]*\})*?"'
            + re.escape(role)
            + r'"\s*:\s*\{(?:[^{}]*?))"backstory_file"\s*:\s*"[^"]*"',
            lambda m: m.group(1) + f'"backstory_file": "{new_key}.txt"',
            text,
            count=1,
        )
    return text


def histogram(counter, pool, top=12, total=None):
    """Render a character-usage counter as aligned report lines.

    Args:
        counter: Per-character config counts.
        pool: Replacement pool, used to resolve display names.
        top: How many characters to list.
        total: Config count used to render a share column; omitted when None.

    Returns:
        The formatted lines joined by newlines.
    """
    lines = []
    for i, (key, n) in enumerate(counter.most_common(top), 1):
        label = pool.get(key, {}).get("name", key)
        share = f"{100 * n / total:5.1f}%" if total else ""
        lines.append(f"   {i:>2} {label:14} {n:>5} {share}")
    return "\n".join(lines)


def main():
    """Parse arguments, report the rebalancing plan and optionally apply it.

    Reporting is the default; nothing is written unless ``--apply`` is passed and
    ``--dry-run`` is not. Each rewritten config is re-parsed before being saved,
    and skipped if the rename would have produced invalid JSON.
    """
    ap = argparse.ArgumentParser(
        description="Flatten character usage across scenario configs."
    )
    ap.add_argument("--config-dir", type=Path, default=Path("dataset/config"))
    ap.add_argument(
        "--block",
        default="multi_user",
        choices=["multi_user", "single_user"],
        help="Which default_characters block to rebalance",
    )
    ap.add_argument(
        "--max-share",
        type=float,
        default=0.05,
        help="Max fraction of configs one character may appear in (default 0.05)",
    )
    ap.add_argument(
        "--protect", default="", help="Comma-separated character keys to never reassign"
    )
    ap.add_argument(
        "--seed",
        default="rebalance-v1",
        help="Changes which files keep the original character",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without it the script only reports.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly report only (this is already the default)",
    )
    ap.add_argument("--report", type=Path, help="Write the rename map as JSON")
    args = ap.parse_args()

    write = args.apply and not args.dry_run
    scenarios_dir = args.config_dir / "scenarios"
    if not scenarios_dir.is_dir():
        sys.exit(f"scenarios dir not found: {scenarios_dir}")

    pool = load_pool(args.config_dir)
    protect = {p.strip().lower() for p in args.protect.split(",") if p.strip()}
    records = scan(scenarios_dir, args.block)
    if not records:
        sys.exit(f"no configs define a '{args.block}' block under {scenarios_dir}")

    name_maps, back_maps, before, after, cap, warns = plan(
        records, pool, args.max_share, args.seed, protect
    )

    n = len(records)
    print(
        f"\nblock={args.block}  configs={n}  characters={len(before)}  "
        f"pool={len(pool)}  cap={cap} files ({args.max_share:.0%})"
    )
    print(f"\nBEFORE (top 12 of {len(before)}):\n{histogram(before, pool, total=n)}")
    after_nz = Counter({k: v for k, v in after.items() if v})
    print(
        f"\nAFTER  (top 12 of {len(after_nz)}):\n{histogram(after_nz, pool, total=n)}"
    )

    def conc(c, k):
        """Return the share of all assignments held by the top ``k`` characters."""
        tot = sum(c.values())
        return 100 * sum(v for _, v in c.most_common(k)) / tot if tot else 0

    print(
        f"\n   top5  {conc(before,5):.1f}% -> {conc(after_nz,5):.1f}%"
        f"   |  top20  {conc(before,20):.1f}% -> {conc(after_nz,20):.1f}%"
        f"   |  distinct {len(before)} -> {len(after_nz)}"
    )

    total_renames = sum(len(m) for m in name_maps.values())
    print(
        f"\n{len(name_maps)} files to change, {total_renames} character slots reassigned"
    )

    risky = {old for m in name_maps.values() for old in m} & RISKY_NAMES
    if risky:
        print(
            f"   ! names that are also common words, check their diffs: {sorted(risky)}"
        )
    for w in warns[:10]:
        print(f"   ! {w}")
    if len(warns) > 10:
        print(f"   ! ...and {len(warns) - 10} more")

    if args.report:
        args.report.write_text(
            json.dumps(
                {str(p): m for p, m in name_maps.items()}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print(f"   rename map written to {args.report}")

    if not write:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write.\n")
        for path, m in list(name_maps.items())[:5]:
            print(
                f"   {path.relative_to(scenarios_dir)}: "
                + ", ".join(f"{o}->{n}" for o, n in m.items())
            )
        if len(name_maps) > 5:
            print(f"   ...and {len(name_maps) - 5} more files")
        return

    changed = 0
    for path, m in name_maps.items():
        text = path.read_text(encoding="utf-8")
        new_text = apply_to_text(text, m, back_maps[path], args.block)
        try:
            json.loads(new_text)
        except json.JSONDecodeError as exc:
            print(
                f"  ! {path} would become invalid JSON, skipped: {exc}", file=sys.stderr
            )
            continue
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
    print(f"\nwrote {changed} files\n")


if __name__ == "__main__":
    main()
