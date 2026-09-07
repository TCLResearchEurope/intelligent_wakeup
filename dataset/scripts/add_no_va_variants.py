#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

No-VA Variant Generator

Adds ``no_va`` twins to couple scenarios: same cast, same setting, same trigger
moment, but the question resolves between the people instead of being put to
Sigma. This is the pattern already established in WatchingTV, where every
``no_va_X.json`` pairs with an ``X.json``.

Twins are the most useful negatives a wakeup model can train on -- the acoustic
scene and the conversational context are identical, and only the device
interaction differs, so the model cannot separate them on room tone or topic.

The mechanical half of the transform is deterministic (name suffix, assistant
role and ``max_turns``). The prose half -- rewriting the prompts so the question
resolves without Sigma -- needs an LLM and therefore an API key.

Usage:
    # what would be created, and the resulting share (no API key needed)
    python -m dataset.scripts.add_no_va_variants --target-share 0.30

    # actually generate
    python -m dataset.scripts.add_no_va_variants --target-share 0.30 --apply
"""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Categories whose premise IS the assistant. A no-VA twin there does not
# describe the same category any more, so they are skipped by default.
# --include-va-roles overrides this; see the module docstring in the report.
VA_CENTRIC = {
    "VoiceControl",
    "VirtualAssistantMultiDeviceTesting",
    "VARoleAdvisor",
    "VARoleArbiter",
    "VARoleDiagnostician",
    "VARoleEditor",
    "VARoleInterpreter",
    "VARoleModerator",
    "VARoleSecretary",
}

REWRITE_SYSTEM = """\
You rewrite conversation scenario BRIEFS so that a voice assistant is never addressed.

A brief is a short instruction to a dialogue generator. It is NOT dialogue itself.

You are given a JSON object of brief fragments. Some describe people reaching a
question and putting it to an assistant called Sigma. Rewrite each fragment so the
same question arises naturally and resolves WITHOUT any assistant: someone suddenly
remembers, another person knows, a character on screen says it, someone checks a
phone or a label, or they simply let it go. Letting it go unresolved is realistic
and often the best choice.

Rules:
- Return a JSON object with EXACTLY the same keys, each value the rewritten text.
- Keep each fragment the same KIND of text and roughly the same length. A one-line
  instruction stays a one-line instruction.
- NEVER write dialogue, quoted speech, or prose narration of a scene.
- NEVER invent character names. Use only names already present in the fragment.
- Keep the same people, place, mood and trigger moment.
- Never mention Sigma, an assistant, a smart speaker, or any voice device.
- Do not comment on the absence of an assistant. It is simply not used.
"""

# Stripping the assistant from a phase's speaker list is not enough on its own:
# a phase still called "assistance_request" makes the user agents summon the
# assistant by name anyway, and it then answers. Assistance-shaped phase names
# are renamed to life-first equivalents, and every phase carries an explicit
# guard, which is how the hand-authored no_va scenarios keep the assistant out.
PHASE_RENAME = {
    "seeking_help": "weighing_options",
    "assistance_request": "comparing_ideas",
    "assistance": "talking_it_over",
    "seeking_guidance": "thinking_it_through",
    "seeking_advice": "thinking_it_through",
    "sigma_moment": "a_passing_question",
    "assistant_query": "a_passing_question",
    "help_request": "weighing_options",
    "recommendation_request": "comparing_ideas",
    "problem": "the_snag",
    # conversation_manager._validate_phase_speakers force-adds the assistant back
    # to any phase named one of these three, whatever the speakers list says.
    # Renaming them is the only way to keep the assistant out.
    "discussion": "talking_it_over",
    "planning": "mapping_it_out",
    "assistance_request ": "comparing_ideas",
}
FORCED_ASSISTANT_PHASES = {"assistance_request", "discussion", "planning"}
ASSIST_PHASE = re.compile(r"assist|help|sigma|recommend|advice|guidance|query", re.I)
PHASE_GUARD = (
    "No assistant is present or addressed in this scenario. The people resolve "
    "everything between themselves, or leave it unresolved. Never say 'Sigma'."
)


NEGATION = re.compile(r"\b(not|never|no|without)\b", re.I)
AFFIRM_ASSISTANT = re.compile(
    r"\b(sigma|(the )?assistant)\b[^.!?]{0,40}?"
    r"\b(answers?|speaks?|gives?|responds?|replies|provides?|offers?|says?"
    r"|is addressed|adds?|confirms?|assigns?|suggests?|recommends?|explains?"
    r"|reminds?|lists?|reads?|sets?|helps?)\b", re.I)
REQUEST_TOPIC = re.compile(
    r"\b(request|receiv|recommend|suggest|advice|advis|guidance|ask|answer"
    r"|provid|instruct|tip|explanat|clarif|help)\w*", re.I)
NO_VA_REVIEW_RULE = "The assistant is never addressed and never speaks in this scenario."


def _drop_affirmative(text):
    """Remove sentences that assert the assistant speaks.

    Args:
        text: Free text, possibly None.

    Returns:
        The text with affirmative assistant sentences removed, or None when the
        input was empty.
    """
    if not text:
        return text
    kept = [s for s in re.split(r"(?<=[.!?])\s+", str(text))
            if not (AFFIRM_ASSISTANT.search(s) and not NEGATION.search(s))]
    return " ".join(kept).strip()


def strip_affirmative_directives(config):
    """Remove every instruction telling the pipeline that the assistant speaks.

    Two fields drive the dialogue enhancer independently of the generated turns:
    ``dialogue_review_rules`` (whose ``assistant_behavior`` entries make it insert
    a Sigma exchange into an otherwise clean take) and each phase's ``note``.
    Both are rewritten to a negative directive.

    Args:
        config: Twin config, modified in place.

    Returns:
        Number of edits made.
    """
    edits = 0
    rules = config.get("dialogue_review_rules")
    if isinstance(rules, dict):
        if rules.get("assistant_behavior") != [NO_VA_REVIEW_RULE]:
            rules["assistant_behavior"] = [NO_VA_REVIEW_RULE]
            edits += 1
        structure = rules.get("conversation_structure")
        if isinstance(structure, list):
            kept = [r for r in structure
                    if not (AFFIRM_ASSISTANT.search(str(r)) and not NEGATION.search(str(r)))]
            if kept != structure:
                rules["conversation_structure"] = kept
                edits += 1
    for variation in config.get("variations") or []:
        for phase in variation.get("scenario_template") or []:
            note = phase.get("note")
            cleaned = _drop_affirmative(note)
            if cleaned != note:
                phase["note"] = cleaned
                edits += 1
    return edits


def deassist_phases(config):
    """Remove every route by which the assistant can enter a scenario.

    Three things are needed, and all three are structural rather than prose:
    the assistant is dropped from each phase's speaker rotation, phase names
    that script an assistance request are renamed, and each phase gains an
    explicit guard note.

    Args:
        config: Twin config, modified in place.

    Returns:
        Number of phases changed.
    """
    changed = 0
    for variation in config.get("variations") or []:
        for phase in variation.get("scenario_template") or []:
            before = json.dumps(phase, sort_keys=True)
            speakers = phase.get("speakers")
            if speakers:
                phase["speakers"] = [
                    s for s in speakers if "assistant" not in str(s).lower()
                ]
            name = str(phase.get("phase", ""))
            if name in PHASE_RENAME:
                phase["phase"] = PHASE_RENAME[name]
            elif ASSIST_PHASE.search(name):
                phase["phase"] = "talking_it_over"
            if phase.get("phase") in FORCED_ASSISTANT_PHASES:
                raise AssertionError(
                    f"phase {phase.get('phase')!r} would have the assistant "
                    "force-added by conversation_manager")
            note = str(phase.get("note") or "").strip()
            if PHASE_GUARD not in note:
                phase["note"] = f"{note} {PHASE_GUARD}".strip()
            if json.dumps(phase, sort_keys=True) != before:
                changed += 1
    return changed


# Only prose that actually references the assistant needs rewriting. Most configs
# are generic briefs that never mention it, and their VA behaviour is carried
# entirely by ``max_turns`` -- rewriting those invites the model to invent a scene.
ASSISTANT_MENTION = re.compile(
    r"\b(sigma|assistant|voice device|smart speaker|voice command)\b", re.I)


def needs_rewrite(text):
    """Report whether a brief fragment references the assistant.

    Args:
        text: Brief fragment, possibly None.

    Returns:
        True when the fragment mentions the assistant and so must be rewritten.
    """
    return bool(text and ASSISTANT_MENTION.search(str(text)))


def assistant_turns(data):
    """Read the assistant's turn budget for a scenario config.

    Args:
        data: Parsed scenario config.

    Returns:
        The assistant's ``max_turns``, or None when the config does not set one.
    """
    return ((data.get("agent_configs") or {}).get("assistant") or {}).get("max_turns")


def is_no_va(path, data):
    """Decide whether a config is already a no-VA scenario.

    ``max_turns == 0`` is the operative marker; the name is only a fallback for
    the handful of configs that omit an assistant budget entirely.

    Args:
        path: Config path, used for the filename fallback.
        data: Parsed scenario config.

    Returns:
        True when the assistant is never addressed in this scenario.
    """
    if assistant_turns(data) == 0:
        return True
    named = "no_va" in str(data.get("name", "")).lower() or path.name.startswith("no_va_")
    return named and assistant_turns(data) is None


def scan(scenarios_dir, block):
    """Inventory one conversation-type block, split into VA and no-VA configs.

    Args:
        scenarios_dir: Root of the scenario tree.
        block: Variant subdirectory to walk, normally ``couple``.

    Returns:
        Tuple of (va, no_va): lists of (path, category, data) records.
    """
    va, novа = [], []
    for path in sorted(scenarios_dir.rglob("*.json")):
        rel = path.relative_to(scenarios_dir)
        if len(rel.parts) < 3 or rel.parts[1] != block:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! skipping unparseable {rel}: {exc}", file=sys.stderr)
            continue
        (novа if is_no_va(path, data) else va).append((path, rel.parts[0], data))
    return va, novа


def twin_path(path):
    """Return the path a config's no-VA twin would occupy.

    Args:
        path: Path of the VA scenario.

    Returns:
        Sibling path with the ``no_va_`` filename prefix.
    """
    return path.with_name(f"no_va_{path.name}")


def needed_for_share(n_no_va, n_total, share):
    """Compute how many twins bring the corpus to a target no-VA share.

    Adding a twin raises both the no-VA count and the total, so this solves
    ``(n + k) / (total + k) = share`` for k.

    Args:
        n_no_va: Current no-VA config count.
        n_total: Current total config count.
        share: Desired no-VA fraction, between 0 and 1.

    Returns:
        Number of twins to add, never negative.
    """
    if share >= 1:
        raise ValueError("target share must be below 1.0")
    return max(0, int(round((share * n_total - n_no_va) / (1 - share))))


def select(va, existing_twins, categories, target, seed):
    """Choose which VA scenarios to twin, spread evenly across categories.

    Categories are taken round-robin so that a large category like MovingHouse
    cannot absorb the whole budget, and configs within a category are ordered by
    a salted hash for reproducibility.

    Args:
        va: VA records from :func:`scan`.
        existing_twins: Paths that already hold a twin and must be skipped.
        categories: Categories eligible for twinning.
        target: How many twins to select.
        seed: Salt for the per-category ordering.

    Returns:
        List of selected VA records, at most ``target`` long.
    """
    buckets = {}
    for rec in va:
        path, cat, _ = rec
        if cat not in categories or twin_path(path) in existing_twins:
            continue
        buckets.setdefault(cat, []).append(rec)
    for cat, items in buckets.items():
        items.sort(key=lambda r: hashlib.sha1(
            f"{seed}:{r[0].name}".encode("utf-8")).hexdigest())

    picked, order = [], sorted(buckets)
    while len(picked) < target and any(buckets.values()):
        for cat in order:
            if not buckets[cat]:
                continue
            picked.append(buckets[cat].pop(0))
            if len(picked) >= target:
                break
    return picked


async def rewrite_batch(client, model, fields):
    """Rewrite every assistant-referencing fragment of one config in a single call.

    Batching keeps the fragments consistent with each other and costs one request
    per config instead of one per field.

    Args:
        client: An ``AsyncOpenAI``-compatible client.
        model: Model name to call.
        fields: Mapping of field key to original text.

    Returns:
        Mapping of field key to rewritten text. Keys the model fails to return are
        omitted so the caller can keep the original.
    """
    if not fields:
        return {}
    resp = await client.chat.completions.create(
        model=model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": json.dumps(fields, ensure_ascii=False)},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    return {k: v for k, v in out.items() if k in fields and isinstance(v, str) and v.strip()}


async def build_twin(client, model, data):
    """Produce the no-VA twin of one scenario config.

    The assistant is left in place but silenced, matching the existing WatchingTV
    twins, so downstream TTS and audio config stay identical across the pair. Only
    fragments that actually reference the assistant are sent to the model; configs
    whose briefs never mention it are converted with no LLM call at all.

    Args:
        client: An ``AsyncOpenAI``-compatible client.
        model: Model name used for the prose rewrites.
        data: Parsed VA scenario config.

    Returns:
        Tuple of (config, n_rewritten, unresolved): the twin, how many fragments
        the model rewrote, and the keys that still mention the assistant after a
        retry.
    """
    out = copy.deepcopy(data)
    name = str(out.get("name", "")).strip()
    if not name.endswith("_no_va"):
        out["name"] = f"{name}_no_va"

    assistant = (out.setdefault("agent_configs", {})).setdefault("assistant", {})
    assistant["role"] = "Ambient household assistant. Not addressed in this scenario."
    assistant["max_turns"] = 0

    deassist_phases(out)

    # Collect only the fragments that mention the assistant, keyed by location.
    fields, slots = {}, {}
    if needs_rewrite(out.get("initial_prompt")):
        fields["initial_prompt"] = out["initial_prompt"]
        slots["initial_prompt"] = lambda v: out.__setitem__("initial_prompt", v)
    custom = out.setdefault("framework_config", {}).setdefault("custom", {})
    if needs_rewrite(custom.get("system_prompt")):
        fields["system_prompt"] = custom["system_prompt"]
        slots["system_prompt"] = lambda v: custom.__setitem__("system_prompt", v)
    for vi, variation in enumerate(out.get("variations") or []):
        for pi, phase in enumerate(variation.get("scenario_template") or []):
            if needs_rewrite(phase.get("topic")):
                key = f"phase_{vi}_{pi}"
                fields[key] = phase["topic"]
                slots[key] = (lambda ph: lambda v: ph.__setitem__("topic", v))(phase)

    rewritten = await rewrite_batch(client, model, fields)

    # The model sometimes drops a key or leaves a mention in a long fragment.
    # Retry just those once, then give up and report rather than write a twin
    # that still tells the generator to address the assistant.
    stubborn = {k: v for k, v in fields.items()
                if needs_rewrite(rewritten.get(k, v))}
    if stubborn:
        retry = await rewrite_batch(client, model, stubborn)
        rewritten.update({k: v for k, v in retry.items() if not needs_rewrite(v)})

    unresolved = [k for k in fields if needs_rewrite(rewritten.get(k, fields[k]))]
    for key, value in rewritten.items():
        if not needs_rewrite(value):
            slots[key](value)
    return out, len(rewritten), unresolved


def make_client(model):
    """Construct an LLM client from the environment.

    Mirrors the provider handling used elsewhere in ``generate_text`` so the same
    ``.env`` works for both.

    Args:
        model: Model name, used to pick the Gemini-compatible endpoint.

    Returns:
        An ``AsyncOpenAI`` client.

    Raises:
        SystemExit: If no API key is present in the environment.
    """
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv()
    if model.startswith("gemini"):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY / GOOGLE_API_KEY not set (add it to .env)")
        return AsyncOpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set (add it to .env)")
    return AsyncOpenAI(api_key=key)


async def run(args):
    """Plan the twins, report the outcome and optionally write them.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    scenarios_dir = args.config_dir / "scenarios"
    if not scenarios_dir.is_dir():
        sys.exit(f"scenarios dir not found: {scenarios_dir}")

    va, novа = scan(scenarios_dir, args.block)
    total, n_nv = len(va) + len(novа), len(novа)
    categories = {c for _, c, _ in va} - (set() if args.include_va_roles else VA_CENTRIC)
    existing = {p for p, _, _ in novа}

    target = args.limit or needed_for_share(n_nv, total, args.target_share)
    picked = select(va, existing, categories, target, args.seed)

    share_now = 100 * n_nv / total if total else 0
    after = 100 * (n_nv + len(picked)) / (total + len(picked)) if total else 0
    print(f"\nblock={args.block}  configs={total}  no_va={n_nv} ({share_now:.1f}%)")
    print(f"eligible categories={len(categories)}"
          f"  excluded={0 if args.include_va_roles else len(VA_CENTRIC)}"
          f"  VA configs available to twin={sum(1 for p,c,_ in va if c in categories and twin_path(p) not in existing)}")
    print(f"target share={args.target_share:.0%} -> need {target} twins, selected {len(picked)}")
    print(f"projected: {n_nv + len(picked)}/{total + len(picked)} = {after:.1f}%\n")

    by_cat = {}
    for path, cat, _ in picked:
        by_cat.setdefault(cat, []).append(path.name)
    for cat in sorted(by_cat):
        print(f"   {cat:36} +{len(by_cat[cat]):>3}")

    if not args.apply:
        print(f"\nDRY RUN -- nothing written. Re-run with --apply to generate.")
        for path, cat, _ in picked[:5]:
            print(f"   would create {twin_path(path).relative_to(scenarios_dir)}")
        if len(picked) > 5:
            print(f"   ...and {len(picked) - 5} more")
        return

    client = make_client(args.model)
    sem = asyncio.Semaphore(args.concurrency)
    done = {"n": 0, "rewritten": 0, "plain": 0}

    async def one(path, data):
        """Generate and write a single twin under the concurrency limit.

        Args:
            path: Path of the source VA config.
            data: Parsed source config.

        Returns:
            None.
        """
        dest = twin_path(path)
        if dest.exists():
            return
        async with sem:
            try:
                twin, n_rw, unresolved = await build_twin(client, args.model, data)
            except Exception as exc:  # one bad rewrite must not stop the run
                print(f"  ! {path.name}: {exc}", file=sys.stderr)
                return
        if unresolved:
            print(f"  ! {path.name}: still mentions the assistant in "
                  f"{unresolved}, not written", file=sys.stderr)
            return
        dest.write_text(json.dumps(twin, ensure_ascii=False, indent=4) + "\n",
                        encoding="utf-8")
        done["n"] += 1
        done["rewritten" if n_rw else "plain"] += 1
        print(f"  [{done['n']}/{len(picked)}] {dest.relative_to(scenarios_dir)}"
              f"{f' ({n_rw} fragments rewritten)' if n_rw else ' (no VA prose, config-only)'}")

    await asyncio.gather(*(one(p, d) for p, _, d in picked))
    print(f"\nwrote {done['n']} twins "
          f"({done['rewritten']} needed prose rewrites, {done['plain']} config-only)\n")


def main():
    """Parse arguments and run the generator."""
    ap = argparse.ArgumentParser(description="Add no-VA twins to couple scenarios.")
    ap.add_argument("--config-dir", type=Path, default=Path("dataset/config"))
    ap.add_argument("--block", default="couple", choices=["couple", "single"])
    ap.add_argument("--target-share", type=float, default=0.30,
                    help="Desired no-VA share of the block (default 0.30)")
    ap.add_argument("--limit", type=int, help="Twin exactly this many, ignoring the share")
    ap.add_argument("--include-va-roles", action="store_true",
                    help="Also twin VoiceControl / VARole* categories")
    ap.add_argument("--model", default="gpt-4o", help="Model used for prose rewrites")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="How many configs to rewrite in parallel (default 8)")
    ap.add_argument("--seed", default="no-va-v1")
    ap.add_argument("--apply", action="store_true",
                    help="Write the twins. Without it the script only reports.")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
