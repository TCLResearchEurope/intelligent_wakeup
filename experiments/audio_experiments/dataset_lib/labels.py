"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Turn labelling logic — mirrors evaluate_corpus.py so labels are identical
to those used in the GPT Realtime baseline evaluation.
"""

from __future__ import annotations

TRIGGER_TYPE_NONE = 0        # non-assistance
TRIGGER_TYPE_DIRECT = 1      # speaker said "sigma"
TRIGGER_TYPE_CONTEXTUAL = 2  # follow-up in active Sigma exchange

TRIGGER_TYPE_MAP = {
    "non-assistance": TRIGGER_TYPE_NONE,
    "direct": TRIGGER_TYPE_DIRECT,
    "contextual": TRIGGER_TYPE_CONTEXTUAL,
}

# Speaker role IDs (role-based, not identity-based → generalises across scenarios)
SPEAKER_PAD = 0      # padding / unknown
SPEAKER_SIGMA = 1    # the VA
SPEAKER_HUMAN_A = 2  # first human speaker in conversation (by order of appearance)
SPEAKER_HUMAN_B = 3  # second human speaker
SPEAKER_HUMAN_C = 4  # third (rare)


def label_turns(turns: list[dict]) -> list[dict]:
    """
    Annotate each turn with:
      expected      – bool, True if the next turn is spoken by Sigma
      trigger_type  – "direct" | "contextual" | "non-assistance"
    """
    labelled = []
    for idx, turn in enumerate(turns):
        next_turn = turns[idx + 1] if idx + 1 < len(turns) else None
        expected = next_turn is not None and next_turn["speaker"] == "Sigma"
        has_sigma_word = "sigma" in turn["content"].lower()

        if not expected:
            trigger_type = "non-assistance"
        elif has_sigma_word:
            trigger_type = "direct"
        else:
            trigger_type = "contextual"

        labelled.append({**turn, "expected": expected, "trigger_type": trigger_type})
    return labelled


def build_speaker_map(conversation: list[dict]) -> dict[str, int]:
    """
    Assign role-based integer IDs to speakers within one conversation.
    Sigma is always 1; humans get 2, 3, 4… by order of first appearance.
    """
    mapping: dict[str, int] = {"Sigma": SPEAKER_SIGMA}
    next_id = SPEAKER_HUMAN_A
    for turn in conversation:
        name = turn["speaker"]
        if name not in mapping:
            mapping[name] = next_id
            next_id += 1
    return mapping
