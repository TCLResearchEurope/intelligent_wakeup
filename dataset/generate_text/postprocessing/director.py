"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Director agent that reviews a completed conversation take and writes notes for
the next take.

Unlike the dialogue enhancer — which rewrites the finished text directly — the
director never edits lines. It reads the whole scene at once and writes notes
that are fed back to the character agents, who then perform the scene again
from a clean slate. The concern is whether the scene reads as a real, inhabited
place rather than disembodied voices exchanging information.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

from ...utils import logger

GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


DIRECTOR_SYSTEM = """\
You are the director of a film, reviewing a take of a single scene.

Your only concern is whether this scene feels REAL and INHABITED — as if a camera
were running in an actual room with actual people. You are not a copy editor and
you do not rewrite lines. You give the actors notes, and they perform it again.

Judge the take against these, in order of importance:

1. GROUNDING — Does the scene establish where we are and what is happening before
   it dives into content? A listener dropped into this scene should quickly sense
   the room, who is present, and what just happened before the camera started.
   Scenes that open cold on abstract business content are the most common failure.
2. PHYSICAL PRESENCE — Do people exist in a place? Do they reference the room, the
   screen share, a document, coffee, someone's connection dropping, arriving late,
   the time? Voices with no bodies and no environment feel like a transcript, not
   a scene.
3. CONTEXT CARRIED IN DIALOGUE — Is what they are talking about explained naturally
   through how people speak, rather than assumed? Real colleagues remind each other
   of things, half-explain, reference last week. A listener with no background should
   be able to follow what is at stake.
4. EMOTIONAL TRUTH — Is anyone actually feeling anything, and does it match the
   scene's stated mood and each character's situation? Look for emotion that is
   named instead of shown ("I'm frustrated") and for flat delivery of lines that
   should carry weight — owning a mistake, being overruled, hearing bad news,
   defending your team, being caught out. Emotion should surface through what
   people choose to say, how much they say, what they avoid, and where they push
   or go quiet.
   Keep it PROPORTIONATE. Most of these scenes are ordinary professional
   situations, and a routine status meeting should feel routine. Restraint,
   politeness masking irritation, or someone deciding not to say the thing is
   usually truer than open conflict. Do not manufacture drama that the scene
   does not call for — if the mood is genuinely low-key, say so and leave it be.
5. CHARACTER DISTINCTNESS — Does each person sound like a specific human with their
   own rhythm, vocabulary, and concerns, consistent with their description? Could you
   tell them apart with the names hidden?
6. HUMAN TEXTURE — Do people hesitate, restart, trail off, react to each other,
   go briefly sideways and get pulled back? Perfectly efficient dialogue is fake.
7. STAKES — Does anything actually matter to anyone here? Do people want things?

Respond with a JSON object:
{
  "thinking": "Your honest analysis of this take: what worked, what felt fake or flat, what is missing and why",
  "general": "One or two concise notes for the whole scene — the most important change for the next take",
  "scene_grounding": "A specific, concrete instruction for how this scene should establish itself early: what physical or situational detail should surface, and roughly when",
  "emotional_arc": "Where the emotional weight of this scene sits and how it should shift across the scene — or, if the scene is genuinely low-key, say plainly that it should stay understated and why",
  "per_character": {
    "CharacterName": "One direct, actionable note for this actor"
  }
}

Rules for your notes:
- Address every character who speaks, by the exact name used in the dialogue.
- Be specific and actionable. "Be more natural" is useless. "You arrive slightly
  late and still catching up — let that show in your first line" is a note.
- For emotion, direct the state and its cause, not the wording: "you already know
  this was your call and you are braced for it" rather than "sound guilty".
- Judge emotion against the mood the scene was written for, which is given to you
  in the scene brief and per-beat directions. If the take is flatter than the
  stated mood, say so; if it is louder, rein it in.
- Never write replacement dialogue. Direct the performance, do not script it.
- Do not ask for length changes or more turns. Same scene, performed better.
- Do not write bracketed tags like [sighs] or [angry] yourself — a later pass adds
  those. But DO name the emotional state plainly in your note, because that pass
  reads your notes to choose the tags. "Amara is braced and a little defensive
  here" gives it what it needs; "make it better" does not.
- Express presence and feeling through speech only. The actors can only write
  spoken words, so do not ask for physical business like closing a laptop —
  ask for what that state makes them SAY, or not say.
- Be direct. Actors need direction, not praise."""


class DirectorAgent:
    """Reviews a completed take and produces notes for the next one."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the director from the postprocessing config file.

        Args:
            config_path: Path to postprocessing.json. The director reuses that
                file's review model/provider settings so it runs on the same
                backend as the dialogue enhancer.
        """
        load_dotenv()
        config: Dict = {}
        if config_path is not None:
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:  # noqa: BLE001 - config is optional
                logger.warning("Director could not load config %s: %s", config_path, e)

        self.model = config.get("director_model", config.get("review_model", "gpt-4o"))
        self.provider = config.get(
            "director_provider", config.get("review_provider", "openai")
        )
        self.temperature = config.get("director_temperature", 0.7)

        if self.provider == "google":
            self.client = AsyncOpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url=GOOGLE_OPENAI_BASE_URL,
            )
        else:
            self.client = AsyncOpenAI()

    def _format_scene_context(
        self,
        variation: Dict,
        scenario_config: Optional[Dict],
    ) -> str:
        """Build the scene brief the director judges the take against.

        Includes the phase ``mood`` and ``note`` fields, which the per-turn
        generation prompt does not currently surface — the director is the one
        consumer that reads the full authored intent of the scene.
        """
        parts = []
        context = variation.get("context", "")
        if context:
            parts.append(f"Scene: {context}")

        setting = variation.get("setting", {})
        for key, label in (
            ("mood", "Mood"),
            ("relationship_dynamic", "Relationships"),
            ("background", "What led up to this"),
            ("conversation_style", "Intended style"),
        ):
            if setting.get(key):
                parts.append(f"{label}: {setting[key]}")

        if scenario_config and scenario_config.get("initial_prompt"):
            parts.append(f"Scene brief: {scenario_config['initial_prompt']}")

        phases = variation.get("scenario_template", [])
        if phases:
            phase_lines = []
            for p in phases:
                speakers = [
                    s["speaker"] if isinstance(s, dict) else s
                    for s in p.get("speakers", [])
                ]
                line = f"  - {p.get('phase', '?')}: {p.get('topic', '')}"
                if p.get("mood"):
                    line += f" [mood: {p['mood']}]"
                if p.get("note"):
                    line += f" [direction: {p['note']}]"
                line += f" (speakers: {', '.join(speakers)})"
                phase_lines.append(line)
            parts.append("Intended beats:\n" + "\n".join(phase_lines))

        return "\n".join(parts)

    def _format_characters(
        self,
        conversation: List[Dict],
        character_info: Optional[Dict],
    ) -> str:
        """Format descriptions for the characters who actually speak in the take."""
        if not character_info:
            return ""

        speakers = {m.get("speaker", "") for m in conversation}
        lines = []
        for name, backstory in character_info.items():
            if not any(name.lower() == s.lower() for s in speakers):
                continue
            text = " ".join(str(backstory).split())
            if len(text) > 700:
                text = text[:700] + "..."
            lines.append(f"  {name}: {text}")

        return "Characters in this scene:\n" + "\n".join(lines) if lines else ""

    async def review_take(
        self,
        conversation: List[Dict],
        take_number: int,
        variation: Dict,
        character_info: Optional[Dict] = None,
        scenario_config: Optional[Dict] = None,
    ) -> Dict:
        """Review one completed take and return notes for the next one.

        Args:
            conversation: The take's turns, each with ``speaker`` and ``content``.
            take_number: 1-based number of the take being reviewed.
            variation: The scenario variation (context, setting, phases).
            character_info: Mapping of character name to backstory text.
            scenario_config: Full scenario config, used for the scene brief.

        Returns:
            Dict with ``thinking``, ``general``, ``scene_grounding``, and
            ``per_character`` keys. Missing keys are filled with empty defaults so
            callers can rely on the shape.
        """
        dialogue = "\n".join(
            f"  {m.get('speaker', '?')}: {m.get('content', '')}" for m in conversation
        )

        prompt_parts = [
            f"TAKE {take_number}",
            self._format_scene_context(variation, scenario_config),
        ]
        chars = self._format_characters(conversation, character_info)
        if chars:
            prompt_parts.append(chars)
        prompt_parts.append(f"Dialogue as performed in this take:\n{dialogue}")
        prompt_parts.append("Write your director notes as JSON.")
        prompt = "\n\n".join(p for p in prompt_parts if p)

        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DIRECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if not self.model.startswith("gpt-5"):
            api_params["temperature"] = self.temperature

        try:
            response = await self.client.chat.completions.create(**api_params)
            raw = response.choices[0].message.content.strip()
            notes = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Director returned non-JSON notes: %s", e)
            notes = {}
        except Exception as e:  # noqa: BLE001 - a failed review must not kill the run
            logger.error("Director review failed: %s", e)
            notes = {}

        notes.setdefault("thinking", "")
        notes.setdefault("general", "")
        notes.setdefault("scene_grounding", "")
        notes.setdefault("emotional_arc", "")
        notes.setdefault("per_character", {})

        if notes.get("general"):
            logger.info("Director note (take %d): %s", take_number, notes["general"])
        if notes.get("scene_grounding"):
            logger.info("Director grounding: %s", notes["scene_grounding"])
        if notes.get("emotional_arc"):
            logger.info("Director emotional arc: %s", notes["emotional_arc"])
        for name, note in notes.get("per_character", {}).items():
            logger.info("  → %s: %s", name, note)

        return notes
