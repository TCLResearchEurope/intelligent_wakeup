"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Module to manage the conversation flow and phases.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ...utils import logger

# Sentinel speaker name marking an ambient/silence segment. Kept in sync with
# generate_speech.generate_audio.BACKGROUND_NOISE_SPEAKER, which consumes it.
BACKGROUND_NOISE_SPEAKER = "background_noise"


@dataclass
class ConversationPhase:
    """Represents a single phase in the conversation.

    ``speakers`` is always the normalized list of plain speaker keys, in
    turn order. A speaker slot may additionally be tagged as an interjection
    in the source config — e.g. ``{"speaker": "user5", "interjection": "agree"}``
    instead of a bare ``"user5"`` — meaning that speaker's turn is an overlap
    onto the turn immediately before it. That tag is recorded in
    ``interjections``, keyed by position in ``speakers``.
    """

    phase: str
    topic: str
    speakers: List[str]
    mood: Optional[str] = None
    note: Optional[str] = None
    interjections: Dict[int, str] = field(default_factory=dict)
    # A silence phase produces ambient sound instead of dialogue. Duration is
    # optional: generate_audio falls back to sound_effects_config, then to its
    # own default, so scenarios may omit it.
    silence: bool = False
    duration: Optional[float] = None

    def is_silence(self) -> bool:
        """Whether this phase is an ambient/silence phase rather than dialogue."""
        return bool(self.silence) or self.phase.lower() == "silence"

    def get_speakers_order(self, assistant_name: str = "Sigma") -> List[str]:
        """Get the ordered list of speakers for this phase."""
        return self.speakers.copy()

    def get_interjection(self, speaker_idx: int) -> Optional[str]:
        """Get the interjection type for a speaker slot, if any."""
        return self.interjections.get(speaker_idx)

    def wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word) + 1

        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def __str__(self) -> str:
        """Create a formatted string representation of the phase."""
        width = 60  # Fixed width for the box
        return f"""\n╔{'═' * (width - 2)}╗
║ {f"Phase: {self.phase}":<{width-4}}║
║ {f"Topic: {self.topic}":<{width-4}}║
║ {f"Speakers: {', '.join(self.speakers)}":<{width-4}}║
║ {f"Mood: {self.mood or 'Not specified'}":<{width-4}}║
║ {f"Note: {self.note or 'Not specified'}":<{width-4}}║
╚{'═' * (width - 2)}╝"""

    def to_summary(self) -> str:
        """Create a compact summary for scenario overview."""
        return f"{self.phase:<15} → {self.topic:<25} [{', '.join(self.speakers)}]"


class ConversationManager:
    """Manages the conversation flow and phases."""

    def __init__(
        self,
        scenario_config: Dict,
        generation_config: Dict,
        conversation_type: str = "multi_user",
    ):
        self.current_phase_idx = 0
        self.turns_in_phase = 0
        self.box_width = 60
        self.conversation_type = conversation_type
        self.assistant_name = generation_config.get("assistant", {}).get(
            "name", "Sigma"
        )
        self.phases = self._load_phases(scenario_config)
        if self.phases:
            self._print_phase_box(self.phases[0])

    def get_current_phase(self) -> ConversationPhase:
        """Get the current conversation phase."""
        if self.current_phase_idx >= len(self.phases):
            return self.phases[-1]
        return self.phases[self.current_phase_idx]

    def _current_speaker_idx(self) -> Optional[int]:
        """Index into the current phase's speaker list for the upcoming turn."""
        speakers = self.get_current_phase().get_speakers_order(self.assistant_name)
        if not speakers:
            return None
        return self.turns_in_phase % len(speakers)

    def get_next_speaker(self) -> str:
        """
        Get the next speaker in the current phase.

        Returns either a speaker name or the BACKGROUND_NOISE_SPEAKER sentinel for
        silence phases, never None.
        """
        current_phase = self.get_current_phase()
        if current_phase.is_silence():
            return BACKGROUND_NOISE_SPEAKER

        speaker_idx = self._current_speaker_idx()
        if speaker_idx is None:
            return BACKGROUND_NOISE_SPEAKER

        return current_phase.get_speakers_order(self.assistant_name)[speaker_idx]

    def get_next_speaker_interjection(self) -> Optional[str]:
        """Get the interjection type for the upcoming turn, if the config tagged it.

        Returns "agree", "disagree", "add", or None for an ordinary turn.
        """
        speaker_idx = self._current_speaker_idx()
        if speaker_idx is None:
            return None
        return self.get_current_phase().get_interjection(speaker_idx)

    def _print_box_line(self, content: str = "", header: bool = False) -> None:
        """Print a line in the box with proper alignment."""
        if header:
            # Center content for headers
            logger.info("║%s║", content.center(self.box_width - 2))
        else:
            # Left-align regular content with proper padding
            padding = " " * (self.box_width - 2 - len(content))
            logger.info("║ %s%s║", content, padding)

    def _print_phase_box(self, phase: ConversationPhase) -> None:
        """Print a phase transition box."""
        logger.info("╔%s╗", "═" * (self.box_width - 1))
        self._print_box_line(f"Phase: {phase.phase}")
        self._print_box_line(f"Topic: {phase.topic}")
        self._print_wrapped_field("Speakers", ", ".join(phase.speakers))
        if phase.mood:
            self._print_wrapped_field("Mood", phase.mood)
        if phase.note:
            self._print_wrapped_field("Note", phase.note)
        logger.info("╚%s╝", "═" * (self.box_width - 1))

    def _print_wrapped_field(self, field_name: str, content: str) -> None:
        """Print a field with content that might need wrapping."""
        if not content:
            self._print_box_line(f"{field_name}: Not specified")
            return

        # Calculate available width for content
        available_width = self.box_width - len(field_name) - 4  # -4 for "║ " and ": "

        # Split content into words
        words = content.split()
        current_line = []
        current_length = 0

        # Handle first line with field name
        while words and (
            current_length + len(words[0]) + (1 if current_line else 0)
            <= available_width
        ):
            word = words.pop(0)
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)

        first_line = f"{field_name}: {' '.join(current_line)}"
        self._print_box_line(first_line)

        # Handle continuation lines if any words remain
        while words:
            current_line = []
            current_length = 0
            while words and (current_length + len(words[0]) + 1 <= available_width):
                word = words.pop(0)
                current_line.append(word)
                current_length += len(word) + 1

            continuation_line = " " * (len(field_name) + 2) + " ".join(current_line)
            self._print_box_line(continuation_line)

    def _validate_phase_speakers(self, phase: ConversationPhase) -> None:
        """Validate that phases include assistant where needed."""
        assistant_required_phases = {"assistance_request", "discussion", "planning"}

        if (
            phase.phase in assistant_required_phases
            and "assistant" not in phase.speakers
        ):
            logger.warning(
                "Phase '%s' should include assistant in speakers. Adding assistant.",
                phase.phase,
            )
            phase.speakers.append("assistant")

    def _normalize_speakers(self, raw_speakers: List) -> tuple[List[str], Dict[int, str]]:
        """Split raw speaker slots into a plain speaker-name list and an
        index→interjection-type map, filtering "user2" out for single_user mode.

        Each raw entry is either a bare speaker key (e.g. ``"user5"``) or a
        dict tagging that slot as an overlap onto the turn before it, e.g.
        ``{"speaker": "user5", "interjection": "agree"}``.
        """
        names: List[str] = []
        interjections: Dict[int, str] = {}
        for entry in raw_speakers:
            if isinstance(entry, dict):
                name = entry["speaker"]
                interjection = entry.get("interjection")
            else:
                name = entry
                interjection = None

            if self.conversation_type == "single_user" and name == "user2":
                continue

            if interjection:
                interjections[len(names)] = interjection
            names.append(name)

        return names, interjections

    def _load_phases(self, scenario_config: Dict) -> List[ConversationPhase]:
        """Load and validate conversation phases from config."""
        phases = []
        template = scenario_config.get("scenario_template", [])

        # Print the conversation flow header
        logger.info("╔%s╗", "═" * (self.box_width - 2))
        self._print_box_line("CONVERSATION FLOW", header=True)
        logger.info("║%s║", "═" * (self.box_width - 2))

        # Print each phase in the flow
        for phase_data in template:
            speakers, interjections = self._normalize_speakers(phase_data["speakers"])

            phase = ConversationPhase(
                phase=phase_data["phase"],
                topic=phase_data["topic"],
                speakers=speakers,
                mood=phase_data.get("mood"),
                note=phase_data.get("note"),
                interjections=interjections,
                silence=phase_data.get("silence", False),
                duration=phase_data.get("duration"),
            )

            # Validate assistant participation
            self._validate_phase_speakers(phase)

            phases.append(phase)

            # Add phase to summary with proper padding
            summary = phase.to_summary()
            if len(summary) > self.box_width - 2:
                # Truncate if too long
                summary = summary[: self.box_width - 5] + "..."
            self._print_box_line(summary)

        # Close the summary box
        logger.info("╚%s╝", "═" * (self.box_width - 2))

        return phases

    def advance_turn(self) -> None:
        """Advance to the next turn, potentially moving to next phase."""
        self.turns_in_phase += 1
        current_phase = self.get_current_phase()
        speakers = current_phase.get_speakers_order(self.assistant_name)

        # If we've gone through all speakers in phase, move to next phase
        if self.turns_in_phase >= len(speakers):
            self.current_phase_idx += 1
            self.turns_in_phase = 0

            if self.current_phase_idx < len(self.phases):
                next_phase = self.phases[self.current_phase_idx]
                # Print phase transition box without extra newline
                self._print_phase_box(next_phase)
