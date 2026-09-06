"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Custom conversation generator implementation with support for variant-specific behavior.
"""

from datetime import datetime
from typing import Dict, List, Optional
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .base import BaseGenerator, GeneratorConfig
from .conversation_manager import (
    ConversationManager,
    ConversationPhase,
    BACKGROUND_NOISE_SPEAKER,
)
from ..agents.custom_agent import CustomAgent
from ..agents.base import AgentConfig, Message, BaseAgent
from ...utils import logger


@dataclass
class TimingConfig:
    """Configuration for conversation timing."""

    chars_per_second: float = 11.0
    min_response_delay: float = 2.0
    max_response_delay: float = 4.0
    thinking_time_variance: float = 1.0


class MessageProcessor:
    """Handles message processing including timing and speaker names."""

    def __init__(self, generation_config: Dict):
        """Initialize with generation configuration."""
        timing_config = generation_config.get("timing", {})
        self.timing = TimingConfig(
            chars_per_second=timing_config.get("chars_per_second", 11.0),
            min_response_delay=timing_config.get("min_response_delay", 2.0),
            max_response_delay=timing_config.get("max_response_delay", 4.0),
            thinking_time_variance=timing_config.get("thinking_time_variance", 1.0),
        )
        self.conversation_start_time = 0.0
        self.last_message_end_time = 0.0
        self.assistant_name = generation_config.get("assistant", {}).get(
            "name", "Sigma"
        )

    def calculate_message_time(self, content: str) -> float:
        """Calculate time for this message based on previous message and content length."""
        # Time to speak the previous message
        speaking_time = len(content) / self.timing.chars_per_second

        # Random thinking/response delay
        response_delay = random.uniform(
            self.timing.min_response_delay, self.timing.max_response_delay
        )

        # Add some random variance to make it more natural
        variance = random.uniform(
            -self.timing.thinking_time_variance, self.timing.thinking_time_variance
        )

        # Calculate total time since conversation start. The variance can be negative,
        # so clamp to zero: a phase opening the scenario must not start before it.
        new_time = max(0.0, self.last_message_end_time + response_delay + variance)

        # Update last message end time for next calculation
        self.last_message_end_time = new_time + speaking_time

        return round(new_time, 1)  # Round to 1 decimal place

    def process_message(
        self,
        message: Dict,
        current_variation: Dict,
        conversation_type: str = "multi_user",
    ) -> Dict:
        """
        Process message to update speaker name and timing.

        Args:
            message: Original message dictionary
            current_variation: Current conversation variation with character info

        Returns:
            Dict: Updated message with correct speaker name and timing
        """
        processed_msg = message.copy()

        logger.debug(
            "Processing message: %s with speaker: %s",
            message,
            message.get("speaker", ""),
        )

        # 1. Get ID directly from source (making sure it's in lowercase).
        # The keys can be present but None, so a get() default is not enough.
        agent_id = (processed_msg.get("agent_id") or "").lower()
        speaker_name = (processed_msg.get("speaker") or "").lower()

        # 2. Handle Assistant
        if agent_id == "assistant" or speaker_name in ["assistant", "sigma"]:
            processed_msg["speaker"] = self.assistant_name
            logger.debug("Mapped assistant to: %s", self.assistant_name)
        elif agent_id:
            # Get default character configuration
            char_config = current_variation.get("default_characters", {}).get(
                conversation_type, {}
            )

            logger.debug("Looking up character for ID %s", agent_id)

            # Direct lookup by specific ID, instead of parsing strings!
            if agent_id in char_config:
                char_info = char_config[agent_id]
                if isinstance(char_info, dict) and "name" in char_info:
                    processed_msg["speaker"] = char_info["name"]
                    logger.debug(
                        "Mapped agent_id %s to name %s",
                        agent_id,
                        processed_msg["speaker"],
                    )

            elif agent_id == "user2":
                variant_type = current_variation.get("variant_type", "")
                if (
                    variant_type in ["couple", "two_noncouple", "two_over_phone"]
                    and "user2" in char_config
                ):
                    processed_msg["speaker"] = char_config["user2"].get("name", "User2")
                else:
                    processed_msg["speaker"] = char_config.get("user2", {}).get(
                        "name", "User2"
                    )

                logger.debug(
                    "Applied fallback for user2 to %s", processed_msg["speaker"]
                )

        # 4. Calculate time (timing)
        processed_msg["time"] = self.calculate_message_time(processed_msg["content"])

        # 5. Cleanup
        processed_msg.pop("timestamp", None)
        processed_msg.pop("agent_id", None)
        if not processed_msg.get("interjection"):
            processed_msg.pop("interjection", None)

        logger.debug("Final processed message: %s", processed_msg)
        return processed_msg


class CustomGenerator(BaseGenerator):
    """Custom generator implementation for direct API usage."""

    DEFAULT_ASSISTANT_NAME = "Sigma"

    def __init__(
        self, config: GeneratorConfig, timeout: int = 60, max_retries: int = 3
    ) -> None:
        """Initialize generator with configuration.

        Args:
            config: Generator configuration
            timeout: Timeout in seconds for API calls (default: 60)
            max_retries: Maximum number of retries for failed calls (default: 3)
        """
        BaseGenerator.__init__(self, config)
        # Initialize basic attributes without setting up agents
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.timeout = timeout
        self.max_retries = max_retries
        # Director notes from the previous take, applied to the next performance.
        # Set externally between takes; None means an unguided first take.
        self.director_notes: Optional[Dict] = None
        if (
            not hasattr(self.config, "agent_configs")
            or self.config.agent_configs is None
        ):
            self.config.agent_configs = {}
        self._validate_config()
        self.generation_config = self._load_generation_config()
        self.assistant_name = self.generation_config.get("assistant", {}).get(
            "name", self.DEFAULT_ASSISTANT_NAME
        )
        self.message_processor = MessageProcessor(self.generation_config)
        # Wake-word policy, from generation.json -> assistant.wake_word_on_first_contact.
        # Defaults to on: the assistant is dormant until named, then stays attentive.
        self.wake_word_on_first_contact = self.generation_config.get(
            "assistant", {}
        ).get("wake_word_on_first_contact", True)
        self.current_variation: Dict = {}
        self._setup_agents()

    def _merge_configs(self, base: Dict, variation: Optional[Dict] = None) -> Dict:
        """
        Merge base config with variation-specific config if available.

        Args:
            base: Base configuration dictionary
            variation: Optional variation-specific configuration

        Returns:
            Dict: Merged configuration
        """
        if not variation:
            return base
        merged = base.copy()
        merged["role"] = variation.get("role", base["role"])
        if "additional_context" in variation:
            merged["role"] = f"{merged['role']} {variation['additional_context']}"
        return merged

    def _load_generation_config(self) -> Dict:
        """Load generation configuration from config file."""
        config_path = Path(self.config.config_path) / "generation.json"
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Generation config not found: %s", config_path)
            return {}

    def _setup_agents(self) -> None:
        """Initialize custom agents based on conversation type."""
        if not self.config.agent_configs:
            logger.warning("No agent configs provided")
            return

        base_configs = self.config.agent_configs
        is_single_user = self.config.conversation_type == "single_user"

        # Get default characters from scenario config
        default_characters = self.current_variation.get("default_characters", {})
        char_type = "single_user" if is_single_user else "multi_user"
        char_config = default_characters.get(char_type, {})

        # Get variant-specific behavior
        variant_type = self.current_variation.get("variant_type", "")
        variant_name = self.current_variation.get("variant_name", "")

        logger.debug(
            "Setting up agents for variant type: %s, name: %s",
            variant_type,
            variant_name,
        )

        for role, base_config in base_configs.items():
            # Skip User2 for single_user conversations or when variant_type is "single"
            if (is_single_user or variant_type == "single") and role == "user2":
                continue

            role_key = role
            config = base_config.copy()
            config["id"] = role

            if role == "assistant":
                if "assistant" in self.generation_config:
                    assistant_config = self.generation_config["assistant"]
                    role_key = assistant_config.get("name_mapping", {}).get(role, role)

                config["name"] = self.assistant_name
                config["character"] = ""

            else:
                character = char_config.get(
                    role, {"name": f"User{role[-1]}", "backstory": ""}
                )
                config["name"] = character.get("name", f"User{role[-1]}")

                # Load character backstory from file if specified
                if "backstory_file" in character:
                    backstory_path = Path(
                        f"{self.config.config_path}/characters/{character['backstory_file']}"
                    )
                    if backstory_path.exists():
                        with open(backstory_path, encoding="utf-8") as f:
                            config["character"] = f.read()
                    else:
                        logger.warning(
                            "Character backstory file not found: %s", backstory_path
                        )
                        config["character"] = character.get("backstory", "")
                else:
                    config["character"] = character.get("backstory", "")

            # Apply variant-specific role modifications
            if variant_type in ["couple", "two_over_phone"]:
                # Enhance role descriptions based on relationship type
                if role.startswith("user"):
                    if (
                        variant_type == "couple"
                        and "couple_context" not in config["role"]
                    ):
                        config[
                            "role"
                        ] += " In a committed relationship with the other participant."
                    elif (
                        variant_type == "two_over_phone"
                        and "phone" not in config["role"]
                    ):
                        config[
                            "role"
                        ] += " Currently having this conversation over the phone."

            variation_config = None
            if self.current_variation and "agent_configs" in self.current_variation:
                variation_config = self.current_variation["agent_configs"].get(role)

            merged_config = self._merge_configs(config, variation_config)
            self.agents[role_key] = CustomAgent(
                AgentConfig(**merged_config),
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

        logger.debug("Initialized agents with keys: %s", list(self.agents.keys()))

    async def _generate_turn(
        self,
        agent: CustomAgent,
        conversation_history: List[Message],
        _prompt,
        variation: Dict,
        *,
        current_phase: ConversationPhase,
        interjection_type: Optional[str] = None,
    ) -> Message:
        """Generate single conversation turn."""
        # Check if this is first time addressing the assistant
        first_interaction = True
        for msg in conversation_history:
            if msg.speaker == self.message_processor.assistant_name:
                first_interaction = False
                break

        assistant_in_phase = self._assistant_in_phase(current_phase)

        user_guidance = self._get_user_guidance(
            agent, first_interaction, assistant_in_phase
        )
        setting_context = self._get_setting_context(agent, variation, current_phase)

        # Add explicit role reminder
        role_reminder = ""

        if "assistant" in agent.config.id.lower():
            role_reminder = (
                f"\nYou are {self.message_processor.assistant_name}. Always respond as "
                f"{self.message_processor.assistant_name} and never as 'Assistant'."
            )
        elif self.is_user(agent):
            if not assistant_in_phase:
                role_reminder = (
                    f"\nYou are a user in the conversation. "
                    f"CRITICAL INSTRUCTION: In this current phase of the conversation, "
                    f"the assistant '{self.message_processor.assistant_name}' is NOT active. "
                    f"Do NOT address them and do NOT ask them any questions. Focus entirely on "
                    f"your own stream of consciousness or talking to other human users."
                )
            else:
                role_reminder = (
                    f"\nYou are a user in the conversation. You can now interact "
                    f"with the assistant. "
                    f"When addressing the assistant, call them "
                    f"'{self.message_processor.assistant_name}'."
                )

            role_reminder += (
                f"\nIMPORTANT: Generate the response ONLY for your character "
                f"({agent.config.name}). "
                f"Under no circumstances should you generate dialogue for any other character."
            )

        # Add variant-specific instructions
        variant_instructions = self._get_variant_instructions(variation, agent)

        system_prompt = self._build_system_prompt(agent, variation)
        user_message = self._build_user_message(
            agent,
            conversation_history,
            current_phase,
            user_guidance,
            setting_context,
            variant_instructions,
            role_reminder,
            interjection_type=interjection_type,
        )

        response = await agent.generate_response(
            user_message, system_prompt=system_prompt
        )

        processed_response = self._process_response(
            response, agent, first_interaction, variation, current_phase=current_phase
        )
        if interjection_type:
            processed_response.interjection = interjection_type

        # Validate response speaker
        if "assistant" in agent.config.name.lower():
            processed_response.speaker = self.message_processor.assistant_name
            logger.debug(
                "Set assistant response speaker to: %s",
                self.message_processor.assistant_name,
            )

        return processed_response

    def _get_variant_instructions(self, variation: Dict, agent: CustomAgent) -> str:
        """Get variant-specific instructions based on the variant type."""
        variant_type = variation.get("variant_type", "")
        is_user = self.is_user(agent)

        if not variant_type or not is_user:
            return ""

        instructions = {
            "single": "You are having a one-on-one conversation with the assistant.",
            "couple": (
                "You are in a committed relationship with the other user in this conversation."
            ),
            "two_noncouple": (
                "You are interacting with another user, but you are not in a romantic relationship."
            ),
            "many": "You are in a group conversation with multiple other users.",
            "two_over_phone": (
                "You are having this conversation over the phone with the other participant."
            ),
        }

        return instructions.get(variant_type, "")

    def _get_user_guidance(
        self,
        agent: CustomAgent,
        first_interaction: bool,
        assistant_in_phase: bool,
    ) -> str:
        """Get appropriate user guidance from config.

        The wake-word instruction is only issued on a turn that is actually
        directed at the assistant — that is, when the assistant is one of the
        speakers of the current phase and has not spoken yet.  Sending it during
        human-only phases would both contradict the role reminder for those
        phases ("do NOT address them") and blunt the instruction by repeating it
        where it cannot be obeyed.
        """
        if not self.is_user(agent):
            return ""

        guidance_config = self.generation_config.get("user_guidance", {})
        if first_interaction and assistant_in_phase:
            return guidance_config.get("first_sigma_interaction", {}).get("user", "")
        return guidance_config.get("subsequent_interaction", {}).get("user", "")

    def _get_setting_context(
        self, agent: CustomAgent, variation: Dict, current_phase: ConversationPhase
    ) -> str:
        """Format setting context from variation."""
        if not self.is_user(agent):
            return ""

        setting = variation.get("setting", {})
        variant_type = variation.get("variant_type", "")

        setting_lines = [
            f"Setting: {setting.get('mood', '')}",
            f"Current dynamic: {setting.get('relationship_dynamic', '')}",
            f"Conversation style: {setting.get('conversation_style', '')}",
            f"Variant type: {variant_type}",
            f"Phase mood: {getattr(current_phase, 'mood', '')}",
        ]

        return "\n".join(setting_lines)

    def _build_system_prompt(self, agent: "CustomAgent", variation: Dict) -> str:
        """Build the static system prompt sent once per conversation per agent.

        Contains everything that does not change turn-to-turn: role, character
        backstory, scenario description, and output rules.  OpenAI caches this
        prefix automatically so it is only charged on the first call.
        """
        output_rule = (
            "Write only the spoken words of your character. Do NOT include stage "
            "directions, action descriptions, internal thoughts, narration, or "
            "attribution phrases such as 'I say', 'she laughs', 'I chuckle', or "
            "'*smiles*'. No quotes around the output."
        )
        parts = [
            f"Role: {agent.config.role}",
            f"Name: {agent.config.name}",
        ]
        character = getattr(agent.config, "character", "")
        if character:
            parts.append(character)

        if self.config.initial_prompt:
            parts.append(f"Scenario: {self.config.initial_prompt}")

        context = variation.get("context", "")
        if context:
            parts.append(f"Context: {context}")

        max_words = variation.get("max_words_per_turn", 20)
        parts.append(f"Maximum words per response: {max_words}")

        if "variant_type" in variation:
            parts.append(f"Conversation structure: {variation['variant_type']}")

        if variation.get("background_type"):
            parts.append(f"Background environment: {variation['background_type']}")

        if self.config.conversation_type == "multi_user":
            parts.append(
                "This is a conversation between multiple users and an assistant."
            )

        if "min_turns" in variation and "max_turns" in variation:
            parts.append(
                f"Generate between {variation['min_turns']} and "
                f"{variation['max_turns']} conversation turns."
            )

        director_block = self._format_director_notes(agent)
        if director_block:
            parts.append(director_block)

        parts.append(f"OUTPUT RULE: {output_rule}")
        return "\n\n".join(parts)

    def _format_director_notes(self, agent: "CustomAgent") -> str:
        """Format the previous take's director notes for this agent's system prompt.

        Returns an empty string on the first (unguided) take, or when the
        director produced nothing usable.

        Args:
            agent: The agent whose personal note should be included.

        Returns:
            A formatted notes block, or "" if there is nothing to say.
        """
        notes = self.director_notes
        if not notes:
            return ""

        lines = [
            "DIRECTOR NOTES — you have performed this scene before. "
            "Perform it again, applying these notes:"
        ]
        if notes.get("general"):
            lines.append(f"- Overall: {notes['general']}")
        if notes.get("scene_grounding"):
            lines.append(f"- Establishing the scene: {notes['scene_grounding']}")
        if notes.get("emotional_arc"):
            lines.append(f"- Emotional arc of the scene: {notes['emotional_arc']}")

        per_character = notes.get("per_character", {}) or {}
        my_note = per_character.get(agent.config.name)
        if my_note is None:
            # Fall back to a case-insensitive match — the director echoes the
            # name as it appeared in the dialogue, which may differ in case.
            for name, note in per_character.items():
                if str(name).lower() == agent.config.name.lower():
                    my_note = note
                    break
        if my_note:
            lines.append(f"- For you ({agent.config.name}): {my_note}")

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _get_exchanges_since_last_turn(
        self, agent: "CustomAgent", conversation: List[Message]
    ) -> List[Message]:
        """Return messages spoken since this agent last spoke.

        Replaces the old 2-turn sliding window: the agent sees everything said
        by others since its previous response, so the growing conversation
        context arrives via the messages array rather than repeated in the
        prompt text.

        Args:
            agent: The agent whose last utterance is used as the start marker.
            conversation: Full conversation history accumulated so far.

        Returns:
            List of Message objects spoken after this agent's last turn.
            If the agent has not spoken yet, returns up to the last 2 messages
            for bootstrapping context.
        """
        last_idx = -1
        for i, msg in enumerate(conversation):
            if msg.agent_id == agent.config.id:
                last_idx = i

        if last_idx == -1:
            # Agent has not spoken yet — provide up to 2 messages for context
            return conversation[-2:] if len(conversation) >= 2 else list(conversation)

        return conversation[last_idx + 1 :]

    # Instructions for a config-tagged interjection turn: a short overlap onto
    # the immediately preceding turn, motivated by agreement, disagreement, or
    # an unrelated-but-relevant idea the speaker feels they must add now.
    _INTERJECTION_INSTRUCTIONS = {
        "agree": (
            "INTERJECTION: You are cutting in on what the previous speaker just "
            "said because you agree with it. Keep this turn very short and "
            "natural — a quick reaction, not a full statement. "
            "E.g. 'Right, exactly!', 'Yeah, I'd agree on that.', 'Same here.'"
        ),
        "disagree": (
            "INTERJECTION: You are cutting in on what the previous speaker just "
            "said because you disagree with it. Keep this turn short and "
            "direct — a quick pushback, not a full counter-argument. "
            "E.g. 'Wait, no—', 'I don't think that's right.', 'Hold on, actually—'"
        ),
        "add": (
            "INTERJECTION: You are cutting in on what the previous speaker just "
            "said because you have a related point you feel you must add right "
            "now. Keep this turn short. "
            "E.g. 'Actually, if I can add—', 'Oh, and one more thing—', 'Right, and also—'"
        ),
    }

    def _build_user_message(
        self,
        agent: "CustomAgent",
        conversation: List[Message],
        current_phase: "ConversationPhase",
        user_guidance: str,
        setting_context: str,
        variant_instructions: str,
        role_reminder: str,
        interjection_type: Optional[str] = None,
    ) -> str:
        """Build the small, dynamic per-turn user message.

        Only contains what changes between turns: recent exchanges from other
        speakers, current phase/topic, and per-turn guidance.  The static
        context (role, backstory, scenario) lives in the system message and is
        not repeated here.

        Args:
            agent: The agent about to generate a response.
            conversation: Full conversation history accumulated so far.
            current_phase: Active conversation phase with phase name and topic.
            user_guidance: Guidance string for how the user should behave this
                turn (e.g. first-interaction vs. subsequent-interaction hints).
            setting_context: Formatted mood/relationship/style context for
                user-role agents; empty string for the assistant.
            variant_instructions: Variant-specific behavioural instructions
                (e.g. couple, phone, group).
            role_reminder: Per-turn reminder about which characters this agent
                may or may not address, based on the active phase speakers.
            interjection_type: If this turn's speaker slot was config-tagged
                as an overlap onto the previous turn — "agree", "disagree", or
                "add" — the matching instruction is appended. None otherwise.

        Returns:
            A formatted string ready to send as the user-role message in the
            OpenAI chat completions API.
        """
        parts = []

        recent = self._get_exchanges_since_last_turn(agent, conversation)
        if recent:
            parts.append(
                "Recent exchanges:\n"
                + "\n".join(f"{m.speaker}: {m.content}" for m in recent)
            )
        elif not conversation:
            parts.append("This is the start of the conversation.")

        if setting_context:
            parts.append(setting_context)

        parts.append(
            f"Phase: {current_phase.phase}\n"
            f"Topic: {current_phase.topic}\n"
            "Keep the conversation natural and focused on topic."
        )

        if user_guidance:
            parts.append(user_guidance)

        if variant_instructions:
            parts.append(variant_instructions)

        if role_reminder:
            parts.append(role_reminder)

        if interjection_type:
            instruction = self._INTERJECTION_INSTRUCTIONS.get(interjection_type)
            if instruction:
                parts.append(instruction)

        return "\n\n".join(filter(None, parts))

    async def generate_conversation(self, variation: Dict) -> Dict:
        """Generate complete conversation using custom implementation.

        Args:
            variation: Dictionary containing conversation variation parameters

        Returns:
            Dict: Generated conversation data

        Raises:
            Exception: If generation fails
        """
        try:
            # Get default_characters from config if available
            if self.config.default_characters is not None:
                variation["default_characters"] = self.config.default_characters

            # Debug log the variation config
            logger.debug("Variation config before processing: %s", variation)

            # Store current variation before setup
            self.current_variation = variation

            # Reset and reinitialize agents with current variation
            self.agents = {}
            self._setup_agents()

            # Reset message processor timing for new conversation
            self.message_processor.conversation_start_time = 0.0
            self.message_processor.last_message_end_time = 0.0

            # Generate raw conversation
            conversation = await self._generate_conversation_async(variation)

            # Process messages with correct names and timing
            processed_messages = []
            for msg in conversation:
                msg_dict = msg.__dict__
                processed_msg = self.message_processor.process_message(
                    msg_dict, variation, self.config.conversation_type
                )
                processed_messages.append(processed_msg)

            # Add variant metadata to output
            result = {
                "scenario_type": self.config.name,
                "conversation_type": self.config.conversation_type,
                "context": variation["context"],
                "conversation": processed_messages,
            }

            # Add variant metadata if available
            if "variant_type" in variation:
                result["variant_type"] = variation["variant_type"]
            if "variant_name" in variation:
                result["variant_name"] = variation["variant_name"]

            return result

        except Exception as e:
            logger.error("Error in custom generation: %s", str(e), exc_info=True)
            raise

    async def _generate_conversation_async(
        self, variation: Dict, _: Optional[str] = None
    ) -> List[Message]:
        """Generate conversation turns asynchronously."""
        conversation = []
        max_turns = variation.get("max_turns", 12)

        # Initialize conversation manager
        conversation_manager = ConversationManager(
            variation, self.generation_config, self.config.conversation_type
        )

        logger.info("=== Starting Conversation Generation ===")

        current_turn = 0
        while current_turn < max_turns:
            # Get current phase and next speaker
            current_phase = conversation_manager.get_current_phase()
            next_speaker = conversation_manager.get_next_speaker()
            interjection_type = conversation_manager.get_next_speaker_interjection()

            if not next_speaker:
                break

            # Log turn information
            logger.info("\nTurn %d/%d", current_turn + 1, max_turns)
            logger.info("Phase: %s", current_phase.phase)
            logger.info("Speaker: %s", next_speaker)
            if interjection_type:
                logger.info("Interjection: %s", interjection_type)

            # A silence phase emits an ambient segment: there is no agent to prompt,
            # the phase topic becomes the sound description for the TTS stage
            if current_phase.is_silence():
                # An empty topic means a plain pause; a topic describes ambience to
                # synthesize. The audio stage distinguishes the two on this content.
                description = (current_phase.topic or "").strip()
                ambient = Message(
                    speaker=BACKGROUND_NOISE_SPEAKER,
                    content=description,
                    timestamp=f"{self.message_processor.calculate_message_time(description)}s",
                    message_type="dialogue",
                    duration=current_phase.duration,
                )
                if current_phase.duration is not None:
                    # Reserve the ambience on the timeline so the next turn starts after it
                    self.message_processor.last_message_end_time += float(
                        current_phase.duration
                    )
                logger.info(
                    "Silence phase: %s (duration=%s)",
                    f"ambient '{description[:60]}'" if description else "plain pause",
                    current_phase.duration,
                )
                conversation.append(ambient)
                current_turn += 1
                conversation_manager.advance_turn()
                continue

            # Map speaker to agent
            agent_key = next_speaker
            if next_speaker == "assistant":
                agent_key = "assistant"

            if agent_key not in self.agents:
                logger.error("Agent not found for speaker: %s", next_speaker)
                current_turn += 1
                continue

            agent = self.agents[agent_key]

            try:
                # Generate and process response
                response = await self._generate_turn(
                    agent,
                    conversation,
                    None,
                    variation,
                    current_phase=current_phase,
                    interjection_type=interjection_type,
                )

                # Log generated response
                logger.info(
                    "Generated response from %s: %s", response.speaker, response.content
                )

                conversation.append(response)
                current_turn += 1

                # Advance to next turn/phase
                conversation_manager.advance_turn()

            except Exception as e:
                logger.error("Error generating turn for %s: %s", next_speaker, str(e))
                raise

        logger.info("=== Conversation Generation Complete ===\n")
        return conversation

    def _process_response(
        self,
        response: Message,
        agent: CustomAgent,
        first_interaction: bool,
        _variation: Dict,
        *,
        current_phase: ConversationPhase,
    ) -> Message:
        """Process and clean up generated response.

        Cleans speaker prefixes and nested exchanges out of the model output, and
        applies the wake word on the first turn addressed to the assistant.
        """
        content = self._strip_meta_wrapper(response.content.strip())
        speaker = agent.config.name

        # Remove any speaker prefixes
        prefixes = [
            agent.config.name,
            "User",
            "Assistant",
            "User1",
            "User2",
            self.assistant_name,
            "Usert",  # Add this to catch malformed cases
        ]
        is_user_turn = self.is_user(agent)
        for name in prefixes:
            if content.startswith(f"{name}:"):
                content = content[len(name) + 1 :].strip()
            if content.startswith(f"{name} :"):
                content = content[len(name) + 2 :].strip()
            # "<name>, ..." is an attribution artefact for every name except the
            # assistant's on a user turn, where it is a vocative — the wake word —
            # and must survive.
            if content.startswith(f"{name},"):
                if is_user_turn and name == self.assistant_name:
                    continue
                content = content[len(name) + 1 :].strip()

        # Ensure assistant name is set correctly
        if agent.config.name.lower() in ["assistant", "sigma", "user"]:
            speaker = self.assistant_name

        # Remove any nested conversation exchanges
        if "\n" in content:
            content = self.remove_nested_exchanges(content, prefixes)

        # Wake word on first contact only.
        #
        # The assistant is treated as always listening but dormant: the first time a
        # user speaks to it in a conversation, the turn must carry the wake word, so
        # the utterance is self-evidently addressed to the assistant.  Every later
        # turn is left alone — once woken, the assistant is expected to infer from
        # context that it is still being spoken to.
        #
        # Conditions are deliberately derived from the conversation itself rather
        # than from phase naming conventions or participant counts:
        #   1. the speaker is a user (the assistant never addresses itself)
        #   2. the assistant has not yet spoken in this conversation
        #   3. the assistant is a participant in the current phase, i.e. this turn
        #      really is directed at it
        # This works for single-user scenes as well as multi-party ones, and does
        # not depend on a phase being named `assistance_request`/`seeking_help`.
        if (
            self.wake_word_on_first_contact
            and self.is_user(agent)
            and first_interaction
            and self._assistant_in_phase(current_phase)
            and not self._has_wake_word(content)
        ):
            content = self._prepend_wake_word(content)
            logger.debug("Applied wake word on first contact: %s", content)

        return Message(
            speaker=speaker,
            content=content,
            timestamp=datetime.now().isoformat(),
            agent_id=agent.config.id,
        )

    # Labels a model sometimes emits around its own output instead of just
    # speaking, e.g.  Original text: "Sigma, what time is it?"
    _META_LABEL = re.compile(
        r"""^\s*(?:original\s+text|revised\s+text|response|reply|output|
             answer|utterance|line|dialogue|text)\s*[:\-–]\s*""",
        re.IGNORECASE | re.VERBOSE,
    )

    @classmethod
    def _strip_meta_wrapper(cls, content: str) -> str:
        """Remove a meta label and any quotes the model wrapped its turn in.

        Models occasionally narrate the act of answering rather than answering:
        ``Original text: "Sigma, what time is it?"``.  Left alone, the label and
        quotes are spoken aloud by the TTS stage and the wake word is no longer
        at the front of the utterance.  Only a wrapper enclosing the *whole*
        turn is removed, so ordinary quoted speech inside a line is untouched.
        """
        cleaned = content.strip()
        for _ in range(3):  # a label may itself be quoted
            before = cleaned
            cleaned = cls._META_LABEL.sub("", cleaned, count=1).strip()
            if len(cleaned) >= 2 and cleaned[0] in cls._WRAPPERS:
                closing = {"\u201c": "\u201d", "\u2018": "\u2019", "\u00ab": "\u00bb"}.get(
                    cleaned[0], cleaned[0]
                )
                if cleaned.endswith(closing) and closing not in cleaned[1:-1]:
                    cleaned = cleaned[1:-1].strip()
            if cleaned == before:
                break
        return cleaned or content.strip()

    @staticmethod
    def _assistant_in_phase(current_phase: ConversationPhase) -> bool:
        """Whether the assistant is one of the speakers of this phase."""
        speakers = getattr(current_phase, "speakers", None) or []
        return any("assistant" in str(s).lower() for s in speakers)

    # Words that may precede the name and still leave it "in the opening":
    # interjections, fillers and connectives a speaker might front an address
    # with.  Anything else pushes the name into the body of the sentence, where
    # it reads as prose *about* the assistant rather than an address *to* it.
    _WAKE_WORD_OPENERS = frozenset(
        """hey hi hello oi yo ok okay right so erm um uh ah oh well and but
        listen look sorry excuse me please now then just also actually""".split()
    )
    # Words at the end of an utterance within which a trailing vocative counts:
    # "..., Sigma?" or "..., ok Sigma?".
    WAKE_WORD_TRAILING_WORDS = 2

    def _tokenize_words(self, content: str) -> List[str]:
        """Split into bare alphabetic words, discarding punctuation entirely.

        Deliberately crude: it exists so that the name's position can be judged
        without any dependence on how the model punctuated the utterance.
        """
        return re.findall(r"[^\W\d_]+", content, re.UNICODE)

    def _names_assistant_in_opening(self, content: str) -> bool:
        """Whether the name leads the utterance, allowing only filler before it.

        Punctuation, casing and quoting are irrelevant, so "Sigma, ...",
        "Sigma... ...", "Sigma. ...", "'Sigma, ...'", "Oi Sigma, ..." and
        "Erm, so, um, Sigma, ..." all resolve alike.  A content word before the
        name ("I asked Sigma ...") means the name is not leading.
        """
        target = self.assistant_name.casefold()
        for word in self._tokenize_words(content):
            folded = word.casefold()
            if folded == target:
                return True
            if folded not in self._WAKE_WORD_OPENERS:
                return False
        return False

    def _names_assistant_in_closing(self, content: str) -> bool:
        """Whether the utterance ends by naming the assistant ("..., Sigma?")."""
        target = self.assistant_name.casefold()
        tail = self._tokenize_words(content)[-self.WAKE_WORD_TRAILING_WORDS :]
        return any(w.casefold() == target for w in tail)

    def _has_wake_word(self, content: str) -> bool:
        """Whether the utterance already names the assistant as an address.

        This answers only the question the caller needs — "would prepending the
        name duplicate it?" — and is decided by *position* rather than by parsing
        grammar, which makes it deterministic and independent of punctuation.

        The name counts when it leads the utterance (bar filler words) or closes
        it as a tag.  A name buried mid-sentence does not count: "I asked Sigma
        about it" talks *about* the assistant and still needs a wake word to be
        directed *at* it.

        Deliberately more permissive than a true vocative test: "Sigma is broken
        again" leads with the name without addressing anyone, and is left alone
        rather than rewritten into a request.  Distinguishing address from
        mention is the prompt's job, not this function's.
        """
        return self._names_assistant_in_opening(
            content
        ) or self._names_assistant_in_closing(content)

    # Quote-like characters a model may wrap an utterance in.  They are held aside
    # while the name is inserted, then restored, so the wake word lands inside the
    # quotation rather than in front of it.
    _WRAPPERS = "\"'“”„‘’«»`"

    def _prepend_wake_word(self, content: str) -> str:
        """Put the assistant's name in front of an utterance as a vocative.

        Deterministic and idempotent: applying it to output it has already
        produced changes nothing, because `_has_wake_word` then reports the name
        in the opening window.

        The first letter of the utterance is lowercased when that is safe, so
        "What should the pressure be?" becomes "Sigma, what should the pressure
        be?" rather than "Sigma, What should ...".  Words capitalised for their
        own sake — "I", acronyms, proper nouns — keep their case.
        """
        stripped = content.strip()
        if not stripped:
            return f"{self.assistant_name}?"

        # Peel any opening/closing quotation so the name goes inside it.
        lead_wrap = ""
        while stripped and stripped[0] in self._WRAPPERS:
            lead_wrap += stripped[0]
            stripped = stripped[1:].lstrip()
        if not stripped:
            return f"{lead_wrap}{self.assistant_name}?"

        first_word = self._tokenize_words(stripped)
        head = first_word[0] if first_word else ""
        keep_case = (
            head == "I"
            or head.isupper()  # acronym, e.g. "OK", "BBC"
            or (len(head) > 1 and head[1:] != head[1:].lower())  # McDonald, iPhone
            # the assistant's own name is a proper noun; leave it capitalised so a
            # sentence that talks *about* it does not become "Sigma, sigma said ..."
            or head.casefold() == self.assistant_name.casefold()
        )
        if not keep_case and stripped[0].isalpha():
            stripped = stripped[0].lower() + stripped[1:]

        return f"{lead_wrap}{self.assistant_name}, {stripped}"

    def _create_enhanced_prompt(self, variation: Dict) -> str:
        """
        Create an enhanced prompt incorporating variation details.

        This method builds a comprehensive prompt by combining the base prompt with
        variation-specific details like context, and background conditions.

        Args:
            variation: Dictionary containing variation parameters including context,
                and optional background type

        Returns:
            str: Enhanced prompt combining base prompt with variation details

        Raises:
            ValueError: If required variation parameters are missing
        """
        if "context" not in variation:
            raise ValueError("Variation must include 'context'")

        prompt_parts = [
            self.config.initial_prompt,
            f"Context: {variation['context']}",
            f"Maximum words per response: {variation.get('max_words_per_turn', 20)}",
        ]

        # Add variant type information
        if "variant_type" in variation:
            prompt_parts.append(f"Conversation structure: {variation['variant_type']}")

        if variation.get("background_type"):
            prompt_parts.append(
                f"Background environment: {variation['background_type']}"
            )

        if self.config.conversation_type == "multi_user":
            prompt_parts.append(
                "This is a conversation between multiple users and an assistant."
            )

        if "min_turns" in variation and "max_turns" in variation:
            prompt_parts.append(
                f"Generate between {variation['min_turns']} and {variation['max_turns']} "
                "conversation turns."
            )

        return " ".join(prompt_parts)

    def add_direction(self, text: str) -> None:
        """Append a stage direction to the current conversation."""
        self.message_processor.last_message_end_time += 0.5  # small gap
        msg = Message(
            speaker="NARRATION",
            content=f"[{text}]",
            timestamp=datetime.utcnow().isoformat(),
            message_type="description",
        )
        self.conversation.append(msg)

    @staticmethod
    def is_user(agent: CustomAgent) -> bool:
        """Determine if the agent is a user based on its configuration."""
        return "user" in agent.config.id.lower()

    @staticmethod
    def remove_nested_exchanges(content: str, prefixes: list[str]) -> str:
        """Remove any nested conversation exchanges from the content."""
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            if any(line.strip().startswith(f"{name}:") for name in prefixes):
                break
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()
