"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Enhances dialogue naturalness through post-processing rules.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from ...utils import logger
from ..generators.conversation_manager import BACKGROUND_NOISE_SPEAKER

GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class DialogueEnhancer:
    """Reviews and enhances dialogue based on instruction-driven approach."""

    def __init__(self, config_path: Path):
        """Initialize with path to instructions configuration file."""
        load_dotenv()
        self.config = self._load_config(config_path)
        self.instructions = self.config.get("dialogue_review_instructions", [])
        self.checklist = self.config.get("verification_checklist", [])
        self.review_model = self.config.get("review_model", "gpt-4o-mini")
        self.review_provider = self.config.get("review_provider", "openai")
        self.review_temperature = self.config.get("review_temperature", 0.7)

        if self.review_provider == "google":
            self.client = AsyncOpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url=GOOGLE_OPENAI_BASE_URL,
            )
        else:
            self.client = AsyncOpenAI()
        self.output_format_instructions = [
            "# Output Format Instructions\n",
            "After your analysis, provide a suggested revised conversation in this exact format:",
            "```json",
            "[",
            "  {",
            '    "speaker": "Character1",',
            '    "content": "Message content",',
            '    "time": 0.0',
            "  },",
            "  {",
            '    "speaker": "Character2",',
            '    "content": "Message content",',
            '    "time": 0.0,',
            '    "interjection": "agree"',
            "  },",
            "  ...",
            "]",
            "```",
            "Ensure the JSON is valid and preserves all conversation timing values.",
            "If an input message has an \"interjection\" field (\"agree\", \"disagree\", or "
            "\"add\"), that message is a required overlap onto the message directly before it "
            "and MUST appear in your output — do not delete it, merge it into another message, "
            "or summarize it away, even for length or flow. Reproduce its \"interjection\" field "
            "with the same value, unchanged. You may lightly polish its wording but it must "
            "remain its own separate message, in the same position relative to the message it "
            "overlaps. Messages without an \"interjection\" field in the input must not have one "
            "in the output either.",
            "Only include the JSON array, no additional text within the code block.",
        ]

    def _load_config(self, config_path: Path) -> Dict:
        """Load instructions from configuration file."""
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            logger.info("Loaded dialogue enhancement instructions from %s", config_path)
            return config
        except Exception as e:
            logger.error("Failed to load dialogue instructions: %s", str(e))
            return {}

    async def enhance_dialogue(
        self,
        conversation_data: Dict,
        character_info: Optional[Dict] = None,
        scenario_variation: Optional[Dict] = None,
        dialogue_review_rules: Optional[Dict] = None,
        director_notes: Optional[Dict] = None,
    ) -> Dict:
        """
        Review and enhance dialogue based on instructions.

        Args:
            conversation_data: Dictionary containing conversation messages
            character_info: Optional dictionary with character backstories
            scenario_variation: Optional dictionary with scenario context
            dialogue_review_rules: Optional dictionary with scenario-specific dialogue review rules
            director_notes: Optional notes from the director pass. Their emotional
                read of the scene guides which emotional tags are chosen.

        Returns:
            Dict: Enhanced conversation data with review comments
        """
        logger.info("=== Starting Dialogue Review and Enhancement ===")

        try:
            # Ambient segments carry no dialogue. The review stage rewrites the
            # conversation wholesale and drops anything that does not read as speech,
            # so hold them out and splice them back afterwards.
            ambient_segments = [
                (i, turn)
                for i, turn in enumerate(conversation_data.get("conversation", []))
                if turn.get("speaker") == BACKGROUND_NOISE_SPEAKER
            ]
            if ambient_segments:
                logger.info(
                    "Holding out %d ambient segment(s) from review", len(ambient_segments)
                )
                conversation_data = {
                    **conversation_data,
                    "conversation": [
                        turn
                        for turn in conversation_data["conversation"]
                        if turn.get("speaker") != BACKGROUND_NOISE_SPEAKER
                    ],
                }

            # Prepare context for review
            context = self._prepare_review_context(
                conversation_data,
                character_info,
                scenario_variation,
                dialogue_review_rules,
                director_notes,
            )

            # Determine conversation type
            conversation_type = conversation_data.get("conversation_type", "multi_user")

            # Get GPT review with appropriate rules
            review_results = await self._get_gpt_review(context, conversation_type)

            # Log review findings
            self._log_review_results(review_results)

            # Apply suggested improvements if any
            if (
                "suggested_conversation" in review_results
                and review_results["suggested_conversation"]
            ):
                try:
                    enhanced_data = conversation_data.copy()
                    enhanced_data["conversation"] = review_results[
                        "suggested_conversation"
                    ]
                except Exception as e:
                    logger.error("Error applying enhancements: %s", str(e))
                    logger.warning("Using original conversation instead")
                    enhanced_data = conversation_data.copy()
            else:
                logger.warning("No valid suggested conversation found in review")
                enhanced_data = conversation_data.copy()

            if ambient_segments:
                enhanced_data["conversation"] = self._restore_ambient_segments(
                    enhanced_data.get("conversation", []), ambient_segments
                )
            return enhanced_data

        except Exception as e:
            logger.error("Error during dialogue review: %s", str(e))
            if ambient_segments:
                conversation_data = {
                    **conversation_data,
                    "conversation": self._restore_ambient_segments(
                        conversation_data.get("conversation", []), ambient_segments
                    ),
                }
            return conversation_data

    @staticmethod
    def _restore_ambient_segments(conversation: list, ambient_segments: list) -> list:
        """
        Splice held-out ambient segments back into a reviewed conversation.

        Args:
            conversation: The reviewed dialogue, without ambient segments.
            ambient_segments: (original index, turn) pairs held out before review.

        Returns:
            list: The conversation with ambient segments back at their positions.
        """
        restored = list(conversation)
        # Review may return a different number of turns than it was given, so an
        # original index is not a usable slot. Ambience is anchored on its timestamp
        # instead, and the result is ordered by time so the scene stays coherent.
        for _, turn in ambient_segments:
            restored.append(turn)

        def turn_time(turn: Dict) -> float:
            value = turn.get("time", turn.get("timestamp"))
            if isinstance(value, str):
                value = value.rstrip("s")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("inf")

        restored.sort(key=turn_time)
        logger.info("Restored %d ambient segment(s) after review", len(ambient_segments))
        return restored

    def _prepare_review_context(
        self,
        conversation_data: Dict,
        character_info: Optional[Dict],
        scenario_variation: Optional[Dict],
        dialogue_review_rules: Optional[Dict],
        director_notes: Optional[Dict] = None,
    ) -> str:
        """Prepare context for GPT review."""
        context_parts = []

        # Add instructions
        context_parts.extend(
            [
                "# Dialogue Review Instructions",
                "\n".join(self.instructions),
                "\n# Verification Checklist",
                "\n".join(self.checklist),
            ]
        )

        # Add output format instructions
        context_parts.append("\n" + "\n".join(self.output_format_instructions))

        # Add only relevant character information (not all characters)
        if character_info and conversation_data.get("conversation"):
            # Extract unique speakers from conversation
            speakers = set()
            for msg in conversation_data["conversation"]:
                speaker = msg.get("speaker", "")
                if speaker:
                    speakers.add(speaker.lower())

            # Filter character_info to only include relevant characters
            relevant_chars = {}
            for char_name, char_backstory in character_info.items():
                if char_name.lower() in speakers:
                    relevant_chars[char_name] = char_backstory

            if relevant_chars:
                context_parts.extend(
                    ["\n# Character Information", json.dumps(relevant_chars, indent=2)]
                )

        # Add only essential scenario context (exclude large nested objects)
        if scenario_variation:
            essential_context = {
                "context": scenario_variation.get("context", ""),
                "mood": scenario_variation.get("setting", {}).get("mood", ""),
                "conversation_style": scenario_variation.get("setting", {}).get(
                    "conversation_style", ""
                ),
            }
            context_parts.extend(
                ["\n# Scenario Context", json.dumps(essential_context, indent=2)]
            )

        # The director's emotional read of the scene, when a director pass ran.
        # This is what emotional tags should be chosen against — it is a
        # scene-level judgement the review pass cannot make on its own.
        if director_notes:
            emo_parts = []
            if director_notes.get("emotional_arc"):
                emo_parts.append(f"Emotional arc: {director_notes['emotional_arc']}")
            per_char = director_notes.get("per_character", {}) or {}
            for name, note in per_char.items():
                emo_parts.append(f"  {name}: {note}")
            if emo_parts:
                context_parts.append(
                    "\n# DIRECTOR'S EMOTIONAL READ OF THIS SCENE\n"
                    "Choose emotional/delivery tags consistent with this. Do not "
                    "contradict it, and do not exceed the intensity it describes.\n"
                    + "\n".join(emo_parts)
                )

        # Add scenario-specific dialogue review rules if present
        if dialogue_review_rules:
            context_parts.append("\n# SCENARIO-SPECIFIC DIALOGUE RULES")
            context_parts.append(
                "These rules MUST be followed for this specific scenario type:\n"
            )

            if "scenario_type" in dialogue_review_rules:
                context_parts.append(
                    f"Scenario Type: {dialogue_review_rules['scenario_type']}"
                )

            if "assistant_behavior" in dialogue_review_rules:
                context_parts.append("\nAssistant Behavior Rules:")
                for rule in dialogue_review_rules["assistant_behavior"]:
                    context_parts.append(f"- {rule}")

            if "conversation_structure" in dialogue_review_rules:
                context_parts.append("\nConversation Structure:")
                for rule in dialogue_review_rules["conversation_structure"]:
                    context_parts.append(f"- {rule}")

        # Add conversation
        context_parts.extend(
            [
                "\n# Conversation to Review",
                json.dumps(conversation_data["conversation"], indent=2),
            ]
        )

        return "\n\n".join(context_parts)

    async def _get_gpt_review(
        self, context: str, conversation_type: str = "multi_user"
    ) -> Dict:
        """Get GPT's review of the dialogue."""
        try:
            # Different prompts based on conversation type
            if conversation_type == "single_user":
                system_content = (
                    "You are an expert dialogue reviewer focusing on natural conversation "
                    "flow between a single user and an assistant. This is a single-user "
                    "conversation where only ONE user interacts with an assistant. Do NOT "
                    "add additional users. Focus on improving the natural flow between "
                    "the existing user and assistant only."
                )
                user_content = (
                    f"Please review this SINGLE-USER dialogue and provide:\n"
                    f"1. Analysis of conversation flow and assistant helpfulness\n"
                    f"2. Specific issues found (but do NOT add additional users)\n"
                    f"3. A revised version that improves the existing user-assistant "
                    f"interaction\n\n"
                    f"IMPORTANT: Keep only the existing single user and assistant. Do not "
                    f"introduce additional speakers.\n\n"
                    f"{context}"
                )
            else:
                system_content = (
                    "You are an expert dialogue reviewer focusing on natural conversation "
                    "flow and character authenticity. After analyzing the dialogue, you "
                    "will provide specific improvements in a valid JSON format that can be "
                    "directly applied to enhance the conversation.\n\n"
                    "IMPORTANT GUIDELINES:\n"
                    "- Avoid adding repetitive phrases like '— sounds good' to responses\n"
                    "- Reduce overly enthusiastic language like 'Absolutely!', 'Perfect!', "
                    "'That sounds amazing!'\n"
                    "- Remove unnecessary agreement markers like 'I like that!', 'I agree', "
                    "'That's great!' when people can build on ideas directly\n"
                    "- Let people contribute to conversations naturally without constant "
                    "validation\n"
                    "- Allow for some disagreement, hesitation, or neutral responses to "
                    "create realistic dynamics\n"
                    "- Natural conversation flows through building on ideas, not explicit "
                    "agreement"
                )
                user_content = (
                    f"Please review this dialogue context and provide:\n"
                    f"1. Analysis of naturalness and character consistency\n"
                    f"2. Specific issues found\n"
                    f"3. A revised version of the conversation that fixes these issues\n\n"
                    f"{context}"
                )

            # Prepare API call parameters
            api_params = {
                "model": self.review_model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            }

            # gpt-5-* models don't support custom temperature
            if not self.review_model.startswith("gpt-5"):
                api_params["temperature"] = self.review_temperature

            # Log the full instructions being sent to the review model
            logger.debug("=== Dialogue Review Model Instructions ===")
            logger.debug("Model: %s", self.review_model)
            logger.debug("System prompt:\n%s", system_content)
            logger.debug("User prompt (with context):\n%s", user_content)
            logger.debug("=" * 50)

            response = await self.client.chat.completions.create(**api_params)

            # Parse GPT's response into structured format
            review_text = response.choices[0].message.content
            return self._parse_review_response(review_text)

        except Exception as e:
            logger.error("Error getting GPT review: %s", str(e))
            return {}

    def _parse_review_response(self, review_text: str) -> Dict:
        """Parse GPT's review into structured format."""
        result = {"analysis": review_text, "suggested_conversation": None}

        # Look for the suggested conversation in JSON format
        json_pattern = r"```json\s*(\[\s*\{.*?\}\s*\])\s*```"
        json_matches = re.search(json_pattern, review_text, re.DOTALL)

        if not json_matches:
            # Try alternative pattern without json tag (just code blocks)
            json_pattern = r"```\s*(\[\s*\{.*?\}\s*\])\s*```"
            json_matches = re.search(json_pattern, review_text, re.DOTALL)

        if json_matches:
            json_str = json_matches.group(1)
            try:
                suggested_conversation = json.loads(json_str)
                result["suggested_conversation"] = suggested_conversation
                logger.info("Successfully parsed suggested conversation")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON conversation: %s", str(e))
                logger.debug("JSON parsing failed for: %s", json_str)
        else:
            logger.warning("No JSON conversation found in review")

        return result

    def _log_review_results(self, review_results: Dict) -> None:
        """Log the results of the dialogue review."""
        logger.info("\nDialogue Review Results:")
        logger.info("-" * 50)

        if "analysis" in review_results:
            logger.info("\nAnalysis:")
            logger.info(review_results["analysis"])

        if (
            "suggested_conversation" in review_results
            and review_results["suggested_conversation"]
        ):
            logger.info("Enhanced conversation extracted successfully")
        else:
            logger.info("\nNo valid conversation improvements could be extracted")

    async def apply_improvements(
        self, conversation_data: Dict, review_results: Dict
    ) -> Dict:
        """Apply suggested improvements to the conversation."""
        if (
            "suggested_conversation" in review_results
            and review_results["suggested_conversation"]
        ):
            enhanced_data = conversation_data.copy()
            enhanced_data["conversation"] = review_results["suggested_conversation"]
            return enhanced_data
        return conversation_data

    async def review_file(
        self,
        input_file_path: Path,
        output_file_path: Optional[Path] = None,
        character_info: Optional[Dict] = None,
        scenario_variation: Optional[Dict] = None,
        dialogue_review_rules: Optional[Dict] = None,
        save_review: bool = False,
    ) -> Tuple[bool, str]:
        """
        Review and enhance a conversation file.

        Args:
            input_file_path: Path to the input JSON file
            output_file_path: Optional path to save enhanced file (defaults to input file)
            character_info: Optional dictionary with character backstories
            scenario_variation: Optional dictionary with scenario context
            dialogue_review_rules: Optional dictionary with scenario-specific dialogue review rules
            save_review: If True, saves both original and enhanced conversations.
                   If False, only saves the enhanced conversation (overwrites original).

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load input file
            with open(input_file_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)

            # Enhance dialogue
            enhanced_data = await self.enhance_dialogue(
                original_data, character_info, scenario_variation, dialogue_review_rules
            )

            # Determine paths and save files based on save_review parameter
            if save_review:
                # Save both original and enhanced files
                if output_file_path is None:
                    # Use default paths based on input file
                    original_path = input_file_path
                    enhanced_stem = f"{input_file_path.stem}-review"
                    enhanced_path = input_file_path.with_stem(enhanced_stem)
                else:
                    # Use provided output path for enhanced version
                    enhanced_path = output_file_path
                    # Use same directory but original filename for original version
                    original_path = enhanced_path.parent / input_file_path.name

                # Save original data
                with open(original_path, "w", encoding="utf-8") as f:
                    json.dump(original_data, f, indent=2, ensure_ascii=False)

                # Save enhanced data
                with open(enhanced_path, "w", encoding="utf-8") as f:
                    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

                return (
                    True,
                    f"Original saved to {original_path}, enhanced saved to {enhanced_path}",
                )

            # Save only the enhanced version (overwriting original if output_file_path is None)
            if output_file_path is None:
                output_file_path = input_file_path

            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

            return True, f"Enhanced conversation saved to {output_file_path}"

        except Exception as e:
            error_msg = f"Error reviewing file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

