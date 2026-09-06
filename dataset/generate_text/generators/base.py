"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Base classes for conversation generators across different frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..agents.base import BaseAgent, Message
from ..agents.custom_agent import CustomAgent


@dataclass
class GeneratorConfig:
    """Configuration for conversation generators."""

    name: str
    conversation_type: str
    chat_loops: int
    initial_prompt: str
    variations: List[Dict]
    agent_configs: Dict[str, Dict]
    config_path: str
    default_characters: Optional[Dict[str, Dict]] = None

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.default_characters is None:
            self.default_characters = {}
        if not isinstance(self.agent_configs, dict):
            self.agent_configs = {}
        if not isinstance(self.variations, list):
            self.variations = []

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback to default."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to config values."""
        return getattr(self, key)


class BaseGenerator(ABC):
    """Abstract base generator defining common interface."""

    def __init__(self, config: GeneratorConfig) -> None:
        """Initialize generator with configuration."""
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        if (
            not hasattr(self.config, "agent_configs")
            or self.config.agent_configs is None
        ):
            self.config.agent_configs = {}
        self._validate_config()
        # Initialize with empty variation instead of None
        self.current_variation: Dict = {}

    def initialize(self) -> None:
        """Initialize the generator after construction."""
        self._setup_agents()

    @abstractmethod
    def _setup_agents(self) -> None:
        """Initialize framework-specific agents."""

    @abstractmethod
    async def generate_conversation(self, variation: Dict) -> Dict:
        """Base generate method to ensure proper logging."""

    def _validate_config(self) -> None:
        """Validate generator configuration."""
        required_fields = {"name", "conversation_type", "agent_configs"}
        if not all(field in vars(self.config) for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

    def _create_message(self, agent: CustomAgent, content: str) -> Message:
        """Create a message with current timestamp."""
        return Message(
            speaker=agent.config.name,
            content=content,
            timestamp=datetime.now().isoformat(),
        )

    def reset_agents(self) -> None:
        """Reset all agents' conversation history."""
        for agent in self.agents.values():
            agent.reset_history()

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
