"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Base classes and interfaces for conversation agents.

This module provides the foundational classes and data structures for implementing conversation
agents across different implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal


MessageType = Literal["dialogue", "description"]


@dataclass
class Message:
    """
    Represents a single message in a conversation.

    Attributes:
        speaker: Name of the speaker/agent
        content: Content of the message
        timestamp: Timestamp of when the message was generated
        message_type: "dialogue" for normal utterances, "description" for stage directions.
        duration: Requested length in seconds, set only for ambient/silence segments.
        interjection: Set when this turn is a config-authored overlap onto the
            immediately preceding turn — "agree", "disagree", or "add". None
            for ordinary turns.
    """

    speaker: str
    content: str
    timestamp: str
    agent_id: Optional[str] = None
    message_type: MessageType = "dialogue"
    # Only set for ambient/silence segments, where it carries the requested length
    # in seconds. None means the audio stage resolves the duration itself.
    duration: Optional[float] = None
    interjection: Optional[str] = None


@dataclass
class AgentConfig:
    """
    Configuration for initializing an agent.

    Attributes:
        name: Name identifier for the agent
        role: Agent's role/backstory for the conversation
        model_config: Configuration for the language model
        history_enabled: Whether to maintain conversation history
        max_turns: Maximum number of conversation turns
        characters: Dictionary of available character profiles
        additional_params: Framework-specific parameters
    """

    id: str
    name: str
    role: str
    model_config: Dict[str, Any]
    history_enabled: bool = True
    max_turns: int = 10
    character: Optional[str] = None
    characters: Dict[str, str] = None
    additional_params: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base class for all conversation agents.

    This class defines the interface that all agent implementations must follow,
    regardless of the underlying implementation.
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize base agent with configuration.

        Args:
            config: Configuration parameters for the agent
        """
        self.config = config
        self.conversation_history: List[Message] = []
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate agent configuration.

        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
        if not self.config.name or not self.config.role:
            raise ValueError("Agent name and role must be provided")
        if not self.config.model_config:
            raise ValueError("Model configuration must be provided")

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> Message:
        """
        Generate agent's response to a prompt.

        Args:
            prompt: Input prompt for the agent
            **kwargs: Additional framework-specific parameters

        Returns:
            Message: Generated response message

        Raises:
            NotImplementedError: Must be implemented by concrete classes
        """

    def add_to_history(self, message: Message) -> None:
        """
        Add a message to conversation history if enabled.

        Args:
            message: Message to add to history
        """
        if self.config.history_enabled:
            self.conversation_history.append(message)

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Message]:
        """
        Get conversation history.

        Returns:
            List[Message]: List of conversation messages
        """
        return self.conversation_history.copy()
