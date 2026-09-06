"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Custom agent implementation using direct OpenAI API calls.

This module provides a simple agent implementation that uses OpenAI's API directly
without additional framework overhead.
"""

import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

from .base import BaseAgent, AgentConfig, Message
from ...utils import logger


class CustomAgent(BaseAgent):
    """
    Custom agent implementation using direct OpenAI API calls.

    This class implements a simple agent that directly uses the OpenAI API
    without additional framework overhead.
    """

    def __init__(
        self, config: AgentConfig, timeout: int = 60, max_retries: int = 3
    ) -> None:
        """
        Initialize custom agent.

        Args:
            config: Agent configuration parameters
            timeout: Timeout in seconds for API calls (default: 60)
            max_retries: Maximum number of retries for failed calls (default: 3)
        """
        super().__init__(config)
        load_dotenv()
        self.client = AsyncOpenAI()
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_history: List[Dict] = []  # OpenAI messages format with full history

    def reset_history(self) -> None:
        """Clear conversation history and API message history."""
        super().reset_history()
        self._api_history = []

    async def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Message:
        """
        Generate response using OpenAI API directly with timeout and retry logic.

        On the first call per conversation, pass system_prompt to initialise the
        persistent message history. Subsequent calls reuse the growing history so
        the static system context is only charged once (and cached by OpenAI).

        Args:
            prompt: Per-turn user message for the agent
            system_prompt: Static system context (role, backstory, scenario).
                           Used only to initialise history on the first call.
            **kwargs: Additional parameters for API call

        Returns:
            Message: Generated response message

        Raises:
            Exception: If API call fails after all retries
        """
        # Initialise history with system message on first call
        if not self._api_history and system_prompt:
            self._api_history = [{"role": "system", "content": system_prompt}]

        # Build full messages list: persistent history + current user turn
        messages = self._api_history + [{"role": "user", "content": prompt}]

        start_time = time.time()
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Get model from config with fallback to gpt-3.5-turbo
                model = self.config.model_config.get("name", "gpt-3.5-turbo")
                logger.debug(
                    "Using model: %s (attempt %d/%d)",
                    model,
                    attempt + 1,
                    self.max_retries,
                )

                # Use asyncio.wait_for to add timeout to the API call
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.config.model_config.get("temperature", 0.7),
                        stream=False,
                    ),
                    timeout=self.timeout,
                )

                logger.info(
                    "GPT inference processing time: %s s", time.time() - start_time
                )

                content = response.choices[0].message.content.strip()
                logger.debug("Response from OpenAI API: %s", content[:100])

                # Commit this exchange to persistent history for future turns
                self._api_history.append({"role": "user", "content": prompt})
                self._api_history.append({"role": "assistant", "content": content})

                message = Message(
                    speaker=self.config.name,
                    content=content,
                    timestamp=datetime.now().isoformat(),
                )

                self.add_to_history(message)
                return message

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(
                    "API call timed out after %d seconds (attempt %d/%d)",
                    self.timeout,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2**attempt
                    logger.info("Waiting %d seconds before retry...", wait_time)
                    await asyncio.sleep(wait_time)

            except Exception as e:
                last_exception = e
                logger.warning(
                    "API call failed: %s (attempt %d/%d)",
                    str(e),
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2**attempt
                    logger.info("Waiting %d seconds before retry...", wait_time)
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        logger.error(
            "Failed to generate response after %d attempts. Last error: %s",
            self.max_retries,
            str(last_exception),
        )
        raise last_exception
