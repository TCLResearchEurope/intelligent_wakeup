"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Generator factory for creating appropriate generator instances.
"""

from typing import Dict, Type

from .custom_generator import CustomGenerator
from .base import BaseGenerator, GeneratorConfig


class GeneratorFactory:
    """Factory class for creating conversation generators."""

    _generators: Dict[str, Type[BaseGenerator]] = {"custom": CustomGenerator}

    @classmethod
    def create_generator(
        cls, config_dict: Dict, timeout: int = 60, max_retries: int = 3
    ) -> BaseGenerator:
        """Create appropriate generator based on configuration.

        Args:
            config_dict: Generator configuration dictionary
            timeout: Timeout in seconds for API calls (default: 60)
            max_retries: Maximum number of retries for failed calls (default: 3)

        Returns:
            Initialized generator instance

        Raises:
            ValueError: If framework type is not supported
        """
        # Extract the default_characters before creating config
        default_characters = config_dict.get("default_characters", {})

        # Handle variant metadata if present
        for variation in config_dict.get("variations", []):
            # Add variant metadata if not already present
            if "variant_type" not in variation and "variant_type" in config_dict:
                variation["variant_type"] = config_dict["variant_type"]
            if "variant_name" not in variation and "variant_name" in config_dict:
                variation["variant_name"] = config_dict["variant_name"]

        # Convert the dictionary to GeneratorConfig
        config = GeneratorConfig(
            name=config_dict["name"],
            conversation_type=config_dict.get("conversation_type", "multi_user"),
            chat_loops=config_dict.get("framework_config", {})
            .get("custom", {})
            .get("chat_loops", 4),
            initial_prompt=config_dict.get("initial_prompt", ""),
            variations=config_dict.get("variations", []),
            agent_configs=config_dict.get("agent_configs", {}),
            config_path=config_dict["config_path"],
            default_characters=default_characters,
        )

        # Fill in the scenario-level cast, but never clobber a variation that
        # names its own. A re-cast variation ("_variant2") exists precisely to
        # run the same scene with a different cast and different voices.
        for variation in config.variations:
            variation.setdefault("default_characters", default_characters)

        return CustomGenerator(config, timeout=timeout, max_retries=max_retries)
