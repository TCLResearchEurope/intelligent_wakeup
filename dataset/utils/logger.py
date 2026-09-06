"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

This module configures the logging for the application.
It ensures that logging is configured only once using the LoggerConfigurator class. The logging
configuration includes setting the logging level, format, and date format. The logger is then
available for use throughout the application with a predefined configuration.
"""
# pylint: disable=too-few-public-methods
import os
import json
import logging
from typing import Optional


# pylint: disable=too-few-public-methods
class OpenAILogFilter(logging.Filter):
    """
    Wrapper around OpenAI logs made to filter http messages (to show less).
    """

    def filter(self, record):
        """Filter out messages from OpenAI client."""
        if record.name == "openai._base_client":
            if "Request options" in record.msg:
                return "'role': 'system'" not in str(record.args)
        return True


class LoggerConfigurator:
    """
    Ensures that logging is configured only once across the application.

    Attributes:
        configured (bool): Tracks whether logging has been configured.
    """

    configured: bool = False

    @staticmethod
    def configure_logger(log_level: Optional[str] = None) -> None:
        """
        Configures the application-wide logging settings if not already configured.

        Args:
            log_level: Optional string specifying the logging level
                      (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if LoggerConfigurator.configured:
            return

        config_path = os.path.join("config", "logging.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                logging_config = json.load(f)

            if log_level:
                # Override root logger and all handlers to use command line level
                logging_config["loggers"][""]["level"] = log_level.upper()
                for handler in logging_config["handlers"].values():
                    handler["level"] = log_level.upper()

            logging.config.dictConfig(logging_config)
        else:
            # Fallback configuration
            logging.basicConfig(
                level=getattr(logging, log_level.upper())
                if log_level
                else logging.INFO,
                format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S %z",
            )

        # Configure OpenAI logger
        openai_logger = logging.getLogger("openai._base_client")
        openai_logger.addFilter(OpenAILogFilter())

        LoggerConfigurator.configured = True


# Create the logger but don't configure it yet
logger = logging.getLogger("iw-generate-text")
