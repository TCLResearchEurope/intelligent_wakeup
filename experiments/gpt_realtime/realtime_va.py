#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

OpenAI Realtime API Virtual Assistant Demo

This script implements a console-based virtual assistant using OpenAI's Realtime API.
It supports two modes:
1. Live mode: Real-time conversation using system microphone
2. Batch mode: Turn-based testing with WAV files

Usage:
    python realtime_va.py --mode live
    python realtime_va.py --mode batch --audio-dir /path/to/wav/files
"""

import sys
import argparse
import logging
from pathlib import Path

from realtime_va.core import RealtimeVACore
from realtime_va.audio_io import AudioInput, AudioOutput
from realtime_va.interface import LiveInterface, BatchInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the Realtime VA application"""
    parser = argparse.ArgumentParser(
        description="OpenAI Realtime API Virtual Assistant Demo"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "batch"],
        required=True,
        help="Operation mode: 'live' for real-time microphone input, 'batch' for WAV file testing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-realtime-mini",
        help="OpenAI model to use (default: gpt-realtime-mini, also available: gpt-realtime)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing WAV files (required for batch mode)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Path to system prompt file (default: prompts/sigma_wakeup.txt)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate batch mode arguments
    if args.mode == "batch":
        if not args.audio_dir:
            logger.error("--audio-dir required for batch mode")
            sys.exit(1)
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            logger.error("Audio directory not found: %s", audio_dir)
            sys.exit(1)
    else:
        audio_dir = None

    # Create components
    try:
        va_core = RealtimeVACore(
            model=args.model,
            prompt_file=args.prompt,
            mode=args.mode,
        )
        audio_input = AudioInput(mode=args.mode)
        audio_output = AudioOutput()

        # Create and run appropriate interface
        if args.mode == "live":
            interface = LiveInterface(va_core, audio_input, audio_output)
        else:
            interface = BatchInterface(va_core, audio_input, audio_output, audio_dir)

        await interface.run()

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to start: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
