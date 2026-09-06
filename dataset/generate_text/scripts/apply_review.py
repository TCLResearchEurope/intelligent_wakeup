#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Standalone script to apply dialogue review and enhancement to conversation files.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from ..postprocessing.dialogue_enhancer import DialogueEnhancer
from ...utils import logger, LoggerConfigurator


def load_json_file(file_path: Path) -> Optional[Dict]:
    """
    Load data from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dict containing JSON data or None if error
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return None
    except json.JSONDecodeError:
        logger.error("Invalid JSON in file: %s", file_path)
        return None
    except Exception as e:
        logger.error("Error loading file %s: %s", file_path, str(e))
        return None


def load_character_info(config_dir: Path) -> Dict[str, str]:
    """
    Load character profiles from config/characters directory.

    Args:
        config_dir: Base config directory

    Returns:
        Dict mapping character names to their backstories
    """
    characters = {}
    characters_dir = config_dir / "characters"

    if not characters_dir.exists():
        logger.warning("Characters directory not found: %s", characters_dir)
        return characters

    for char_file in characters_dir.glob("*.txt"):
        try:
            with open(char_file, encoding="utf-8") as f:
                characters[char_file.stem] = f.read()
            logger.debug("Loaded character profile: %s", char_file.stem)
        except Exception as e:
            logger.error("Error loading character %s: %s", char_file.name, str(e))
            continue

    return characters


async def process_file(
    input_file: Path,
    output_file: Path,
    rules_path: Path,
    characters: Dict,
    scenario_file: Optional[Path] = None,
    save_review: bool = False,
) -> bool:
    """
    Process a single conversation file through dialogue enhancement.

    Args:
        input_file: Path to input conversation JSON file
        output_file: Path to output enhanced JSON file
        rules_path: Path to postprocessing rules configuration
        characters: Dictionary of character profiles
        scenario_file: Optional path to scenario configuration file
        save_review: If True, creates a new file with '-review' suffix instead of overwriting

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize dialogue enhancer
        enhancer = DialogueEnhancer(rules_path)

        # Load scenario configuration if provided
        scenario_variation = None
        dialogue_review_rules = None
        if scenario_file and scenario_file.exists():
            scenario_data = load_json_file(scenario_file)
            if scenario_data:
                # Extract variations
                if "variations" in scenario_data and scenario_data["variations"]:
                    scenario_variation = scenario_data["variations"][0]
                # Extract dialogue_review_rules from scenario level
                if "dialogue_review_rules" in scenario_data:
                    dialogue_review_rules = scenario_data["dialogue_review_rules"]

        # Apply review
        success, message = await enhancer.review_file(
            input_file,
            output_file,
            characters,
            scenario_variation,
            dialogue_review_rules,
            save_review,
        )

        if success:
            logger.info(message)
            return True

        logger.error(message)
        return False

    except Exception as e:
        logger.error("Error processing file %s: %s", input_file, str(e))
        return False


async def batch_process(
    input_dir: Path,
    output_dir: Path,
    rules_path: Path,
    config_dir: Path,
    pattern: str = "*.json",
    save_review: bool = False,
) -> None:
    """
    Process multiple conversation files in a directory.

    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        rules_path: Path to postprocessing rules
        config_dir: Directory with configuration files
        pattern: File pattern to match (default: "*.json")
        save_review: If True, creates a new file with '-review' suffix instead of overwriting
    """
    # Make sure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load character information
    characters = load_character_info(config_dir)

    # Get list of input files
    input_files = list(input_dir.glob(pattern))
    if not input_files:
        logger.warning("No files matching pattern %s found in %s", pattern, input_dir)
        return

    logger.info("Processing %d files...", len(input_files))

    # Process each file
    successful = 0
    for input_file in input_files:
        # Determine output path
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to find matching scenario file
        scenario_name = input_file.stem.split("_")[0]
        scenario_file = config_dir / "scenarios" / f"{scenario_name}.json"

        # Process the file
        logger.info("Processing %s -> %s", input_file, output_file)
        if await process_file(
            input_file, output_file, rules_path, characters, scenario_file, save_review
        ):
            successful += 1

    logger.info("Processed %d/%d files successfully", successful, len(input_files))


async def main_async():
    """Main asynchronous execution function."""
    parser = argparse.ArgumentParser(
        description="Apply dialogue review and enhancement to conversation files."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input file or directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file or directory (defaults to overwriting input)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("config/postprocessing.json"),
        help="Path to dialogue enhancement rules",
    )
    parser.add_argument(
        "--scenario", type=Path, help="Path to scenario configuration file"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process all files in input directory"
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="File pattern to match when using batch mode",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--save-review",
        action="store_true",
        help="Create a new file with -review suffix instead of overwriting output",
    )

    args = parser.parse_args()

    # Configure logger with specified level
    LoggerConfigurator.configure_logger(args.log_level)

    # Determine output path if not specified
    if args.output is None:
        args.output = args.input

    # Load character information
    characters = load_character_info(args.config_dir)

    # Process in batch mode or single file mode
    if args.batch:
        if not args.input.is_dir():
            logger.error("Input must be a directory when using batch mode")
            sys.exit(1)
        await batch_process(
            args.input,
            args.output,
            args.rules,
            args.config_dir,
            args.pattern,
            args.save_review,
        )
    else:
        if not args.input.is_file():
            logger.error("Input must be a file when not using batch mode")
            sys.exit(1)

        # Ensure output directory exists
        if args.output.is_dir():
            args.output.mkdir(parents=True, exist_ok=True)
            output_file = args.output / args.input.name
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_file = args.output

        # Process the file
        success = await process_file(
            args.input,
            output_file,
            args.rules,
            characters,
            args.scenario,
            args.save_review,
        )

        if not success:
            sys.exit(1)


def main():
    """Main entry point to run async code."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
