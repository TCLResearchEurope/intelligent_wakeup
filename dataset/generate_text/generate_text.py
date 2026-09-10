"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Main script for running conversation generation across different frameworks.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import asyncio

from .generators.factory import GeneratorFactory
from .postprocessing.dialogue_enhancer import DialogueEnhancer
from .postprocessing.director import DirectorAgent
from .versioning import VersionManager
from .version import get_version
from ..utils import logger, LoggerConfigurator


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dict containing configuration data

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in configuration file %s: %s", config_path, str(e))
        raise


def load_characters(config_dir: Path) -> Dict[str, str]:
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


async def generate_conversations(
    scenario_config: Dict,
    output_dir: Path,
    conversation_type: str,
    characters: Dict[str, str],
    config_dir: Path,
    postprocessing_rules_path: Path,
    save_review: bool = False,
    timeout: int = 60,
    max_retries: int = 3,
    takes: int = 1,
    save_takes: bool = False,
    skip_long_variants: bool = False,
    only_missing: bool = False,
) -> int:
    """
    Generate conversations using specified framework.

    Args:
        scenario_config: Scenario configuration
        output_dir: Directory for output files
        conversation_type: Type of conversation to generate
        characters: Dictionary of available character profiles
        postprocessing_rules_path: Path to config with dialog postprocessing rules
        save_review: If True, creates a new file with '-review' suffix instead of overwriting
        timeout: Timeout in seconds for API calls (default: 60)
        max_retries: Maximum number of retries for failed calls (default: 3)
        takes: Number of performances of each scene. 1 (default) keeps the
            original single-pass behaviour. With 2+, a director agent reviews
            each take and the characters perform the scene again from a clean
            slate using its notes.
        skip_long_variants: Skip variations whose variant_name ends in "_long".
        only_missing: Generate only variations with no output file yet.
        save_takes: If True, write every intermediate take and the director
            notes alongside the final output for comparison.
    """
    try:
        # Add debug logging for configuration
        logger.debug("Full scenario config: %s", scenario_config)
        logger.debug(
            "Default characters from config: %s",
            scenario_config.get("default_characters"),
        )

        logger.info(
            "Generating conversations for scenario: %s", scenario_config["name"]
        )
        logger.info("Conversation type: %s", conversation_type)
        logger.info("Number of variations: %s", len(scenario_config["variations"]))
        logger.info("Available characters: %s", list(characters.keys()))

        # Create generator config
        generator_config = {
            "name": scenario_config["name"],
            "conversation_type": conversation_type,
            "chat_loops": scenario_config["framework_config"].get("chat_loops", 3),
            "initial_prompt": scenario_config["initial_prompt"],
            "variations": scenario_config["variations"],
            "agent_configs": scenario_config["agent_configs"],
            "default_characters": scenario_config.get("default_characters"),
            "config_path": config_dir,
        }

        # Debug log the generator config
        logger.debug("Created generator config: %s", generator_config)

        # Create and initialize generator with timeout and retry settings
        generator = GeneratorFactory.create_generator(
            generator_config, timeout=timeout, max_retries=max_retries
        )
        enhancer = DialogueEnhancer(postprocessing_rules_path)
        director = (
            DirectorAgent(postprocessing_rules_path) if takes > 1 else None
        )

        # Extract dialogue_review_rules from scenario config if present
        dialogue_review_rules = scenario_config.get("dialogue_review_rules", None)

        # Generate conversations for each variation
        successful_saves = 0
        for idx, variation in enumerate(scenario_config["variations"]):
            try:
                # variant_name is filled from the file stem earlier when the
                # config omits it, so this sees the same name the output file
                # gets. Anchored on the suffix: ten scenarios have "long"
                # elsewhere in the name (long_run_pacing, LongDistance, ...).
                if skip_long_variants and str(
                    variation.get("variant_name", "")
                ).endswith("_long"):
                    logger.info(
                        "Skipping long variant %s (%s turns)",
                        variation.get("variant_name"),
                        variation.get("max_turns", 12),
                    )
                    continue
                # Version tracking is keyed on the config FILE, so once one
                # variation of a multi-variation config is recorded the whole
                # file counts as done and its siblings become unreachable.
                # --only-missing works at variation granularity instead.
                if only_missing:
                    existing = (
                        output_dir
                        / scenario_config["name"]
                        / variation.get("variant_type", "default")
                        / f"{variation.get('variant_name', f'variation_{idx}')}.json"
                    )
                    if existing.exists():
                        logger.info(
                            "Skipping %s (--only-missing: output exists)",
                            variation.get("variant_name", f"variation_{idx}"),
                        )
                        continue

                logger.info(
                    "Processing variation %s / %s",
                    idx + 1,
                    len(scenario_config["variations"]),
                )
                logger.info("Context: %s", variation["context"])

                # Extract scenario details for organized output paths
                variant_type = variation.get("variant_type", "default")
                variant_name = variation.get("variant_name", f"variation_{idx}")

                # Perform the scene. With takes > 1 the director reviews each
                # take and the characters perform it again from a clean slate —
                # agents are rebuilt inside generate_conversation, so each take
                # starts with no memory of the previous one, only the notes.
                generator.director_notes = None
                conversation = None
                for take in range(1, takes + 1):
                    if takes > 1:
                        logger.info("=== Take %d/%d ===", take, takes)
                    conversation = await generator.generate_conversation(variation)

                    if save_takes and takes > 1:
                        take_dir = output_dir / scenario_config["name"] / variant_type
                        take_dir.mkdir(parents=True, exist_ok=True)
                        take_path = take_dir / f"{variant_name}-take{take}.json"
                        with open(take_path, "w", encoding="utf-8") as f:
                            json.dump(conversation, f, indent=2, ensure_ascii=False)
                        logger.info("Take %d saved to %s", take, take_path)

                    # No review after the final take — nothing would consume it.
                    if director is None or take == takes:
                        break

                    notes = await director.review_take(
                        conversation.get("conversation", []),
                        take_number=take,
                        variation=variation,
                        character_info=characters,
                        scenario_config=scenario_config,
                    )
                    generator.director_notes = notes

                    if save_takes:
                        notes_path = (
                            output_dir
                            / scenario_config["name"]
                            / variant_type
                            / f"{variant_name}-director_notes{take}.json"
                        )
                        with open(notes_path, "w", encoding="utf-8") as f:
                            json.dump(notes, f, indent=2, ensure_ascii=False)

                # Keep the last director notes so the enhancer picks emotional
                # tags that match the emotional read the scene was performed to.
                final_director_notes = generator.director_notes
                generator.director_notes = None

                # Apply dialogue enhancement
                enhanced_conversation = await enhancer.enhance_dialogue(
                    conversation,
                    character_info=characters,
                    scenario_variation=variation,
                    dialogue_review_rules=dialogue_review_rules,
                    director_notes=final_director_notes,
                )

                # Create output subdirectories for organized storage
                scenario_output_dir = (
                    output_dir / scenario_config["name"] / variant_type
                )
                scenario_output_dir.mkdir(parents=True, exist_ok=True)

                # Determine file paths
                base_path = scenario_output_dir / f"{variant_name}.json"
                enhanced_path = scenario_output_dir / f"{variant_name}-review.json"

                if save_review:
                    # Save both original and enhanced conversations
                    with open(base_path, "w", encoding="utf-8") as f:
                        json.dump(conversation, f, indent=2, ensure_ascii=False)
                    logger.info("Original conversation saved to %s", base_path)

                    with open(enhanced_path, "w", encoding="utf-8") as f:
                        json.dump(
                            enhanced_conversation, f, indent=2, ensure_ascii=False
                        )
                    logger.info("Enhanced conversation saved to %s", enhanced_path)
                else:
                    # Save only the enhanced conversation (in place of the original)
                    with open(base_path, "w", encoding="utf-8") as f:
                        json.dump(
                            enhanced_conversation, f, indent=2, ensure_ascii=False
                        )
                    logger.info("Enhanced conversation saved to %s", base_path)

                successful_saves += 1

            except Exception as e:
                logger.error("Error generating variation %s: %s", idx, str(e))
                continue

        return successful_saves

    except Exception as e:
        logger.error("Error in generate_conversations: %s", str(e))
        raise


def prompt_user_regenerate(scenario_key: str, reason: str) -> bool:
    """
    Prompt user whether to regenerate a scenario.

    Args:
        scenario_key: Scenario identifier
        reason: Reason for regeneration

    Returns:
        True if user wants to regenerate, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_key}")
    print(f"Reason: {reason}")
    print(f"{'='*60}")

    while True:
        try:
            response = input("Regenerate this scenario? [y/N]: ").strip().lower()
        except EOFError:
            # No terminal attached (piped or CI run): take the default rather
            # than dying part-way through a long generation.
            print("n  (no input available, taking the default)")
            return False
        if response in ("y", "yes"):
            return True

        if response in ("n", "no", ""):
            return False

        print("Please answer 'y' or 'n'")


def find_scenario_variants(
    scenario_dir: Path, variant: str = "all", conversation_type: str = "single_user"
) -> List[Tuple[Path, str, str]]:
    """
    Find all variant configurations for a specific scenario.

    Args:
        scenario_dir: Directory of the scenario
        variant: Specific variant to find or "all" for all variants
        conversation_type: Type of conversation to filter variants by

    Returns:
        List of tuples (config_path, variant_type, variant_name)
    """
    variants = []

    # Check if directory exists
    if not scenario_dir.exists():
        logger.warning("Scenario directory not found: %s", scenario_dir)
        return variants

    # Map conversation types to directory names
    variant_type_mapping = {
        "single_user": "single",
        "multi_user": "couple",
        "background_noise": "couple",  # Use couple for background noise as well
    }

    # Get the appropriate variant type directory based on conversation type
    target_variant_type = variant_type_mapping.get(conversation_type, "single")

    # Find all variant types (subdirectories) - filter by conversation type
    for variant_type_dir in scenario_dir.glob("*"):
        if not variant_type_dir.is_dir():
            continue

        variant_type = variant_type_dir.name

        # Skip directories that don't match the conversation type
        if variant_type != target_variant_type:
            continue

        # Find all variant configs in this type directory
        for variant_config in variant_type_dir.glob("*.json"):
            variant_name = variant_config.stem

            # If specific variant requested, check if this matches
            if variant not in ("all", variant_name):
                continue

            variants.append((variant_config, variant_type, variant_name))

    return variants


async def main_async():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate conversations using LLM agents."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated conversations",
    )
    parser.add_argument(
        "--postprocessing-rules",
        type=Path,
        default=None,
        help="Path to dialogue enhancement rules",
    )
    parser.add_argument(
        "--scenario", default="all", help='Specific scenario to generate or "all"'
    )
    parser.add_argument(
        "--variant", default="all", help='Specific variant to generate or "all"'
    )
    parser.add_argument(
        "--conversation-type",
        default="single_user",
        choices=["single_user", "multi_user", "background_noise"],
        help="Type of conversation to generate",
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
    parser.add_argument(
        "--takes",
        type=int,
        default=1,
        help=(
            "Number of performances per scene (default: 1, unchanged behaviour). "
            "With 2 or more, a director agent reviews each take and the characters "
            "perform the scene again from a clean slate using its notes."
        ),
    )
    parser.add_argument(
        "--save-takes",
        action="store_true",
        help="Save every intermediate take and the director notes for comparison",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for API calls (default: 60)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API calls (default: 3)",
    )
    parser.add_argument(
        "-n",
        "--no",
        action="store_true",
        help="Automatically answer no to all regeneration prompts: keep the "
        "existing output and skip those scenarios (non-interactive)",
    )
    parser.add_argument(
        "--skip-long-variants",
        action="store_true",
        help="Skip variations whose variant_name ends in '_long', leaving the "
        "expensive long variants for a separate run. Note this skips only the "
        "added long variants, not scenarios that are naturally long.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Generate only variations that have no output file yet, ignoring "
        "version data. Use this to fill gaps -- notably the second variation of "
        "a config whose first one is already recorded as up-to-date.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of all scenarios, ignoring version checks",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer yes to all prompts (non-interactive mode)",
    )

    args = parser.parse_args()

    if args.yes and args.no:
        parser.error("--yes and --no are mutually exclusive")

    # Configure logger with specified level
    LoggerConfigurator.configure_logger(args.log_level)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize version manager
        version_manager = VersionManager(args.output_dir)

        # Log version information
        logger.info("Text generation version: %s", get_version())
        if args.force_regenerate:
            logger.info("Force regeneration enabled - skipping version checks")

        # Set default postprocessing rules path if not provided
        if args.postprocessing_rules is None:
            args.postprocessing_rules = args.config_dir / "postprocessing.json"

        # Load character profiles
        characters = load_characters(args.config_dir)
        if not characters and args.conversation_type == "multi_user":
            logger.error("No character profiles found for multi-user conversation")
            sys.exit(1)

        # Define scenarios base directory with new structure
        scenarios_base_dir = args.config_dir / "scenarios"

        # Collect scenarios to process
        scenarios_to_process = []
        if args.scenario == "all":
            # Process all scenario directories
            for scenario_dir in scenarios_base_dir.glob("*"):
                if scenario_dir.is_dir():
                    scenarios_to_process.append(scenario_dir.name)
        else:
            scenarios_to_process.append(args.scenario)

        # Process each requested scenario
        for scenario_name in scenarios_to_process:
            scenario_dir = scenarios_base_dir / scenario_name

            if not scenario_dir.exists():
                logger.warning("Scenario directory not found: %s", scenario_dir)
                continue

            logger.info("Processing scenario: %s", scenario_name)

            # Find all variants for this scenario
            variants = find_scenario_variants(
                scenario_dir, args.variant, args.conversation_type
            )

            if not variants:
                logger.warning("No variants found for scenario: %s", scenario_name)
                continue

            logger.info("Found %d variants to process", len(variants))

            # Process each variant
            for variant_config_path, variant_type, variant_name in variants:
                try:
                    scenario_key = f"{scenario_name}/{variant_type}/{variant_name}"
                    logger.info("Processing variant: %s", scenario_key)

                    # Check if regeneration is needed (unless --force-regenerate is set)
                    # --only-missing decides per variation below, so the
                    # file-level version gate has to be stepped over or the
                    # config is skipped before its variations are read.
                    if not args.force_regenerate and not args.only_missing:
                        (
                            needs_regeneration,
                            reason,
                            is_first_generation,
                        ) = version_manager.check_scenario_version(
                            scenario_name,
                            variant_type,
                            variant_name,
                            args.config_dir,
                            args.conversation_type,
                        )

                        if not needs_regeneration:
                            logger.info(
                                "Scenario %s is up-to-date, skipping generation",
                                scenario_key,
                            )
                            continue

                        # Scenario needs regeneration
                        if is_first_generation:
                            # First time generation - no prompt needed
                            logger.info(
                                "Generating scenario %s for the first time",
                                scenario_key,
                            )
                        else:
                            # Regeneration due to version/config change
                            logger.info("Regeneration needed: %s", reason)

                            # Auto-skip mode: keep whatever is already there.
                            if args.no:
                                logger.info(
                                    "Skipping scenario %s (--no): %s",
                                    scenario_key,
                                    reason,
                                )
                                continue

                            # Prompt user if not in auto-yes mode
                            if not args.yes:
                                if not prompt_user_regenerate(scenario_key, reason):
                                    logger.info("Skipping scenario %s", scenario_key)
                                    continue

                            logger.info("Regenerating scenario %s", scenario_key)

                    # Load variant configuration
                    scenario_config = load_config(variant_config_path)

                    # Add variant metadata to config for organization
                    for variation in scenario_config["variations"]:
                        variation["variant_type"] = variant_type
                        if "variant_name" not in variation:
                            variation["variant_name"] = variant_name

                    # Set output directory directly (no extra layers)
                    output_dir = args.output_dir
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Generate conversations for this variant
                    successful_saves = await generate_conversations(
                        scenario_config,
                        output_dir,
                        args.conversation_type,
                        characters,
                        args.config_dir,
                        args.postprocessing_rules,
                        args.save_review,
                        args.timeout,
                        args.max_retries,
                        args.takes,
                        args.save_takes,
                        args.skip_long_variants,
                        args.only_missing,
                    )

                    if successful_saves == 0:
                        logger.warning(
                            "No conversations saved for %s, skipping version update",
                            scenario_key,
                        )
                        continue

                    # Update version data after successful generation
                    conversation_count = successful_saves
                    version_manager.update_scenario_version(
                        scenario_name,
                        variant_type,
                        variant_name,
                        args.config_dir,
                        conversation_count,
                        args.conversation_type,
                    )
                    logger.info("Updated version data for scenario %s", scenario_key)

                except Exception as e:
                    logger.error(
                        "Error processing variant %s/%s: %s",
                        variant_type,
                        variant_name,
                        str(e),
                    )
                    continue

    except Exception as e:
        logger.error("Error during generation process: %s", str(e))
        sys.exit(1)


def main():
    """Main entry point to run async code."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
