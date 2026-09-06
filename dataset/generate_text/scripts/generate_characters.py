#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Character Generator

This script generates unique character profiles based on a template structure.
It can also analyze existing characters to ensure diversity in the newly generated profiles.

Usage:
    python generate_characters.py --count 3 --output-dir ./config/new_characters \
            --existing-dir ./config/characters

The script will create unique character profiles with distinctive traits, backgrounds,
and interaction styles while avoiding similarities with existing characters.
"""

import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
import asyncio
import openai
from dotenv import load_dotenv

from ...utils import logger, LoggerConfigurator


# ======================================================
# CHARACTER GENERATION CONFIGURATION
# ======================================================
# Edit these instructions to change the character generation guidance

CHARACTER_TEMPLATE_INSTRUCTION = """
You are tasked with creating unique character profiles for dialogue simulations.
These characters will be used in simulated conversations where their distinct traits,
backgrounds, and interaction styles should be clearly visible.

Please create a character profile following this structure:
1. Context / Background: Include essential facts about upbringing, education, environment, and formative events.
2. Life Agenda (Motivation / Goals): Describe what drives them day-to-day and hint at deeper motivations.
3. Personality & Distinguishing Traits: List key personality traits and unique quirks or habits.
4. Interaction Style: Describe how they typically communicate and their attitude toward others.
5. Relationships & Emotional Stakes: Identify significant personal connections and emotional influences.

IMPORTANT GUIDELINES:
- Create natural but somewhat heightened characters whose traits would be noticeable in dialogue.
- Include at least one distinctive quirk that would be visible in conversation (speech pattern, reference style, etc.)
- Give them specific interests (e.g., cooking style, entertainment preferences, hobby focus)
- Create a mix of ages, backgrounds, professions, and temperaments
- Avoid stereotypes while still creating distinctive personalities
- Make their drives and motivations psychologically realistic

The character should feel like someone from a well-written novel or TV show - realistic but with clear defining traits.
"""

# List of occupation categories to ensure diversity
OCCUPATION_CATEGORIES = [
    "creative professional",
    "healthcare worker",
    "educator",
    "technology professional",
    "service industry worker",
    "corporate professional",
    "tradesperson",
    "entrepreneur",
    "scientist/researcher",
    "student",
    "retiree",
    "government worker",
    "non-profit worker",
]

# List of personality dimension pairs to ensure variety
PERSONALITY_DIMENSIONS = [
    ("introverted", "extroverted"),
    ("analytical", "intuitive"),
    ("practical", "idealistic"),
    ("cautious", "adventurous"),
    ("organized", "spontaneous"),
    ("traditional", "unconventional"),
    ("reserved", "expressive"),
    ("competitive", "cooperative"),
]

# ======================================================
# HELPER FUNCTIONS
# ======================================================


def load_template_file(file_path: Path) -> str:
    """
    Load the character template structure from a file.

    Args:
        file_path: Path to template file

    Returns:
        String containing the template structure
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Template file not found: %s", file_path)
        return ""
    except Exception as e:
        logger.error("Error loading template file: %s", str(e))
        return ""


def load_existing_characters(directory: Path) -> List[Dict]:
    """
    Load existing character profiles to analyze for similarities.

    Args:
        directory: Directory containing character files

    Returns:
        List of character data dictionaries
    """
    characters = []

    if not directory.exists():
        logger.warning("Existing characters directory not found: %s", directory)
        return characters

    for char_file in directory.glob("*.txt"):
        try:
            with open(char_file, encoding="utf-8") as f:
                characters.append({"name": char_file.stem, "content": f.read()})
            logger.debug("Loaded existing character: %s", char_file.stem)
        except Exception as e:
            logger.error("Error loading character %s: %s", char_file.name, str(e))
            continue

    return characters


def extract_key_traits(characters: List[Dict]) -> Dict[str, Set[str]]:
    """
    Extract key traits from existing characters to avoid duplication.

    Args:
        characters: List of character data dictionaries

    Returns:
        Dictionary of trait categories and sets of existing traits
    """
    traits = {
        "occupations": set(),
        "backgrounds": set(),
        "personalities": set(),
        "interests": set(),
        "quirks": set(),
    }

    # This is a simplified approach - in a real implementation,
    # you might want to use NLP to extract these traits more accurately
    for character in characters:
        content = character["content"].lower()

        # Extract occupation keywords
        for occupation in OCCUPATION_CATEGORIES:
            if occupation in content:
                traits["occupations"].add(occupation)

        # Extract personality traits
        for pair in PERSONALITY_DIMENSIONS:
            for trait in pair:
                if trait in content:
                    traits["personalities"].add(trait)

    return traits


async def generate_character(
    client, template: str, existing_traits: Dict[str, Set[str]], index: int
) -> Dict:
    """
    Generate a unique character profile using OpenAI.

    Args:
        client: OpenAI client
        template: Character template structure
        existing_traits: Dictionary of existing character traits to avoid
        index: Index number for this character

    Returns:
        Dictionary with generated character data
    """
    # Create a prompt that includes existing traits to avoid
    avoid_traits = ""
    if existing_traits:
        avoid_traits = "AVOID THESE TRAITS FROM EXISTING CHARACTERS:\n"
        for category, traits in existing_traits.items():
            if traits:
                avoid_traits += f"- {category.capitalize()}: {', '.join(traits)}\n"

    # Select traits to encourage for this character to ensure diversity
    encourage_traits = "TRAITS TO INCLUDE:\n"

    # Select an occupation category not heavily used
    available_occupations = [
        o
        for o in OCCUPATION_CATEGORIES
        if o not in existing_traits.get("occupations", set())
    ]
    if not available_occupations:
        available_occupations = OCCUPATION_CATEGORIES
    occupation = random.choice(available_occupations)
    encourage_traits += f"- Occupation category: {occupation}\n"

    # Select a personality dimension
    personality_pair = random.choice(PERSONALITY_DIMENSIONS)
    trait = random.choice(personality_pair)
    encourage_traits += f"- Personality tendency: {trait}\n"

    full_prompt = f"{CHARACTER_TEMPLATE_INSTRUCTION}\n\n{template}\n\n{avoid_traits}\n{encourage_traits}"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",  # or another appropriate model
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.8,
        )

        character_content = response.choices[0].message.content.strip()

        # Generate a unique name based on traits
        name_prompt = (
            f"Based on this character description, suggest a first name that fits their "
            f"personality and background. Provide ONLY the name, nothing else:\n\n"
            f"{character_content}"
        )

        name_response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for this simple task
            messages=[{"role": "user", "content": name_prompt}],
            temperature=0.7,
            max_tokens=20,
        )

        name = (
            name_response.choices[0].message.content.strip().split()[0]
        )  # Take the first word
        name = "".join(c for c in name if c.isalnum())  # Sanitize the name

        return {"name": name, "content": character_content}

    except Exception as e:
        logger.error("Error generating character: %s", str(e))
        return {
            "name": f"character_{index}",
            "content": "Error generating character profile.",
        }


def save_character(character: Dict, output_dir: Path) -> bool:
    """
    Save a generated character to a file.

    Args:
        character: Dictionary with character data
        output_dir: Directory to save the character file

    Returns:
        Boolean indicating success
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename, ensuring it's unique
        base_name = character["name"].lower()
        counter = 0
        file_path = output_dir / f"{base_name}.txt"

        while file_path.exists():
            counter += 1
            file_path = output_dir / f"{base_name}_{counter}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(character["content"])

        logger.info("Saved character %s to %s", character["name"], file_path)
        return True

    except Exception as e:
        logger.error("Error saving character %s: %s", character["name"], str(e))
        return False


async def generate_characters(
    count: int,
    template_file: Path,
    output_dir: Path,
    existing_dir: Optional[Path] = None,
) -> None:
    """
    Generate multiple character profiles.

    Args:
        count: Number of characters to generate
        template_file: Path to template file
        output_dir: Directory to save generated profiles
        existing_dir: Optional directory containing existing profiles
    """
    # Load OpenAI API key
    load_dotenv()
    client = openai.AsyncOpenAI()  # Uses API key from environment variable

    # Load template and existing characters
    template = load_template_file(template_file)
    if not template:
        logger.error("Failed to load template. Exiting.")
        sys.exit(1)

    existing_characters = []
    if existing_dir:
        existing_characters = load_existing_characters(existing_dir)
        logger.info(
            "Loaded %s existing characters for reference", len(existing_characters)
        )

    # Extract traits to avoid duplication
    existing_traits = extract_key_traits(existing_characters)

    # Generate requested number of characters
    successful = 0
    logger.info("Generating %s new characters...", count)

    for i in range(count):
        logger.info("Generating character %s/%s...", i + 1, count)
        character = await generate_character(client, template, existing_traits, i)

        if character["content"] != "Error generating character profile.":
            if save_character(character, output_dir):
                successful += 1

            # Update traits to avoid for next generation
            new_traits = extract_key_traits([character])
            for category, traits in new_traits.items():
                existing_traits[category].update(traits)

    logger.info("Successfully generated %s/%s characters", successful, count)


async def main_async():
    """Main asynchronous execution function."""
    parser = argparse.ArgumentParser(
        description="Generate unique character profiles for dialogue simulations."
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of characters to generate"
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("characters.txt"),
        help="Template file with character structure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_characters"),
        help="Output directory for generated characters",
    )
    parser.add_argument(
        "--existing-dir",
        type=Path,
        help="Directory containing existing character profiles (optional)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Configure logger with specified level
    LoggerConfigurator.configure_logger(args.log_level)

    await generate_characters(
        args.count, args.template, args.output_dir, args.existing_dir
    )


def main():
    """Main entry point to run async code."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
