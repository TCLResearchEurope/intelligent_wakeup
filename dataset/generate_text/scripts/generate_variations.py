"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Main script for creating variations of the scenarios from the text corpus.
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from ...utils import logger, LoggerConfigurator


def get_prompt(name, var, n):
    return f"""
			You are generating **new, original dialogue scenario outlines** for simulation training.

			You will be given:

			1. An example scenario (or multiple scenarios)
			2. A target category

			#YOUR TASK:

			* Create NEW scenarios that follow the SAME JSON STRUCTURE
			* Keep the SAME GENERAL DOMAIN as the category
			* DO NOT copy or slightly rephrase the example
			* Ensure each scenario is meaningfully different in context, interaction, and flow

			##STRICT REQUIREMENTS:

			* Output ONLY a JSON array (starting with '[' and ending with ']')
			* Do NOT include any explanations, comments, or markdown
			* Each scenario MUST include all required fields:
			context, setting, scenario_template, complexity, max_turns
			* Preserve the schema exactly

			##VARIATION RULES:

			* Change the **context** to a new but related situation within the category
			* Vary the **setting** (mood, relationship_dynamic, background, conversation_style)
			* Introduce a **different type of challenge/problem**
			* Ensure the **assistant plays a meaningful role**, not just a generic reminder
			* Vary phases:

			* Use different phase names
			* Change the flow of interaction
			* Add or remove phases if appropriate (minimum 3, maximum 6)
			* Keep realistic human interactions

			##CATEGORY:
			{name}

			##EXAMPLES:
			{var}

			Now generate {n} new unique scenarios.
			Return ONLY the JSON array.
			"""


def extract_json(text):
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


async def generate_variations(input_dir, N, add_inplace):
    client = AsyncOpenAI()  # Uses API key from environment variable

    for file_path in Path(input_dir).rglob("*.json"):
        filename = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        variations = data.get("variations", [])
        category_name = data.get("name", "general")

        full_prompt = get_prompt(category_name, variations, N)

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.8,
        )

        content = response.choices[0].message.content.strip()
        content = extract_json(content)

        try:
            new_variations = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON for {filename}")
            logger.error(content)
            continue

        if not isinstance(new_variations, list):
            logger.error(f"Output is not a list for {filename}")
            logger.error(content)
            continue

        data["variations"] = variations + new_variations

        # add variant_name for each variation
        for i in range(len(data["variations"])):
            if "variant_name" not in data["variations"][i]:
                data["variations"][i]["variant_name"] = category_name + f"_{i}"
            elif f"_{i}" not in data["variations"][i]["variant_name"]:
                data["variations"][i]["variant_name"] += f"_{i}"

        if add_inplace:
            output_path = file_path
        else:
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(input_dir, f"{name}_augmented{ext}")

        # save file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"Processed {filename} -> {output_path}")
    logger.info("Finished!")


async def main_async():
    parser = argparse.ArgumentParser(
        description="Generate unique scenario variations for dialogue simulations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing existing scenarios.",
        required=True,
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of variations to generate."
    )
    parser.add_argument(
        "--add-inplace",
        type=bool,
        default=False,
        help="If True, the variations will be added directly to the files, modifying them inplace. If False, new files in corresponding directories will be used.",
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

    await generate_variations(args.input_dir, args.count, args.add_inplace)


def main():
    """Main entry point to run async code."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
