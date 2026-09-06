"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Makes dialogue more natural through LLM-based hedging and micro-expansions.
"""

import re
from typing import Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from ...utils import logger


class Naturalizer:
    """Adds natural hesitations and expansions to dialogue using LLM."""

    def __init__(self, naturalization_rate: float = 0.4):
        """Initialize with naturalization probability."""
        load_dotenv()
        self.client = AsyncOpenAI()
        self.naturalization_rate = naturalization_rate

        self.system_prompt = """You are an expert in natural human dialogue. Your job is to make
conversations sound more human by adding subtle natural speech patterns.

GOAL: Add natural hesitations to rejections/counter-proposals and expand brief affirmations with micro-reasons.

RULES:
1. For rejections/disagreements (no, but, can't, etc.): Add brief hedges like "mm", "uh", "I guess", "maybe", "I mean"
2. For short affirmations (yes, ok, sure, etc.): Add tiny reasons like "— makes sense", "— good idea", "— that works", "— I think so", "— fair enough"
3. Keep changes SUBTLE - don't change meaning or add major content
4. Preserve the speaker's personality and tone
5. Don't hedge every response (aim for natural variation)
6. Skip if text already has natural hedging
7. AVOID repetitive phrases like "sounds good" - use variety in expansions
8. AVOID overly enthusiastic or positive language unless it fits the speaker's personality
9. REMOVE unnecessary agreement markers like "I like that!", "I agree", "That's great!" when the response can build on ideas directly
10. Let conversation flow naturally - people don't always need to explicitly validate before contributing

OUTPUT: Return ONLY the improved text, or return the original text unchanged if no
improvement is needed."""

    # A label the model may echo back from the prompt instead of answering with
    # the line alone, e.g.  Original text: "Sigma, what time is it?"
    _ECHOED_LABEL = re.compile(
        r'^\s*(?:original|revised|improved|new)?\s*'
        r'(?:text|line|version|output|response)\s*[:\-\u2013]\s*',
        re.IGNORECASE,
    )
    _QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb"

    @classmethod
    def _unwrap(cls, raw: str) -> str:
        """Strip an echoed prompt label and surrounding quotes from model output.

        Applied twice because the echoed label is itself sometimes quoted.
        Returns an empty string if nothing usable survives, which the caller
        treats as "leave the original line alone".
        """
        cleaned = (raw or "").strip()
        for _ in range(2):
            before = cleaned
            cleaned = cls._ECHOED_LABEL.sub("", cleaned, count=1).strip()
            if len(cleaned) >= 2 and cleaned[0] in cls._QUOTES:
                closing = {"\u201c": "\u201d", "\u2018": "\u2019", "\u00ab": "\u00bb"}.get(
                    cleaned[0], cleaned[0]
                )
                if cleaned.endswith(closing) and closing not in cleaned[1:-1]:
                    cleaned = cleaned[1:-1].strip()
            if cleaned == before:
                break
        return cleaned

    async def naturalize_message(
        self, text: str, context: str = "", speaker: str = ""
    ) -> Tuple[str, bool]:
        """
        Naturalize a single message using LLM.

        Args:
            text: The original dialogue text
            context: Optional context about the conversation
            speaker: Optional speaker name for personality consistency

        Returns:
            Tuple of (revised text, changed flag)
        """
        try:
            # Skip empty text
            if not text.strip():
                return text, False

            # Prepare user prompt.  The line is labelled but the model is told
            # plainly not to echo the label — gpt-4o-mini otherwise sometimes
            # replies with 'Original text: "..."' verbatim, and that string then
            # ends up spoken aloud by the TTS stage.
            user_prompt = f'Original text: "{text}"'
            if context:
                user_prompt += f"\nContext: {context}"
            if speaker:
                user_prompt += f"\nSpeaker: {speaker}"

            user_prompt += (
                "\n\nImprove this to sound more naturally human, or return unchanged "
                "if already natural. Reply with the line itself only: no label, no "
                "quotation marks, no commentary."
            )

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective for this task
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for consistent, subtle changes
                max_tokens=200,
            )

            improved_text = self._unwrap(response.choices[0].message.content)

            # A model that ignored the instruction entirely, or returned nothing
            # usable, must not be allowed to replace the line with rubbish.
            if not improved_text:
                return text, False

            # Check if text was actually changed
            changed = improved_text != text.strip()

            if changed:
                logger.debug(
                    "Naturalized: '%s' -> '%s'",
                    text[:50] + "..." if len(text) > 50 else text,
                    improved_text[:50] + "..."
                    if len(improved_text) > 50
                    else improved_text,
                )

            return improved_text, changed

        except Exception as e:
            logger.error("Error in LLM naturalization: %s", str(e))
            return text, False


async def naturalize_text(
    text: str, context: str = "", speaker: str = "", naturalization_rate: float = 0.4
) -> Tuple[str, bool]:
    """
    Convenience function for naturalizing dialogue with LLM.

    Args:
        text: The original dialogue text
        context: Optional conversation context
        speaker: Optional speaker name for personality consistency
        naturalization_rate: Probability of attempting naturalization (0.0 to 1.0)

    Returns:
        Tuple of (revised text, changed flag)
    """
    naturalizer = Naturalizer(naturalization_rate=naturalization_rate)
    return await naturalizer.naturalize_message(text, context, speaker)
