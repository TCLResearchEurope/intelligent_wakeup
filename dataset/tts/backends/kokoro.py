"""Copyright © 2026 TCL Research Europe.

SPDX-License-Identifier: Apache-2.0
"""

import random
import torch
import numpy as np
from kokoro import KPipeline


class Kokoro:
    """
    A class for generating text-to-speech (TTS) audio using the Kokoro model.

    Attributes:
        device (str): The device to run the model on, either "cuda" (GPU) or "cpu".
        pipeline (KPipeline): The KPipeline object for generating speech from text.
    """

    def __init__(self, language_code: str) -> None:
        """
        Initializes the Kokoro instance by selecting the model based on the specified language.

        Args:
            language_code (str): The language code for the TTS model ('es', 'pl', or 'en').
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline: KPipeline = None
        self.load_model(language_code=language_code)

    def load_model(self, language_code: str) -> None:
        """
        Loads the TTS model based on the provided language code.

        Args:
            language_code (str): The language code (e.g., 'en', 'es', 'pl') for selecting the model.

        Raises:
            ValueError: If the language code is not supported by the model.
        """
        language_codes_map: dict[str, str] = {
            "es": "e",
            "fr": "f",
            "hi": "h",
            "it": "i",
            "pt": "p",
            "en": "a",
            "jp": "j",
            "zh": "z",
        }

        if language_code not in language_codes_map:
            raise ValueError(
                f"Kokoro model does not handle specified language: {language_code}"
            )

        repo_id: str = "hexgrad/Kokoro-82M"
        if language_codes_map[language_code] == "zh":
            repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh"
        en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

        def en_callable(text: str):
            if text == "Kokoro":
                return "kˈOkəɹO"
            if text == "Sol":
                return "sˈOl"
            return next(en_pipeline(text)).phonemes

        self.pipeline = KPipeline(
            lang_code=language_codes_map[language_code],
            repo_id=repo_id,
            device=self.device,
            en_callable=en_callable,
        )

    def generate_audio(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
        """
        Generates speech from the input text using the specified voice and speed.

        Args:
            text (str): The text to be converted into speech.
            voice (str): Voice which is to be used for the synthesis.
            speed (float): Speed of the generated audio.

        Returns:
            tuple[bytes, int]: A tuple containing the generated audio and the sampling rate.
        """
        generator = self.pipeline(text, voice=voice, speed=speed, split_pattern="")

        all_audio = []
        for _, _, audio in generator:
            audio_array: bytes = (audio.numpy() * 32767).astype("int16")
            all_audio.append(audio_array)
        concatenated_audio = np.concatenate(all_audio)
        sample_rate: int = 24000
        return concatenated_audio, sample_rate


def generate_audio_adapter(model: Kokoro, text: str) -> tuple[bytes, int, dict[str]]:
    """
    Adapter function to generate audio using Kokoro with a randomly selected voice and speed.

    Args:
        model (Kokoro): The Kokoro instance.
        text (str): The text to be converted to speech.

    Returns:
        tuple[bytes, int, dict[str]]:
            A tuple containing the generated audio in bytes, the sampling rate and
            for the debugging purposes parameters provided for the synthesis.
    """

    voices: list[str] = [
        "af_maple",
        "af_sol",
        "bf_vale",
        "zf_001",
        "zf_002",
        "zf_003",
        "zf_004",
        "zf_005",
        "zf_006",
        "zf_007",
        "zf_008",
        "zf_017",
        "zf_018",
        "zf_019",
        "zf_021",
        "zf_022",
        "zf_023",
        "zf_024",
        "zf_026",
        "zf_027",
        "zf_028",
        "zf_032",
        "zf_036",
        "zf_038",
        "zf_039",
        "zf_040",
        "zf_042",
        "zf_043",
        "zf_044",
        "zf_046",
        "zf_047",
        "zf_048",
        "zf_049",
        "zf_051",
        "zf_059",
        "zf_060",
        "zf_067",
        "zf_070",
        "zf_071",
        "zf_072",
        "zf_073",
        "zf_074",
        "zf_075",
        "zf_076",
        "zf_077",
        "zf_078",
        "zf_079",
        "zf_083",
        "zf_084",
        "zf_085",
        "zf_086",
        "zf_087",
        "zf_088",
        "zf_090",
        "zf_092",
        "zf_093",
        "zf_094",
        "zf_099",
        "zm_009",
        "zm_010",
        "zm_011",
        "zm_012",
        "zm_013",
        "zm_014",
        "zm_015",
        "zm_016",
        "zm_020",
        "zm_025",
        "zm_029",
        "zm_030",
        "zm_031",
        "zm_033",
        "zm_034",
        "zm_035",
        "zm_037",
        "zm_041",
        "zm_045",
        "zm_050",
        "zm_052",
        "zm_053",
        "zm_054",
        "zm_055",
        "zm_056",
        "zm_057",
        "zm_058",
        "zm_061",
        "zm_062",
        "zm_063",
        "zm_064",
        "zm_065",
        "zm_066",
        "zm_068",
        "zm_069",
        "zm_080",
        "zm_081",
        "zm_082",
        "zm_089",
        "zm_091",
        "zm_095",
        "zm_096",
        "zm_097",
        "zm_098",
        "zm_100",
    ]

    voice: str = random.choice(voices)
    speed: float = random.random() * 0.6 + 0.7
    parameters: dict[str] = {"voice": voice, "speed": speed}
    return *model.generate_audio(text=text, voice=voice, speed=speed), parameters
