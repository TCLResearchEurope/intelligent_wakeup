# Third-party notices

Material in this repository was produced using the corpora, audio and hosted
services below. This file records what was used and under which terms, and
flags the items that still need confirmation before redistribution.

## Speech corpora (referenced, not redistributed)

These are downloaded at run time by the pipeline; no audio from them is
committed to this repository.

| Corpus | Used by | License |
|---|---|---|
| LibriTTS-R (Koizumi et al., 2023), `train-clean-100` | `experiments/tts_voice_analysis/generate/libritts_reference.py` | CC BY 4.0 |
| VCTK (via the `sanchit-gandhi/vctk` mirror) | `experiments/tts_voice_analysis/generate/vctk_reference.py` | **Not recorded in this repository — confirm against the upstream corpus before relying on it.** |

## Freesound background audio

`dataset/background_sounds/freesound_db/` records sound IDs and metadata; the
audio itself is downloaded by `download_audio_from_freesound.py` and is not
committed here. Sounds referenced by `smart_cooking.json`:

| Sound | Author | Link |
|---|---|---|
| Cooking | aliha | https://freesound.org/people/aliha/sounds/143911/ |
| Kitchen sounds | leon_den_engelsen | https://freesound.org/people/leon_den_engelsen/sounds/275954/ |
| Kitchen Exhaust Fan Low Power Setting | Joao_Janz | https://freesound.org/people/Joao_Janz/sounds/473722/ |
| Cooking and Preparing Soup | SpaceJoe | https://freesound.org/people/SpaceJoe/sounds/484373/ |
| binaural_kitchen_peeling_cutting_frying_onions | vorreiter_sound | https://freesound.org/people/vorreiter_sound/sounds/501347/ |
| Cooking frying wok sizzle fan kitchen | TRP | https://freesound.org/people/TRP/sounds/577769/ |
| preparing breakfast | lonerdroner | https://freesound.org/people/lonerdroner/sounds/710641/ |
| random sound recorded in my kitchen | naotokui | https://freesound.org/people/naotokui/sounds/759115/ |

> **Unresolved.** Freesound licenses each sound individually (CC0, CC BY,
> CC BY-NC or Sampling+), and the metadata files here do not record which
> applies to each. Two consequences:
>
> 1. Sounds under CC BY require attribution wherever derived audio is
>    published; sounds under CC BY-NC additionally forbid commercial use.
> 2. `dataset/generate_final_audio_scenes/create_audio_scene.py` mixes a
>    background track into rendered scenes, so any such obligation can
>    propagate into published scene audio.
>
> Confirm the per-sound licenses, and confirm whether the committed
> `docs/assets/scenes/*.mp3` files embed this background audio, before
> redistributing them.

## Hosted model services

Speech and text in this dataset were generated using third-party services.
Their output remains subject to the provider's terms, independently of the
licenses in this repository:

- **ElevenLabs** — voice design and TTS (`dataset/tts/backends/elevenlabs.py`)
- **OpenAI** — text generation and TTS (`dataset/tts/backends/openai_*.py`)
- **Qwen3-TTS** — voice cloning and voice design, via an internally hosted
  server (`experiments/tts_voice_analysis/lib/qwen_tts_client.py`)

## Software dependencies

Dependencies declared in `requirements.txt`, `environment.yml` and
`experiments/*/requirements.txt` are installed from their upstream sources and
are not redistributed here. Each remains under its own license.
