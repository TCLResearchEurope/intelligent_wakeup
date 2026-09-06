# Intelligent Wakeup Paper

This repository contains the code to reproduce results from the paper "Intelligent Wakeup for Seamless Virtual Assistant". It provides a modular framework for dataset generation, model training, and experiments.

## Project Structure

```
.
├── dataset                  # Scripts and utilities for dataset creation
│   ├── generate_text         # Text generation using LLM
│   ├── synthesize_speech     # Speech synthesis (Kokoro, ElevenLabs, OpenAI)
│   ├── create_audio_scenes   # Audio scene generation and manipulation
│   ├── generate_dataset.sh   # Entry point for dataset generation
│   └── README.md             # Instructions for dataset generation
├── models                   # Pre-trained models and model training scripts
│   └── README.md             # Details on models and usage
├── experiments              # Scripts to reproduce paper experiments
│   └── README.md             # Experiment setup and execution guide
├── environment.yml          # Base environment definition for the repository
└── README.md                # Root repository documentation
```

## Introduction

This repository is designed with modularity in mind. Each component (`dataset`, `models`, `experiments`) operates independently, with clear interfaces and separate dependencies:

- **Dataset Module:** Code for creating, preprocessing, and uploading datasets. Includes modules for text generation, speech synthesis, and audio scene generation.
- **Models Module:** Pre-trained models and scripts for training or fine-tuning models.
- **Experiments Module:** Scripts to run experiments, utilizing datasets and models.

Each module has:
- Its own `requirements.txt` for managing dependencies.
- A dedicated `README.md` with usage instructions and module-specific details.

## Installation

### 1. Prepare the virtual environment

```sh
micromamba env create -f environment.yml
micromamba activate intelligent_wakeup
```

### 2. Install Python dependencies

```sh
pip install -r requirements.txt
```

### 3. Export API keys

```sh
export ELEVENLABS_API_KEY=<your_elevenlabs_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
```

### 4. (Optional) Authenticate with Hugging Face

Agree to the [terms of Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0), then log in:

```sh
huggingface-cli login
```

## Running

Run the whole pipeline
```sh
./dataset/generate_dataset.sh
```

Run text dialogs generation
```sh
./dataset/generate_text/build.sh
```

Run dialogs synthesis
```sh
./dataset/generate_speech/build.sh
```

## Audio Synthesis Models

Currently following TTS models are supported:
  - `ElevenLabs`
  - `OpenAI_TTS`
  - `OpenAI_Audio`
  - `OpenAI_Audio_mini`
  - `Kokoro`

and following sound effects models:
  - `ElevenLabs`
  - `StableAudio`

One shall specify which model to use in json files from `config/scenarios` directory.

## Issues
- For now only single utterances are provided to be synthesized, which may result in inconsistency across the whole dialog.
Maybe it would be worth considering providing the whole preceeding dialog to the OpenAI's audio model.
- ElevenLabs may generate sound effect of length only up to 22 seconds while Stable Audio Open up to 74 seconds.


## Citation

If you use this code or dataset, please cite:
```
@inproceedings{sowanski2026intelligentwakeup,
  title={Training Intelligent Voice Assistant Wakeup with Controllable Synthetic Conversations},
  author={Sowa{\'n}ski, Marcin and Leszczy{\'n}ski, Kacper and Krzywicki, Kacper and Wodnicki, Krzysztof},
  year={2026},
}
```

## License

This repository is released under two licenses, which is the usual split for a
paper artifact that ships both code and data:

| Part | License |
|---|---|
| Source code | [Apache License 2.0](LICENSE) |
| Dataset artifacts — character and scenario configurations, generated audio, demo assets | [CC BY-NC 4.0](LICENSE-DATA) |

In short: the code may be used commercially, with a patent grant and the
attribution requirements of Apache-2.0. The dataset and generated audio are
for **non-commercial use** — research, teaching and evaluation — and require
attribution.

Source files carry an `SPDX-License-Identifier` header. Files without one fall
under the split described in [NOTICE](NOTICE).

Third-party corpora, background audio and hosted TTS services carry their own
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Note in particular
that the Freesound background sounds are licensed individually and their
per-sound licenses are not yet recorded.
