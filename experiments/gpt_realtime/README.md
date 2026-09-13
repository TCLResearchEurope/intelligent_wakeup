# OpenAI Realtime API Virtual Assistant Demo

Console-based virtual assistant with live microphone and batch WAV file testing modes.

## Setup

1. Install PortAudio (Ubuntu/Debian):
```bash
sudo apt-get install portaudio19-dev
```

2. Set API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Install dependencies:
```bash
uv sync
```

## Usage

**Live mode** (real-time microphone):
```bash
./run_live.sh
```

**Batch mode** (test with WAV files):
```bash
./run_batch.sh /path/to/wav/files
# Then enter filenames like: 1.wav, 2.wav, etc.
# Type 'EOD' to quit
```

## WAV File Requirements

- Format: PCM 16-bit, 24000 Hz, Mono

Convert with ffmpeg:
```bash
ffmpeg -i input.wav -ar 24000 -ac 1 -sample_fmt s16 output.wav
```

## Custom System Prompts

System prompts are loaded from `prompts/sigma_wakeup.txt` by default. Use `--prompt` to specify a different prompt file.

## Options

```bash
uv run realtime_va.py --mode {live|batch} --audio-dir DIR [--prompt FILE] [--verbose]
```

## Corpus Evaluation

The wakeup baseline. It pulls conversations from the published dataset
[`TCLResearchEurope/intelligent_wakeup`](https://huggingface.co/datasets/TCLResearchEurope/intelligent_wakeup),
cuts each session into turns at the recorded onsets, sends every non-assistant
turn to the Realtime API, and checks whether the model answers or stays silent.

Nothing but an API key is needed — the corpus is fetched from the Hub and
cached by `datasets`.

### Setup

```bash
export OPENAI_API_KEY='your-api-key-here'
pip install torch torchcodec --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt      # datasets, websockets, ...
```

`torchcodec` decodes the audio column and needs `torch` plus FFmpeg. Install
torch and torchcodec from the same index — the torchcodec wheel on PyPI links
against a CUDA build of torch and will not load beside a CPU one. The CPU build
is enough; nothing is trained here. On conda, take FFmpeg from conda-forge
(see `environment.yml`) rather than the system, or its newer `libstdc++`
requirement will clash with the one conda ships.

### Running

Evaluate the test split:
```bash
python evaluate_corpus.py \
  --split test \
  --model gpt-realtime-1.5 \
  --output results/eval_test.json
```

One category, or one conversation:
```bash
python evaluate_corpus.py --split test --category EducationLearning ...
python evaluate_corpus.py --id EducationLearning/couple/homework_help ...
```

Only the sessions that never address the assistant, which is where false
accepts come from:
```bash
python evaluate_corpus.py --split test --no-va --output results/eval_negatives.json
```

Stream rather than download the split first, and stop after a few
conversations — useful for a smoke test, since the test split is ~2 GB:
```bash
python evaluate_corpus.py --split test --streaming --limit 3 --output /tmp/smoke.json
```

Resume an interrupted run (skips turns already in the output file):
```bash
python evaluate_corpus.py ... --resume
```

### Dataset revision

`--revision` defaults to `v1.0.1`. **Do not evaluate against `v1.0.0`**: its
turn times were planned during text generation rather than measured from the
audio, so slicing a session by them yields the wrong audio for most turns. The
script warns when it sees onsets past the end of a session.

### Output

A single JSON file containing:
- `model`, `prompt`, `evaluated_at`
- `dataset` — repo, revision, split and any filters, so a run is reproducible
- `conversations` — how many sessions contributed
- `metrics` — TP/FP/TN/FN, precision, recall and F1 by `overall` / `direct` /
  `contextual` / `non-assistance`, plus `false_accepts`
- `results` — per-turn details (speaker, text, span, model response, outcome)

### What is tested

Each non-assistant turn is sent as audio. A turn **should trigger** the VA if
the next turn in the session is spoken by Sigma. Responses are classified as:

- **TP** — correctly responded when expected
- **TN** — correctly stayed silent when not expected
- **FP** — responded when it should have stayed silent
- **FN** — stayed silent when it should have responded

Turns are categorised by how the VA was invoked:

- **direct** — the utterance explicitly contains the word "Sigma"
- **contextual** — no explicit invocation, but the VA is expected to respond
  (a follow-up in an ongoing exchange)
- **non-assistance** — no response expected; the model should stay silent

`false_accepts` reports FPs per hour of audio the model should have ignored.
Counting false-positive *turns* alone understates the problem, because sessions
differ in how much speech they carry; an always-on assistant is judged per hour.

After each evaluated turn, the corpus Sigma response is injected as an
assistant message, so the model accumulates conversation history across the
session.

### A note on the audio

Turn *i* is the audio between its onset and the next turn's, and the final turn
runs to the end of the session. That slice comes from the mixed render, so it
carries room tone, background and — in scenarios with `overlapping_speech` — a
little of a neighbouring speaker. Earlier revisions of this script read clean,
isolated per-utterance wavs that only ever existed on the generating machine
and were never published. Expect slightly worse numbers here, and slightly more
honest ones.
