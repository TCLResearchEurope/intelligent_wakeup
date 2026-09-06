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

Evaluates the model against a synthetic speech corpus by sending each non-VA utterance to the Realtime API and checking whether the model responds or stays silent.

### Setup

```bash
export OPENAI_API_KEY='your-api-key-here'
export INPUT_BASE_DIR=/path/to/text_corpora
export OUTPUT_BASE_DIR=/path/to/your/results
```

### Running

Evaluate a full scenario:
```bash
python evaluate_corpus.py \
  --text-corpora $INPUT_BASE_DIR \
  --scenario EducationLearning \
  --model gpt-realtime-1.5 \
  --output $OUTPUT_BASE_DIR/eval_EducationLearning.json
```

Evaluate a single conversation:
```bash
python evaluate_corpus.py \
  --text-corpora $INPUT_BASE_DIR \
  --file EducationLearning/couple/homework_help \
  --model gpt-realtime-1.5 \
  --output $OUTPUT_BASE_DIR/eval_homework_help.json
```

Resume an interrupted run:
```bash
python evaluate_corpus.py ... --resume
```

### Output

A single JSON file containing:
- `model` - model used
- `evaluated_at` - timestamp
- `dataset_versions` - per-scenario corpus version and content hashes
- `metrics` - TP/FP/TN/FN, precision, recall, F1 broken down by `overall` / `direct` / `contextual` / `non-assistance`
- `results` - per-turn details (speaker, text, model response, outcome)

### What is tested

Each non-Sigma turn in the corpus is sent as audio to the model. A turn **should trigger** the VA if the next turn in the corpus is spoken by Sigma (i.e. the VA was expected to respond). The model's response is classified as:

- **TP** - correctly responded when expected
- **TN** - correctly stayed silent when not expected
- **FP** - responded when it should have stayed silent
- **FN** - stayed silent when it should have responded

Turns are categorised by how the VA was invoked:

- **direct** - the utterance explicitly contains the word "Sigma"
- **contextual** - no explicit invocation, but the VA is expected to respond (follow-up in an ongoing exchange)
- **non-assistance** - VA response not expected; model should stay silent

After each evaluated turn the corpus Sigma response is injected as context, so the model accumulates conversation history naturally across the session.
