# TTS Voice Analysis

**Step 1** generates 2 ElevenLabs samples per character voice in
`dataset/config/voice_mapping.json`: a shared phoneme-rich passage (for acoustic/ML
voice comparison) and a short LLM-written character introduction.

**Step 2** (see `experiment-setup.md`) tests whether the ElevenLabs voices are genuinely
distinct speaker identities, not just acoustically different recordings: it samples
several lexically-varied utterances per voice, embeds them with two independent
speaker encoders, and compares intra-speaker vs. inter-speaker distances against a
natural multi-speaker reference corpus (steps 1-4 of the doc). It also runs the
channel-confound controls: synthetic channel perturbations, channel-normalized
replication, and interpretable acoustic descriptors (steps 5-7).

## Layout

All commands below assume you've `cd`'d into this directory:

```bash
cd experiments/tts_voice_analysis
```

Every script is run as a module (`python3 -m <package>.<name>`), not as a bare file,
so imports resolve correctly across the package split:

- `lib/` — shared library code (metrics, encoders, channel transforms, caching,
  provenance, API clients). Not runnable directly.
- `generate/` — produce a corpus: ElevenLabs samples/utterances, the VCTK and
  LibriTTS natural/reference sets, the two Qwen3-TTS systems.
- `channel/` — apply a channel transform (normalization or perturbation) to an
  existing corpus.
- `features/` — extract per-utterance features (speaker embeddings or interpretable
  acoustic descriptors) from a corpus.
- `analyze/` — compare corpora using the extracted features: diversity, channel
  confounds, voice quality/prosody.
- `pipeline/` — `run_pipeline.sh` (the end-to-end orchestrator), `pipeline_config.env`
  (single source of truth for run parameters), `write_pipeline_provenance.py`.
- `unofficial/` — ad hoc side experiments, not part of `run_pipeline.sh`.

## Setup

```bash
export ELEVENLABS_API_KEY=...
export OPENAI_API_KEY=...
pip install speechbrain praat-parselmouth  # encoders + F0/HNR (transformers/datasets/librosa already required)
```

## Reproduce everything in one command

```bash
./pipeline/run_pipeline.sh
```

Runs all 26 sub-steps below, parameterized by `pipeline/pipeline_config.env` (the
single source of truth for every run parameter — utterance counts, VCTK speaker
target, encoders, perturbation subset size). Writes `output/pipeline_provenance.json`
at the end: git commit, the config's own hash, and the sample sizes actually achieved
(VCTK exhausts its ~109-110 real speakers well before hitting most requested targets).
To reproduce `RESULTS.md` exactly, check out the `git_commit` recorded there and
re-run with `pipeline/pipeline_config.env` unchanged — **note this only works once
the scripts themselves are committed**, since `git_commit` alone doesn't capture
untracked files.

The sections below document each step individually (what it does, its own flags) for
running pieces standalone rather than the full pipeline.

### Custom-scale / smoke-test runs

Every `pipeline/pipeline_config.env` value is itself overridable via environment
variable, and output goes wherever `OUT` points (default `output/`), so a full
end-to-end run at a smaller scale doesn't require editing anything or touching real
results:

```bash
# Tiny wiring smoke test (1 of everything), written to output_test/ instead of output/
CONFIG_FILE=pipeline/pipeline_config.test.env OUT=output_test ./pipeline/run_pipeline.sh

# Custom scale without a separate config file - just override what you want
N_UTTERANCES_PER_VOICE=2 VCTK_N_SPEAKERS=5 QWEN_N_SPEAKERS=5 CHARACTER_LIMIT=5 \
    OUT=output_custom ./pipeline/run_pipeline.sh
```

`CHARACTER_LIMIT=N` restricts the 3 ElevenLabs-character-based steps
(`generate.diversity_utterances`, `generate.qwen_design_instructions`,
`generate.qwen_designed_speakers`) to the first N characters, consistently across
all three. `CHARACTER_NAMES="Timo Aisha"` restricts to specific named characters
instead (takes priority over `CHARACTER_LIMIT` if both are set). See the comment
block at the top of `pipeline/run_pipeline.sh` for the full list of knobs.

## Generate the full set (as in paper)

```bash
python3 -m generate.voice_samples
```

Outputs go to `output/`:
- `phoneme/` — one sample per unique voice_id
- `intro/` — one sample per character
- `intro_texts/` — cached LLM-generated intro text (editable, reused on rerun)
- `manifest.json` — per-character paths, description source, intro text
- `provenance.json` — git commit + file hashes of the inputs used

Already-generated files are skipped on rerun. Useful flags: `--dry-run`, `--limit N`,
`--character NAME` (repeatable), `--force` / `--force-text` to regenerate.

## Reproduce the samples "as in paper"

```bash
git checkout 6f444e5f3afaa0beae22a86e9357aca2d77f4a84  # experiments/el_voice_analysis
python3 experiments/tts_voice_analysis/generate_voice_samples.py  # pre-refactor path, at that commit
```

Same inputs + same `generation_settings` (prompt, models) reproduce the same intro
text and near-identical audio (ElevenLabs synthesis isn't bit-exact deterministic).
For any other run, check `git_commit` in that run's `output/provenance.json` instead.

## Step 2: voice-diversity analysis

```bash
# 1. N lexically-varied utterances per character (default 8), synthesized via ElevenLabs
python3 -m generate.diversity_utterances

# 2. Natural multi-speaker reference (VCTK, streamed - only pulls as many speakers as asked)
python3 -m generate.vctk_reference  # default: 150 requested (VCTK only has ~109-110, so this gets all of them)

# 3. Speaker embeddings (ECAPA-TDNN + WavLM-SV) for both corpora
python3 -m features.extract_embeddings \
    --diversity-manifest output/diversity_manifest.json \
    --output-dir output/embeddings
python3 -m features.extract_embeddings \
    --index output/vctk/index.json --audio-root output/vctk \
    --output-dir output/vctk_embeddings

# 4. Intra-/inter-speaker distances, collision check, identification accuracy, plots
python3 -m analyze.voice_diversity
```

Results land in `output/analysis/`: `<encoder>_summary.json` (the 4 distance
distributions + collision count + ID accuracy), `<encoder>_nearest_neighbor.json`
(closest other voice per character), `<encoder>_distributions.png`. Every step is
resumable (skips what's already generated/cached; `--force`/`--force-text` to redo).
All scripts support `--limit N` / `--n-speakers` / `--n-utterances` for a quick
smoke test before running the full 169-voice batch.

## Step 2b: channel-confound controls

```bash
# 5. Perturb a subset of voices (bandwidth, EQ, codec, noise, reverb) + embed + compare
python3 -m channel.apply_perturbations  # default: 20 voices x 3 utterances
python3 -m features.extract_embeddings \
    --index output/channel_variants/index.json --audio-root output/channel_variants \
    --output-dir output/channel_variants_embeddings
python3 -m analyze.channel_perturbation

# 6. Normalize channel (loudness + bandwidth) for both corpora + re-embed + compare
python3 -m channel.apply_normalization \
    --index output/embeddings/index.json --audio-root output \
    --output-dir output/diversity_normalized
python3 -m channel.apply_normalization \
    --index output/vctk/index.json --audio-root output/vctk \
    --output-dir output/vctk_normalized
python3 -m features.extract_embeddings \
    --index output/diversity_normalized/index.json --audio-root output/diversity_normalized \
    --output-dir output/embeddings_normalized
python3 -m features.extract_embeddings \
    --index output/vctk_normalized/index.json --audio-root output/vctk_normalized \
    --output-dir output/vctk_embeddings_normalized
python3 -m analyze.channel_normalization

# 7. Interpretable acoustic features (F0, HNR, jitter/shimmer, formants, bandwidth, SNR, ...)
python3 -m features.extract_acoustic_features \
    --index output/embeddings/index.json --audio-root output \
    --output output/acoustic_features/elevenlabs.csv
python3 -m features.extract_acoustic_features \
    --index output/vctk/index.json --audio-root output/vctk \
    --output output/acoustic_features/natural.csv
python3 -m analyze.acoustic_confound

# 7b. Same acoustic features for the Qwen systems (once generated, see below),
#     then the jitter/shimmer/formants/pitch-CV/voiced-fraction comparison across all 4
python3 -m features.extract_acoustic_features \
    --index output/qwen/index.json --audio-root output/qwen \
    --output output/acoustic_features/qwen_cloned.csv
python3 -m features.extract_acoustic_features \
    --index output/qwen_designed/index.json --audio-root output/qwen_designed \
    --output output/acoustic_features/qwen_designed.csv
python3 -m analyze.voice_quality_multi_system
```

- **Step 5** (`analyze/channel_perturbation.py`): for each perturbation, the "same
  voice, channel-shifted" distance should sit below the inter-speaker baseline
  (ideally near the intra-speaker baseline) — printed per perturbation with an
  OK/WARNING flag, plotted in `<encoder>_channel_perturbation.png`.
- **Step 6** (`analyze/channel_normalization.py`): compares the intra-/inter-speaker
  separation ratio unnormalized vs. normalized for both corpora; reports what
  fraction of the ElevenLabs corpus's separation is retained (HOLDS/WEAKENED/COLLAPSED).
- **Step 7** (`analyze/acoustic_confound.py`): prints acoustic feature means
  (ElevenLabs vs. Natural (VCTK)) and Pearson correlations between pairwise embedding
  distance and pairwise channel-feature differences — a low `|r|` (< 0.3, plotted
  in `acoustic_confound_correlations.png`) supports separation being identity-driven.
- **Step 7b** (`analyze/voice_quality_multi_system.py`): compares jitter, shimmer,
  formants (F1-F3), pitch coefficient of variation and voicing fraction across
  all 4 systems (ElevenLabs, Qwen (Cloned), Qwen (Designed), Natural (VCTK)) — box plots in
  `voice_quality_{jitter_shimmer,formants,pitch_cv,voiced_fraction}.png/.pdf`, summary
  stats in `voice_quality_multi_system.json`.

Correlation/collision numbers are only meaningful at real scale - on a small
`--limit`/`--n-voices` smoke test, `n` is tiny and these will look noisy; that's
expected, not a bug.
