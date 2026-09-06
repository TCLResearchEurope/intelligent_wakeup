#!/usr/bin/env bash
# Copyright © 2026 TCL Research Europe.
# SPDX-License-Identifier: Apache-2.0
#
# Runs the full tts_voice_analysis pipeline end-to-end, parameterized entirely
# by pipeline_config.env (the single source of truth for reproducing
# RESULTS.md). See README.md for what each step produces.
#
# Requires ELEVENLABS_API_KEY and OPENAI_API_KEY in the environment.
#
# Usage:
#   ./pipeline/run_pipeline.sh
#
# Every pipeline_config.env value is itself overridable via environment variable
# (e.g. N_UTTERANCES_PER_VOICE=3, VCTK_N_SPEAKERS=10, QWEN_N_SPEAKERS=5 - see that
# file for the full list). A few extra knobs exist only here, not in the config:
#   CONFIG_FILE=pipeline/pipeline_config.test.env   # use a different config file entirely
#   OUT=output_custom                               # keep this run's output separate from output/
#   CHARACTER_LIMIT=5                                # restrict the 3 character-based steps
#                                                    # (diversity_utterances, qwen_design_instructions,
#                                                    # qwen_designed_speakers) to the first N
#                                                    # characters, the same N at every step
#   CHARACTER_NAMES="Timo Aisha"                     # ...or restrict to specific character(s) by
#                                                    # name instead (space-separated); takes
#                                                    # priority over CHARACTER_LIMIT if both are set
# Leave CHARACTER_LIMIT/CHARACTER_NAMES unset for a full run (all 169 characters).
#
# Example smoke test (tiny scale, separate output dir, one character):
#   CONFIG_FILE=pipeline/pipeline_config.test.env OUT=output_test ./pipeline/run_pipeline.sh
#
# Example custom-scale run without a separate config file:
#   N_UTTERANCES_PER_VOICE=2 VCTK_N_SPEAKERS=5 QWEN_N_SPEAKERS=5 CHARACTER_LIMIT=5 \
#       OUT=output_custom ./pipeline/run_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG_FILE="${CONFIG_FILE:-pipeline/pipeline_config.env}"
OUT="${OUT:-output}"
CHARACTER_LIMIT="${CHARACTER_LIMIT:-}"
CHARACTER_NAMES="${CHARACTER_NAMES:-}"
source "$CONFIG_FILE"
export CONFIG_FILE
export TTS_OUTPUT_DIR="$(pwd)/$OUT"

character_flag=()
if [ -n "$CHARACTER_NAMES" ]; then
    for name in $CHARACTER_NAMES; do
        character_flag+=(--character "$name")
    done
elif [ -n "$CHARACTER_LIMIT" ]; then
    character_flag=(--limit "$CHARACTER_LIMIT")
fi

step() { echo; echo "===== [$(date +%H:%M:%S)] $1 =====" ; }

step "1/26 generate.diversity_utterances"
python3 -m generate.diversity_utterances \
    --n-utterances "$N_UTTERANCES_PER_VOICE" --llm-model "$LLM_MODEL" --tts-model-id "$TTS_MODEL_ID" \
    "${character_flag[@]}"

step "2/26 generate.vctk_reference"
python3 -m generate.vctk_reference \
    --n-speakers "$VCTK_N_SPEAKERS" --n-utterances "$VCTK_N_UTTERANCES" --revision "$VCTK_REVISION"

step "3/26 features.extract_embeddings (elevenlabs)"
python3 -m features.extract_embeddings \
    --diversity-manifest "$OUT/diversity_manifest.json" \
    --output-dir "$OUT/embeddings" --encoders $ENCODERS

step "4/26 features.extract_embeddings (natural)"
python3 -m features.extract_embeddings \
    --index "$OUT/vctk/index.json" --audio-root "$OUT/vctk" \
    --output-dir "$OUT/vctk_embeddings" --encoders $ENCODERS

step "5/26 analyze.voice_diversity"
python3 -m analyze.voice_diversity --encoders $ENCODERS

step "6/26 channel.apply_perturbations"
python3 -m channel.apply_perturbations \
    --n-voices "$PERTURB_N_VOICES" --n-utterances-per-voice "$PERTURB_N_UTTERANCES_PER_VOICE"

step "7/26 features.extract_embeddings (channel variants)"
python3 -m features.extract_embeddings \
    --index "$OUT/channel_variants/index.json" --audio-root "$OUT/channel_variants" \
    --output-dir "$OUT/channel_variants_embeddings" --encoders $ENCODERS

step "8/26 analyze.channel_perturbation"
python3 -m analyze.channel_perturbation --encoders $ENCODERS

step "9/26 channel.apply_normalization (elevenlabs)"
python3 -m channel.apply_normalization \
    --index "$OUT/embeddings/index.json" --audio-root "$OUT" \
    --output-dir "$OUT/diversity_normalized"

step "10/26 channel.apply_normalization (natural)"
python3 -m channel.apply_normalization \
    --index "$OUT/vctk/index.json" --audio-root "$OUT/vctk" --output-dir "$OUT/vctk_normalized"

step "11/26 features.extract_embeddings (elevenlabs, normalized)"
python3 -m features.extract_embeddings \
    --index "$OUT/diversity_normalized/index.json" --audio-root "$OUT/diversity_normalized" \
    --output-dir "$OUT/embeddings_normalized" --encoders $ENCODERS

step "12/26 features.extract_embeddings (natural, normalized)"
python3 -m features.extract_embeddings \
    --index "$OUT/vctk_normalized/index.json" --audio-root "$OUT/vctk_normalized" \
    --output-dir "$OUT/vctk_embeddings_normalized" --encoders $ENCODERS

step "13/26 analyze.channel_normalization"
python3 -m analyze.channel_normalization --encoders $ENCODERS

step "14/26 features.extract_acoustic_features (elevenlabs)"
python3 -m features.extract_acoustic_features \
    --index "$OUT/embeddings/index.json" --audio-root "$OUT" \
    --output "$OUT/acoustic_features/elevenlabs.csv"

step "15/26 features.extract_acoustic_features (natural)"
python3 -m features.extract_acoustic_features \
    --index "$OUT/vctk/index.json" --audio-root "$OUT/vctk" \
    --output "$OUT/acoustic_features/natural.csv"

step "16/26 analyze.acoustic_confound"
python3 -m analyze.acoustic_confound

step "17/26 generate.libritts_reference"
python3 -m generate.libritts_reference --n-speakers "$QWEN_N_SPEAKERS"

step "18/26 generate.qwen_speakers (sequential Qwen3-TTS calls, slow)"
python3 -m generate.qwen_speakers \
    --n-utterances "$QWEN_N_UTTERANCES" --llm-model "$LLM_MODEL" --qwen-base-url "$QWEN_BASE_URL"

step "19/26 features.extract_embeddings (Qwen-cloned)"
python3 -m features.extract_embeddings \
    --index "$OUT/qwen/index.json" --audio-root "$OUT/qwen" \
    --output-dir "$OUT/qwen_embeddings" --encoders $ENCODERS

step "20/26 features.extract_acoustic_features (Qwen-cloned)"
python3 -m features.extract_acoustic_features \
    --index "$OUT/qwen/index.json" --audio-root "$OUT/qwen" \
    --output "$OUT/acoustic_features/qwen_cloned.csv"

step "21/26 generate.qwen_design_instructions"
python3 -m generate.qwen_design_instructions --llm-model "$QWEN_DESIGN_LLM_MODEL" \
    "${character_flag[@]}"

step "22/26 generate.qwen_designed_speakers (sequential Qwen3-TTS calls, slow)"
python3 -m generate.qwen_designed_speakers --qwen-base-url "$QWEN_BASE_URL" \
    "${character_flag[@]}"

step "23/26 features.extract_embeddings (Qwen-designed)"
python3 -m features.extract_embeddings \
    --index "$OUT/qwen_designed/index.json" --audio-root "$OUT/qwen_designed" \
    --output-dir "$OUT/qwen_designed_embeddings" --encoders $ENCODERS

step "24/26 features.extract_acoustic_features (Qwen-designed)"
python3 -m features.extract_acoustic_features \
    --index "$OUT/qwen_designed/index.json" --audio-root "$OUT/qwen_designed" \
    --output "$OUT/acoustic_features/qwen_designed.csv"

step "25/26 analyze.multi_system_diversity"
python3 -m analyze.multi_system_diversity --encoders $ENCODERS

step "26/26 analyze.voice_quality_multi_system"
python3 -m analyze.voice_quality_multi_system

step "writing pipeline_provenance.json"
python3 -m pipeline.write_pipeline_provenance

echo
echo "PIPELINE_COMPLETE $(date +%H:%M:%S)"
