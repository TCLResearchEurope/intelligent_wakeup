#!/bin/bash
# This code was developed by TCL Research Europe.
# Build script for speech synthesis with retry logic and configurable paths.

set -e  # Exit on error

# Exit cleanly on Ctrl+C or SIGTERM — don't retry
trap 'echo ""; echo "Interrupted, exiting."; exit 130' INT TERM

# Configuration
MAX_RETRIES=${MAX_RETRIES:-3}
RETRY_DELAY=${RETRY_DELAY:-10}  # seconds
GENERATION_TIMEOUT=${GENERATION_TIMEOUT:-7200}  # 2 hours per scenario

# Get parameters from environment variables (with sensible local defaults)
SCENARIOS=${SCENARIOS:-"all"}
INPUT_BASE_DIR=${INPUT_BASE_DIR:-"dataset/data/generated_text"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"dataset/data/generated_audio"}
CONFIG_DIR=${CONFIG_DIR:-"dataset/config"}
TTS_MODEL=${TTS_MODEL:-""}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Speech Corpora Generation"
echo "=========================================="
echo "Scenarios: $SCENARIOS"
echo "Input Directory: $INPUT_BASE_DIR"
echo "Output Directory: $OUTPUT_BASE_DIR"
echo "Config Directory: $CONFIG_DIR"
echo "TTS Model: ${TTS_MODEL:-"(from scenario config)"}"
echo "=========================================="

# Check for required API keys
if [ -z "$ELEVENLABS_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: At least one of ELEVENLABS_API_KEY or OPENAI_API_KEY must be set${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Track overall success/failure
TOTAL_SCENARIOS=0
SUCCESSFUL_SCENARIOS=0
FAILED_SCENARIOS=0
FAILED_SCENARIO_LIST=()

# Build optional TTS model arg
TTS_MODEL_ARG=""
if [ -n "$TTS_MODEL" ]; then
    TTS_MODEL_ARG="--tts-model $TTS_MODEL"
fi

# Function to generate audio for a single scenario
generate_single() {
    local scenario=$1
    local input_dir=$2
    local output_dir=$3
    local attempt=1

    echo ""
    echo -e "${YELLOW}>>> Generating audio: scenario=$scenario${NC}"
    echo "    Input:  $input_dir"
    echo "    Output: $output_dir"

    mkdir -p "$output_dir"

    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Attempt $attempt of $MAX_RETRIES..."

        # shellcheck disable=SC2086
        if timeout $GENERATION_TIMEOUT python -m dataset.generate_speech.generate_audio \
            --config-dir "$CONFIG_DIR" \
            --scenario "$scenario" \
            --input-dir "$input_dir" \
            --output-dir "$output_dir" \
            $TTS_MODEL_ARG \
            --log-level info; then
            echo -e "${GREEN}✓ Success: scenario=$scenario${NC}"
            return 0
        else
            exit_code=$?
            echo -e "${RED}✗ Failed with exit code $exit_code${NC}"

            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi

        attempt=$((attempt + 1))
    done

    echo -e "${RED}✗ Failed after $MAX_RETRIES attempts: scenario=$scenario${NC}"
    return 1
}

# Process scenarios
if [ "$SCENARIOS" = "all" ]; then
    # Extract scenario names from the JSON config files (the "name" field),
    # which matches how text corpora directories are named on NAS.
    mapfile -t SCENARIO_LIST < <(python3 -c "
import json, glob
names = set()
for f in glob.glob('$CONFIG_DIR/scenarios/**/*.json', recursive=True):
    try:
        d = json.load(open(f))
        if 'name' in d:
            names.add(d['name'])
    except Exception:
        pass
print('\n'.join(sorted(names)))
")
else
    IFS=',' read -ra SCENARIO_LIST <<< "$SCENARIOS"
fi

echo ""
echo "Will generate audio for ${#SCENARIO_LIST[@]} scenario(s)"
echo ""

# Generate audio for each scenario
for scenario in "${SCENARIO_LIST[@]}"; do
    scenario=$(echo "$scenario" | xargs)

    TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + 1))

    input_dir="$INPUT_BASE_DIR/$scenario"
    output_dir="$OUTPUT_BASE_DIR/$scenario"

    if [ ! -d "$input_dir" ]; then
        echo -e "${YELLOW}⚠ Input directory not found for scenario '$scenario': $input_dir — skipping${NC}"
        FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
        FAILED_SCENARIO_LIST+=("$scenario (no input)")
        continue
    fi

    if generate_single "$scenario" "$input_dir" "$output_dir"; then
        SUCCESSFUL_SCENARIOS=$((SUCCESSFUL_SCENARIOS + 1))
    else
        FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
        FAILED_SCENARIO_LIST+=("$scenario")
    fi
done

# Print summary
echo ""
echo "=========================================="
echo "Generation Summary"
echo "=========================================="
echo "Total scenarios: $TOTAL_SCENARIOS"
echo -e "${GREEN}Successful: $SUCCESSFUL_SCENARIOS${NC}"
if [ $FAILED_SCENARIOS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_SCENARIOS${NC}"
    echo ""
    echo "Failed scenarios:"
    for failed in "${FAILED_SCENARIO_LIST[@]}"; do
        echo "  - $failed"
    done
else
    echo "Failed: 0"
fi
echo "=========================================="

# Create generation metadata
METADATA_FILE="$OUTPUT_BASE_DIR/generation_metadata.json"
cat > "$METADATA_FILE" <<EOF
{
  "git_commit": "${CI_COMMIT_SHA:-unknown}",
  "git_commit_short": "${CI_COMMIT_SHORT_SHA:-unknown}",
  "git_branch": "${CI_COMMIT_BRANCH:-unknown}",
  "pipeline_id": "${CI_PIPELINE_ID:-unknown}",
  "job_id": "${CI_JOB_ID:-unknown}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scenarios_requested": "$SCENARIOS",
  "tts_model": "${TTS_MODEL:-"from_config"}",
  "input_base_dir": "$INPUT_BASE_DIR",
  "output_base_dir": "$OUTPUT_BASE_DIR",
  "total_scenarios": $TOTAL_SCENARIOS,
  "successful": $SUCCESSFUL_SCENARIOS,
  "failed": $FAILED_SCENARIOS,
  "failed_list": [$(printf '"%s",' "${FAILED_SCENARIO_LIST[@]}" | sed 's/,$//')]
}
EOF

echo "Metadata saved to: $METADATA_FILE"

# Non-fatal: let validation stage assess partial results
if [ $FAILED_SCENARIOS -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}WARNING: Some scenarios failed. Continuing to validation stage...${NC}"
fi

echo ""
echo -e "${GREEN}Speech generation completed.${NC}"
exit 0
