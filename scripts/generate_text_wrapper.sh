#!/bin/bash
# This code was developed by TCL Research Europe.
# Wrapper script for text generation with error handling and retry logic.

set -e  # Exit on error

# Configuration
MAX_RETRIES=3
RETRY_DELAY=10  # seconds
GENERATION_TIMEOUT=3600  # 1 hour per scenario

# Get parameters from environment variables
SCENARIOS=${SCENARIOS:-"all"}
CONVERSATION_TYPES=${CONVERSATION_TYPES:-"single_user"}
VARIANTS=${VARIANTS:-"all"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"dataset/data/generated_text"}
CONFIG_DIR=${CONFIG_DIR:-"dataset/config"}

echo "=========================================="
echo "Text Corpora Generation Wrapper"
echo "=========================================="
echo "Scenarios: $SCENARIOS"
echo "Conversation Types: $CONVERSATION_TYPES"
echo "Variants: $VARIANTS"
echo "Output Directory: $OUTPUT_BASE_DIR"
echo "Config Directory: $CONFIG_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Track overall success/failure
TOTAL_SCENARIOS=0
SUCCESSFUL_SCENARIOS=0
FAILED_SCENARIOS=0
FAILED_SCENARIO_LIST=()

# Function to generate for a single scenario and conversation type
generate_single() {
    local scenario=$1
    local conv_type=$2
    local variant=$3
    local attempt=1

    echo ""
    echo ">>> Generating: scenario=$scenario, type=$conv_type, variant=$variant"

    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Attempt $attempt of $MAX_RETRIES..."

        if timeout $GENERATION_TIMEOUT python -m dataset.generate_text.generate_text \
            --scenario "$scenario" \
            --variant "$variant" \
            --conversation-type "$conv_type" \
            --output-dir "$OUTPUT_BASE_DIR" \
            --config-dir "$CONFIG_DIR" \
            --log-level info; then
            echo "✓ Success: scenario=$scenario, type=$conv_type, variant=$variant"
            return 0
        else
            exit_code=$?
            echo "✗ Failed with exit code $exit_code"

            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi

        attempt=$((attempt + 1))
    done

    echo "✗ Failed after $MAX_RETRIES attempts: scenario=$scenario, type=$conv_type, variant=$variant"
    return 1
}

# Split comma-separated conversation types
IFS=',' read -ra CONV_TYPE_ARRAY <<< "$CONVERSATION_TYPES"

# Process scenarios
if [ "$SCENARIOS" = "all" ]; then
    # Get all scenario directories from config
    SCENARIO_DIRS=$(find "$CONFIG_DIR/scenarios" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
    SCENARIO_LIST=($SCENARIO_DIRS)
else
    # Split comma-separated scenarios
    IFS=',' read -ra SCENARIO_LIST <<< "$SCENARIOS"
fi

echo ""
echo "Will generate for ${#SCENARIO_LIST[@]} scenario(s) and ${#CONV_TYPE_ARRAY[@]} conversation type(s)"
echo ""

# Generate for each combination
for scenario in "${SCENARIO_LIST[@]}"; do
    for conv_type in "${CONV_TYPE_ARRAY[@]}"; do
        # Trim whitespace
        scenario=$(echo "$scenario" | xargs)
        conv_type=$(echo "$conv_type" | xargs)

        TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + 1))

        if generate_single "$scenario" "$conv_type" "$VARIANTS"; then
            SUCCESSFUL_SCENARIOS=$((SUCCESSFUL_SCENARIOS + 1))
        else
            FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
            FAILED_SCENARIO_LIST+=("$scenario/$conv_type")
        fi
    done
done

# Print summary
echo ""
echo "=========================================="
echo "Generation Summary"
echo "=========================================="
echo "Total combinations: $TOTAL_SCENARIOS"
echo "Successful: $SUCCESSFUL_SCENARIOS"
echo "Failed: $FAILED_SCENARIOS"

if [ $FAILED_SCENARIOS -gt 0 ]; then
    echo ""
    echo "Failed scenarios:"
    for failed in "${FAILED_SCENARIO_LIST[@]}"; do
        echo "  - $failed"
    done
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
  "conversation_types": "$CONVERSATION_TYPES",
  "variants": "$VARIANTS",
  "total_combinations": $TOTAL_SCENARIOS,
  "successful": $SUCCESSFUL_SCENARIOS,
  "failed": $FAILED_SCENARIOS,
  "failed_list": [$(printf '"%s",' "${FAILED_SCENARIO_LIST[@]}" | sed 's/,$//')]
}
EOF

echo "Metadata saved to: $METADATA_FILE"

# Exit with error if any scenario failed (but allow pipeline to continue)
if [ $FAILED_SCENARIOS -gt 0 ]; then
    echo ""
    echo "WARNING: Some scenarios failed to generate. Check logs above."
    echo "Continuing pipeline to validate partial results..."
    # Don't exit with error - let validation stage assess quality
fi

echo ""
echo "Generation wrapper completed."
exit 0
