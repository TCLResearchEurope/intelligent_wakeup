#!/bin/bash -ex

# Configuration
OUTPUT_DIR="dataset/data/generated_text"
CONFIG_DIR="dataset/config"
SCENARIOS_DIR="$CONFIG_DIR/scenarios"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting text generation process...${NC}"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the generate_text generation script for each scenario and conversation type
echo -e "${YELLOW}Generating conversations...${NC}"

# Generate conversations
python -m dataset.generate_text.generate_text \
    --scenario all \
    --conversation-type multi_user \
    --output-dir "$OUTPUT_DIR/multi_user" \
    --config-dir $CONFIG_DIR

echo -e "${GREEN}Completed multi-user conversations${NC}"

# Verify generation
echo -e "${YELLOW}Verifying generated files...${NC}"
EXPECTED_FILES=$(($(ls $SCENARIOS_DIR | wc -l) * 3))  # 3 conversation types for each scenario
ACTUAL_FILES=$(find "$OUTPUT_DIR" -type f -name "*.json" | wc -l)

if [ "$ACTUAL_FILES" -eq "$EXPECTED_FILES" ]; then
    echo -e "${GREEN}Successfully generated $ACTUAL_FILES conversation files${NC}"
else
    echo "Warning: Expected $EXPECTED_FILES files but found $ACTUAL_FILES"
    echo "Some scenarios might have failed to generate"
    exit 1
fi

# Create a summary file
echo -e "${YELLOW}Creating generation summary...${NC}"
cat << EOF > "$OUTPUT_DIR/generation_summary.json"
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "total_files": $ACTUAL_FILES,
    "scenarios": $(ls $SCENARIOS_DIR | wc -l),
    "conversation_types": 3
}

echo -e "${GREEN}Text generation completed successfully!${NC}"
echo "Generated files can be found in: $OUTPUT_DIR"
EOF
