#!/bin/bash -ex

# Configuration
INPUT_DIR="../generate_text/output/"
OUTPUT_DIR="../dataset"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Merge synthesized conversation turns with background noise in virtual audio enviroment...${NC}"

python create_audio_scene.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --room-width 5.0 \
    --room-height 7.0 \
    --rt60 0.5

echo -e "${GREEN}Final audio generation completed successfully!${NC}"
echo "Generated files can be found in: $OUTPUT_DIR"
