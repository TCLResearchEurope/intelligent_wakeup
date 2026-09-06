#!/bin/bash

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Installing beehive-ai without dependencies..."
pip install beehive-ai==0.0.2 --no-deps

echo "Installing dependencies required by beehive-ai..."
pip install -r requirements_beehive.txt

echo "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo -e "${YELLOW}====================================${NC}"
echo -e "${RED}IMPORTANT NOTICE:${NC}"
echo -e "${YELLOW}You may see warnings about missing or incompatible dependencies for beehive-ai.${NC}"
echo -e "${YELLOW}These warnings are expected and safe to ignore, as the script bypasses${NC}"
echo -e "${YELLOW}beehive-ai's dependencies intentionally to resolve version conflicts.${NC}"
echo -e "${YELLOW}Please report any runtime issues if they arise!${NC}"
echo -e "${YELLOW}====================================${NC}"
echo ""

# Define checkpoint file/folder
CHECKPOINT_FOLDER="synthesize_speech/checkpoints_v2_0417"
CHECKPOINT_ARCHIVE="synthesize_speech/checkpoints_v2_0417.zip"
CHECKPOINT_URL="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

# Check if the folder already exists
if [ -d "$CHECKPOINT_FOLDER" ]; then
    echo -e "${GREEN}Checkpoints already downloaded and extracted at ${CHECKPOINT_FOLDER}.${NC}"
else
    echo "Downloading OpenVoice v2 checkpoints..."
    if [ ! -f "$CHECKPOINT_ARCHIVE" ]; then
        wget "$CHECKPOINT_URL" -O "$CHECKPOINT_ARCHIVE"
    else
        echo -e "${YELLOW}Archive $CHECKPOINT_ARCHIVE already exists, skipping download.${NC}"
    fi

    echo "Extracting checkpoints..."
    unzip -o "$CHECKPOINT_ARCHIVE" -d "$CHECKPOINT_FOLDER"
    echo -e "${GREEN}Checkpoints downloaded and extracted to ${CHECKPOINT_FOLDER}.${NC}"
fi
