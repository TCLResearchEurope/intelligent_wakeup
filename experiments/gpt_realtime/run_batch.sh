#!/bin/bash
# Run OpenAI Realtime Virtual Assistant in BATCH mode
# This allows you to test with pre-recorded WAV files

set -e

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Default audio directory
AUDIO_DIR="${1:-./test_audio}"

if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Audio directory not found: $AUDIO_DIR"
    echo ""
    echo "Usage: $0 [audio_directory]"
    echo "Example: $0 /path/to/wav/files"
    exit 1
fi

echo "Starting OpenAI Realtime Virtual Assistant in BATCH mode..."
echo "Audio directory: $AUDIO_DIR"
echo ""

uv run realtime_va.py --mode batch --audio-dir "$AUDIO_DIR" "${@:2}"
