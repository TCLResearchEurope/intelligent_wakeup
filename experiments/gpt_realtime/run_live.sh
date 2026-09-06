#!/bin/bash
# Run OpenAI Realtime Virtual Assistant in LIVE mode
# This will use your system microphone for real-time conversation

set -e

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "Starting OpenAI Realtime Virtual Assistant in LIVE mode..."
echo "Press Ctrl+C to stop"
echo ""

uv run realtime_va.py --mode live "$@"
