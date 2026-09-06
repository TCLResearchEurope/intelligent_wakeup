#!/bin/bash -ex

echo "Starting Intelligent Wakeup dataset generation..."

echo "Step 1: Generating conversation texts..."
./dataset/generate_text/build.sh

echo "Step 2: Synthesizing speech..."
./dataset/generate_text/build.sh

echo "Step 3: Creating audio scenes..."
cd generate_final_audio_scenes
./build.sh
cd ..

echo "Dataset generation complete!"
