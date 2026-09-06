"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Script that merge all sounds to single audio file.
"""
#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf
import pyroomacoustics as pra

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> float:
    """Convert timestamp string to seconds"""
    try:
        # Remove any parenthetical time (like "(12:00)")
        ts = ts.split("(")[0].strip()
        # Remove 's' suffix and convert to float
        return float(ts.replace("s", ""))
    except Exception as e:
        logger.error("Error parsing timestamp %s: %s", ts, e)
        raise


def load_audio_file(file_path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file and return signal and sample rate"""
    try:
        signal, sr = sf.read(str(file_path))
        # Convert stereo to mono if necessary
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)
        # Convert to float32 to save memory
        return signal.astype(np.float32), sr
    except Exception as e:
        logger.error("Error loading audio file %s: %s", file_path, e)
        raise


def load_and_trim_audio(
    file_path: Path, max_duration: float, sample_rate: int
) -> np.ndarray:
    """Load and trim audio to a maximum duration."""
    signal, _ = load_audio_file(file_path)
    max_samples = int(max_duration * sample_rate)
    if len(signal) > max_samples:
        logger.warning("Trimming audio file %s to %s seconds", file_path, max_duration)
    return signal[:max_samples]  # Trim to maximum length


def create_room(room_dim: List[float], sample_rate: int) -> pra.Room:
    """Create a pyroomacoustics room with specified dimensions and reverberation time"""
    corners = np.array(
        [
            [0, 0],  # bottom-left
            [0, room_dim[1]],  # top-left
            [room_dim[0], room_dim[1]],  # top-right
            [room_dim[0], 0],  # bottom-right
        ]
    ).T

    room = pra.Room.from_corners(
        corners,
        fs=sample_rate,
        materials=pra.Material(energy_absorption=0.2),
        max_order=3,
        ray_tracing=False,
        air_absorption=False,
    )

    # Add the microphone array
    mic_loc = np.array([[room_dim[0] / 2], [room_dim[1] / 2]])  # Center of the room
    room.add_microphone(mic_loc)

    return room


def process_conversation(
    json_path: Path,
    output_dir: Path,
    room_dim: List[float] = None,
    rt60: float = 0.5,
    background_path: Path = None,
) -> None:
    """Process a single conversation JSON file and create the audio scene."""
    if room_dim is None:
        room_dim = [8.0, 6.0]

    try:
        # Load conversation JSON
        with open(json_path, encoding="utf-8") as f:
            conversation_data = json.load(f)

        # Determine audio directory from JSON path
        audio_dir = json_path.parent / json_path.stem
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # Load background sound first to determine duration and sample rate
        if background_path:
            background_signal, sample_rate = load_audio_file(background_path)
            max_duration = len(background_signal) / sample_rate
            logger.info("Using background duration: %s seconds", max_duration)
        else:
            # If no background, get sample rate from first audio file
            first_audio = list(audio_dir.glob("*.wav"))[0]
            _, sample_rate = load_audio_file(first_audio)
            # Set max duration based on last timestamp plus buffer
            max_timestamp = max(
                parse_timestamp(turn["timestamp"])
                for turn in conversation_data["conversation"]
            )
            max_duration = max_timestamp + 5.0  # Add 5-second buffer

        # Initialize output array
        total_samples = int(max_duration * sample_rate)
        output_signal = np.zeros(total_samples, dtype=np.float32)

        # Create room
        create_room(room_dim, sample_rate)

        # Process each turn separately to maintain correct timing
        for i, turn in enumerate(conversation_data["conversation"]):
            speaker = turn["speaker"]
            timestamp = parse_timestamp(turn["timestamp"])
            start_sample = int(timestamp * sample_rate)

            # Skip if timestamp is beyond max duration
            if start_sample >= total_samples:
                logger.warning(
                    "Skipping turn %s: timestamp %ss exceeds max duration %ss",
                    i + 1,
                    timestamp,
                    max_duration,
                )
                continue

            # Ambient segments are non-directional: insert them without room simulation
            if speaker == "background_noise":
                logger.debug("Processing background noise segment at %.2fs", timestamp)
                audio_file = audio_dir / f"turn_{i+1}-speaker_{speaker}.wav"
                signal, _ = load_audio_file(audio_file)

                end_sample = start_sample + len(signal)
                if end_sample > total_samples:
                    signal = signal[: total_samples - start_sample]
                    end_sample = total_samples

                output_signal[start_sample:end_sample] += signal
                continue

            # Define speaker positions
            if speaker == "User1":
                position = [1.0, room_dim[1] / 2]  # Left-middle
            elif speaker == "Assistant":
                position = [room_dim[0] - 1.0, room_dim[1] / 2]  # Right-middle
            else:
                position = [room_dim[0] / 2, room_dim[1] / 2]  # Center

            # Create a new room for each turn to maintain timing
            turn_room = create_room(room_dim, sample_rate)

            # Load and process audio for this turn
            audio_file = audio_dir / f"turn_{i+1}-speaker_{speaker}.wav"
            signal, _ = load_audio_file(audio_file)

            # Add the source to the turn's room
            turn_room.add_source(position, signal=signal)

            # Compute RIR and simulate for this turn
            turn_room.compute_rir()
            turn_room.simulate()

            # Add the simulated audio at the correct timestamp
            turn_signal = turn_room.mic_array.signals[0]
            end_sample = start_sample + len(turn_signal)
            if end_sample > total_samples:
                turn_signal = turn_signal[: total_samples - start_sample]
                end_sample = total_samples

            output_signal[start_sample:end_sample] += turn_signal

        # Add background sound if provided
        if background_path:
            background_room = create_room(room_dim, sample_rate)
            background_position = [room_dim[0] / 2, room_dim[1] - 1.0]  # Center-back
            background_signal = load_and_trim_audio(
                background_path, max_duration, sample_rate
            )
            background_room.add_source(background_position, signal=background_signal)
            background_room.compute_rir()
            background_room.simulate()

            # Mix background with conversation
            background_output = background_room.mic_array.signals[0]
            min_length = min(len(output_signal), len(background_output))
            output_signal[:min_length] += (
                background_output[:min_length] * 0.3
            )  # Reduce background volume

        # Normalize output
        max_amplitude = np.max(np.abs(output_signal))
        if max_amplitude > 0:
            output_signal = output_signal / max_amplitude * 0.9  # Leave some headroom

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the result
        output_file = output_dir / f"{json_path.stem}_scene.wav"
        sf.write(str(output_file), output_signal, sample_rate)
        logger.info("Created audio scene: %s", output_file)

    except Exception as e:
        logger.error("Error processing %s: %s", json_path, e)
        raise


def main():
    """
    Create audio scenes from input conversation files
    """
    parser = argparse.ArgumentParser(
        description="Create audio scenes from conversation files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing JSON conversation files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output audio scenes",
    )
    parser.add_argument(
        "--room-width", type=float, default=5.0, help="Room width in meters"
    )
    parser.add_argument(
        "--room-height", type=float, default=4.0, help="Room height in meters"
    )
    parser.add_argument(
        "--rt60",
        type=float,
        default=0.3,
        help="Desired RT60 (reverberation time) in seconds",
    )
    parser.add_argument(
        "--background-sound", type=Path, help="Path to a background sound file"
    )

    args = parser.parse_args()

    try:
        # Process all JSON files in input directory
        json_files = list(args.input_dir.glob("**/*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {args.input_dir}")

        for json_file in json_files:
            logger.info("Processing %s", json_file)
            process_conversation(
                json_file,
                args.output_dir,
                room_dim=[args.room_width, args.room_height],
                rt60=args.rt60,
                background_path=args.background_sound,
            )

    except Exception as e:
        logger.error("Error in main execution: %s", e)
        raise


if __name__ == "__main__":
    main()
