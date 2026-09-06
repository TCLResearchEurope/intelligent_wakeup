"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Divide longer audio files with background sounds into chunks of defined size (e.g. 1 minute).
"""

import argparse
from pathlib import Path
from pydub import AudioSegment


def process_audio_files(input_dir: str, output_dir: str) -> None:
    """
    Process audio files: convert to single channel, 44.1 kHz, 192 kbps, and split into 1-minute
    segments.

    Args:
        input_dir (str): Path to the directory containing input audio files.
        output_dir (str): Path to the directory to save processed audio files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    failed_files = []

    for audio_file in input_path.iterdir():
        if audio_file.suffix.lower() not in {".mp3", ".wav", ".flac"}:
            print(f"Skipping unsupported file: {audio_file}")
            continue

        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_file)
            original_sample_rate = audio.frame_rate
            original_bit_rate = audio.frame_width * 8 * audio.frame_rate

            # Check if the audio meets requirements
            if original_sample_rate < 44100 or original_bit_rate < 192000:
                print(f"File not suitable for reformatting: {audio_file.name}")
                failed_files.append(str(audio_file.name))
                continue

            # Reformat audio
            audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)

            # Split audio into 1-minute chunks
            duration_seconds = len(audio) // 1000
            chunks_count = duration_seconds // 60
            for i in range(chunks_count):
                chunk_start = i * 60 * 1000
                chunk_end = chunk_start + 60 * 1000
                chunk = audio[chunk_start:chunk_end]

                # Save chunk
                chunk_name = f"{audio_file.stem}_part{i+1:03d}of{chunks_count:03d}.wav"
                chunk_path = output_path / chunk_name
                chunk.export(chunk_path, format="wav", bitrate="192k")
                print(f"Saved: {chunk_path}")
        except Exception as e:
            print(f"Failed to process file {audio_file.name}: {e}")
            failed_files.append(str(audio_file.name))

    # Save failed files
    if failed_files:
        failed_log_path = output_path / "failed_files.txt"
        with open(failed_log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(failed_files))
        print(f"Saved failed files log: {failed_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat and split audio files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory with audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for processed files.",
    )
    args = parser.parse_args()

    process_audio_files(args.input_dir, args.output_dir)
