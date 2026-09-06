# Background Sound Processing Scripts

This repository contains two Python scripts designed for processing audio data:
1. `download_audio_from_freesound.py`: Downloads audio files from Freesound.org using their API and an access token.
2. `split_audio_to_fragments.py`: Splits long audio files into 1-minute segments while reformatting them as single-channel WAV files at 44.1 kHz and 192 kbps.

## Prerequisites

1. Install Python dependencies:
   ```bash
   pip install requests requests_oauthlib pydub
   ```
2. Ensure `ffmpeg` is installed on your system:
   - For Linux:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - For MacOS:
     ```bash
     brew install ffmpeg
     ```

## Usage

### 1. Download Audio from Freesound
The `download_audio_from_freesound.py` script downloads audio files specified in a JSON file using a valid Freesound API access token.

#### Example JSON File
```json
{
    "audio_files": [
        {
            "orig_filename": "143911__aliha__cooking.mp3",
            "link": "https://freesound.org/people/aliha/sounds/143911/"
        }
    ]
}
```

#### Run the Script
```bash
python download_audio_from_freesound.py \
    --json_file /path/to/json_file.json \
    --output_dir /path/to/output_directory \
    --token YOUR_ACCESS_TOKEN
```

- Replace `YOUR_ACCESS_TOKEN` with your valid access token obtained via the Freesound OAuth2 process.
- Audio files will be downloaded to the specified output directory with their original filenames.

### 2. Split Audio into Fragments
The `split_audio_to_fragments.py` script splits long audio files into 1-minute segments, reformats them to a single channel, and ensures a sample rate of 44.1 kHz and a bitrate of 192 kbps.

#### File Structure
Place your audio files in the input directory, e.g.:
```
/input_directory/
    143911__aliha__cooking.mp3
    275954__leon_den_engelsen__kitchen-sounds.wav
```

#### Run the Script
```bash
python split_audio_to_fragments.py \
    --input_dir /path/to/input_directory \
    --output_dir /path/to/output_directory
```

- The output directory will contain split audio files named in the format: `<original_name>_part001of005.wav`.
- Any files that cannot be processed due to incompatible formats will be logged in `failed_files.txt`.

## Notes
- Ensure you have a valid Freesound access token before running the download script.
- Check the JSON file format for the `download_audio_from_freesound.py` script to match your requirements.
- For any issues or further modifications, feel free to contribute to the repository or open an issue.
