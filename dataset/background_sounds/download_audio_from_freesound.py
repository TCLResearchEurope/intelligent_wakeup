"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Replication script used to download background audio files in the paper.
"""

import json
import argparse
from pathlib import Path
import requests

API_BASE_URL = "https://freesound.org/apiv2"


def extract_sound_id(link: str) -> str:
    """
    Extract the sound ID from a Freesound link.

    Args:
        link (str): The Freesound URL of the audio file.

    Returns:
        str: The extracted sound ID.
    """
    return link.rstrip("/").split("/")[-1]


def download_sound(
    sound_id: str, output_dir: Path, token: str, orig_filename: str
) -> None:
    """
    Download a sound file from Freesound using its sound ID and save it with the original filename.

    Args:
        sound_id (str): The ID of the sound to download.
        output_dir (Path): The directory where the downloaded file will be saved.
        token (str): The access token for API authentication.
        orig_filename (str): The original filename for the downloaded sound.
    """
    url = f"{API_BASE_URL}/sounds/{sound_id}/download/"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers, stream=True, timeout=30)
    if response.status_code == 200:
        output_file = output_dir / orig_filename
        with open(output_file, "wb", encoding="utf-8") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {output_file}")
    else:
        print(f"Failed to download {sound_id}: {response.status_code} {response.text}")


def download_from_json(json_file: str, output_dir: str, token: str) -> None:
    """
    Download audio files from Freesound based on a JSON configuration file.

    Args:
        json_file (str): Path to the JSON file containing Freesound metadata.
        output_dir (str): Path to the directory where downloaded files will be stored.
        token (str): The access token for API authentication.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for audio_file in data["audio_files"]:
        link = audio_file["link"]
        orig_filename = audio_file["orig_filename"]
        sound_id = extract_sound_id(link)
        print(f"Processing: {link} (Sound ID: {sound_id})")
        download_sound(sound_id, output_path, token, orig_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download audio files from Freesound using an access token."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the JSON file containing Freesound metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where downloaded files will be stored.",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Access token for Freesound API authentication.",
    )
    args = parser.parse_args()

    download_from_json(args.json_file, args.output_dir, args.token)
