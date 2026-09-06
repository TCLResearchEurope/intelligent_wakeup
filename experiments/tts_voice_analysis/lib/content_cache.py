"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Lightweight content-hash-based cache invalidation shared by every pipeline
stage that would otherwise cache purely by name/utterance_id and so silently
serve stale output after an upstream input changes (e.g. editing a voice_id
in voice_mapping.json). Each stage keeps a small JSON sidecar next to its real
output, mapping a cache key to a hash of whatever inputs determined that
output; on the next run, if the current hash of those inputs doesn't match,
the cached output is stale and gets regenerated - no manual --force needed.
"""

import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    """Compute the content hash of a file.

    Reads the file in chunks so large audio files don't need a full RAM load.

    Args:
        path: Path to the file to hash.

    Returns:
        The hex-encoded SHA-256 digest of the file's contents.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(*parts: str) -> str:
    """Compute the content hash of one or more strings (e.g. voice_id + model + text).

    Args:
        parts: Strings to hash together, joined with "|" before hashing.

    Returns:
        The hex-encoded SHA-256 digest of the joined parts.
    """
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load_hash_cache(path: Path) -> dict:
    """Load a {key: hash} sidecar file.

    Args:
        path: Path to the sidecar JSON file.

    Returns:
        The cached {key: hash} mapping, or {} if the sidecar doesn't exist yet.
    """
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_hash_cache(path: Path, cache: dict) -> None:
    """Write a {key: hash} sidecar file.

    Args:
        path: Path to the sidecar JSON file to write.
        cache: The {key: hash} mapping to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
        f.write("\n")
