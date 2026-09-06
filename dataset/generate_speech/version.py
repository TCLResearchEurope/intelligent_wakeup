"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Version management for speech generation pipeline.
"""

# Semantic versioning for speech generation code
# Format: MAJOR.MINOR.PATCH
#
# MAJOR: Breaking changes to output format or audio pipeline
# MINOR: New features, TTS model additions, non-breaking improvements (triggers regeneration)
# PATCH: Bug fixes, minor parameter tweaks (no regeneration required)
VERSION = "0.1.0"


def get_version() -> str:
    """
    Get the current version of the speech generation pipeline.

    Returns:
        Version string in semantic versioning format (MAJOR.MINOR.PATCH)
    """
    return VERSION


def get_version_tuple() -> tuple[int, int, int]:
    """
    Get the current version as a tuple of integers.

    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    major, minor, patch = VERSION.split(".")
    return int(major), int(minor), int(patch)


def should_regenerate(old_version: str, new_version: str) -> bool:
    """
    Determine if regeneration is required based on version change.

    Only MAJOR and MINOR version changes trigger regeneration.
    PATCH changes do not require regeneration.

    Args:
        old_version: Previous version string
        new_version: Current version string

    Returns:
        True if regeneration is required, False otherwise
    """
    try:
        old_major, old_minor, _ = map(int, old_version.split("."))
        new_major, new_minor, _ = map(int, new_version.split("."))
        return old_major != new_major or old_minor != new_minor
    except (ValueError, AttributeError):
        return True
