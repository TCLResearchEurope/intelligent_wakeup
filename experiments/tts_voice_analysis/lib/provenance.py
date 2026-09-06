"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Reproducibility helpers for generate/voice_samples.py.

Records exactly which version of voice_mapping.json and the character backstory
files were used to generate a batch of audio samples, so results can be traced
back to (and reproduced from) a specific git commit even if those files change
later. Falls back to content hashing when a file is uncommitted or the repo
state can't be determined, so provenance is always recorded even mid-experiment.
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Optional


def _run_git(repo_root: Path, *args: str) -> Optional[str]:
    """Run a git command in repo_root.

    Args:
        repo_root: Path to the git repository to run the command in.
        *args: Git subcommand and its arguments (e.g. "rev-parse", "HEAD").

    Returns:
        The command's stripped stdout, or None if git failed, wasn't found, or timed
        out.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_head_commit(repo_root: Path) -> Optional[str]:
    """Get the repo's current HEAD commit SHA.

    Args:
        repo_root: Path to the git repository.

    Returns:
        The current HEAD commit SHA, or None if repo_root is outside a git repo.
    """
    return _run_git(repo_root, "rev-parse", "HEAD")


def git_repo_dirty(repo_root: Path) -> bool:
    """Check whether the repo has any uncommitted changes.

    Args:
        repo_root: Path to the git repository.

    Returns:
        True if the repo has any uncommitted changes (tracked or untracked).
    """
    status = _run_git(repo_root, "status", "--porcelain")
    return bool(status)


def git_last_commit_for_path(repo_root: Path, path: Path) -> Optional[str]:
    """Get the SHA of the last commit that touched path.

    Args:
        repo_root: Path to the git repository.
        path: File or directory whose commit history to look up.

    Returns:
        The SHA of the last commit that touched path, or None if it was never committed.
    """
    rel = str(path.relative_to(repo_root))
    return _run_git(repo_root, "log", "-1", "--format=%H", "--", rel)


def git_path_dirty(repo_root: Path, path: Path) -> bool:
    """Check whether path has uncommitted changes.

    Args:
        repo_root: Path to the git repository.
        path: File or directory to check.

    Returns:
        True if path has uncommitted changes vs HEAD, or is untracked/new.
    """
    rel = str(path.relative_to(repo_root))
    status = _run_git(repo_root, "status", "--porcelain", "--", rel)
    return status is not None and len(status) > 0


def sha256_file(path: Path) -> str:
    """Hash a file's contents, independent of git state.

    Args:
        path: File to hash.

    Returns:
        The file's content hash, as a hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hash an in-memory string (e.g. a hardcoded prompt).

    Args:
        text: Text to hash.

    Returns:
        The text's content hash, as a hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_provenance(repo_root: Path, path: Path) -> Optional[dict]:
    """Build a provenance record for a single file.

    Args:
        repo_root: Path to the git repository containing path.
        path: File to record provenance for.

    Returns:
        A dict with path, sha256, git_last_commit, and git_dirty for the file, or None
        if the file is missing.
    """
    if not path.exists():
        return None
    return {
        "path": str(path.relative_to(repo_root)),
        "sha256": sha256_file(path),
        "git_last_commit": git_last_commit_for_path(repo_root, path),
        "git_dirty": git_path_dirty(repo_root, path),
    }


def directory_provenance(repo_root: Path, directory: Path, pattern: str = "*") -> dict:
    """Aggregate provenance for a directory.

    The combined hash changes if any file inside is added, removed, or edited,
    without needing per-file git history for a coarse "did anything change" check.

    Args:
        repo_root: Path to the git repository containing directory.
        directory: Directory to record provenance for.
        pattern: Glob pattern selecting which files inside directory to hash.

    Returns:
        A dict with path, n_files, a combined content hash of all matching files, and
        the directory's own git_last_commit/git_dirty state.
    """
    files = {}
    for path in sorted(directory.glob(pattern)):
        if path.is_file():
            files[path.name] = sha256_file(path)
    combined = "\n".join(f"{name}:{h}" for name, h in sorted(files.items()))
    return {
        "path": str(directory.relative_to(repo_root)),
        "n_files": len(files),
        "combined_sha256": sha256_text(combined),
        "git_last_commit": git_last_commit_for_path(repo_root, directory),
        "git_dirty": git_path_dirty(repo_root, directory),
    }


def build_run_provenance(repo_root: Path, config_dir: Path) -> dict:
    """Build the top-level provenance record for one generation run.

    Args:
        repo_root: Path to the git repository the run was executed from.
        config_dir: Path to the config directory holding voice_mapping.json and the
            characters directory.

    Returns:
        A dict combining the repo's git state with provenance for voice_mapping.json
        and the characters directory.
    """
    return {
        "repo_root": str(repo_root),
        "git_commit": git_head_commit(repo_root),
        "git_dirty": git_repo_dirty(repo_root),
        "voice_mapping": file_provenance(repo_root, config_dir / "voice_mapping.json"),
        "characters_dir": directory_provenance(repo_root, config_dir / "characters", "*.txt"),
    }
