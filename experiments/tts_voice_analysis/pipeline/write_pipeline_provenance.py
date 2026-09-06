#!/usr/bin/env python3
"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Writes output/pipeline_provenance.json: the git commit and pipeline_config.env
values that produced the current output/ contents, plus the sample sizes
actually achieved (VCTK exhausts before reaching most requested speaker
counts, so "requested" and "achieved" can differ). Run automatically at the
end of run_pipeline.sh; safe to re-run standalone at any time.

CONFIG_PATH honors the CONFIG_FILE environment variable (set by
run_pipeline.sh) so a test run using pipeline_config.test.env records the
config it actually used, not always the production pipeline_config.env.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from lib.provenance import git_head_commit, git_repo_dirty, sha256_file
from lib.tts_common import DEFAULT_OUTPUT_DIR, REPO_ROOT

CONFIG_PATH = Path(
    os.environ.get("CONFIG_FILE") or (Path(__file__).resolve().parent / "pipeline_config.env")
)


def parse_env_file(path: Path) -> dict:
    """Parse a simple KEY=VALUE config file, ignoring comments and blank lines.

    Args:
        path: Path to the KEY=VALUE config file to parse.

    Returns:
        Mapping from key to its (quote-stripped) string value.
    """
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def main() -> None:
    """Assemble and write output/pipeline_provenance.json."""
    config = parse_env_file(CONFIG_PATH)
    output_dir = DEFAULT_OUTPUT_DIR

    diversity_manifest = json.loads((output_dir / "diversity_manifest.json").read_text())
    vctk_provenance = json.loads((output_dir / "vctk" / "provenance.json").read_text())

    achieved = {
        "n_elevenlabs_voices": len(diversity_manifest),
        "n_vctk_speakers_requested": vctk_provenance["n_speakers_requested"],
        "n_vctk_speakers_found": vctk_provenance["n_speakers_found"],
        "n_vctk_utterances_total": vctk_provenance["n_utterances_total"],
    }
    libritts_provenance_path = output_dir / "libritts_reference" / "provenance.json"
    if libritts_provenance_path.exists():
        libritts_provenance = json.loads(libritts_provenance_path.read_text())
        qwen_manifest_path = output_dir / "qwen" / "manifest.json"
        qwen_manifest = (
            json.loads(qwen_manifest_path.read_text()) if qwen_manifest_path.exists() else []
        )
        achieved.update(
            {
                "n_libritts_speakers_requested": libritts_provenance["n_speakers_requested"],
                "n_libritts_speakers_found": libritts_provenance["n_speakers_found"],
                "n_qwen_cloned_speakers": sum(1 for e in qwen_manifest if "audio_paths" in e),
            }
        )

    qwen_designed_manifest_path = output_dir / "qwen_designed" / "manifest.json"
    if qwen_designed_manifest_path.exists():
        qwen_designed_manifest = json.loads(qwen_designed_manifest_path.read_text())
        achieved["n_qwen_designed_voices"] = sum(
            1 for e in qwen_designed_manifest if "audio_paths" in e
        )

    record = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": git_head_commit(REPO_ROOT),
        "git_dirty": git_repo_dirty(REPO_ROOT),
        "config_file": CONFIG_PATH.name,
        "config_sha256": sha256_file(CONFIG_PATH),
        "config": config,
        "achieved": achieved,
    }

    out_path = output_dir / "pipeline_provenance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
