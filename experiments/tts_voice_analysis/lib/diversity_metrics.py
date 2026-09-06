"""
Copyright © 2026 TCL Research Europe.
SPDX-License-Identifier: Apache-2.0

Shared speaker-embedding distance/centroid metrics, used by analyze/voice_diversity.py
(phase 1) and the channel-perturbation / channel-normalization analyses (phase 2) so
"within-voice" / "between-voice" / "collision" / "identification accuracy" all mean
exactly the same thing everywhere they're computed.
"""

import itertools
import json
from pathlib import Path

import numpy as np


def load_embeddings(embeddings_dir: Path, encoder: str) -> dict[str, dict[str, np.ndarray]]:
    """Load embedding vectors grouped by speaker/voice group.

    Args:
        embeddings_dir: Output directory of features/extract_embeddings.py, containing
            index.json and one .npz file per encoder.
        encoder: Name of the encoder whose embeddings to load (selects the .npz file).

    Returns:
        A mapping of {group: {utterance_id: vector}}.
    """
    index = json.loads((embeddings_dir / "index.json").read_text(encoding="utf-8"))
    with np.load(embeddings_dir / f"{encoder}.npz") as data:
        vectors = {k: data[k] for k in data.files}

    by_group: dict[str, dict[str, np.ndarray]] = {}
    for rec in index:
        uid = rec["utterance_id"]
        if uid not in vectors:
            continue
        by_group.setdefault(rec["group"], {})[uid] = vectors[uid]
    return by_group


def centroids_of(by_group: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Compute the mean-then-renormalize centroid for each group.

    Args:
        by_group: Mapping of {group: {utterance_id: vector}}.

    Returns:
        A mapping of {group: centroid vector}, each centroid L2-normalized.
    """
    out = {}
    for group, utterances in by_group.items():
        mean = np.mean(list(utterances.values()), axis=0)
        out[group] = mean / np.linalg.norm(mean)
    return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1 - cosine similarity for two already L2-normalized vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        The cosine distance between a and b.
    """
    return 1.0 - float(np.dot(a, b))


def within_group_distances(by_group: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Compute all pairwise utterance-level distances within each group, pooled across
    groups.

    Args:
        by_group: Mapping of {group: {utterance_id: vector}}.

    Returns:
        A flat array of cosine distances between all utterance pairs within each group.
    """
    distances = []
    for utterances in by_group.values():
        vecs = list(utterances.values())
        for a, b in itertools.combinations(vecs, 2):
            distances.append(cosine_distance(a, b))
    return np.array(distances)


def between_group_distances(centroids: dict[str, np.ndarray]) -> np.ndarray:
    """Compute all pairwise centroid-to-centroid distances between different groups.

    Args:
        centroids: Mapping of {group: centroid vector}.

    Returns:
        A flat array of cosine distances between every pair of group centroids.
    """
    vecs = list(centroids.values())
    return np.array([cosine_distance(a, b) for a, b in itertools.combinations(vecs, 2)])


def nearest_neighbor_analysis(
    centroids: dict[str, np.ndarray], collision_threshold: float
) -> list[dict]:
    """Find each group's closest other group centroid and flag possible collisions.

    Args:
        centroids: Mapping of {group: centroid vector}.
        collision_threshold: Distance at or below which two groups are flagged as a
            possible collision (i.e. not reliably distinguishable).

    Returns:
        A list of per-group records (nearest other group, distance, and whether it is
        a possible collision), sorted by ascending distance.
    """
    groups = list(centroids)
    results = []
    for g in groups:
        best_other, best_dist = None, float("inf")
        for other in groups:
            if other == g:
                continue
            d = cosine_distance(centroids[g], centroids[other])
            if d < best_dist:
                best_dist, best_other = d, other
        results.append(
            {
                "group": g,
                "nearest_other": best_other,
                "distance": best_dist,
                "possible_collision": best_dist <= collision_threshold,
            }
        )
    return sorted(results, key=lambda r: r["distance"])


def identification_accuracy(
    by_group: dict[str, dict[str, np.ndarray]], centroids: dict[str, np.ndarray]
) -> dict:
    """Compute the fraction of utterances whose nearest centroid (any group) is their
    own group.

    Args:
        by_group: Mapping of {group: {utterance_id: vector}}.
        centroids: Mapping of {group: centroid vector}.

    Returns:
        A dict with "accuracy" (fraction correctly identified, or None if there were no
        utterances) and "n" (total number of utterances evaluated).
    """
    groups = list(centroids)
    centroid_matrix = np.stack([centroids[g] for g in groups])
    correct, total = 0, 0
    for true_group, utterances in by_group.items():
        for vec in utterances.values():
            sims = centroid_matrix @ vec
            predicted = groups[int(np.argmax(sims))]
            correct += int(predicted == true_group)
            total += 1
    return {"accuracy": correct / total if total else None, "n": total}


def summarize(dist: np.ndarray) -> dict:
    """Compute basic distribution summary statistics.

    Args:
        dist: Array of distance values to summarize.

    Returns:
        A dict of summary stats (n, mean, std, median, p5, p95), or just {"n": 0} if
        dist is empty.
    """
    if len(dist) == 0:
        return {"n": 0}
    return {
        "n": len(dist),
        "mean": float(np.mean(dist)),
        "std": float(np.std(dist)),
        "median": float(np.median(dist)),
        "p5": float(np.percentile(dist, 5)),
        "p95": float(np.percentile(dist, 95)),
    }


def fmt_stat(stat: dict, key: str, spec: str = ".3f") -> str:
    """Format one summarize() stat for printing, tolerating a missing/empty result.

    At very small sample sizes (e.g. a single-voice/single-utterance smoke test)
    within- or between-group distances can be empty, so summarize() returns just
    {"n": 0} with no "mean"/"median" - formatting that missing value directly
    would raise TypeError instead of printing something readable.

    Args:
        stat: A dict as returned by summarize().
        key: Which stat to format (e.g. "mean", "median").
        spec: Format spec to apply to the value, if present.

    Returns:
        The formatted value, or "n/a" if key is missing from stat.
    """
    value = stat.get(key)
    return f"{value:{spec}}" if value is not None else "n/a"
