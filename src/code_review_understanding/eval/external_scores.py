from __future__ import annotations

import json
from pathlib import Path

from code_review_understanding.data.io_utils import open_maybe_gzip


def load_score_feature_map(path: str | Path) -> dict[str, dict[str, float]]:
    cache_path = Path(path)
    with open_maybe_gzip(cache_path, "rt") as handle:
        payload = json.load(handle)

    scores = payload["scores"]
    sample_ids = payload["sample_ids"]
    context_ids_by_group = payload["context_ids_by_group"]
    group_sizes = payload["group_sizes"]

    feature_map: dict[str, dict[str, float]] = {}
    offset = 0
    for sample_id, context_ids, group_size in zip(
        sample_ids,
        context_ids_by_group,
        group_sizes,
        strict=True,
    ):
        next_offset = offset + int(group_size)
        feature_map[sample_id] = {
            context_id: float(score)
            for context_id, score in zip(context_ids, scores[offset:next_offset], strict=True)
        }
        offset = next_offset
    return feature_map


def _load_score_cache_payload(path: str | Path) -> dict:
    """Load a score cache while retaining ordering metadata for validation."""
    cache_path = Path(path)
    with open_maybe_gzip(cache_path, "rt") as handle:
        return json.load(handle)


def augment_feature_rows_with_external_scores(
    feature_rows_by_sample_id: dict[str, list[dict]] | None,
    *,
    score_cache_path: str | Path,
    feature_name: str,
) -> dict[str, list[dict]]:
    if feature_rows_by_sample_id is None:
        raise ValueError("external features require feature_row_cache to be loaded first")

    # Keep the cache's ordered context lists available for a strict alignment
    # check.  Mapping only by context_id can silently accept reordered rows (or
    # a cache with a missing/extra sample), which is especially dangerous when
    # score caches are regenerated under a new dataset namespace.
    payload = _load_score_cache_payload(score_cache_path)
    sample_ids = [str(value) for value in payload.get("sample_ids", [])]
    context_ids_by_group = payload.get("context_ids_by_group", [])
    group_sizes = [int(value) for value in payload.get("group_sizes", [])]
    scores = payload.get("scores", [])
    if len(sample_ids) != len(context_ids_by_group) or len(sample_ids) != len(group_sizes):
        raise ValueError(f"Malformed score cache metadata: {score_cache_path}")
    if len(scores) != sum(group_sizes):
        raise ValueError(f"Score cache length does not match group_sizes: {score_cache_path}")

    score_feature_map = load_score_feature_map(score_cache_path)
    expected_sample_ids = list(feature_rows_by_sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Score cache contains duplicate sample_ids: {score_cache_path}")
    missing = sorted(set(expected_sample_ids) - set(sample_ids))[:5]
    if missing:
        raise ValueError(
            f"Score cache sample_ids do not align for {feature_name}; "
            f"missing={missing}"
        )

    cache_context_ids = {
        sample_id: [str(value) for value in context_ids]
        for sample_id, context_ids in zip(sample_ids, context_ids_by_group, strict=True)
    }
    for sample_id, feature_rows in feature_rows_by_sample_id.items():
        sample_scores = score_feature_map.get(sample_id)
        if sample_scores is None:
            raise KeyError(f"Missing score-cache sample_id={sample_id} for feature '{feature_name}'")
        expected_context_ids = [str(row["context_id"]) for row in feature_rows]
        if cache_context_ids[sample_id] != expected_context_ids:
            raise ValueError(
                f"Score cache context ordering does not align for sample_id={sample_id} "
                f"feature '{feature_name}'"
            )
        for row in feature_rows:
            context_id = row["context_id"]
            if context_id not in sample_scores:
                raise KeyError(
                    f"Missing score-cache context_id={context_id} for sample_id={sample_id} feature '{feature_name}'"
                )
            row["features"][feature_name] = float(sample_scores[context_id])
    return feature_rows_by_sample_id


def validate_feature_rows_have_features(
    feature_rows_by_sample_id: dict[str, list[dict]] | None,
    *,
    required_feature_names: list[str],
) -> None:
    if feature_rows_by_sample_id is None:
        return

    for sample_id, feature_rows in feature_rows_by_sample_id.items():
        for row in feature_rows:
            missing = [name for name in required_feature_names if name not in row["features"]]
            if missing:
                raise KeyError(
                    f"Missing required features for sample_id={sample_id} context_id={row['context_id']}: {missing}"
                )
