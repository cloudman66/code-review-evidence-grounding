from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from code_review_understanding.eval.baseline import build_context_feature_rows, extract_context_path
from code_review_understanding.data.io_utils import load_jsonl, write_jsonl


def build_feature_matrix(
    feature_rows: list[dict],
    *,
    base_feature_names: list[str] | None = None,
    feature_transform: str = "raw_plus_relative",
) -> tuple[np.ndarray, list[str], list[str]]:
    if not feature_rows:
        return np.zeros((0, 0), dtype=np.float32), [], []

    resolved_base_names = list(base_feature_names or feature_rows[0]["features"].keys())
    raw_matrix = np.asarray(
        [
            [float(row["features"].get(feature_name, 0.0)) for feature_name in resolved_base_names]
            for row in feature_rows
        ],
        dtype=np.float32,
    )

    if feature_transform == "raw":
        return raw_matrix, resolved_base_names, list(resolved_base_names)

    if feature_transform not in {
        "raw_plus_relative",
        "raw_plus_relative_plus_context_buckets",
        "raw_plus_relative_plus_file_context",
        "raw_plus_relative_plus_context_buckets_plus_file_context",
        "raw_plus_relative_plus_context_slices_plus_file_context",
    }:
        raise ValueError(f"Unsupported feature transform: {feature_transform}")

    transformed_names = list(resolved_base_names)
    blocks: list[np.ndarray] = [raw_matrix]

    max_values = raw_matrix.max(axis=0)
    safe_max = np.where(max_values > 0.0, max_values, 1.0)
    max_norm = raw_matrix / safe_max
    blocks.append(max_norm.astype(np.float32))
    transformed_names.extend(f"{feature_name}_max_norm" for feature_name in resolved_base_names)

    if len(feature_rows) == 1:
        rank_pct = np.ones_like(raw_matrix, dtype=np.float32)
    else:
        rank_pct = np.zeros_like(raw_matrix, dtype=np.float32)
        denom = float(len(feature_rows) - 1)
        for column_index in range(raw_matrix.shape[1]):
            order = np.argsort(-raw_matrix[:, column_index], kind="stable")
            rank_pct[order, column_index] = 1.0 - (
                np.arange(len(feature_rows), dtype=np.float32) / denom
            )
    blocks.append(rank_pct.astype(np.float32))
    transformed_names.extend(f"{feature_name}_rank_pct" for feature_name in resolved_base_names)

    if feature_transform in {
        "raw_plus_relative_plus_context_buckets",
        "raw_plus_relative_plus_context_buckets_plus_file_context",
    }:
        group_size = len(feature_rows)
        context_bucket_flags = {
            "ctx_gt20": 1.0 if group_size > 20 else 0.0,
            "ctx_gt40": 1.0 if group_size > 40 else 0.0,
        }
        for bucket_name, flag in context_bucket_flags.items():
            interaction_block = raw_matrix * np.float32(flag)
            blocks.append(interaction_block.astype(np.float32))
            transformed_names.extend(
                f"{feature_name}_x_{bucket_name}" for feature_name in resolved_base_names
            )

    if feature_transform == "raw_plus_relative_plus_context_slices_plus_file_context":
        group_size = len(feature_rows)
        context_slice_flags = {
            "ctx_11_20": 1.0 if 11 <= group_size <= 20 else 0.0,
            "ctx_21_40": 1.0 if 21 <= group_size <= 40 else 0.0,
            "ctx_gt40": 1.0 if group_size > 40 else 0.0,
        }
        for slice_name, flag in context_slice_flags.items():
            interaction_block = raw_matrix * np.float32(flag)
            blocks.append(interaction_block.astype(np.float32))
            transformed_names.extend(
                f"{feature_name}_x_{slice_name}" for feature_name in resolved_base_names
            )

    if feature_transform in {
        "raw_plus_relative_plus_file_context",
        "raw_plus_relative_plus_context_buckets_plus_file_context",
        "raw_plus_relative_plus_context_slices_plus_file_context",
    }:
        context_paths = [extract_context_path(row["text"]) for row in feature_rows]
        file_max = np.zeros_like(raw_matrix, dtype=np.float32)
        within_file_rank = np.ones_like(raw_matrix, dtype=np.float32)
        path_to_indices: dict[str, list[int]] = {}
        for index, path in enumerate(context_paths):
            path_to_indices.setdefault(path, []).append(index)
        for indices in path_to_indices.values():
            file_values = raw_matrix[indices]
            file_max[indices] = file_values.max(axis=0)
            if len(indices) > 1:
                denom = float(len(indices) - 1)
                for column_index in range(raw_matrix.shape[1]):
                    order = np.argsort(-file_values[:, column_index], kind="stable")
                    rank_values = np.zeros(len(indices), dtype=np.float32)
                    rank_values[order] = 1.0 - (np.arange(len(indices), dtype=np.float32) / denom)
                    within_file_rank[np.asarray(indices), column_index] = rank_values
        blocks.append(file_max.astype(np.float32))
        transformed_names.extend(f"{feature_name}_file_max" for feature_name in resolved_base_names)
        blocks.append(within_file_rank.astype(np.float32))
        transformed_names.extend(
            f"{feature_name}_within_file_rank_pct" for feature_name in resolved_base_names
        )

    matrix = np.concatenate(blocks, axis=1)
    return matrix.astype(np.float32), resolved_base_names, transformed_names


def build_feature_row_cache_records(samples: list[dict]) -> list[dict]:
    return [
        {
            "sample_id": sample["sample_id"],
            "feature_rows": build_context_feature_rows(sample),
        }
        for sample in samples
    ]


def save_feature_row_cache(samples: list[dict], path: str | Path) -> None:
    write_jsonl(build_feature_row_cache_records(samples), path)


def load_feature_row_cache(path: str | Path) -> dict[str, list[dict]]:
    return {
        record["sample_id"]: record["feature_rows"]
        for record in load_jsonl(path)
    }


def build_labeled_ranking_dataset(
    samples: list[dict],
    *,
    base_feature_names: list[str] | None = None,
    feature_transform: str = "raw_plus_relative",
    feature_rows_by_sample_id: dict[str, list[dict]] | None = None,
) -> dict:
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    group_sizes: list[int] = []
    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    resolved_base_names = list(base_feature_names or [])
    resolved_feature_names: list[str] = []

    for sample in samples:
        if feature_rows_by_sample_id is None:
            feature_rows = build_context_feature_rows(sample)
        else:
            feature_rows = feature_rows_by_sample_id.get(sample["sample_id"])
            if feature_rows is None:
                raise KeyError(f"Missing cached feature rows for sample_id={sample['sample_id']}")
        matrix, current_base_names, current_feature_names = build_feature_matrix(
            feature_rows,
            base_feature_names=resolved_base_names or None,
            feature_transform=feature_transform,
        )
        if not resolved_base_names:
            resolved_base_names = current_base_names
            resolved_feature_names = current_feature_names
        gold_contexts = set(sample["gold_context_ids"])
        matrices.append(matrix)
        labels.extend(int(row["context_id"] in gold_contexts) for row in feature_rows)
        group_sizes.append(len(feature_rows))
        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append([row["context_id"] for row in feature_rows])
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))

    stacked_matrix = (
        np.concatenate(matrices, axis=0).astype(np.float32)
        if matrices
        else np.zeros((0, len(resolved_feature_names)), dtype=np.float32)
    )
    return {
        "X": stacked_matrix,
        "y": np.asarray(labels, dtype=np.int8),
        "group_sizes": group_sizes,
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "base_feature_names": resolved_base_names,
        "feature_names": resolved_feature_names,
        "feature_transform": feature_transform,
    }


def ranking_metrics_from_group_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int],
    *,
    top_k: int,
) -> dict[str, float]:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    offset = 0

    for group_size in group_sizes:
        next_offset = offset + group_size
        group_scores = scores[offset:next_offset]
        group_labels = labels[offset:next_offset]
        ranked_indices = np.argsort(-group_scores, kind="stable")
        ranked_labels = group_labels[ranked_indices]

        hit1 += int(bool(ranked_labels.size) and ranked_labels[0] == 1)
        hitk += int(bool(np.any(ranked_labels[:top_k] == 1)))

        positive_positions = np.flatnonzero(ranked_labels == 1)
        if len(positive_positions):
            mrr += 1.0 / float(positive_positions[0] + 1)

        offset = next_offset

    total = len(group_sizes) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
    }


def fit_pointwise_logistic_ranker(
    train_samples: list[dict],
    dev_samples: list[dict],
    *,
    top_k: int,
    training_config: dict | None = None,
    train_feature_rows_by_sample_id: dict[str, list[dict]] | None = None,
    dev_feature_rows_by_sample_id: dict[str, list[dict]] | None = None,
) -> dict:
    config = training_config or {}
    feature_transform = str(config.get("feature_transform", "raw_plus_relative"))
    candidate_c_values = [float(value) for value in config.get("candidate_c_values", [0.1, 0.25, 0.5, 1.0, 2.0])]
    max_iter = int(config.get("max_iter", 1000))
    solver = str(config.get("solver", "liblinear"))
    penalty = str(config.get("penalty", "l2"))
    random_state = int(config.get("random_state", 42))
    class_weight = config.get("class_weight", "balanced")

    train_data = build_labeled_ranking_dataset(
        train_samples,
        feature_transform=feature_transform,
        feature_rows_by_sample_id=train_feature_rows_by_sample_id,
    )
    dev_data = build_labeled_ranking_dataset(
        dev_samples,
        base_feature_names=train_data["base_feature_names"],
        feature_transform=feature_transform,
        feature_rows_by_sample_id=dev_feature_rows_by_sample_id,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data["X"])
    X_dev = scaler.transform(dev_data["X"])

    best_search: dict | None = None
    search_results: list[dict] = []
    best_classifier: LogisticRegression | None = None

    for c_value in candidate_c_values:
        classifier_kwargs = {
            "C": c_value,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "solver": solver,
            "random_state": random_state,
        }
        # In recent scikit-learn releases explicitly passing the default
        # ``penalty='l2'`` emits a deprecation warning.  Omitting it retains
        # the same solver default while keeping non-default penalties explicit.
        if penalty != "l2":
            classifier_kwargs["penalty"] = penalty
        classifier = LogisticRegression(**classifier_kwargs)
        classifier.fit(X_train, train_data["y"])
        dev_scores = classifier.decision_function(X_dev)
        dev_metrics = ranking_metrics_from_group_scores(
            dev_scores,
            dev_data["y"],
            dev_data["group_sizes"],
            top_k=top_k,
        )
        result = {
            "C": c_value,
            "dev_metrics": dev_metrics,
        }
        search_results.append(result)
        if best_search is None or dev_metrics["hit@1"] > best_search["dev_metrics"]["hit@1"] or (
            dev_metrics["hit@1"] == best_search["dev_metrics"]["hit@1"]
            and dev_metrics["mrr"] > best_search["dev_metrics"]["mrr"]
        ):
            best_search = result
            best_classifier = classifier

    if best_search is None or best_classifier is None:
        raise RuntimeError("No candidate models were trained.")

    model = {
        "model_type": "pointwise_logistic_regression",
        "feature_transform": feature_transform,
        "base_feature_names": train_data["base_feature_names"],
        "feature_names": train_data["feature_names"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": best_classifier.coef_[0].tolist(),
        "intercept": float(best_classifier.intercept_[0]),
        "search": {
            "candidate_c_values": candidate_c_values,
            "selected_C": best_search["C"],
            "solver": solver,
            "penalty": penalty,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state,
        },
        "best_dev_metrics": best_search["dev_metrics"],
    }

    return {
        "model": model,
        "search_results": search_results,
        "train_dataset": train_data,
        "dev_dataset": dev_data,
    }


def linear_model_scores(matrix: np.ndarray, model: dict) -> np.ndarray:
    scaler_mean = np.asarray(model["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(model["scaler_scale"], dtype=np.float32)
    safe_scale = np.where(scaler_scale != 0.0, scaler_scale, 1.0)
    coefficients = np.asarray(model["coefficients"], dtype=np.float32)
    intercept = float(model["intercept"])
    normalized = (matrix.astype(np.float32) - scaler_mean) / safe_scale
    return (normalized @ coefficients) + intercept


def evaluate_ranking_dataset(dataset: dict, model: dict, *, top_k: int) -> dict:
    scores = linear_model_scores(dataset["X"], model)
    hit1 = 0
    hitk = 0
    mrr = 0.0
    ranked_predictions = []
    offset = 0

    for sample_id, group_size, context_ids, gold_context_ids in zip(
        dataset["sample_ids"],
        dataset["group_sizes"],
        dataset["context_ids_by_group"],
        dataset["gold_context_ids_by_group"],
        strict=True,
    ):
        next_offset = offset + group_size
        group_scores = scores[offset:next_offset]
        group_labels = dataset["y"][offset:next_offset]
        ranked_indices = np.argsort(-group_scores, kind="stable")
        ranked_context_ids = [context_ids[index] for index in ranked_indices]
        ranked_labels = group_labels[ranked_indices]
        gold = set(gold_context_ids)

        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))

        positive_positions = np.flatnonzero(ranked_labels == 1)
        if len(positive_positions):
            mrr += 1.0 / float(positive_positions[0] + 1)

        ranked_predictions.append(
            {
                "sample_id": sample_id,
                "gold_context_ids": gold_context_ids,
                "ranked_context_ids": ranked_context_ids,
            }
        )
        offset = next_offset

    total = len(dataset["group_sizes"]) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
        "ranked_predictions": ranked_predictions,
    }


def rank_feature_rows_with_model(feature_rows: list[dict], model: dict) -> list[dict]:
    matrix, _, _ = build_feature_matrix(
        feature_rows,
        base_feature_names=model["base_feature_names"],
        feature_transform=model["feature_transform"],
    )
    scores = linear_model_scores(matrix, model)
    ranked = [
        {
            "context_id": row["context_id"],
            "source": row["source"],
            "text": row["text"],
            "score": float(score),
            "features": row["features"],
        }
        for row, score in zip(feature_rows, scores, strict=True)
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def rank_contexts_with_model(sample: dict, model: dict) -> list[dict]:
    return rank_feature_rows_with_model(build_context_feature_rows(sample), model)


def evidence_metrics_with_model(samples: list[dict], model: dict, *, top_k: int) -> dict:
    dataset = build_labeled_ranking_dataset(
        samples,
        base_feature_names=model["base_feature_names"],
        feature_transform=model["feature_transform"],
    )
    return evaluate_ranking_dataset(dataset, model, top_k=top_k)


def load_linear_ranker_model(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_linear_ranker_model(model: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, indent=2)
