from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    linear_model_scores,
    load_linear_ranker_model,
)


def _open_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_score_cache(path: Path) -> dict:
    with _open_maybe_gzip(path, "rt") as handle:
        payload = json.load(handle)
    return {
        "model_path": payload.get("model_path", ""),
        "dataset": {
            "sample_ids": payload["sample_ids"],
            "context_ids_by_group": payload["context_ids_by_group"],
            "gold_context_ids_by_group": payload["gold_context_ids_by_group"],
            "group_sizes": payload["group_sizes"],
        },
        "scores": np.asarray(payload["scores"], dtype=np.float32),
    }


def write_score_cache(path: Path, *, model_path: str, dataset: dict, scores: np.ndarray) -> None:
    payload = {
        "model_path": model_path,
        "sample_ids": dataset["sample_ids"],
        "context_ids_by_group": dataset["context_ids_by_group"],
        "gold_context_ids_by_group": dataset["gold_context_ids_by_group"],
        "group_sizes": dataset["group_sizes"],
        "scores": scores.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_maybe_gzip(path, "wt") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def normalize_group_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    arr = scores.astype(np.float32)
    if mode == "raw":
        return arr
    if mode == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-6:
            return arr - mean
        return (arr - mean) / std
    if mode == "minmax":
        lower = float(arr.min())
        upper = float(arr.max())
        if upper - lower < 1e-6:
            return arr - lower
        return (arr - lower) / (upper - lower)
    if mode == "rank":
        order = np.argsort(-arr, kind="stable")
        ranked = np.zeros_like(arr, dtype=np.float32)
        if len(arr) == 1:
            ranked[0] = 1.0
            return ranked
        ranked[order] = 1.0 - (np.arange(len(arr), dtype=np.float32) / float(len(arr) - 1))
        return ranked
    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_score_cache(samples: list[dict], model_specs: list[dict]) -> tuple[dict, list[dict]]:
    cache: dict[str, dict] = {}
    for index, spec in enumerate(model_specs):
        score_cache_path = spec.get("score_cache_path")
        if score_cache_path:
            cached = load_score_cache(Path(score_cache_path))
            model_path = Path(cached["model_path"]) if cached["model_path"] else Path(str(score_cache_path))
            dataset = cached["dataset"]
            scores = cached["scores"]
        else:
            model_path = Path(spec["model_path"])
            model = load_linear_ranker_model(model_path)
            dataset = build_labeled_ranking_dataset(
                samples,
                base_feature_names=model["base_feature_names"],
                feature_transform=model["feature_transform"],
            )
            scores = linear_model_scores(dataset["X"], model)
        key = str(index)
        cache[key] = {
            "model_path": str(model_path),
            "dataset": dataset,
            "scores": scores,
            "weight": float(spec["weight"]),
        }

    cache_items = list(cache.values())
    if not cache_items:
        raise ValueError("At least one model spec is required for score fusion.")

    reference = cache_items[0]["dataset"]
    for item in cache_items[1:]:
        dataset = item["dataset"]
        if reference["sample_ids"] != dataset["sample_ids"]:
            raise ValueError("Sample ids do not align across fusion components.")
        if reference["context_ids_by_group"] != dataset["context_ids_by_group"]:
            raise ValueError("Context ordering does not align across fusion components.")
    return cache, cache_items


def resolve_model_specs(model_specs: list[dict], split_name: str) -> list[dict]:
    resolved = []
    for spec in model_specs:
        item = dict(spec)
        cache_by_split = item.pop("score_cache_path_by_split", None)
        if cache_by_split:
            if split_name not in cache_by_split:
                raise ValueError(f"Missing score cache for split '{split_name}'.")
            item["score_cache_path"] = cache_by_split[split_name]
        resolved.append(item)
    return resolved


def fused_evidence_metrics(
    samples: list[dict],
    *,
    model_specs: list[dict],
    normalization: str,
    top_k: int,
) -> dict:
    _, cache_items = build_score_cache(samples, model_specs)
    reference_dataset = cache_items[0]["dataset"]

    hit1 = 0
    hitk = 0
    mrr = 0.0
    ranked_predictions = []
    offset = 0

    for sample, context_ids, gold_context_ids, group_size in zip(
        samples,
        reference_dataset["context_ids_by_group"],
        reference_dataset["gold_context_ids_by_group"],
        reference_dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        fused_scores = np.zeros(group_size, dtype=np.float32)
        for item in cache_items:
            group_scores = item["scores"][offset:next_offset]
            fused_scores += item["weight"] * normalize_group_scores(group_scores, normalization)

        ranked_indices = np.argsort(-fused_scores, kind="stable")
        ranked_context_ids = [context_ids[index] for index in ranked_indices]
        gold = set(gold_context_ids)

        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))

        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_context_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank

        ranked_predictions.append(
            {
                "sample_id": sample["sample_id"],
                "gold_context_ids": sample["gold_context_ids"],
                "ranked_context_ids": ranked_context_ids,
            }
        )
        offset = next_offset

    total = len(samples) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
        "ranked_predictions": ranked_predictions,
    }


def run_fusion_experiment(config: dict) -> dict:
    train_path = config["dataset"].get("train_path")
    dev_path = config["dataset"]["dev_path"]
    test_path = config["dataset"]["test_path"]
    top_k = int(config["retrieval"]["top_k"])
    normalization = str(config["fusion"]["normalization"])
    model_specs = config["fusion"]["models"]

    train_samples = load_jsonl(train_path) if train_path else []
    dev_samples = load_jsonl(dev_path)
    test_samples = load_jsonl(test_path)

    dev_metrics = fused_evidence_metrics(
        dev_samples,
        model_specs=resolve_model_specs(model_specs, "dev"),
        normalization=normalization,
        top_k=top_k,
    )
    test_metrics = fused_evidence_metrics(
        test_samples,
        model_specs=resolve_model_specs(model_specs, "test"),
        normalization=normalization,
        top_k=top_k,
    )

    predictions = []
    ranked_test = {item["sample_id"]: item for item in test_metrics["ranked_predictions"]}
    for sample in test_samples:
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_test[sample["sample_id"]]["ranked_context_ids"][:top_k],
            }
        )

    return {
        "metrics": {
            "dataset_sizes": {
                "train": len(train_samples),
                "dev": len(dev_samples),
                "test": len(test_samples),
            },
            "fusion": {
                "normalization": normalization,
                "models": model_specs,
            },
            "retrieval_dev": {
                "hit@1": dev_metrics["hit@1"],
                f"hit@{top_k}": dev_metrics[f"hit@{top_k}"],
                "mrr": dev_metrics["mrr"],
            },
            "retrieval_test": {
                "hit@1": test_metrics["hit@1"],
                f"hit@{top_k}": test_metrics[f"hit@{top_k}"],
                "mrr": test_metrics["mrr"],
            },
        },
        "predictions": predictions,
    }
