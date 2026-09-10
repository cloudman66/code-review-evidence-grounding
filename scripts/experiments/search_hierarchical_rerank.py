from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_context_path, extract_quoted_spans
from code_review_understanding.models.fusion import write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_feature_matrix,
    linear_model_scores,
    load_feature_row_cache,
    load_linear_ranker_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--dev-feature-row-cache", required=True)
    parser.add_argument("--test-feature-row-cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--external-score",
        action="append",
        default=[],
        help="Attach external scores as name=dev_path=test_path or name=dev_path::test_path",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def open_maybe_gzip(path: str | Path, mode: str):
    resolved = Path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, mode, encoding="utf-8")
    return resolved.open(mode, encoding="utf-8")


def parse_external_score_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid external-score spec: {spec}")
    name, payload = spec.split("=", 1)
    if "::" in payload:
        dev_path, test_path = payload.split("::", 1)
    elif "=" in payload:
        dev_path, test_path = payload.split("=", 1)
    else:
        raise ValueError(f"Invalid external-score payload: {payload}")
    return name.strip(), dev_path.strip(), test_path.strip()


def load_score_feature_map(path: str | Path) -> dict[str, dict[str, float]]:
    with open_maybe_gzip(path, "rt") as handle:
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


def augment_feature_rows_with_external_scores(
    feature_rows_by_sample_id: dict[str, list[dict]],
    *,
    feature_name: str,
    score_cache_path: str,
) -> dict[str, list[dict]]:
    score_map = load_score_feature_map(score_cache_path)
    for sample_id, feature_rows in feature_rows_by_sample_id.items():
        sample_scores = score_map.get(sample_id)
        if sample_scores is None:
            raise KeyError(f"Missing score-cache sample_id={sample_id} for feature {feature_name}")
        for row in feature_rows:
            context_id = row["context_id"]
            if context_id not in sample_scores:
                raise KeyError(
                    f"Missing context_id={context_id} for sample_id={sample_id} feature={feature_name}"
                )
            row["features"][feature_name] = float(sample_scores[context_id])
    return feature_rows_by_sample_id


def bucket_context_count(count: int) -> str:
    if count <= 10:
        return "<=10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    return ">40"


def policy_names() -> list[str]:
    return [
        "always",
        "ctx_gt20",
        "ctx_gt40",
        "quoted_no",
        "quoted_no_or_ctx_gt40",
    ]


def should_apply_policy(sample: dict, name: str) -> bool:
    context_count = len(sample["contexts"])
    quoted = bool(extract_quoted_spans(sample["comment"]))
    if name == "always":
        return True
    if name == "ctx_gt20":
        return context_count > 20
    if name == "ctx_gt40":
        return context_count > 40
    if name == "quoted_no":
        return not quoted
    if name == "quoted_no_or_ctx_gt40":
        return (not quoted) or context_count > 40
    raise ValueError(f"Unsupported policy: {name}")


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [1.0 for _ in values]
    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    if mode == "max":
        return ordered[0]
    if mode == "mean":
        return sum(ordered) / len(ordered)
    if mode == "top2_mean":
        return sum(ordered[:2]) / min(len(ordered), 2)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def build_scored_samples(
    samples: list[dict],
    *,
    feature_rows_by_sample_id: dict[str, list[dict]],
    model: dict,
) -> list[dict]:
    scored_samples: list[dict] = []
    for sample in samples:
        feature_rows = feature_rows_by_sample_id[sample["sample_id"]]
        matrix, _, _ = build_feature_matrix(
            feature_rows,
            base_feature_names=model["base_feature_names"],
            feature_transform=model["feature_transform"],
        )
        model_scores = linear_model_scores(matrix, model)
        model_scores_norm = minmax(model_scores.tolist())
        semantic_values = [float(row["features"].get("semantic_score", 0.0)) for row in feature_rows]
        semantic_norm = minmax(semantic_values)

        contexts = []
        for row, model_score, model_score_norm, semantic_score, semantic_score_norm in zip(
            feature_rows,
            model_scores,
            model_scores_norm,
            semantic_values,
            semantic_norm,
            strict=True,
        ):
            contexts.append(
                {
                    "context_id": row["context_id"],
                    "file_path": extract_context_path(row["text"]),
                    "model_score": float(model_score),
                    "model_score_norm": float(model_score_norm),
                    "semantic_score": float(semantic_score),
                    "semantic_score_norm": float(semantic_score_norm),
                }
            )
        scored_samples.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": list(sample["gold_context_ids"]),
                "contexts": contexts,
                "context_bucket": bucket_context_count(len(sample["contexts"])),
                "quoted_span": bool(extract_quoted_spans(sample["comment"])),
            }
        )
    return scored_samples


def file_source_value(context: dict, source: str) -> float:
    if source == "model":
        return float(context["model_score"])
    if source == "semantic":
        return float(context["semantic_score"])
    if source == "hybrid_norm":
        return 0.5 * (float(context["model_score_norm"]) + float(context["semantic_score_norm"]))
    raise ValueError(f"Unsupported file source: {source}")


def rerank_context_ids(
    sample: dict,
    *,
    policy: str,
    file_source: str,
    agg_mode: str,
    alpha: float,
    top_files: int,
    outside_penalty: float,
) -> tuple[list[str], list[float]]:
    contexts = sample["contexts"]
    if not should_apply_policy(sample, policy):
        ranked = sorted(contexts, key=lambda item: item["model_score"], reverse=True)
        scores_by_context_id = {
            context["context_id"]: float(context["model_score"]) for context in contexts
        }
        return [context["context_id"] for context in ranked], [
            scores_by_context_id[context["context_id"]]
            for context in contexts
        ]

    path_to_values: dict[str, list[float]] = defaultdict(list)
    for context in contexts:
        path_to_values[context["file_path"]].append(file_source_value(context, file_source))

    file_paths = list(path_to_values.keys())
    file_scores = [aggregate(path_to_values[path], agg_mode) for path in file_paths]
    file_scores_norm = minmax(file_scores)
    file_score_map = {
        path: score for path, score in zip(file_paths, file_scores_norm, strict=True)
    }
    ranked_files = [
        path for path, _ in sorted(file_score_map.items(), key=lambda item: item[1], reverse=True)
    ]
    allowed_files = set(ranked_files[:top_files]) if top_files > 0 else set(ranked_files)

    reranked = []
    score_by_context_id: dict[str, float] = {}
    for context in contexts:
        file_bonus = alpha * file_score_map.get(context["file_path"], 0.0)
        adjusted_score = float(context["model_score"]) + file_bonus
        if top_files > 0 and context["file_path"] not in allowed_files:
            adjusted_score -= outside_penalty
        score_by_context_id[context["context_id"]] = adjusted_score
        reranked.append(
            {
                "context_id": context["context_id"],
                "score": adjusted_score,
            }
        )
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return [context["context_id"] for context in reranked], [
        score_by_context_id[context["context_id"]]
        for context in contexts
    ]


def evaluate_policy(
    samples: list[dict],
    *,
    policy: str,
    file_source: str,
    agg_mode: str,
    alpha: float,
    top_files: int,
    outside_penalty: float,
    top_k: int,
) -> dict:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    predictions: list[dict] = []
    score_blocks: list[float] = []

    for sample in samples:
        ranked_context_ids, aligned_scores = rerank_context_ids(
            sample,
            policy=policy,
            file_source=file_source,
            agg_mode=agg_mode,
            alpha=alpha,
            top_files=top_files,
            outside_penalty=outside_penalty,
        )
        score_blocks.extend(aligned_scores)
        gold = set(sample["gold_context_ids"])
        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))
        positive_positions = [
            index for index, context_id in enumerate(ranked_context_ids) if context_id in gold
        ]
        if positive_positions:
            mrr += 1.0 / float(positive_positions[0] + 1)

        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[:top_k],
            }
        )

    total = len(samples) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
        "predictions": predictions,
        "scores": np.asarray(score_blocks, dtype=np.float32),
    }


def build_score_dataset(samples: list[dict]) -> dict:
    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "context_ids_by_group": [
            [context["context_id"] for context in sample["contexts"]]
            for sample in samples
        ],
        "gold_context_ids_by_group": [sample["gold_context_ids"] for sample in samples],
        "group_sizes": [len(sample["contexts"]) for sample in samples],
    }


def search_policies(dev_samples: list[dict], *, top_k: int) -> list[dict]:
    results: list[dict] = []
    for policy in policy_names():
        for file_source in ("model", "semantic", "hybrid_norm"):
            for agg_mode in ("max", "mean", "top2_mean"):
                for alpha in (0.1, 0.25, 0.5, 1.0, 1.5):
                    for top_files in (0, 1, 2, 3, 4):
                        for outside_penalty in (0.0, 0.25, 0.5, 1.0):
                            metrics = evaluate_policy(
                                dev_samples,
                                policy=policy,
                                file_source=file_source,
                                agg_mode=agg_mode,
                                alpha=alpha,
                                top_files=top_files,
                                outside_penalty=outside_penalty,
                                top_k=top_k,
                            )
                            results.append(
                                {
                                    "policy": policy,
                                    "file_source": file_source,
                                    "agg_mode": agg_mode,
                                    "alpha": alpha,
                                    "top_files": top_files,
                                    "outside_penalty": outside_penalty,
                                    "dev_metrics": {
                                        "hit@1": metrics["hit@1"],
                                        f"hit@{top_k}": metrics[f"hit@{top_k}"],
                                        "mrr": metrics["mrr"],
                                    },
                                }
                            )
    results.sort(
        key=lambda item: (
            item["dev_metrics"]["hit@1"],
            item["dev_metrics"]["mrr"],
            item["dev_metrics"][f"hit@{top_k}"],
        ),
        reverse=True,
    )
    return results


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(best: dict, dev_metrics: dict, test_metrics: dict, top_k: int) -> str:
    lines = [
        "# Hierarchical File-First Rerank Search",
        "",
        "## Selected Policy",
        "",
        f"- policy: {best['policy']}",
        f"- file_source: {best['file_source']}",
        f"- agg_mode: {best['agg_mode']}",
        f"- alpha: {best['alpha']}",
        f"- top_files: {best['top_files']}",
        f"- outside_penalty: {best['outside_penalty']}",
        "",
        "## Metrics",
        "",
        "| split | hit@1 | hit@3 | mrr |",
        "| --- | ---: | ---: | ---: |",
        f"| dev | {dev_metrics['hit@1']:.4f} | {dev_metrics[f'hit@{top_k}']:.4f} | {dev_metrics['mrr']:.4f} |",
        f"| test | {test_metrics['hit@1']:.4f} | {test_metrics[f'hit@{top_k}']:.4f} | {test_metrics['mrr']:.4f} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    model = load_linear_ranker_model(args.model)

    dev_samples_raw = load_jsonl(args.dev_dataset)
    test_samples_raw = load_jsonl(args.test_dataset)
    dev_feature_rows = load_feature_row_cache(args.dev_feature_row_cache)
    test_feature_rows = load_feature_row_cache(args.test_feature_row_cache)

    for spec in args.external_score:
        feature_name, dev_score_path, test_score_path = parse_external_score_spec(spec)
        dev_feature_rows = augment_feature_rows_with_external_scores(
            dev_feature_rows,
            feature_name=feature_name,
            score_cache_path=dev_score_path,
        )
        test_feature_rows = augment_feature_rows_with_external_scores(
            test_feature_rows,
            feature_name=feature_name,
            score_cache_path=test_score_path,
        )

    dev_samples = build_scored_samples(
        dev_samples_raw,
        feature_rows_by_sample_id=dev_feature_rows,
        model=model,
    )
    test_samples = build_scored_samples(
        test_samples_raw,
        feature_rows_by_sample_id=test_feature_rows,
        model=model,
    )

    search_results = search_policies(dev_samples, top_k=args.top_k)
    best = search_results[0]
    dev_metrics = evaluate_policy(
        dev_samples,
        policy=best["policy"],
        file_source=best["file_source"],
        agg_mode=best["agg_mode"],
        alpha=float(best["alpha"]),
        top_files=int(best["top_files"]),
        outside_penalty=float(best["outside_penalty"]),
        top_k=args.top_k,
    )
    test_metrics = evaluate_policy(
        test_samples,
        policy=best["policy"],
        file_source=best["file_source"],
        agg_mode=best["agg_mode"],
        alpha=float(best["alpha"]),
        top_files=int(best["top_files"]),
        outside_penalty=float(best["outside_penalty"]),
        top_k=args.top_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "selected_policy": best,
        "retrieval_dev": {
            "hit@1": dev_metrics["hit@1"],
            f"hit@{args.top_k}": dev_metrics[f"hit@{args.top_k}"],
            "mrr": dev_metrics["mrr"],
        },
        "retrieval_test": {
            "hit@1": test_metrics["hit@1"],
            f"hit@{args.top_k}": test_metrics[f"hit@{args.top_k}"],
            "mrr": test_metrics["mrr"],
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "selected_policy.json", best)
    write_json(output_dir / "search_results_top20.json", search_results[:20])
    write_jsonl(output_dir / "predictions_test.jsonl", test_metrics["predictions"])
    write_score_cache(
        output_dir / "dev_scores.json.gz",
        model_path=str(args.model),
        dataset=build_score_dataset(dev_samples),
        scores=dev_metrics["scores"],
    )
    write_score_cache(
        output_dir / "test_scores.json.gz",
        model_path=str(args.model),
        dataset=build_score_dataset(test_samples),
        scores=test_metrics["scores"],
    )
    (output_dir / "report.md").write_text(
        build_report(best, dev_metrics, test_metrics, args.top_k),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
