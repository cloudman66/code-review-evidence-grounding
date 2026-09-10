from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_quoted_spans
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for split in ("train", "dev", "test"):
        parser.add_argument(f"--{split}-dataset", required=True)
        parser.add_argument(f"--{split}-caches", default="")
        parser.add_argument(f"--{split}-cache-a", default="")
        parser.add_argument(f"--{split}-cache-b", default="")
        parser.add_argument(f"--{split}-cache-c", default="")
        parser.add_argument(f"--{split}-cache-d", default="")
    parser.add_argument("--model-names", default="word,dual,char")
    parser.add_argument("--c-values", default="0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def resolve_cache_paths(args: argparse.Namespace, split: str, expected_count: int) -> list[str]:
    combined = getattr(args, f"{split}_caches")
    if combined:
        paths = [part.strip() for part in combined.split(",") if part.strip()]
    else:
        paths = [
            getattr(args, f"{split}_cache_a"),
            getattr(args, f"{split}_cache_b"),
            getattr(args, f"{split}_cache_c"),
            getattr(args, f"{split}_cache_d"),
        ]
        paths = [path for path in paths if path]
    if len(paths) != expected_count:
        raise ValueError(
            f"{split} cache count {len(paths)} does not match model count {expected_count}"
        )
    return paths


def bucket_context_count(count: int) -> str:
    if count <= 10:
        return "<=10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    return ">40"


def sample_meta(sample: dict) -> dict[str, float]:
    comment = sample["comment"]
    comment_lower = comment.lower()
    context_bucket = bucket_context_count(len(sample["contexts"]))
    word_count = len(comment.split())
    return {
        "context_bucket_<=10": 1.0 if context_bucket == "<=10" else 0.0,
        "context_bucket_11_20": 1.0 if context_bucket == "11-20" else 0.0,
        "context_bucket_21_40": 1.0 if context_bucket == "21-40" else 0.0,
        "context_bucket_gt40": 1.0 if context_bucket == ">40" else 0.0,
        "code_fence": 1.0 if "```" in comment else 0.0,
        "suggestion": 1.0 if "suggestion" in comment.lower() else 0.0,
        "mention": 1.0 if "@" in comment else 0.0,
        "quoted_span": 1.0 if bool(extract_quoted_spans(comment)) else 0.0,
        "short_comment": 1.0 if word_count <= 6 else 0.0,
        "long_comment": 1.0 if word_count >= 30 else 0.0,
        "question": 1.0 if "?" in comment else 0.0,
        "intent_typo": 1.0
        if any(token in comment_lower for token in ("typo", "spelled wrong", "spelling", "misspell"))
        else 0.0,
        "intent_naming": 1.0
        if any(token in comment_lower for token in ("variable name", "nicer variable", "rename", "name ", "named "))
        else 0.0,
        "intent_remove": 1.0
        if any(token in comment_lower for token in ("remove", "delete", "redundant", "unnecessary", "comment this out"))
        else 0.0,
        "intent_simplify": 1.0
        if any(
            token in comment_lower
            for token in ("simplify", "just ", "instead", "nicer", "can we", "possible to")
        )
        else 0.0,
        "intent_exception": 1.0
        if any(
            token in comment_lower
            for token in ("exception", "try catch", "catch ", "raise ", "error", "typeerror")
        )
        else 0.0,
        "intent_docs_text": 1.0
        if any(
            token in comment_lower
            for token in ("docs", "documentation", "link", "wording", "style guide", "spelled", "word ")
        )
        else 0.0,
        "intent_api_usage": 1.0
        if any(
            token in comment_lower
            for token in ("import", "dependency", "undefined", "parameter", "argument", "callable", "function")
        )
        else 0.0,
        "word_count": float(word_count),
        "context_count": float(len(sample["contexts"])),
    }


def topk_stats(scores: np.ndarray, context_ids: list[str], gold: set[str]) -> dict[str, float | str]:
    ranked_indices = np.argsort(-scores, kind="stable")
    ranked_context_ids = [context_ids[index] for index in ranked_indices]
    ranked_scores = scores[ranked_indices]
    top1_score = float(ranked_scores[0]) if len(ranked_scores) else 0.0
    top2_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else top1_score
    margin = top1_score - top2_score
    top1_id = ranked_context_ids[0] if ranked_context_ids else ""
    top1_correct = 1.0 if top1_id in gold else 0.0

    reciprocal_rank = 0.0
    for index, context_id in enumerate(ranked_context_ids, start=1):
        if context_id in gold:
            reciprocal_rank = 1.0 / index
            break

    return {
        "top1_id": top1_id,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "top1_correct": top1_correct,
        "mrr": reciprocal_rank,
        "top3_hit": 1.0 if any(context_id in gold for context_id in ranked_context_ids[:3]) else 0.0,
        "ranked_context_ids": ranked_context_ids,
    }


def score_key(stats: dict[str, float | str]) -> tuple[float, float, float]:
    return (
        float(stats["top1_correct"]),
        float(stats["mrr"]),
        float(stats["top3_hit"]),
    )


def build_sample_rows(
    samples: list[dict],
    caches: list[dict],
    model_names: list[str],
) -> list[dict]:
    dataset = caches[0]["dataset"]
    for cache in caches[1:]:
        if dataset["sample_ids"] != cache["dataset"]["sample_ids"]:
            raise ValueError("sample_ids do not align")
        if dataset["context_ids_by_group"] != cache["dataset"]["context_ids_by_group"]:
            raise ValueError("context ordering does not align")

    rows: list[dict] = []
    offset = 0
    for sample, sample_id, context_ids, gold_context_ids, group_size in zip(
        samples,
        dataset["sample_ids"],
        dataset["context_ids_by_group"],
        dataset["gold_context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        gold = set(gold_context_ids)
        stats_map = {
            model_name: topk_stats(cache["scores"][offset:next_offset], context_ids, gold)
            for model_name, cache in zip(model_names, caches, strict=True)
        }
        best_model = max(model_names, key=lambda name: score_key(stats_map[name]))
        features = sample_meta(sample)

        for name in model_names:
            stats = stats_map[name]
            features[f"{name}_top1_score"] = float(stats["top1_score"])
            features[f"{name}_margin"] = float(stats["margin"])

        for left_name in model_names:
            for right_name in model_names:
                if left_name == right_name:
                    continue
                left_stats = stats_map[left_name]
                right_stats = stats_map[right_name]
                features[f"{left_name}_minus_{right_name}_top1_score"] = float(left_stats["top1_score"]) - float(right_stats["top1_score"])
                features[f"{left_name}_minus_{right_name}_margin"] = float(left_stats["margin"]) - float(right_stats["margin"])
                features[f"{left_name}_same_top1_as_{right_name}"] = 1.0 if left_stats["top1_id"] == right_stats["top1_id"] else 0.0

        rows.append(
            {
                "sample_id": sample_id,
                "comment": sample["comment"],
                "gold_context_ids": list(sample["gold_context_ids"]),
                "features": features,
                "stats": stats_map,
                "label": best_model,
            }
        )
        offset = next_offset
    return rows


def rows_to_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )


def evaluate_predictions(
    rows: list[dict],
    predicted_indices: np.ndarray,
    class_names: list[str],
) -> dict[str, float | dict[str, int] | list[dict]]:
    hit1 = 0
    hit3 = 0
    mrr = 0.0
    usage = {name: 0 for name in class_names}
    predictions: list[dict] = []
    for row, predicted_index in zip(rows, predicted_indices, strict=True):
        selected_name = class_names[int(predicted_index)]
        usage[selected_name] += 1
        stats = row["stats"][selected_name]
        hit1 += int(float(stats["top1_correct"]) > 0.0)
        hit3 += int(float(stats["top3_hit"]) > 0.0)
        mrr += float(stats["mrr"])
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "comment": row["comment"],
                "gold_context_ids": row["gold_context_ids"],
                "predicted_context_ids_topk": list(stats["ranked_context_ids"][:3]),
                "selected_model": selected_name,
            }
        )
    total = len(rows) or 1
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
        "usage": usage,
        "predictions": predictions,
    }


def build_report(
    *,
    model_names: list[str],
    selected_c: float,
    search_rows: list[dict],
    dev_metrics: dict,
    test_metrics: dict,
    usage_train: dict[str, int],
    coefficients: dict[str, list[tuple[str, float]]],
) -> str:
    lines = ["# Multimodel Router", "", "## Metrics", ""]
    lines.append(f"- models: `{', '.join(model_names)}`")
    lines.append(f"- selected C: `{selected_c}`")
    lines.append(f"- dev: `{dev_metrics['hit@1']:.4f} / {dev_metrics['hit@3']:.4f} / {dev_metrics['mrr']:.4f}`")
    lines.append(f"- test: `{test_metrics['hit@1']:.4f} / {test_metrics['hit@3']:.4f} / {test_metrics['mrr']:.4f}`")
    lines.append("")
    lines.append("## C Search")
    lines.append("")
    lines.append("| C | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in search_rows:
        lines.append(
            f"| {row['c']:.2f} | {row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    lines.append("")
    lines.append("## Train Usage")
    lines.append("")
    for name in model_names:
        lines.append(f"- {name}: `{usage_train[name]}`")
    lines.append("")
    lines.append("## Top Coefficients")
    lines.append("")
    for name in model_names:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| feature | coefficient |")
        lines.append("| --- | ---: |")
        for feature_name, value in coefficients[name]:
            lines.append(f"| {feature_name} | {value:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    model_names = [part.strip() for part in args.model_names.split(",") if part.strip()]
    if len(model_names) < 3:
        raise ValueError("--model-names must contain at least three comma-separated names")
    c_values = [float(part) for part in args.c_values.split(",") if part.strip()]

    train_caches = [load_score_cache(Path(path)) for path in resolve_cache_paths(args, "train", len(model_names))]
    dev_caches = [load_score_cache(Path(path)) for path in resolve_cache_paths(args, "dev", len(model_names))]
    test_caches = [load_score_cache(Path(path)) for path in resolve_cache_paths(args, "test", len(model_names))]

    train_rows = build_sample_rows(
        load_jsonl(args.train_dataset),
        train_caches,
        model_names,
    )
    dev_rows = build_sample_rows(
        load_jsonl(args.dev_dataset),
        dev_caches,
        model_names,
    )
    test_rows = build_sample_rows(
        load_jsonl(args.test_dataset),
        test_caches,
        model_names,
    )

    feature_names = sorted(train_rows[0]["features"].keys())
    class_names = list(model_names)
    label_to_index = {name: index for index, name in enumerate(class_names)}

    X_train = rows_to_matrix(train_rows, feature_names)
    y_train = np.asarray([label_to_index[row["label"]] for row in train_rows], dtype=np.int8)
    X_dev = rows_to_matrix(dev_rows, feature_names)
    X_test = rows_to_matrix(test_rows, feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    best: dict | None = None
    search_rows: list[dict] = []
    for c_value in c_values:
        classifier = LogisticRegression(
            C=c_value,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        classifier.fit(X_train_scaled, y_train)

        train_predictions = classifier.predict(X_train_scaled)
        dev_predictions = classifier.predict(X_dev_scaled)
        test_predictions = classifier.predict(X_test_scaled)

        train_metrics = evaluate_predictions(train_rows, train_predictions, class_names)
        dev_metrics = evaluate_predictions(dev_rows, dev_predictions, class_names)
        test_metrics = evaluate_predictions(test_rows, test_predictions, class_names)
        row = {
            "c": c_value,
            "train": train_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
            "classifier": classifier,
        }
        search_rows.append(row)
        if best is None or (
            dev_metrics["hit@1"],
            dev_metrics["mrr"],
            dev_metrics["hit@3"],
        ) > (
            best["dev"]["hit@1"],
            best["dev"]["mrr"],
            best["dev"]["hit@3"],
        ):
            best = row

    if best is None:
        raise RuntimeError("No router configurations were evaluated.")

    classifier = best["classifier"]
    train_usage = best["train"]["usage"]
    dev_metrics = best["dev"]
    test_metrics = best["test"]

    coefficients: dict[str, list[tuple[str, float]]] = {}
    for class_index, class_name in enumerate(class_names):
        ranked = sorted(
            zip(feature_names, classifier.coef_[class_index], strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:12]
        coefficients[class_name] = [(feature_name, float(value)) for feature_name, value in ranked]

    payload = {
        "models": class_names,
        "selected_c": best["c"],
        "search": [
            {
                "c": row["c"],
                "train": row["train"],
                "dev": row["dev"],
                "test": row["test"],
            }
            for row in search_rows
        ],
        "train": {
            "usage": train_usage,
        },
        "dev": dev_metrics,
        "test": test_metrics,
        "feature_names": feature_names,
        "coefficients": {name: [{"feature": feature, "coefficient": value} for feature, value in values] for name, values in coefficients.items()},
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_predictions_dev = output_json.parent / "predictions_dev.jsonl"
    output_predictions = output_json.parent / "predictions_test.jsonl"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(
        build_report(
            model_names=class_names,
            selected_c=best["c"],
            search_rows=[
                {
                    "c": row["c"],
                    "dev": row["dev"],
                    "test": row["test"],
                }
                for row in search_rows
            ],
            dev_metrics=dev_metrics,
            test_metrics=test_metrics,
            usage_train=train_usage,
            coefficients=coefficients,
        ),
        encoding="utf-8",
    )
    with output_predictions_dev.open("w", encoding="utf-8") as handle:
        for row in dev_metrics["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output_predictions.open("w", encoding="utf-8") as handle:
        for row in test_metrics["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "selected_c": best["c"],
                "dev": {
                    "hit@1": dev_metrics["hit@1"],
                    "hit@3": dev_metrics["hit@3"],
                    "mrr": dev_metrics["mrr"],
                    "usage": dev_metrics["usage"],
                },
                "test": {
                    "hit@1": test_metrics["hit@1"],
                    "hit@3": test_metrics["hit@3"],
                    "mrr": test_metrics["mrr"],
                    "usage": test_metrics["usage"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")
    print(f"Saved dev predictions to {output_predictions_dev}")
    print(f"Saved predictions to {output_predictions}")


if __name__ == "__main__":
    main()
