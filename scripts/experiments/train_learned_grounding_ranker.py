from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.external_scores import (
    augment_feature_rows_with_external_scores,
)
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    fit_pointwise_logistic_ranker,
    load_feature_row_cache,
    save_linear_ranker_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def coefficient_rows(model: dict) -> list[dict]:
    rows = []
    for feature_name, coefficient in zip(
        model["feature_names"],
        model["coefficients"],
        strict=True,
    ):
        rows.append(
            {
                "feature": feature_name,
                "coefficient": float(coefficient),
                "abs_coefficient": abs(float(coefficient)),
            }
        )
    rows.sort(key=lambda item: item["abs_coefficient"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_samples = load_jsonl(config["dataset"]["train_path"])
    dev_samples = load_jsonl(config["dataset"]["dev_path"])
    test_samples = load_jsonl(config["dataset"]["test_path"])
    top_k = int(config["retrieval"]["top_k"])
    feature_cache_config = config.get("feature_row_cache", {})

    train_feature_cache = (
        load_feature_row_cache(feature_cache_config["train_path"])
        if feature_cache_config.get("train_path")
        else None
    )
    dev_feature_cache = (
        load_feature_row_cache(feature_cache_config["dev_path"])
        if feature_cache_config.get("dev_path")
        else None
    )
    test_feature_cache = (
        load_feature_row_cache(feature_cache_config["test_path"])
        if feature_cache_config.get("test_path")
        else None
    )
    external_features = config.get("external_features", config.get("benchmarks_features", []))
    if external_features:
        for item in external_features:
            feature_name = str(item["name"])
            train_feature_cache = augment_feature_rows_with_external_scores(
                train_feature_cache,
                score_cache_path=str(item["train_score_cache_path"]),
                feature_name=feature_name,
            )
            dev_feature_cache = augment_feature_rows_with_external_scores(
                dev_feature_cache,
                score_cache_path=str(item["dev_score_cache_path"]),
                feature_name=feature_name,
            )
            test_feature_cache = augment_feature_rows_with_external_scores(
                test_feature_cache,
                score_cache_path=str(item["test_score_cache_path"]),
                feature_name=feature_name,
            )
    print("Loaded train/dev/test samples.")
    print("Building train/dev features and selecting best linear ranker...")
    training_results = fit_pointwise_logistic_ranker(
        train_samples,
        dev_samples,
        top_k=top_k,
        training_config=config.get("training"),
        train_feature_rows_by_sample_id=train_feature_cache,
        dev_feature_rows_by_sample_id=dev_feature_cache,
    )
    model = training_results["model"]
    print(f"Selected C={model['search']['selected_C']}. Reusing cached dev features.")
    dev_metrics = evaluate_ranking_dataset(
        training_results["dev_dataset"],
        model,
        top_k=top_k,
    )
    print("Building test features for final evaluation...")
    test_dataset = build_labeled_ranking_dataset(
        test_samples,
        base_feature_names=model["base_feature_names"],
        feature_transform=model["feature_transform"],
        feature_rows_by_sample_id=test_feature_cache,
    )
    test_metrics = evaluate_ranking_dataset(test_dataset, model, top_k=top_k)

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.json"
    save_linear_ranker_model(model, model_path)

    metrics = {
        "dataset_sizes": {
            "train": len(train_samples),
            "dev": len(dev_samples),
            "test": len(test_samples),
        },
        "training": model["search"],
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
    }

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

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    predictions_path = output_dir / "predictions_test.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    search_path = output_dir / "search_results.json"
    with search_path.open("w", encoding="utf-8") as handle:
        json.dump(training_results["search_results"], handle, ensure_ascii=False, indent=2)

    coefficients_path = output_dir / "coefficients.json"
    with coefficients_path.open("w", encoding="utf-8") as handle:
        json.dump(coefficient_rows(model), handle, ensure_ascii=False, indent=2)

    coefficients_md_path = output_dir / "coefficients.md"
    with coefficients_md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Learned Ranker Coefficients\n\n")
        handle.write("| feature | coefficient |\n")
        handle.write("| --- | ---: |\n")
        for row in coefficient_rows(model):
            handle.write(f"| {row['feature']} | {row['coefficient']:.4f} |\n")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
