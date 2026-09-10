from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import warnings

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.external_scores import load_score_feature_map
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    fit_pointwise_logistic_ranker,
    load_feature_row_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--feature-specs", required=True)
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=5)
    parser.add_argument("--must-include", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def clone_feature_cache(cache: dict[str, list[dict]]) -> dict[str, list[dict]]:
    cloned: dict[str, list[dict]] = {}
    for sample_id, feature_rows in cache.items():
        cloned[sample_id] = [
            {
                "context_id": row["context_id"],
                "text": row["text"],
                "features": dict(row["features"]),
            }
            for row in feature_rows
        ]
    return cloned


def apply_score_maps(
    cache: dict[str, list[dict]],
    score_maps_by_feature: dict[str, dict[str, dict[str, float]]],
    feature_names: list[str],
) -> dict[str, list[dict]]:
    augmented = clone_feature_cache(cache)
    for feature_name in feature_names:
        score_map = score_maps_by_feature[feature_name]
        for sample_id, feature_rows in augmented.items():
            sample_scores = score_map[sample_id]
            for row in feature_rows:
                row["features"][feature_name] = float(sample_scores[row["context_id"]])
    return augmented


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
    )
    args = parse_args()
    with open(args.base_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_samples = load_jsonl(config["dataset"]["train_path"])
    dev_samples = load_jsonl(config["dataset"]["dev_path"])
    test_samples = load_jsonl(config["dataset"]["test_path"])
    feature_cache_config = config["feature_row_cache"]
    train_base_cache = load_feature_row_cache(feature_cache_config["train_path"])
    dev_base_cache = load_feature_row_cache(feature_cache_config["dev_path"])
    test_base_cache = load_feature_row_cache(feature_cache_config["test_path"])
    feature_transform = config["training"]["feature_transform"]
    top_k = int(config["retrieval"]["top_k"])

    with open(args.feature_specs, "r", encoding="utf-8") as handle:
        feature_specs = yaml.safe_load(handle)

    score_maps_by_feature: dict[str, dict[str, dict[str, float]]] = {}
    for feature_name, spec in feature_specs.items():
        score_maps_by_feature[feature_name] = {
            split_name: load_score_feature_map(spec[f"{split_name}_score_cache_path"])
            for split_name in ("train", "dev", "test")
        }

    feature_names = sorted(score_maps_by_feature.keys())
    must_include = [item.strip() for item in args.must_include.split(",") if item.strip()]
    for feature_name in must_include:
        if feature_name not in feature_names:
            raise KeyError(f"Unknown must-include feature: {feature_name}")
    combinations_to_run = []
    for size in range(args.min_size, min(args.max_size, len(feature_names)) + 1):
        for combo in itertools.combinations(feature_names, size):
            if must_include and not all(feature_name in combo for feature_name in must_include):
                continue
            combinations_to_run.append(combo)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best = None
    for combo in combinations_to_run:
        combo_list = list(combo)
        print(f"Running combo: {combo_list}", flush=True)
        train_cache = apply_score_maps(
            train_base_cache,
            {name: score_maps_by_feature[name]["train"] for name in combo_list},
            combo_list,
        )
        dev_cache = apply_score_maps(
            dev_base_cache,
            {name: score_maps_by_feature[name]["dev"] for name in combo_list},
            combo_list,
        )
        test_cache = apply_score_maps(
            test_base_cache,
            {name: score_maps_by_feature[name]["test"] for name in combo_list},
            combo_list,
        )

        training_results = fit_pointwise_logistic_ranker(
            train_samples,
            dev_samples,
            top_k=top_k,
            training_config=config["training"],
            train_feature_rows_by_sample_id=train_cache,
            dev_feature_rows_by_sample_id=dev_cache,
        )
        model = training_results["model"]
        test_dataset = build_labeled_ranking_dataset(
            test_samples,
            base_feature_names=model["base_feature_names"],
            feature_transform=feature_transform,
            feature_rows_by_sample_id=test_cache,
        )
        test_metrics = evaluate_ranking_dataset(test_dataset, model, top_k=top_k)
        row = {
            "features": combo_list,
            "selected_C": model["search"]["selected_C"],
            "dev": {
                "hit@1": training_results["model"]["best_dev_metrics"]["hit@1"],
                "hit@3": training_results["model"]["best_dev_metrics"][f"hit@{top_k}"],
                "mrr": training_results["model"]["best_dev_metrics"]["mrr"],
            },
            "test": {
                "hit@1": test_metrics["hit@1"],
                "hit@3": test_metrics[f"hit@{top_k}"],
                "mrr": test_metrics["mrr"],
            },
        }
        results.append(row)
        if best is None or (
            row["dev"]["hit@1"],
            row["dev"]["mrr"],
            row["dev"]["hit@3"],
        ) > (
            best["row"]["dev"]["hit@1"],
            best["row"]["dev"]["mrr"],
            best["row"]["dev"]["hit@3"],
        ):
            best = {
                "row": row,
                "model": model,
                "test_metrics": test_metrics,
            }

        partial_results = sorted(
            results,
            key=lambda item: (
                item["dev"]["hit@1"],
                item["dev"]["mrr"],
                item["dev"]["hit@3"],
            ),
            reverse=True,
        )
        (output_dir / "search.partial.json").write_text(
            json.dumps(
                {
                    "must_include": must_include,
                    "completed": len(results),
                    "total": len(combinations_to_run),
                    "best": best["row"] if best else None,
                    "results": partial_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    results.sort(
        key=lambda item: (
            item["dev"]["hit@1"],
            item["dev"]["mrr"],
            item["dev"]["hit@3"],
        ),
        reverse=True,
    )

    (output_dir / "search.json").write_text(
        json.dumps(
            {
                "must_include": must_include,
                "best": best["row"] if best else None,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# External Feature Combo Search",
        "",
        f"- base config: `{args.base_config}`",
        f"- feature transform: `{feature_transform}`",
        "",
        "## Results",
        "",
        "| features | C | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {', '.join(row['features'])} | {row['selected_C']:.2f} | "
            f"{row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    (output_dir / "search.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
