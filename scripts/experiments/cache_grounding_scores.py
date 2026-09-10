from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.external_scores import (
    augment_feature_rows_with_external_scores,
    validate_feature_rows_have_features,
)
from code_review_understanding.models.fusion import build_score_cache, write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    linear_model_scores,
    load_feature_row_cache,
    load_linear_ranker_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to learned ranker model.json.")
    parser.add_argument("--dataset", required=True, help="Path to train/dev/test jsonl.")
    parser.add_argument("--output", required=True, help="Path to output score cache (.json or .json.gz).")
    parser.add_argument(
        "--feature-row-cache",
        default="",
        help="Optional feature-row cache jsonl(.gz). Required when the model depends on external features.",
    )
    parser.add_argument(
        "--external-score",
        action="append",
        default=[],
        help="Attach external score feature as name=path_to_score_cache.json(.gz).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke checks.")
    return parser.parse_args()


def parse_external_score_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --external-score spec: {spec}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]

    output_path = Path(args.output)
    if args.feature_row_cache or args.external_score:
        if not args.feature_row_cache:
            raise ValueError("--feature-row-cache is required when using --external-score.")

        model = load_linear_ranker_model(args.model_path)
        feature_rows_by_sample_id = load_feature_row_cache(args.feature_row_cache)
        for spec in args.external_score:
            feature_name, score_cache_path = parse_external_score_spec(spec)
            feature_rows_by_sample_id = augment_feature_rows_with_external_scores(
                feature_rows_by_sample_id,
                score_cache_path=score_cache_path,
                feature_name=feature_name,
            )
        validate_feature_rows_have_features(
            feature_rows_by_sample_id,
            required_feature_names=model["base_feature_names"],
        )
        dataset = build_labeled_ranking_dataset(
            samples,
            base_feature_names=model["base_feature_names"],
            feature_transform=model["feature_transform"],
            feature_rows_by_sample_id=feature_rows_by_sample_id,
        )
        scores = linear_model_scores(dataset["X"], model)
        write_score_cache(
            output_path,
            model_path=str(args.model_path),
            dataset=dataset,
            scores=scores,
        )
        print(
            {
                "samples": len(dataset["sample_ids"]),
                "contexts": len(scores),
                "output": str(output_path),
                "mode": "feature_row_cache",
                "benchmarks_features": [spec.split("=", 1)[0] for spec in args.external_score],
            }
        )
        return

    _, items = build_score_cache(samples, [{"model_path": args.model_path, "weight": 1.0}])
    item = items[0]
    write_score_cache(output_path, model_path=str(item["model_path"]), dataset=item["dataset"], scores=item["scores"])
    print({"samples": len(item["dataset"]["sample_ids"]), "contexts": len(item["scores"]), "output": str(output_path), "mode": "model_only"})


if __name__ == "__main__":
    main()
