from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_context_path
from code_review_understanding.models.fusion import load_score_cache, write_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--input-score-cache", required=True, help="Input score cache to aggregate.")
    parser.add_argument("--output", required=True, help="Output aggregated score cache.")
    parser.add_argument("--agg", choices=("max", "mean", "top2_mean"), default="mean")
    parser.add_argument(
        "--normalize",
        choices=("none", "minmax", "zscore"),
        default="none",
        help="Optional within-sample normalization applied after file aggregation.",
    )
    return parser.parse_args()


def aggregate(values: list[float], mode: str) -> float:
    ordered = sorted(values, reverse=True)
    if mode == "max":
        return float(ordered[0])
    if mode == "mean":
        return float(sum(ordered) / len(ordered))
    if mode == "top2_mean":
        return float(sum(ordered[:2]) / min(2, len(ordered)))
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def normalize_scores(values: list[float], mode: str) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    if mode == "none":
        return arr.tolist()
    if mode == "minmax":
        lower = float(arr.min())
        upper = float(arr.max())
        if upper - lower < 1e-6:
            return np.ones_like(arr, dtype=np.float32).tolist()
        return ((arr - lower) / (upper - lower)).astype(np.float32).tolist()
    if mode == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-6:
            return (arr - mean).astype(np.float32).tolist()
        return ((arr - mean) / std).astype(np.float32).tolist()
    raise ValueError(f"Unsupported normalization mode: {mode}")


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    cache = load_score_cache(Path(args.input_score_cache))
    dataset = cache["dataset"]

    sample_ids = [sample["sample_id"] for sample in samples]
    if sample_ids != dataset["sample_ids"]:
        raise ValueError("Dataset sample_ids do not align with score cache.")

    aggregated_scores: list[float] = []
    offset = 0
    for sample, context_ids, group_size in zip(
        samples,
        dataset["context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        group_scores = cache["scores"][offset:next_offset]
        contexts = sample["contexts"]
        if len(contexts) != group_size:
            raise ValueError(f"Context count mismatch for sample_id={sample['sample_id']}")

        path_to_scores: dict[str, list[float]] = defaultdict(list)
        context_paths: list[str] = []
        for context, score in zip(contexts, group_scores, strict=True):
            file_path = extract_context_path(context["text"])
            context_paths.append(file_path)
            path_to_scores[file_path].append(float(score))

        file_score_map = {
            path: aggregate(scores, args.agg) for path, scores in path_to_scores.items()
        }
        aligned = [file_score_map[path] for path in context_paths]
        aggregated_scores.extend(normalize_scores(aligned, args.normalize))
        offset = next_offset

    output_path = Path(args.output)
    write_score_cache(
        output_path,
        model_path=f"{cache['model_path']}::file_prior::{args.agg}::{args.normalize}",
        dataset=dataset,
        scores=np.asarray(aggregated_scores, dtype=np.float32),
    )
    print(
        {
            "samples": len(dataset["sample_ids"]),
            "contexts": len(aggregated_scores),
            "agg": args.agg,
            "normalize": args.normalize,
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
