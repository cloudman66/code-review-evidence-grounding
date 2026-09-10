from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def ranking_metrics(predicted_context_ids_by_sample: dict[str, list[str]], gold_by_sample: dict[str, set[str]], *, top_k: int) -> dict[str, float]:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    total = len(predicted_context_ids_by_sample) or 1
    for sample_id, ranked_context_ids in predicted_context_ids_by_sample.items():
        gold = gold_by_sample[sample_id]
        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))
        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_context_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
    }


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    cache = load_score_cache(Path(args.score_cache))
    top_k = int(args.top_k)

    sample_ids = [sample["sample_id"] for sample in samples]
    if sample_ids != cache["dataset"]["sample_ids"]:
        raise ValueError("dataset and score-cache sample_ids do not align")

    predicted_context_ids_by_sample: dict[str, list[str]] = {}
    gold_by_sample: dict[str, set[str]] = {}
    predictions: list[dict] = []
    offset = 0
    for sample, context_ids, group_size in zip(
        samples,
        cache["dataset"]["context_ids_by_group"],
        cache["dataset"]["group_sizes"],
        strict=True,
    ):
        next_offset = offset + int(group_size)
        scores = cache["scores"][offset:next_offset]
        ranked_indices = sorted(
            range(int(group_size)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )
        ranked_context_ids = [context_ids[index] for index in ranked_indices]
        predicted_context_ids_by_sample[sample["sample_id"]] = ranked_context_ids
        gold_by_sample[sample["sample_id"]] = set(sample["gold_context_ids"])
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[:top_k],
            }
        )
        offset = next_offset

    metrics = ranking_metrics(
        predicted_context_ids_by_sample,
        gold_by_sample,
        top_k=top_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model_path": cache["model_path"],
        "dataset_sizes": {"samples": len(samples)},
        "retrieval_test": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved predictions to {output_dir / 'predictions_test.jsonl'}")


if __name__ == "__main__":
    main()
