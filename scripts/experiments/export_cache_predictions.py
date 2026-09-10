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
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    cache = load_score_cache(Path(args.score_cache))

    sample_ids = cache["dataset"]["sample_ids"]
    context_ids_by_group = cache["dataset"]["context_ids_by_group"]
    group_sizes = cache["dataset"]["group_sizes"]

    if [sample["sample_id"] for sample in samples] != sample_ids:
        raise ValueError("dataset and score-cache sample_ids do not align")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample, context_ids, group_size in zip(
            samples,
            context_ids_by_group,
            group_sizes,
            strict=True,
        ):
            next_offset = offset + int(group_size)
            scores = cache["scores"][offset:next_offset]
            ranked_indices = sorted(
                range(int(group_size)),
                key=lambda index: float(scores[index]),
                reverse=True,
            )
            ranked_context_ids = [context_ids[index] for index in ranked_indices]
            row = {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[: args.top_k],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            offset = next_offset

    print(
        json.dumps(
            {
                "samples": len(samples),
                "top_k": args.top_k,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
