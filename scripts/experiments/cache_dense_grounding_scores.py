from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.models.dense_retrieval import build_dense_score_cache
from code_review_understanding.models.fusion import write_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset jsonl to encode into a score cache.")
    parser.add_argument("--output", required=True, help="Output score cache (.json or .json.gz).")
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--query-mode", default="expanded")
    parser.add_argument("--context-mode", default="full")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--context-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]

    dataset, scores = build_dense_score_cache(
        samples,
        model_name=args.model_name,
        query_mode=args.query_mode,
        context_mode=args.context_mode,
        query_prefix=args.query_prefix,
        context_prefix=args.context_prefix,
        batch_size=int(args.batch_size),
        parallel=(None if args.parallel <= 0 else int(args.parallel)),
        cache_dir=(args.cache_dir or None),
        threads=(None if args.threads <= 0 else int(args.threads)),
    )
    output_path = Path(args.output)
    write_score_cache(
        output_path,
        model_path=(
            f"dense::{args.model_name}"
            f"::query_mode={args.query_mode}"
            f"::context_mode={args.context_mode}"
            f"::query_prefix={args.query_prefix or '<none>'}"
            f"::context_prefix={args.context_prefix or '<none>'}"
        ),
        dataset=dataset,
        scores=scores,
    )

    print(
        {
            "samples": len(dataset["sample_ids"]),
            "contexts": len(scores),
            "model_name": args.model_name,
            "query_mode": args.query_mode,
            "context_mode": args.context_mode,
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
