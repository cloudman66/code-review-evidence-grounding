from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import save_feature_row_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to train/dev/test jsonl.")
    parser.add_argument("--output", required=True, help="Path to output feature-row cache jsonl or jsonl.gz.")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    save_feature_row_cache(samples, args.output)
    print({"samples": len(samples), "output": args.output})


if __name__ == "__main__":
    main()
