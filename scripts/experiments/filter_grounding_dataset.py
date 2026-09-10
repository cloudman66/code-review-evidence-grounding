from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-contexts", type=int, default=0)
    parser.add_argument("--max-contexts", type=int, default=0)
    return parser.parse_args()


def include_sample(sample: dict, *, min_contexts: int, max_contexts: int) -> bool:
    context_count = len(sample["contexts"])
    if context_count < min_contexts:
        return False
    if max_contexts > 0 and context_count > max_contexts:
        return False
    return True


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    filtered = [
        sample
        for sample in samples
        if include_sample(
            sample,
            min_contexts=args.min_contexts,
            max_contexts=args.max_contexts,
        )
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in filtered:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        {
            "input": args.input,
            "output": args.output,
            "kept": len(filtered),
            "min_contexts": args.min_contexts,
            "max_contexts": args.max_contexts,
        }
    )


if __name__ == "__main__":
    main()
