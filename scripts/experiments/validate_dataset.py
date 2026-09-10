from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl


REQUIRED_FIELDS = {"sample_id", "comment", "intent", "gold_context_ids", "contexts"}
REQUIRED_CONTEXT_FIELDS = {"context_id", "text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to a jsonl file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    records = load_jsonl(path)

    if not records:
        raise SystemExit(f"No records found in {path}")

    intent_counts: Counter[str] = Counter()
    context_count_sum = 0

    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise SystemExit(f"Record {index} missing required fields: {sorted(missing)}")

        if not isinstance(record["gold_context_ids"], list):
            raise SystemExit(f"Record {index} has non-list gold_context_ids")
        if not isinstance(record["contexts"], list) or not record["contexts"]:
            raise SystemExit(f"Record {index} has invalid contexts")

        for context in record["contexts"]:
            context_missing = REQUIRED_CONTEXT_FIELDS - set(context)
            if context_missing:
                raise SystemExit(
                    f"Record {index} context missing required fields: {sorted(context_missing)}"
                )

        intent_counts[record["intent"]] += 1
        context_count_sum += len(record["contexts"])

    avg_contexts = context_count_sum / len(records)
    print(f"Validated {len(records)} records from {path}")
    print(f"Unique intents: {len(intent_counts)}")
    print(f"Intent distribution: {dict(intent_counts)}")
    print(f"Average contexts per sample: {avg_contexts:.2f}")


if __name__ == "__main__":
    main()
