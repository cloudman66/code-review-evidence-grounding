from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-parquet", default="benchmarks/swe_care/dev.parquet")
    parser.add_argument("--test-parquet", default="benchmarks/swe_care/test.parquet")
    parser.add_argument("--output-dir", default="data/processed/swe_care_grounding")
    parser.add_argument("--language", default="Python")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_comment_text(text: str) -> str:
    text = text.replace("\r", "").strip()
    text = re.split(r"\n@author:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"^@\w+:\s*", "", text)
    return text.strip()


def split_patch_to_hunks(patch: str) -> list[dict]:
    contexts: list[dict] = []
    current_file = "unknown"
    current_lines: list[str] = []
    current_hunk_id = 0

    def flush() -> None:
        nonlocal current_lines, current_hunk_id
        if current_lines:
            current_hunk_id += 1
            contexts.append(
                {
                    "context_id": f"{current_file}::hunk_{current_hunk_id}",
                    "source": "diff_hunk",
                    "text": f"path: {current_file}\n" + "\n".join(current_lines).strip(),
                }
            )
            current_lines = []

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            flush()
        elif line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            flush()
            current_lines = [line]
        elif current_lines:
            current_lines.append(line)

    flush()
    return contexts


def match_gold_context_ids(contexts: list[dict], diff_hunk: str, path: str | None) -> list[str]:
    gold_ids = []
    needle = (diff_hunk or "").strip()
    for context in contexts:
        if needle and needle in context["text"]:
            gold_ids.append(context["context_id"])

    if gold_ids:
        return gold_ids

    if path:
        prefix = f"path: {path}\n"
        for context in contexts:
            if context["text"].startswith(prefix):
                gold_ids.append(context["context_id"])
                break

    return gold_ids


def load_grounding_records(parquet_path: str, language: str) -> list[dict]:
    table = pq.read_table(
        parquet_path,
        columns=["instance_id", "repo", "language", "commit_to_review", "reference_review_comments", "metadata"],
    )

    records: list[dict] = []
    for row in table.to_pylist():
        if row["language"] != language:
            continue

        patch = row["commit_to_review"]["patch_to_review"]
        contexts = split_patch_to_hunks(patch)
        if len(contexts) < 2:
            continue

        for idx, review in enumerate(row["reference_review_comments"]):
            comment_text = clean_comment_text(review.get("text") or "")
            if not comment_text:
                continue

            gold_context_ids = match_gold_context_ids(contexts, review.get("diff_hunk") or "", review.get("path"))
            if not gold_context_ids:
                continue

            records.append(
                {
                    "sample_id": f"{row['instance_id']}::comment_{idx}",
                    "comment": comment_text,
                    "intent": "review_comment",
                    "gold_context_ids": gold_context_ids,
                    "contexts": contexts,
                    "metadata": {
                        "repo": row["repo"],
                        "difficulty": row["metadata"]["difficulty"],
                        "problem_domain": row["metadata"]["problem_domain"],
                        "path": review.get("path"),
                    },
                }
            )

    return records


def split_dev_train(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    items = records[:]
    rng.shuffle(items)
    n_train = int(len(items) * 0.9)
    return items[:n_train], items[n_train:]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_records = load_grounding_records(args.dev_parquet, language=args.language)
    test_records = load_grounding_records(args.test_parquet, language=args.language)
    train_records, heldout_dev_records = split_dev_train(dev_records, seed=args.seed)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "dev.jsonl", heldout_dev_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    summary = {
        "train": len(train_records),
        "dev": len(heldout_dev_records),
        "test": len(test_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
