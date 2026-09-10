from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_quoted_spans
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the evaluation jsonl dataset.")
    parser.add_argument("--predictions", required=True, help="Path to predictions_test.jsonl.")
    parser.add_argument("--output-json", required=True, help="Path to output analysis json.")
    parser.add_argument("--output-md", required=True, help="Path to output markdown report.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--failure-limit", type=int, default=15)
    return parser.parse_args()


def bucket_context_count(count: int) -> str:
    if count <= 10:
        return "<=10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    return ">40"


def sample_slices(sample: dict) -> dict[str, bool | str]:
    comment = sample["comment"]
    word_count = len(comment.split())
    context_count = len(sample["contexts"])
    return {
        "code_fence": "```" in comment,
        "suggestion": "suggestion" in comment.lower(),
        "mention": "@" in comment,
        "quoted_span": bool(extract_quoted_spans(comment)),
        "short_comment": word_count <= 6,
        "long_comment": word_count >= 30,
        "context_bucket": bucket_context_count(context_count),
    }


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_report(analysis: dict, failures: list[dict]) -> str:
    lines: list[str] = []
    overall = analysis["overall"]
    hit_k_key = f"hit@{overall['top_k']}"
    lines.append("# SWE-CARE Grounding Slice Analysis")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- samples: {overall['samples']}")
    lines.append(f"- hit@1: {overall['hit@1']:.4f}")
    lines.append(f"- {hit_k_key}: {overall[hit_k_key]:.4f}")
    lines.append("")
    lines.append("## Slice Metrics")
    lines.append("")

    for slice_name, slice_metrics in analysis["slices"].items():
        lines.append(f"### {slice_name}")
        lines.append("")
        for key, metrics in slice_metrics.items():
            if metrics["samples"] == 0:
                continue
            lines.append(
                f"- {key}: samples={metrics['samples']}, hit@1={metrics['hit@1']:.4f}, "
                f"{hit_k_key}={metrics[hit_k_key]:.4f}"
            )
        lines.append("")

    lines.append("## Failure Examples")
    lines.append("")
    for failure in failures:
        lines.append(f"### {failure['sample_id']}")
        lines.append("")
        lines.append(f"- comment: {failure['comment']}")
        lines.append(f"- gold: {', '.join(failure['gold_context_ids'])}")
        lines.append(f"- predicted_topk: {', '.join(failure['predicted_context_ids_topk'])}")
        lines.append(f"- context_count: {failure['context_count']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    dataset = {sample["sample_id"]: sample for sample in load_jsonl(args.dataset)}
    predictions = load_jsonl(args.predictions)

    overall_hits_1 = 0
    overall_hits_k = 0
    slice_stats: dict[str, dict[str, dict[str, int]]] = {
        "binary": defaultdict(lambda: {"samples": 0, "hit@1": 0, f"hit@{args.top_k}": 0}),
        "context_bucket": defaultdict(lambda: {"samples": 0, "hit@1": 0, f"hit@{args.top_k}": 0}),
    }
    failures: list[dict] = []

    for prediction in predictions:
        sample = dataset[prediction["sample_id"]]
        gold = set(prediction["gold_context_ids"])
        predicted = prediction["predicted_context_ids_topk"]
        hit_1 = int(bool(predicted) and predicted[0] in gold)
        hit_k = int(any(context_id in gold for context_id in predicted[: args.top_k]))

        overall_hits_1 += hit_1
        overall_hits_k += hit_k

        slices = sample_slices(sample)
        for slice_name in ("code_fence", "suggestion", "mention", "quoted_span", "short_comment", "long_comment"):
            bucket = "yes" if slices[slice_name] else "no"
            slice_stats["binary"][(slice_name, bucket)]["samples"] += 1
            slice_stats["binary"][(slice_name, bucket)]["hit@1"] += hit_1
            slice_stats["binary"][(slice_name, bucket)][f"hit@{args.top_k}"] += hit_k

        context_bucket = str(slices["context_bucket"])
        slice_stats["context_bucket"][context_bucket]["samples"] += 1
        slice_stats["context_bucket"][context_bucket]["hit@1"] += hit_1
        slice_stats["context_bucket"][context_bucket][f"hit@{args.top_k}"] += hit_k

        if not hit_k and len(failures) < args.failure_limit:
            failures.append(
                {
                    "sample_id": sample["sample_id"],
                    "comment": sample["comment"].replace("\n", "\\n")[:400],
                    "gold_context_ids": sample["gold_context_ids"],
                    "predicted_context_ids_topk": predicted[: args.top_k],
                    "context_count": len(sample["contexts"]),
                }
            )

    def normalize(group: dict) -> dict:
        normalized = {}
        for key, metrics in group.items():
            samples = metrics["samples"]
            normalized_key = "::".join(key) if isinstance(key, tuple) else str(key)
            normalized[normalized_key] = {
                "samples": samples,
                "hit@1": safe_div(metrics["hit@1"], samples),
                f"hit@{args.top_k}": safe_div(metrics[f"hit@{args.top_k}"], samples),
            }
        return dict(sorted(normalized.items()))

    analysis = {
        "overall": {
            "samples": len(predictions),
            "top_k": args.top_k,
            "hit@1": safe_div(overall_hits_1, len(predictions)),
            f"hit@{args.top_k}": safe_div(overall_hits_k, len(predictions)),
        },
        "slices": {
            "binary": normalize(slice_stats["binary"]),
            "context_bucket": normalize(slice_stats["context_bucket"]),
        },
        "failures": failures,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_report(analysis, failures), encoding="utf-8")

    print(json.dumps(analysis["overall"], ensure_ascii=False, indent=2))
    print(f"Saved JSON analysis to {output_json}")
    print(f"Saved Markdown report to {output_md}")


if __name__ == "__main__":
    main()
