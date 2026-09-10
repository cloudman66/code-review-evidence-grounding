from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_quoted_spans
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--dev-cache-a", required=True)
    parser.add_argument("--test-cache-a", required=True)
    parser.add_argument("--dev-cache-b", required=True)
    parser.add_argument("--test-cache-b", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-test-predictions", required=True)
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


def all_policies() -> list[dict]:
    binary_features = [
        "code_fence",
        "suggestion",
        "mention",
        "quoted_span",
        "short_comment",
        "long_comment",
    ]
    policies = [{"name": "always_a", "kind": "always_a"}]
    for feature_name in binary_features:
        policies.append(
            {
                "name": f"route_b_if_{feature_name}_yes",
                "kind": "binary",
                "feature": feature_name,
                "value": True,
            }
        )
        policies.append(
            {
                "name": f"route_b_if_{feature_name}_no",
                "kind": "binary",
                "feature": feature_name,
                "value": False,
            }
        )
    for bucket in ("<=10", "11-20", "21-40", ">40"):
        policies.append(
            {
                "name": f"route_b_if_context_bucket_{bucket.replace('>', 'gt').replace('<', 'le').replace('-', '_')}",
                "kind": "context_bucket",
                "value": bucket,
            }
        )
    return policies


def use_cache_b(policy: dict, sample: dict) -> bool:
    if policy["kind"] == "always_a":
        return False
    slices = sample_slices(sample)
    if policy["kind"] == "binary":
        return bool(slices[policy["feature"]]) == bool(policy["value"])
    if policy["kind"] == "context_bucket":
        return slices["context_bucket"] == policy["value"]
    raise ValueError(f"Unsupported policy kind: {policy['kind']}")


def evaluate_policy(samples: list[dict], cache_a: dict, cache_b: dict, policy: dict) -> tuple[dict, list[dict]]:
    dataset_a = cache_a["dataset"]
    dataset_b = cache_b["dataset"]
    if dataset_a["sample_ids"] != dataset_b["sample_ids"]:
        raise ValueError("sample_ids do not align")
    if dataset_a["context_ids_by_group"] != dataset_b["context_ids_by_group"]:
        raise ValueError("context_ids do not align")

    hit1 = 0
    hit3 = 0
    mrr = 0.0
    predictions: list[dict] = []
    offset = 0

    for sample, sample_id, context_ids, gold_context_ids, group_size in zip(
        samples,
        dataset_a["sample_ids"],
        dataset_a["context_ids_by_group"],
        dataset_a["gold_context_ids_by_group"],
        dataset_a["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        scores = cache_b["scores"][offset:next_offset] if use_cache_b(policy, sample) else cache_a["scores"][offset:next_offset]
        ranked_indices = sorted(range(group_size), key=lambda index: float(scores[index]), reverse=True)
        ranked_context_ids = [context_ids[index] for index in ranked_indices]
        gold = set(gold_context_ids)

        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hit3 += int(any(context_id in gold for context_id in ranked_context_ids[:3]))

        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_context_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank
        predictions.append(
            {
                "sample_id": sample_id,
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[:3],
                "policy_used": policy["name"],
                "used_model": "b" if use_cache_b(policy, sample) else "a",
            }
        )
        offset = next_offset

    total = len(samples) or 1
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
    }, predictions


def build_report(rows: list[dict], best: dict) -> str:
    lines = ["# Sample Routing Search", "", "## Best", ""]
    lines.append(f"- policy: `{best['policy']}`")
    lines.append(f"- dev: `{best['dev']['hit@1']:.4f} / {best['dev']['hit@3']:.4f} / {best['dev']['mrr']:.4f}`")
    lines.append(f"- test: `{best['test']['hit@1']:.4f} / {best['test']['hit@3']:.4f} / {best['test']['mrr']:.4f}`")
    lines.append("")
    lines.append("## All Policies")
    lines.append("")
    lines.append("| policy | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dev_samples = load_jsonl(args.dev_dataset)
    test_samples = load_jsonl(args.test_dataset)
    dev_cache_a = load_score_cache(Path(args.dev_cache_a))
    test_cache_a = load_score_cache(Path(args.test_cache_a))
    dev_cache_b = load_score_cache(Path(args.dev_cache_b))
    test_cache_b = load_score_cache(Path(args.test_cache_b))

    rows: list[dict] = []
    best: dict | None = None
    best_test_predictions: list[dict] = []
    for policy in all_policies():
        dev_metrics, _ = evaluate_policy(dev_samples, dev_cache_a, dev_cache_b, policy)
        test_metrics, test_predictions = evaluate_policy(test_samples, test_cache_a, test_cache_b, policy)
        row = {"policy": policy["name"], "dev": dev_metrics, "test": test_metrics}
        rows.append(row)
        if best is None or dev_metrics["hit@1"] > best["dev"]["hit@1"] or (
            dev_metrics["hit@1"] == best["dev"]["hit@1"]
            and dev_metrics["mrr"] > best["dev"]["mrr"]
        ) or (
            dev_metrics["hit@1"] == best["dev"]["hit@1"]
            and dev_metrics["mrr"] == best["dev"]["mrr"]
            and dev_metrics["hit@3"] > best["dev"]["hit@3"]
        ):
            best = row
            best_test_predictions = test_predictions

    if best is None:
        raise RuntimeError("No routing policies evaluated.")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_predictions = Path(args.output_test_predictions)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"best": best, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_report(rows, best), encoding="utf-8")
    with output_predictions.open("w", encoding="utf-8") as handle:
        for row in best_test_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")
    print(f"Saved best-policy predictions to {output_predictions}")


if __name__ == "__main__":
    main()
