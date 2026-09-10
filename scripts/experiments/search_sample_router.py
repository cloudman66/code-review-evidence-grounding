from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_quoted_spans
from code_review_understanding.models.fusion import load_score_cache, normalize_group_scores
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--dev-cache-a", required=True)
    parser.add_argument("--test-cache-a", required=True)
    parser.add_argument("--dev-cache-b", required=True)
    parser.add_argument("--test-cache-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="-1.0,-0.5,-0.25,-0.1,-0.05,0.0,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--normalizations", default="raw,zscore,minmax,rank")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def make_condition_label(condition: tuple[str, str, str] | None) -> str:
    if condition is None:
        return "always"
    return f"{condition[0]}::{condition[1]}::{condition[2]}"


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


def list_conditions(samples: list[dict]) -> list[tuple[str, str, str]]:
    conditions: list[tuple[str, str, str]] = []
    for sample in samples:
        slices = sample_slices(sample)
        for name in ("code_fence", "suggestion", "mention", "quoted_span", "short_comment", "long_comment"):
            conditions.append(("binary", name, "yes" if slices[name] else "no"))
        conditions.append(("context_bucket", "context_bucket", str(slices["context_bucket"])))
    deduped = sorted(set(conditions))
    return deduped


def match_condition(sample: dict, condition: tuple[str, str, str] | None) -> bool:
    if condition is None:
        return True
    group, name, value = condition
    slices = sample_slices(sample)
    if group == "binary":
        return ("yes" if slices[name] else "no") == value
    if group == "context_bucket":
        return str(slices["context_bucket"]) == value
    raise ValueError(f"Unsupported condition group: {group}")


def build_sample_records(samples: list[dict], cache: dict, *, normalization: str) -> list[dict]:
    if [sample["sample_id"] for sample in samples] != cache["dataset"]["sample_ids"]:
        raise ValueError("Sample ids do not align with dataset order.")

    records: list[dict] = []
    offset = 0
    for sample, context_ids, gold_context_ids, group_size in zip(
        samples,
        cache["dataset"]["context_ids_by_group"],
        cache["dataset"]["gold_context_ids_by_group"],
        cache["dataset"]["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        scores = normalize_group_scores(cache["scores"][offset:next_offset], normalization)
        ranked_indices = np.argsort(-scores, kind="stable")
        ranked_context_ids = [context_ids[index] for index in ranked_indices]
        top1 = float(scores[ranked_indices[0]]) if len(ranked_indices) else 0.0
        top2 = float(scores[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
        records.append(
            {
                "sample": sample,
                "gold": set(gold_context_ids),
                "ranked_context_ids": ranked_context_ids,
                "margin": top1 - top2,
            }
        )
        offset = next_offset
    return records


def evaluate_router(
    records_a: list[dict],
    records_b: list[dict],
    *,
    top_k: int,
    condition: tuple[str, str, str] | None,
    threshold: float | None,
) -> dict:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    predictions: list[dict] = []

    for record_a, record_b in zip(records_a, records_b, strict=True):
        sample = record_a["sample"]
        use_b = False
        if condition is None:
            use_b = threshold is None or (record_b["margin"] - record_a["margin"]) >= threshold
        else:
            if match_condition(sample, condition):
                use_b = threshold is None or (record_b["margin"] - record_a["margin"]) >= threshold
        selected = record_b if use_b else record_a
        ranked_context_ids = selected["ranked_context_ids"]
        gold = selected["gold"]

        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))

        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_context_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[:top_k],
                "selected_model": "b" if use_b else "a",
            }
        )

    total = len(records_a) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
        "predictions": predictions,
    }


def score_key(result: dict, *, top_k: int) -> tuple[float, float, float]:
    return (result["hit@1"], result["mrr"], result[f"hit@{top_k}"])


def build_report(search_results: list[dict], best: dict, *, top_k: int) -> str:
    lines = ["# Sample Router Search", "", "## Best", ""]
    lines.append(f"- normalization: `{best['normalization']}`")
    lines.append(f"- condition: `{best['condition_label']}`")
    lines.append(f"- threshold: `{best['threshold']}`")
    lines.append(
        f"- dev: `{best['dev']['hit@1']:.4f} / {best['dev'][f'hit@{top_k}']:.4f} / {best['dev']['mrr']:.4f}`"
    )
    lines.append(
        f"- test: `{best['test']['hit@1']:.4f} / {best['test'][f'hit@{top_k}']:.4f} / {best['test']['mrr']:.4f}`"
    )
    lines.append("")
    lines.append("## Top Runs")
    lines.append("")
    lines.append("| normalization | condition | threshold | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(search_results, key=lambda item: score_key(item["dev"], top_k=top_k), reverse=True)[:20]:
        lines.append(
            f"| {row['normalization']} | {row['condition_label']} | {row['threshold']:.2f} | "
            f"{row['dev']['hit@1']:.4f} | {row['dev'][f'hit@{top_k}']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test'][f'hit@{top_k}']:.4f} | {row['test']['mrr']:.4f} |"
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

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    normalizations = [item.strip() for item in args.normalizations.split(",") if item.strip()]
    conditions = [None] + list_conditions(dev_samples)

    search_results: list[dict] = []
    best: dict | None = None
    best_test_predictions: list[dict] = []

    for normalization in normalizations:
        dev_records_a = build_sample_records(dev_samples, dev_cache_a, normalization=normalization)
        dev_records_b = build_sample_records(dev_samples, dev_cache_b, normalization=normalization)
        test_records_a = build_sample_records(test_samples, test_cache_a, normalization=normalization)
        test_records_b = build_sample_records(test_samples, test_cache_b, normalization=normalization)

        for condition in conditions:
            for threshold in thresholds:
                dev_result = evaluate_router(
                    dev_records_a,
                    dev_records_b,
                    top_k=args.top_k,
                    condition=condition,
                    threshold=threshold,
                )
                test_result = evaluate_router(
                    test_records_a,
                    test_records_b,
                    top_k=args.top_k,
                    condition=condition,
                    threshold=threshold,
                )
                row = {
                    "normalization": normalization,
                    "condition": condition,
                    "condition_label": make_condition_label(condition),
                    "threshold": threshold,
                    "dev": {
                        "hit@1": dev_result["hit@1"],
                        f"hit@{args.top_k}": dev_result[f"hit@{args.top_k}"],
                        "mrr": dev_result["mrr"],
                    },
                    "test": {
                        "hit@1": test_result["hit@1"],
                        f"hit@{args.top_k}": test_result[f"hit@{args.top_k}"],
                        "mrr": test_result["mrr"],
                    },
                }
                search_results.append(row)
                if best is None or score_key(dev_result, top_k=args.top_k) > score_key(best["dev"], top_k=args.top_k):
                    best = row
                    best_test_predictions = test_result["predictions"]

    if best is None:
        raise RuntimeError("No routing results produced.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search.json").write_text(
        json.dumps({"best": best, "results": search_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(search_results, best, top_k=args.top_k),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "router": {
                    "normalization": best["normalization"],
                    "condition": best["condition"],
                    "condition_label": best["condition_label"],
                    "threshold": best["threshold"],
                },
                "retrieval_dev": best["dev"],
                "retrieval_test": best["test"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in best_test_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved search to {output_dir / 'search.json'}")
    print(f"Saved report to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
