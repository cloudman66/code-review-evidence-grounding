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

from code_review_understanding.models.fusion import load_score_cache, normalize_group_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-cache-a", required=True)
    parser.add_argument("--test-cache-a", required=True)
    parser.add_argument("--dev-cache-b", required=True)
    parser.add_argument("--test-cache-b", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--weights", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5")
    parser.add_argument("--normalizations", default="raw,zscore,minmax,rank")
    return parser.parse_args()


def evaluate_two_way(cache_a: dict, cache_b: dict, *, normalization: str, weight_b: float) -> dict[str, float]:
    dataset = cache_a["dataset"]
    if dataset["sample_ids"] != cache_b["dataset"]["sample_ids"]:
        raise ValueError("sample_ids do not align")
    if dataset["context_ids_by_group"] != cache_b["dataset"]["context_ids_by_group"]:
        raise ValueError("context_ids do not align")

    hit1 = 0
    hit3 = 0
    mrr = 0.0
    offset = 0
    for context_ids, gold_context_ids, group_size in zip(
        dataset["context_ids_by_group"],
        dataset["gold_context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        scores_a = normalize_group_scores(cache_a["scores"][offset:next_offset], normalization)
        scores_b = normalize_group_scores(cache_b["scores"][offset:next_offset], normalization)
        fused_scores = scores_a + (weight_b * scores_b)
        ranked_indices = np.argsort(-fused_scores, kind="stable")
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
        offset = next_offset

    total = len(dataset["sample_ids"]) or 1
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
    }


def build_report(results: list[dict], best: dict) -> str:
    lines = ["# Two-Way Fusion Search", "", "## Best", ""]
    lines.append(
        f"- normalization: `{best['normalization']}`"
    )
    lines.append(
        f"- weight_b: `{best['weight_b']}`"
    )
    lines.append(
        f"- dev: `{best['dev']['hit@1']:.4f} / {best['dev']['hit@3']:.4f} / {best['dev']['mrr']:.4f}`"
    )
    lines.append(
        f"- test: `{best['test']['hit@1']:.4f} / {best['test']['hit@3']:.4f} / {best['test']['mrr']:.4f}`"
    )
    lines.append("")
    lines.append("## All Runs")
    lines.append("")
    lines.append("| normalization | weight_b | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        lines.append(
            f"| {row['normalization']} | {row['weight_b']:.2f} | "
            f"{row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dev_cache_a = load_score_cache(Path(args.dev_cache_a))
    test_cache_a = load_score_cache(Path(args.test_cache_a))
    dev_cache_b = load_score_cache(Path(args.dev_cache_b))
    test_cache_b = load_score_cache(Path(args.test_cache_b))

    weight_values = [float(item) for item in args.weights.split(",") if item.strip()]
    normalizations = [item.strip() for item in args.normalizations.split(",") if item.strip()]

    results: list[dict] = []
    best: dict | None = None
    for normalization in normalizations:
        for weight_b in weight_values:
            dev_metrics = evaluate_two_way(dev_cache_a, dev_cache_b, normalization=normalization, weight_b=weight_b)
            test_metrics = evaluate_two_way(test_cache_a, test_cache_b, normalization=normalization, weight_b=weight_b)
            row = {
                "normalization": normalization,
                "weight_b": weight_b,
                "dev": dev_metrics,
                "test": test_metrics,
            }
            results.append(row)
            if best is None or dev_metrics["hit@1"] > best["dev"]["hit@1"] or (
                dev_metrics["hit@1"] == best["dev"]["hit@1"]
                and dev_metrics["mrr"] > best["dev"]["mrr"]
            ) or (
                dev_metrics["hit@1"] == best["dev"]["hit@1"]
                and dev_metrics["mrr"] == best["dev"]["mrr"]
                and dev_metrics["hit@3"] > best["dev"]["hit@3"]
            ):
                best = row

    if best is None:
        raise RuntimeError("No search results produced.")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"best": best, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_report(results, best), encoding="utf-8")

    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


if __name__ == "__main__":
    main()
