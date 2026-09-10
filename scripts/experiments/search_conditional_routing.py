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
    parser.add_argument("--dev-primary-cache", required=True)
    parser.add_argument("--test-primary-cache", required=True)
    parser.add_argument("--dev-secondary-cache", required=True)
    parser.add_argument("--test-secondary-cache", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def bucket_context_count(count: int) -> str:
    if count <= 10:
        return "<=10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    return ">40"


def sample_tags(sample: dict) -> set[str]:
    comment = sample["comment"]
    word_count = len(comment.split())
    tags = {
        f"context_bucket::{bucket_context_count(len(sample['contexts']))}",
        f"code_fence::{'yes' if '```' in comment else 'no'}",
        f"suggestion::{'yes' if 'suggestion' in comment.lower() else 'no'}",
        f"mention::{'yes' if '@' in comment else 'no'}",
        f"quoted_span::{'yes' if bool(extract_quoted_spans(comment)) else 'no'}",
        f"short_comment::{'yes' if word_count <= 6 else 'no'}",
        f"long_comment::{'yes' if word_count >= 30 else 'no'}",
    }
    return tags


def build_tag_map(samples: list[dict]) -> dict[str, set[str]]:
    return {sample["sample_id"]: sample_tags(sample) for sample in samples}


def candidate_rules(tag_map: dict[str, set[str]]) -> list[str]:
    all_tags: set[str] = set()
    for tags in tag_map.values():
        all_tags.update(tags)
    return sorted(all_tags)


def evaluate_route(
    samples: list[dict],
    primary_cache: dict,
    secondary_cache: dict,
    tag_map: dict[str, set[str]],
    *,
    rule: str,
) -> dict[str, float]:
    dataset = primary_cache["dataset"]
    if dataset["sample_ids"] != secondary_cache["dataset"]["sample_ids"]:
        raise ValueError("sample_ids do not align")
    if dataset["context_ids_by_group"] != secondary_cache["dataset"]["context_ids_by_group"]:
        raise ValueError("context_ids do not align")

    hit1 = 0
    hit3 = 0
    mrr = 0.0
    switched = 0
    offset = 0
    for sample_id, context_ids, gold_context_ids, group_size in zip(
        dataset["sample_ids"],
        dataset["context_ids_by_group"],
        dataset["gold_context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        use_secondary = rule != "__primary__" and rule in tag_map[sample_id]
        scores = (
            secondary_cache["scores"][offset:next_offset]
            if use_secondary
            else primary_cache["scores"][offset:next_offset]
        )
        if use_secondary:
            switched += 1

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
        offset = next_offset

    total = len(samples) or 1
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
        "switched": switched,
        "switch_rate": switched / total,
    }


def build_report(results: list[dict], best: dict) -> str:
    lines = ["# Conditional Routing Search", "", "## Best Rule", ""]
    lines.append(f"- rule: `{best['rule']}`")
    lines.append(f"- switched: `{best['dev']['switched']}` samples on dev")
    lines.append(f"- dev: `{best['dev']['hit@1']:.4f} / {best['dev']['hit@3']:.4f} / {best['dev']['mrr']:.4f}`")
    lines.append(f"- test: `{best['test']['hit@1']:.4f} / {best['test']['hit@3']:.4f} / {best['test']['mrr']:.4f}`")
    lines.append("")
    lines.append("## Top Rules")
    lines.append("")
    lines.append("| rule | dev switched | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results[:12]:
        lines.append(
            f"| {row['rule']} | {row['dev']['switched']} | "
            f"{row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dev_samples = load_jsonl(args.dev_dataset)
    test_samples = load_jsonl(args.test_dataset)
    dev_tag_map = build_tag_map(dev_samples)
    test_tag_map = build_tag_map(test_samples)

    dev_primary_cache = load_score_cache(Path(args.dev_primary_cache))
    test_primary_cache = load_score_cache(Path(args.test_primary_cache))
    dev_secondary_cache = load_score_cache(Path(args.dev_secondary_cache))
    test_secondary_cache = load_score_cache(Path(args.test_secondary_cache))

    rules = ["__primary__"] + candidate_rules(dev_tag_map)
    results: list[dict] = []
    best: dict | None = None
    for rule in rules:
        dev_metrics = evaluate_route(
            dev_samples,
            dev_primary_cache,
            dev_secondary_cache,
            dev_tag_map,
            rule=rule,
        )
        test_metrics = evaluate_route(
            test_samples,
            test_primary_cache,
            test_secondary_cache,
            test_tag_map,
            rule=rule,
        )
        row = {
            "rule": rule,
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
        raise RuntimeError("No routing results produced.")

    results.sort(
        key=lambda row: (
            row["dev"]["hit@1"],
            row["dev"]["mrr"],
            row["dev"]["hit@3"],
        ),
        reverse=True,
    )

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
