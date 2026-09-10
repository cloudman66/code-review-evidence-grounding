from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import build_context_feature_rows, rank_feature_rows
from code_review_understanding.data.io_utils import load_jsonl


VARIANTS = {
    "lexical_base": {
        "inline_code_weight": 0.0,
        "inline_path_weight": 0.0,
        "inline_line_weight": 0.0,
        "path_token_weight": 0.0,
        "path_exact_weight": 0.0,
        "word_tfidf_weight": 0.0,
        "char_tfidf_weight": 0.0,
        "quoted_span_weight": 0.0,
        "quoted_line_weight": 0.0,
    },
    "inline_code": {
        "inline_code_weight": 1.5,
        "inline_path_weight": 0.75,
        "inline_line_weight": 2.5,
        "path_token_weight": 0.0,
        "path_exact_weight": 0.0,
        "word_tfidf_weight": 0.0,
        "char_tfidf_weight": 0.0,
        "quoted_span_weight": 0.0,
        "quoted_line_weight": 0.0,
    },
    "inline_code+path": {
        "inline_code_weight": 1.5,
        "inline_path_weight": 0.75,
        "inline_line_weight": 2.5,
        "path_token_weight": 2.0,
        "path_exact_weight": 2.0,
        "word_tfidf_weight": 0.0,
        "char_tfidf_weight": 0.0,
        "quoted_span_weight": 0.0,
        "quoted_line_weight": 0.0,
    },
    "inline_code+char_tfidf": {
        "inline_code_weight": 1.5,
        "inline_path_weight": 0.75,
        "inline_line_weight": 2.5,
        "path_token_weight": 0.0,
        "path_exact_weight": 0.0,
        "word_tfidf_weight": 0.0,
        "char_tfidf_weight": 1.0,
        "quoted_span_weight": 0.0,
        "quoted_line_weight": 0.0,
    },
    "inline_code+word_char_tfidf": {
        "inline_code_weight": 1.5,
        "inline_path_weight": 0.75,
        "inline_line_weight": 2.5,
        "path_token_weight": 0.0,
        "path_exact_weight": 0.0,
        "word_tfidf_weight": 1.5,
        "char_tfidf_weight": 1.0,
        "quoted_span_weight": 0.0,
        "quoted_line_weight": 0.0,
    },
    "inline_code+word_char_tfidf+quoted": {
        "inline_code_weight": 1.5,
        "inline_path_weight": 0.75,
        "inline_line_weight": 2.5,
        "path_token_weight": 0.0,
        "path_exact_weight": 0.0,
        "word_tfidf_weight": 1.5,
        "char_tfidf_weight": 1.0,
        "quoted_span_weight": 3.0,
        "quoted_line_weight": 2.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Grounding yaml config.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "# Grounding Ranker Ablation",
        "",
        "| Variant | Dev Hit@1 | Dev Hit@3 | Dev MRR | Test Hit@1 | Test Hit@3 | Test MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | "
            f"{row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_cached(feature_cache: list[tuple[dict, list[dict]]], ranking_config: dict, top_k: int) -> dict:
    hit1 = 0
    hitk = 0
    mrr = 0.0

    for sample, feature_rows in feature_cache:
        ranked = rank_feature_rows(feature_rows, ranking_config=ranking_config)
        gold = set(sample["gold_context_ids"])
        ranked_ids = [row["context_id"] for row in ranked]
        hit1 += int(bool(ranked_ids) and ranked_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_ids[:top_k]))

        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank

    total = len(feature_cache) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
    }


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dev_samples = load_jsonl(config["dataset"]["dev_path"])
    test_samples = load_jsonl(config["dataset"]["test_path"])
    top_k = int(config["retrieval"]["top_k"])
    dev_cache = [(sample, build_context_feature_rows(sample)) for sample in dev_samples]
    test_cache = [(sample, build_context_feature_rows(sample)) for sample in test_samples]

    rows: list[dict] = []
    for variant_name, ranking_config in VARIANTS.items():
        dev = evaluate_cached(dev_cache, ranking_config=ranking_config, top_k=top_k)
        test = evaluate_cached(test_cache, ranking_config=ranking_config, top_k=top_k)
        rows.append(
            {
                "variant": variant_name,
                "ranking": ranking_config,
                "dev": {
                    "hit@1": dev["hit@1"],
                    "hit@3": dev[f"hit@{top_k}"],
                    "mrr": dev["mrr"],
                },
                "test": {
                    "hit@1": test["hit@1"],
                    "hit@3": test[f"hit@{top_k}"],
                    "mrr": test["mrr"],
                },
            }
        )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown_table(rows), encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved ablation json to {output_json}")
    print(f"Saved ablation markdown to {output_md}")


if __name__ == "__main__":
    main()
