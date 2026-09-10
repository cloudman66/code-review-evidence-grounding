from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_quoted_spans
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--train-primary-cache", required=True)
    parser.add_argument("--dev-primary-cache", required=True)
    parser.add_argument("--test-primary-cache", required=True)
    parser.add_argument("--train-secondary-cache", required=True)
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


def sample_meta(sample: dict) -> dict[str, float]:
    comment = sample["comment"]
    comment_lower = comment.lower()
    context_bucket = bucket_context_count(len(sample["contexts"]))
    word_count = len(comment.split())
    return {
        "context_bucket_<=10": 1.0 if context_bucket == "<=10" else 0.0,
        "context_bucket_11_20": 1.0 if context_bucket == "11-20" else 0.0,
        "context_bucket_21_40": 1.0 if context_bucket == "21-40" else 0.0,
        "context_bucket_gt40": 1.0 if context_bucket == ">40" else 0.0,
        "code_fence": 1.0 if "```" in comment else 0.0,
        "suggestion": 1.0 if "suggestion" in comment.lower() else 0.0,
        "mention": 1.0 if "@" in comment else 0.0,
        "quoted_span": 1.0 if bool(extract_quoted_spans(comment)) else 0.0,
        "short_comment": 1.0 if word_count <= 6 else 0.0,
        "long_comment": 1.0 if word_count >= 30 else 0.0,
        "question": 1.0 if "?" in comment else 0.0,
        "intent_typo": 1.0
        if any(token in comment_lower for token in ("typo", "spelled wrong", "spelling", "misspell"))
        else 0.0,
        "intent_naming": 1.0
        if any(token in comment_lower for token in ("variable name", "nicer variable", "rename", "name ", "named "))
        else 0.0,
        "intent_remove": 1.0
        if any(token in comment_lower for token in ("remove", "delete", "redundant", "unnecessary", "comment this out"))
        else 0.0,
        "intent_simplify": 1.0
        if any(
            token in comment_lower
            for token in ("simplify", "just ", "instead", "nicer", "can we", "possible to")
        )
        else 0.0,
        "intent_exception": 1.0
        if any(
            token in comment_lower
            for token in ("exception", "try catch", "catch ", "raise ", "error", "typeerror")
        )
        else 0.0,
        "intent_docs_text": 1.0
        if any(
            token in comment_lower
            for token in ("docs", "documentation", "link", "wording", "style guide", "spelled", "word ")
        )
        else 0.0,
        "intent_api_usage": 1.0
        if any(
            token in comment_lower
            for token in ("import", "dependency", "undefined", "parameter", "argument", "callable", "function")
        )
        else 0.0,
        "word_count": float(word_count),
        "context_count": float(len(sample["contexts"])),
    }


def topk_stats(scores: np.ndarray, context_ids: list[str], gold: set[str]) -> dict[str, float | str]:
    ranked_indices = np.argsort(-scores, kind="stable")
    ranked_context_ids = [context_ids[index] for index in ranked_indices]
    ranked_scores = scores[ranked_indices]
    top1_score = float(ranked_scores[0]) if len(ranked_scores) else 0.0
    top2_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else top1_score
    margin = top1_score - top2_score
    top1_id = ranked_context_ids[0] if ranked_context_ids else ""
    top1_correct = 1.0 if top1_id in gold else 0.0

    reciprocal_rank = 0.0
    for index, context_id in enumerate(ranked_context_ids, start=1):
        if context_id in gold:
            reciprocal_rank = 1.0 / index
            break

    return {
        "top1_id": top1_id,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "top1_correct": top1_correct,
        "mrr": reciprocal_rank,
        "top3_hit": 1.0 if any(context_id in gold for context_id in ranked_context_ids[:3]) else 0.0,
    }


def build_sample_rows(samples: list[dict], primary_cache: dict, secondary_cache: dict) -> list[dict]:
    if primary_cache["dataset"]["sample_ids"] != secondary_cache["dataset"]["sample_ids"]:
        raise ValueError("sample_ids do not align")

    rows: list[dict] = []
    offset = 0
    for sample, sample_id, context_ids, gold_context_ids, group_size in zip(
        samples,
        primary_cache["dataset"]["sample_ids"],
        primary_cache["dataset"]["context_ids_by_group"],
        primary_cache["dataset"]["gold_context_ids_by_group"],
        primary_cache["dataset"]["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        gold = set(gold_context_ids)
        primary_scores = primary_cache["scores"][offset:next_offset]
        secondary_scores = secondary_cache["scores"][offset:next_offset]
        primary_stats = topk_stats(primary_scores, context_ids, gold)
        secondary_stats = topk_stats(secondary_scores, context_ids, gold)
        meta = sample_meta(sample)
        meta.update(
            {
                "primary_top1_score": float(primary_stats["top1_score"]),
                "primary_margin": float(primary_stats["margin"]),
                "secondary_top1_score": float(secondary_stats["top1_score"]),
                "secondary_margin": float(secondary_stats["margin"]),
                "top1_same": 1.0 if primary_stats["top1_id"] == secondary_stats["top1_id"] else 0.0,
                "score_gap_top1": float(primary_stats["top1_score"]) - float(secondary_stats["top1_score"]),
                "margin_gap": float(primary_stats["margin"]) - float(secondary_stats["margin"]),
            }
        )
        rows.append(
            {
                "sample_id": sample_id,
                "features": meta,
                "primary": primary_stats,
                "secondary": secondary_stats,
                "label_secondary_better_top1": 1.0
                if secondary_stats["top1_correct"] > primary_stats["top1_correct"]
                else 0.0,
            }
        )
        offset = next_offset
    return rows


def rows_to_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )


def evaluate_threshold(rows: list[dict], probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    hit1 = 0
    hit3 = 0
    mrr = 0.0
    switched = 0
    for row, probability in zip(rows, probabilities, strict=True):
        use_secondary = float(probability) >= threshold
        stats = row["secondary"] if use_secondary else row["primary"]
        if use_secondary:
            switched += 1
        hit1 += int(float(stats["top1_correct"]) > 0.0)
        hit3 += int(float(stats["top3_hit"]) > 0.0)
        mrr += float(stats["mrr"])
    total = len(rows) or 1
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
        "switched": switched,
        "switch_rate": switched / total,
    }


def build_report(best: dict, threshold_rows: list[dict]) -> str:
    lines = ["# Score Router Search", "", "## Best", ""]
    lines.append(f"- selected threshold: `{best['threshold']:.2f}`")
    lines.append(f"- dev: `{best['dev']['hit@1']:.4f} / {best['dev']['hit@3']:.4f} / {best['dev']['mrr']:.4f}`")
    lines.append(f"- test: `{best['test']['hit@1']:.4f} / {best['test']['hit@3']:.4f} / {best['test']['mrr']:.4f}`")
    lines.append("")
    lines.append("## Threshold Sweep")
    lines.append("")
    lines.append("| threshold | dev switched | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in threshold_rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['dev']['switched']} | "
            f"{row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_samples = load_jsonl(args.train_dataset)
    dev_samples = load_jsonl(args.dev_dataset)
    test_samples = load_jsonl(args.test_dataset)

    train_rows = build_sample_rows(
        train_samples,
        load_score_cache(Path(args.train_primary_cache)),
        load_score_cache(Path(args.train_secondary_cache)),
    )
    dev_rows = build_sample_rows(
        dev_samples,
        load_score_cache(Path(args.dev_primary_cache)),
        load_score_cache(Path(args.dev_secondary_cache)),
    )
    test_rows = build_sample_rows(
        test_samples,
        load_score_cache(Path(args.test_primary_cache)),
        load_score_cache(Path(args.test_secondary_cache)),
    )

    feature_names = sorted(train_rows[0]["features"].keys())
    X_train = rows_to_matrix(train_rows, feature_names)
    y_train = np.asarray([int(row["label_secondary_better_top1"]) for row in train_rows], dtype=np.int8)
    X_dev = rows_to_matrix(dev_rows, feature_names)
    X_test = rows_to_matrix(test_rows, feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression(
        C=0.25,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    classifier.fit(X_train_scaled, y_train)

    dev_probabilities = classifier.predict_proba(X_dev_scaled)[:, 1]
    test_probabilities = classifier.predict_proba(X_test_scaled)[:, 1]
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    threshold_rows: list[dict] = []
    best: dict | None = None
    for threshold in thresholds:
        dev_metrics = evaluate_threshold(dev_rows, dev_probabilities, threshold)
        test_metrics = evaluate_threshold(test_rows, test_probabilities, threshold)
        row = {
            "threshold": threshold,
            "dev": dev_metrics,
            "test": test_metrics,
        }
        threshold_rows.append(row)
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
        raise RuntimeError("No thresholds evaluated.")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "best": best,
                "thresholds": threshold_rows,
                "feature_names": feature_names,
                "router_coefficients": classifier.coef_[0].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_md.write_text(build_report(best, threshold_rows), encoding="utf-8")

    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


if __name__ == "__main__":
    main()
