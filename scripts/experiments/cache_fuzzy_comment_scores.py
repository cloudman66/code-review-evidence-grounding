from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import (
    clean_comment_text,
    extract_changed_lines,
    extract_context_path,
    extract_inline_code_spans,
    extract_path_mentions,
    extract_quoted_spans,
    extract_replacement_pairs,
    split_path_parts,
    tokenize_retrieval,
)
from code_review_understanding.models.fusion import write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.semantic_retrieval import (
    dedupe_keep_order,
    filtered_query_terms,
    review_intent_hints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--output", required=True, help="Output score cache path.")
    parser.add_argument(
        "--min-token-len",
        type=int,
        default=3,
        help="Minimum token length kept for fuzzy matching.",
    )
    parser.add_argument(
        "--focus-mode",
        choices=("all", "focused"),
        default="all",
        help="Whether to score against all query tokens or prioritize quoted/inline/replacement/path tokens.",
    )
    return parser.parse_args()


def keep_token(token: str, *, min_token_len: int) -> bool:
    return len(token) >= min_token_len and not token.isdigit()


def comment_query_tokens(comment: str, *, min_token_len: int) -> list[str]:
    tokens: list[str] = []
    base_text = clean_comment_text(comment)
    tokens.extend(filtered_query_terms(base_text))
    for span in extract_inline_code_spans(comment):
        tokens.extend(filtered_query_terms(span))
    for span in extract_quoted_spans(comment):
        tokens.extend(filtered_query_terms(span))
    for old, new in extract_replacement_pairs(comment):
        tokens.extend(filtered_query_terms(old))
        tokens.extend(filtered_query_terms(new))
    for path in extract_path_mentions(comment):
        tokens.extend(split_path_parts(path))
    for hint in review_intent_hints(comment):
        tokens.extend(filtered_query_terms(hint))
    filtered = [token for token in dedupe_keep_order(tokens) if keep_token(token, min_token_len=min_token_len)]
    return filtered


def comment_focus_tokens(comment: str, *, min_token_len: int) -> list[str]:
    tokens: list[str] = []
    for span in extract_inline_code_spans(comment):
        tokens.extend(filtered_query_terms(span))
    for span in extract_quoted_spans(comment):
        tokens.extend(filtered_query_terms(span))
    for old, new in extract_replacement_pairs(comment):
        tokens.extend(filtered_query_terms(old))
        tokens.extend(filtered_query_terms(new))
    for path in extract_path_mentions(comment):
        tokens.extend(split_path_parts(path))
    filtered = [token for token in dedupe_keep_order(tokens) if keep_token(token, min_token_len=min_token_len)]
    return filtered


def context_candidate_tokens(context_text: str, *, min_token_len: int) -> list[str]:
    path = extract_context_path(context_text)
    added_lines, removed_lines = extract_changed_lines(context_text)
    tokens: list[str] = []
    tokens.extend(split_path_parts(path))
    tokens.extend(tokenize_retrieval(" ".join(sorted(added_lines | removed_lines))))
    filtered = [token for token in dedupe_keep_order(tokens) if keep_token(token, min_token_len=min_token_len)]
    return filtered


def token_similarity(query_token: str, candidate_tokens: list[str]) -> float:
    if not candidate_tokens:
        return 0.0
    best = 0.0
    query_len = len(query_token)
    for candidate in candidate_tokens:
        if candidate == query_token:
            return 1.0
        if abs(len(candidate) - query_len) > max(2, query_len // 2):
            continue
        if query_token[0] != candidate[0] and query_len >= 5:
            shared = len(set(query_token) & set(candidate))
            if shared < max(2, min(len(set(query_token)), len(set(candidate))) // 2):
                continue
        score = SequenceMatcher(None, query_token, candidate).ratio()
        if score > best:
            best = score
    return best


def build_score_cache(
    samples: list[dict],
    *,
    min_token_len: int,
    focus_mode: str,
) -> tuple[dict, np.ndarray]:
    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    group_sizes: list[int] = []
    score_blocks: list[np.ndarray] = []

    for sample in samples:
        query_tokens = comment_query_tokens(sample["comment"], min_token_len=min_token_len)
        focus_tokens = comment_focus_tokens(sample["comment"], min_token_len=min_token_len)
        lowered_comment = sample["comment"].lower()
        typo_or_naming = any(
            needle in lowered_comment
            for needle in ("typo", "spelling", "spelled", "misspell", "rename", "name ", "named ")
        )

        sample_scores: list[float] = []
        context_ids: list[str] = []
        for context in sample["contexts"]:
            context_ids.append(context["context_id"])
            candidate_tokens = context_candidate_tokens(context["text"], min_token_len=min_token_len)
            if not query_tokens or not candidate_tokens:
                sample_scores.append(0.0)
                continue

            best_scores = [token_similarity(token, candidate_tokens) for token in query_tokens]
            exact_hits = sum(1 for score in best_scores if score >= 0.999)
            near_hits = sum(1 for score in best_scores if score >= 0.84)
            fuzzy_mean = float(sum(best_scores) / len(best_scores))
            exact_ratio = exact_hits / len(query_tokens)
            near_ratio = near_hits / len(query_tokens)

            score = (0.45 * exact_ratio) + (0.35 * near_ratio) + (0.20 * fuzzy_mean)
            if typo_or_naming:
                score = (0.25 * exact_ratio) + (0.45 * near_ratio) + (0.30 * fuzzy_mean)

            if focus_mode == "focused" and focus_tokens:
                focus_scores = [token_similarity(token, candidate_tokens) for token in focus_tokens]
                focus_exact_ratio = sum(1 for score in focus_scores if score >= 0.999) / len(focus_tokens)
                focus_near_ratio = sum(1 for score in focus_scores if score >= 0.84) / len(focus_tokens)
                focus_fuzzy_mean = float(sum(focus_scores) / len(focus_scores))
                focus_score = (0.25 * focus_exact_ratio) + (0.40 * focus_near_ratio) + (0.35 * focus_fuzzy_mean)
                score = (0.60 * score) + (0.40 * focus_score)
                if typo_or_naming:
                    score = (0.45 * score) + (0.55 * focus_score)
            sample_scores.append(float(score))

        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append(context_ids)
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))
        group_sizes.append(len(context_ids))
        score_blocks.append(np.asarray(sample_scores, dtype=np.float32))

    scores = np.concatenate(score_blocks, axis=0) if score_blocks else np.zeros((0,), dtype=np.float32)
    dataset = {
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "group_sizes": group_sizes,
    }
    return dataset, scores


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    dataset, scores = build_score_cache(
        samples,
        min_token_len=int(args.min_token_len),
        focus_mode=str(args.focus_mode),
    )
    output_path = Path(args.output)
    write_score_cache(
        output_path,
        model_path=f"fuzzy_comment_score::min_token_len={int(args.min_token_len)}::focus_mode={args.focus_mode}",
        dataset=dataset,
        scores=scores,
    )
    print(
        {
            "samples": len(dataset["sample_ids"]),
            "contexts": int(len(scores)),
            "output": str(output_path),
            "focus_mode": args.focus_mode,
        }
    )


if __name__ == "__main__":
    main()
