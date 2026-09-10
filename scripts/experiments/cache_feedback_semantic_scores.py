from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import (
    extract_changed_lines,
    extract_context_path,
    split_path_parts,
    tokenize_retrieval,
)
from code_review_understanding.models.fusion import load_score_cache, write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.semantic_retrieval import (
    REVIEW_QUERY_STOPWORDS,
    dedupe_keep_order,
    encode_semantic_texts,
    filtered_query_terms,
    load_semantic_retriever,
    semantic_context_text,
    semantic_query_text,
)


CODE_PRF_STOPWORDS = {
    "args",
    "call",
    "class",
    "cls",
    "code",
    "context",
    "data",
    "def",
    "false",
    "file",
    "files",
    "function",
    "line",
    "lines",
    "list",
    "module",
    "name",
    "none",
    "path",
    "result",
    "return",
    "self",
    "string",
    "test",
    "text",
    "true",
    "type",
    "value",
    "values",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--base-score-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feedback-top-k", type=int, default=3)
    parser.add_argument("--max-feedback-terms", type=int, default=12)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument(
        "--diversify-by-file",
        action="store_true",
        help="Prefer top contexts from different files before filling remaining slots.",
    )
    return parser.parse_args()


def keep_feedback_token(token: str, *, base_terms: set[str]) -> bool:
    return (
        len(token) >= 2
        and not token.isdigit()
        and token not in REVIEW_QUERY_STOPWORDS
        and token not in CODE_PRF_STOPWORDS
        and token not in base_terms
    )


def context_feedback_terms(
    context_text: str,
    *,
    base_terms: set[str],
) -> tuple[list[str], list[str], str]:
    path = extract_context_path(context_text)
    added_lines, removed_lines = extract_changed_lines(context_text)
    changed_text = " ".join(sorted(added_lines | removed_lines))

    path_terms = [
        token
        for token in dedupe_keep_order(split_path_parts(path))
        if keep_feedback_token(token, base_terms=base_terms)
    ]
    changed_terms = [
        token
        for token in dedupe_keep_order(filtered_query_terms(changed_text))
        if keep_feedback_token(token, base_terms=base_terms)
    ]
    return path_terms, changed_terms, path


def sample_base_terms(comment: str, *, query_mode: str) -> set[str]:
    base_text = semantic_query_text(comment, mode=query_mode)
    return set(filtered_query_terms(base_text))


def ranked_context_indices(scores: np.ndarray) -> list[int]:
    return list(np.argsort(-scores, kind="stable"))


def selected_feedback_contexts(
    sample: dict,
    *,
    base_scores: np.ndarray,
    top_k: int,
    diversify_by_file: bool,
) -> list[dict]:
    ranked = ranked_context_indices(base_scores)
    contexts = sample["contexts"]
    if not diversify_by_file:
        return [contexts[index] for index in ranked[:top_k]]

    selected: list[dict] = []
    selected_indices: set[int] = set()
    seen_paths: set[str] = set()

    for index in ranked:
        context = contexts[index]
        path = extract_context_path(context["text"])
        normalized_path = path.lower()
        if normalized_path and normalized_path in seen_paths:
            continue
        selected.append(context)
        selected_indices.add(index)
        if normalized_path:
            seen_paths.add(normalized_path)
        if len(selected) >= top_k:
            return selected

    for index in ranked:
        if index in selected_indices:
            continue
        selected.append(contexts[index])
        if len(selected) >= top_k:
            break
    return selected


def build_feedback_query(
    sample: dict,
    *,
    query_mode: str,
    selected_contexts: list[dict],
    max_feedback_terms: int,
    max_paths: int,
) -> str:
    base_query = semantic_query_text(sample["comment"], mode=query_mode)
    base_terms = sample_base_terms(sample["comment"], query_mode=query_mode)

    weighted_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    paths: list[str] = []

    for rank, context in enumerate(selected_contexts, start=1):
        weight = 1.0 / float(rank)
        path_terms, changed_terms, path = context_feedback_terms(
            context["text"],
            base_terms=base_terms,
        )
        candidate_terms = dedupe_keep_order(path_terms[:8] + changed_terms[:16])
        for token in path_terms[:8]:
            weighted_counts[token] += 2.0 * weight
        for token in changed_terms[:16]:
            weighted_counts[token] += 1.0 * weight
        for token in candidate_terms:
            doc_counts[token] += 1
        if path:
            paths.append(path.lower())

    ranked_terms = sorted(
        weighted_counts,
        key=lambda token: (
            doc_counts[token] >= 2,
            weighted_counts[token],
            len(token),
            token,
        ),
        reverse=True,
    )
    chosen_terms = ranked_terms[:max_feedback_terms]
    chosen_paths = dedupe_keep_order(paths)[:max_paths]

    parts = [base_query]
    if chosen_terms:
        parts.append("feedback_terms " + " ".join(chosen_terms))
    if chosen_paths:
        parts.append("feedback_paths " + " ".join(chosen_paths))
        path_parts = dedupe_keep_order(
            token
            for path in chosen_paths
            for token in split_path_parts(path)
            if keep_feedback_token(token, base_terms=base_terms)
        )
        if path_parts:
            parts.append("feedback_path_parts " + " ".join(path_parts[:max_feedback_terms]))
    return " ".join(part for part in parts if part).strip()


def cache_to_group_scores(payload: dict) -> dict[str, np.ndarray]:
    dataset = payload["dataset"]
    group_scores: dict[str, np.ndarray] = {}
    offset = 0
    for sample_id, group_size in zip(dataset["sample_ids"], dataset["group_sizes"], strict=True):
        next_offset = offset + int(group_size)
        group_scores[sample_id] = payload["scores"][offset:next_offset]
        offset = next_offset
    return group_scores


def build_score_cache(
    samples: list[dict],
    *,
    semantic_model: dict,
    base_group_scores: dict[str, np.ndarray],
    feedback_top_k: int,
    max_feedback_terms: int,
    max_paths: int,
    diversify_by_file: bool,
) -> tuple[dict, np.ndarray]:
    query_mode = str(semantic_model.get("query_mode", "normalized"))
    context_mode = str(semantic_model.get("context_mode", "full"))

    unique_context_texts: list[str] = []
    context_seen: set[str] = set()
    feedback_queries: list[str] = []
    for sample in samples:
        feedback_queries.append(
            build_feedback_query(
                sample,
                query_mode=query_mode,
                selected_contexts=selected_feedback_contexts(
                    sample,
                    base_scores=base_group_scores[sample["sample_id"]],
                    top_k=feedback_top_k,
                    diversify_by_file=diversify_by_file,
                ),
                max_feedback_terms=max_feedback_terms,
                max_paths=max_paths,
            )
        )
        for context in sample["contexts"]:
            context_text = semantic_context_text(context["text"], mode=context_mode)
            if context_text not in context_seen:
                context_seen.add(context_text)
                unique_context_texts.append(context_text)

    query_vectors = encode_semantic_texts(feedback_queries, semantic_model)
    context_vectors = encode_semantic_texts(unique_context_texts, semantic_model)
    context_vector_map = {
        text: vector for text, vector in zip(unique_context_texts, context_vectors, strict=True)
    }

    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    group_sizes: list[int] = []
    score_blocks: list[np.ndarray] = []

    for sample, query_vector in zip(samples, query_vectors, strict=True):
        context_ids: list[str] = []
        current_vectors: list[np.ndarray] = []
        for context in sample["contexts"]:
            context_ids.append(context["context_id"])
            current_vectors.append(
                context_vector_map[semantic_context_text(context["text"], mode=context_mode)]
            )
        context_matrix = (
            np.vstack(current_vectors).astype(np.float32)
            if current_vectors
            else np.zeros((0, query_vector.shape[0]), dtype=np.float32)
        )
        score_blocks.append((context_matrix @ query_vector).astype(np.float32))
        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append(context_ids)
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))
        group_sizes.append(len(context_ids))

    dataset = {
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "group_sizes": group_sizes,
    }
    scores = np.concatenate(score_blocks, axis=0) if score_blocks else np.zeros((0,), dtype=np.float32)
    return dataset, scores


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    semantic_model = load_semantic_retriever(args.semantic_model)
    base_payload = load_score_cache(Path(args.base_score_cache))
    base_group_scores = cache_to_group_scores(base_payload)
    dataset, scores = build_score_cache(
        samples,
        semantic_model=semantic_model,
        base_group_scores=base_group_scores,
        feedback_top_k=int(args.feedback_top_k),
        max_feedback_terms=int(args.max_feedback_terms),
        max_paths=int(args.max_paths),
        diversify_by_file=bool(args.diversify_by_file),
    )
    model_path = (
        f"semantic_feedback::{Path(args.semantic_model).name}"
        f"::topk={int(args.feedback_top_k)}"
        f"::terms={int(args.max_feedback_terms)}"
        f"::paths={int(args.max_paths)}"
        f"::diversify={int(bool(args.diversify_by_file))}"
    )
    output_path = Path(args.output)
    write_score_cache(
        output_path,
        model_path=model_path,
        dataset=dataset,
        scores=scores,
    )
    print(
        {
            "samples": len(dataset["sample_ids"]),
            "contexts": int(len(scores)),
            "output": str(output_path),
            "feedback_top_k": int(args.feedback_top_k),
            "max_feedback_terms": int(args.max_feedback_terms),
            "diversify_by_file": bool(args.diversify_by_file),
        }
    )


if __name__ == "__main__":
    main()
