from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.models.dense_retrieval import (
    build_dense_encoder,
    encode_dense_texts,
)
from code_review_understanding.models.fusion import load_score_cache, normalize_group_scores, write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.semantic_retrieval import semantic_context_text, semantic_query_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-score-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--query-mode", default="expanded")
    parser.add_argument("--context-mode", default="path_plus_diff")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--context-prefix", default="")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def prepare_text(text: str, *, prefix: str = "") -> str:
    normalized = text.strip()
    if not prefix:
        return normalized
    return f"{prefix.strip()} {normalized}".strip()


def selected_indices_from_base(scores: np.ndarray, *, top_n: int) -> list[int]:
    ranked = np.argsort(-scores, kind="stable")
    return ranked[: min(top_n, len(ranked))].tolist()


def normalize_selected(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower < 1e-6:
        return np.ones_like(values, dtype=np.float32)
    return ((values - lower) / (upper - lower)).astype(np.float32)


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    base_cache = load_score_cache(Path(args.base_score_cache))
    if [sample["sample_id"] for sample in samples] != base_cache["dataset"]["sample_ids"]:
        raise ValueError("dataset and base-score-cache sample_ids do not align")

    model = build_dense_encoder(
        model_name=args.model_name,
        cache_dir=(args.cache_dir or None),
        threads=(None if args.threads <= 0 else int(args.threads)),
    )
    parallel = None if args.parallel <= 0 else int(args.parallel)

    query_texts: list[str] = []
    query_seen: set[str] = set()
    context_texts: list[str] = []
    context_seen: set[str] = set()
    selection_specs: list[dict] = []

    offset = 0
    for sample, context_ids, group_size in zip(
        samples,
        base_cache["dataset"]["context_ids_by_group"],
        base_cache["dataset"]["group_sizes"],
        strict=True,
    ):
        next_offset = offset + int(group_size)
        base_scores = base_cache["scores"][offset:next_offset]
        selected_indices = selected_indices_from_base(base_scores, top_n=int(args.top_n))

        query_text = prepare_text(
            semantic_query_text(sample["comment"], mode=args.query_mode),
            prefix=args.query_prefix,
        )
        if query_text not in query_seen:
            query_seen.add(query_text)
            query_texts.append(query_text)

        selected_context_texts: dict[int, str] = {}
        for selected_index in selected_indices:
            context_text = prepare_text(
                semantic_context_text(sample["contexts"][selected_index]["text"], mode=args.context_mode),
                prefix=args.context_prefix,
            )
            selected_context_texts[selected_index] = context_text
            if context_text not in context_seen:
                context_seen.add(context_text)
                context_texts.append(context_text)

        selection_specs.append(
            {
                "sample_id": sample["sample_id"],
                "query_text": query_text,
                "selected_indices": selected_indices,
                "selected_context_texts": selected_context_texts,
                "base_scores": np.asarray(base_scores, dtype=np.float32),
                "context_ids": list(context_ids),
            }
        )
        offset = next_offset

    query_vectors = encode_dense_texts(
        model,
        query_texts,
        batch_size=int(args.batch_size),
        parallel=parallel,
    )
    context_vectors = encode_dense_texts(
        model,
        context_texts,
        batch_size=int(args.batch_size),
        parallel=parallel,
    )
    query_map = {text: vector for text, vector in zip(query_texts, query_vectors, strict=True)}
    context_map = {text: vector for text, vector in zip(context_texts, context_vectors, strict=True)}

    score_blocks: list[np.ndarray] = []
    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    group_sizes: list[int] = []

    for sample, spec in zip(samples, selection_specs, strict=True):
        base_scores = spec["base_scores"]
        base_norm = normalize_group_scores(base_scores, "minmax")
        reranked_scores = (-1.0 + 0.01 * base_norm).astype(np.float32)
        selected_indices = spec["selected_indices"]
        if selected_indices:
            query_vector = query_map[spec["query_text"]]
            dense_raw = np.asarray(
                [
                    float(context_map[spec["selected_context_texts"][index]] @ query_vector)
                    for index in selected_indices
                ],
                dtype=np.float32,
            )
            dense_norm = normalize_selected(dense_raw)
            selected_base_norm = base_norm[np.asarray(selected_indices, dtype=np.int32)]
            blended = (
                float(args.blend_alpha) * dense_norm
                + (1.0 - float(args.blend_alpha)) * selected_base_norm
            ).astype(np.float32)
            for index, score in zip(selected_indices, blended, strict=True):
                reranked_scores[index] = float(score)

        score_blocks.append(reranked_scores)
        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append(spec["context_ids"])
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))
        group_sizes.append(len(spec["context_ids"]))

    dataset = {
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "group_sizes": group_sizes,
    }
    scores = np.concatenate(score_blocks, axis=0) if score_blocks else np.zeros((0,), dtype=np.float32)
    output_path = Path(args.output)
    write_score_cache(
        output_path,
        model_path=(
            f"dense_topn_rerank::{args.model_name}"
            f"::query_mode={args.query_mode}"
            f"::context_mode={args.context_mode}"
            f"::top_n={int(args.top_n)}"
            f"::blend_alpha={float(args.blend_alpha):.2f}"
        ),
        dataset=dataset,
        scores=scores,
    )
    print(
        {
            "samples": len(sample_ids),
            "contexts": len(scores),
            "top_n": int(args.top_n),
            "blend_alpha": float(args.blend_alpha),
            "model_name": args.model_name,
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
