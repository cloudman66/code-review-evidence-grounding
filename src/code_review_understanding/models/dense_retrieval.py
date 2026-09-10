from __future__ import annotations

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from code_review_understanding.models.semantic_retrieval import (
    semantic_context_text,
    semantic_query_text,
)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    return (matrix / safe_norms).astype(np.float32)


def _prepare_dense_text(text: str, *, prefix: str = "") -> str:
    normalized = text.strip()
    if not prefix:
        return normalized
    return f"{prefix.strip()} {normalized}".strip()


def build_dense_encoder(
    *,
    model_name: str,
    cache_dir: str | Path | None = None,
    threads: int | None = None,
) -> TextEmbedding:
    return TextEmbedding(
        model_name=model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        threads=threads,
    )


def encode_dense_texts(
    model: TextEmbedding,
    texts: list[str],
    *,
    batch_size: int,
    parallel: int | None,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = list(model.embed(texts, batch_size=batch_size, parallel=parallel))
    matrix = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    return _normalize_rows(matrix)


def build_dense_score_cache(
    samples: list[dict],
    *,
    model_name: str,
    query_mode: str = "expanded",
    context_mode: str = "full",
    query_prefix: str = "",
    context_prefix: str = "",
    batch_size: int = 128,
    parallel: int | None = None,
    cache_dir: str | Path | None = None,
    threads: int | None = None,
) -> tuple[dict, np.ndarray]:
    model = build_dense_encoder(
        model_name=model_name,
        cache_dir=cache_dir,
        threads=threads,
    )

    query_texts: list[str] = []
    query_seen: set[str] = set()
    context_texts: list[str] = []
    context_seen: set[str] = set()

    for sample in samples:
        query_text = _prepare_dense_text(
            semantic_query_text(sample["comment"], mode=query_mode),
            prefix=query_prefix,
        )
        if query_text not in query_seen:
            query_seen.add(query_text)
            query_texts.append(query_text)
        for context in sample["contexts"]:
            context_text = _prepare_dense_text(
                semantic_context_text(context["text"], mode=context_mode),
                prefix=context_prefix,
            )
            if context_text not in context_seen:
                context_seen.add(context_text)
                context_texts.append(context_text)

    query_vectors = encode_dense_texts(
        model,
        query_texts,
        batch_size=batch_size,
        parallel=parallel,
    )
    context_vectors = encode_dense_texts(
        model,
        context_texts,
        batch_size=batch_size,
        parallel=parallel,
    )

    query_map = {text: vector for text, vector in zip(query_texts, query_vectors, strict=True)}
    context_map = {text: vector for text, vector in zip(context_texts, context_vectors, strict=True)}

    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    group_sizes: list[int] = []
    score_blocks: list[np.ndarray] = []

    for sample in samples:
        query_text = _prepare_dense_text(
            semantic_query_text(sample["comment"], mode=query_mode),
            prefix=query_prefix,
        )
        query_vector = query_map[query_text]
        current_context_ids: list[str] = []
        current_vectors: list[np.ndarray] = []
        for context in sample["contexts"]:
            context_text = _prepare_dense_text(
                semantic_context_text(context["text"], mode=context_mode),
                prefix=context_prefix,
            )
            current_context_ids.append(context["context_id"])
            current_vectors.append(context_map[context_text])

        if current_vectors:
            context_matrix = np.vstack(current_vectors).astype(np.float32)
            scores = (context_matrix @ query_vector).astype(np.float32)
        else:
            scores = np.zeros((0,), dtype=np.float32)

        score_blocks.append(scores)
        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append(current_context_ids)
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))
        group_sizes.append(len(current_context_ids))

    dataset = {
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "group_sizes": group_sizes,
    }
    dense_scores = np.concatenate(score_blocks, axis=0) if score_blocks else np.zeros((0,), dtype=np.float32)
    return dataset, dense_scores
