from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from code_review_understanding.eval.baseline import (
    clean_comment_text,
    extract_changed_lines,
    extract_context_path,
    extract_inline_code_spans,
    extract_path_mentions,
    extract_quoted_spans,
    extract_replacement_pairs,
    normalize_line,
    split_path_parts,
    tokenize_retrieval,
)


REVIEW_QUERY_STOPWORDS = {
    "a",
    "about",
    "actually",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "get",
    "go",
    "had",
    "has",
    "have",
    "here",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "ll",
    "make",
    "maybe",
    "might",
    "more",
    "much",
    "need",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "please",
    "probably",
    "really",
    "see",
    "should",
    "so",
    "some",
    "still",
    "sure",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "up",
    "use",
    "using",
    "very",
    "was",
    "we",
    "what",
    "when",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def review_intent_hints(comment: str) -> list[str]:
    lowered = comment.lower()
    hints: list[str] = []
    keyword_sets = [
        (("typo", "spelling", "spelled", "misspell"), "typo spelling naming"),
        (("rename", "name", "variable name"), "rename naming identifier"),
        (("delete", "remove", "redundant", "unnecessary", "cleanup"), "delete remove cleanup redundant"),
        (("except", "exception", "raise", "catch", "try"), "exception error handling raise catch"),
        (("sort", "ordering", "order"), "sort ordering order"),
        (("dependency", "import"), "dependency import module"),
        (("link", "docs", "documentation"), "docs link documentation"),
    ]
    for needles, hint_text in keyword_sets:
        if any(needle in lowered for needle in needles):
            hints.append(hint_text)
    return hints


def intent_expansion_hints(comment: str) -> list[str]:
    lowered = comment.lower()
    hints: list[str] = []
    keyword_sets = [
        (
            ("typo", "spelling", "spelled", "misspell", "grammar", "wording", "phrase"),
            "intent typo spelling wording text literal string message docs",
        ),
        (
            ("rename", "renamed", "naming", "variable name", "function name", "class name"),
            "intent rename naming identifier symbol variable function class api",
        ),
        (
            ("delete", "remove", "cleanup", "clean up", "redundant", "unnecessary", "unused"),
            "intent delete remove cleanup redundant unused simplify",
        ),
        (
            ("import", "dependency", "module", "package"),
            "intent import dependency module package",
        ),
        (
            ("except", "exception", "raise", "throw", "catch", "error handling", "try"),
            "intent exception error handling raise throw catch try",
        ),
        (
            ("sort", "ordering", "order", "deterministic"),
            "intent ordering sort deterministic stable",
        ),
        (
            ("comment", "doc", "docstring", "documentation", "readme"),
            "intent documentation comment docstring readme text",
        ),
        (
            ("optional", "none", "null", "nullable", "missing"),
            "intent optional none null missing default",
        ),
        (
            ("format", "style", "lint", "pep8", "spacing", "indent"),
            "intent formatting style lint spacing indent",
        ),
    ]
    for needles, hint_text in keyword_sets:
        if any(needle in lowered for needle in needles):
            hints.append(hint_text)
    return hints


def filtered_query_terms(text: str) -> list[str]:
    filtered: list[str] = []
    for token in tokenize_retrieval(text):
        if len(token) <= 1:
            continue
        if token.isdigit():
            continue
        if token in REVIEW_QUERY_STOPWORDS:
            continue
        filtered.append(token)
    return dedupe_keep_order(filtered)


def semantic_query_text(comment: str, *, mode: str = "normalized") -> str:
    normalized_lines = [
        normalized
        for raw_line in clean_comment_text(comment).splitlines()
        if (normalized := normalize_line(raw_line))
    ]
    base_text = " ".join(normalized_lines) or clean_comment_text(comment).lower()
    if mode == "normalized":
        return base_text
    if mode == "expanded":
        inline_spans = dedupe_keep_order(extract_inline_code_spans(comment))
        quoted_spans = dedupe_keep_order(extract_quoted_spans(comment))
        path_mentions = dedupe_keep_order(extract_path_mentions(comment))
        replacement_pairs = extract_replacement_pairs(comment)

        parts: list[str] = [base_text]
        if inline_spans:
            inline_text = " ".join(span.lower() for span in inline_spans)
            parts.append("inline " + inline_text)
            parts.append("inline_tokens " + " ".join(tokenize_retrieval(inline_text)))
        if quoted_spans:
            quoted_text = " ".join(span.lower() for span in quoted_spans)
            parts.append("quoted " + quoted_text)
            parts.append("quoted_tokens " + " ".join(tokenize_retrieval(quoted_text)))
        if path_mentions:
            parts.append("paths " + " ".join(path_mentions))
            parts.append(
                "path_parts "
                + " ".join(
                    token
                    for path in path_mentions
                    for token in split_path_parts(path)
                )
            )
        if replacement_pairs:
            replacement_parts = []
            for old, new in replacement_pairs:
                replacement_parts.append(f"replace {old.lower()} with {new.lower()}")
                replacement_parts.extend(tokenize_retrieval(f"{old} {new}"))
            parts.append(" ".join(replacement_parts))
        parts.extend(review_intent_hints(comment))
        return " ".join(part for part in parts if part).strip()
    if mode == "procedural":
        inline_spans = dedupe_keep_order(extract_inline_code_spans(comment))
        quoted_spans = dedupe_keep_order(extract_quoted_spans(comment))
        path_mentions = dedupe_keep_order(extract_path_mentions(comment))
        replacement_pairs = extract_replacement_pairs(comment)

        focus_terms = filtered_query_terms(base_text)
        target_terms = dedupe_keep_order(
            token
            for span in inline_spans + quoted_spans
            for token in filtered_query_terms(span)
        )
        path_parts = dedupe_keep_order(
            token
            for path in path_mentions
            for token in split_path_parts(path)
        )

        action_terms: list[str] = []
        for old, new in replacement_pairs:
            action_terms.extend(filtered_query_terms(old))
            action_terms.extend(filtered_query_terms(new))
            action_terms.append("replace")
        for hint in review_intent_hints(comment):
            action_terms.extend(filtered_query_terms(hint))
        action_terms = dedupe_keep_order(action_terms)

        parts: list[str] = []
        if focus_terms:
            parts.append("focus " + " ".join(focus_terms))
        if target_terms:
            parts.append("targets " + " ".join(target_terms))
        if path_mentions:
            parts.append("paths " + " ".join(path_mentions))
        if path_parts:
            parts.append("path_parts " + " ".join(path_parts))
        if action_terms:
            parts.append("actions " + " ".join(action_terms))
        if not parts:
            return semantic_query_text(comment, mode="expanded")
        return " ".join(parts).strip()
    if mode == "intent_expanded":
        inline_spans = dedupe_keep_order(extract_inline_code_spans(comment))
        quoted_spans = dedupe_keep_order(extract_quoted_spans(comment))
        path_mentions = dedupe_keep_order(extract_path_mentions(comment))
        replacement_pairs = extract_replacement_pairs(comment)

        focus_terms = filtered_query_terms(base_text)
        symbol_terms = dedupe_keep_order(
            token
            for span in inline_spans + quoted_spans
            for token in filtered_query_terms(span)
        )
        path_parts = dedupe_keep_order(
            token
            for path in path_mentions
            for token in split_path_parts(path)
        )
        replacement_terms = dedupe_keep_order(
            token
            for old, new in replacement_pairs
            for token in filtered_query_terms(f"{old} {new}")
        )

        parts: list[str] = [base_text]
        if focus_terms:
            parts.append("focus " + " ".join(focus_terms))
        if symbol_terms:
            parts.append("symbols " + " ".join(symbol_terms))
        if path_mentions:
            parts.append("paths " + " ".join(path_mentions))
        if path_parts:
            parts.append("path_parts " + " ".join(path_parts))
        if replacement_pairs:
            replacement_text = []
            for old, new in replacement_pairs:
                replacement_text.append(f"replace {old.lower()} with {new.lower()}")
                replacement_text.append(f"rename {old.lower()} {new.lower()}")
            parts.append(" ".join(replacement_text))
        if replacement_terms:
            parts.append("replacement_terms " + " ".join(replacement_terms))
        parts.extend(review_intent_hints(comment))
        parts.extend(intent_expansion_hints(comment))
        return " ".join(part for part in parts if part).strip()
    raise ValueError(f"Unsupported semantic query mode: {mode}")


def semantic_context_text(text: str, *, mode: str = "full") -> str:
    path = extract_context_path(text)
    path_parts = " ".join(split_path_parts(path))
    if mode == "full":
        normalized_lines = [
            normalized
            for raw_line in text.splitlines()
            if not raw_line.startswith(("path: ", "@@", "+++", "---"))
            if (normalized := normalize_line(raw_line))
        ]
    elif mode == "path_plus_diff":
        added_lines, removed_lines = extract_changed_lines(text)
        normalized_lines = sorted(added_lines) + sorted(removed_lines)
    else:
        raise ValueError(f"Unsupported semantic context mode: {mode}")

    parts: list[str] = []
    if path:
        parts.append(path.lower())
    if path_parts:
        parts.append(path_parts)
    if normalized_lines:
        parts.append(" ".join(normalized_lines))
    return " ".join(parts) or text.lower()


def build_training_corpus(
    samples: list[dict],
    *,
    query_mode: str = "normalized",
    context_mode: str = "full",
) -> list[str]:
    seen: set[str] = set()
    corpus: list[str] = []

    for sample in samples:
        query_text = semantic_query_text(sample["comment"], mode=query_mode)
        if query_text and query_text not in seen:
            seen.add(query_text)
            corpus.append(query_text)
        for context in sample["contexts"]:
            context_text = semantic_context_text(context["text"], mode=context_mode)
            if context_text and context_text not in seen:
                seen.add(context_text)
                corpus.append(context_text)
    return corpus


def fit_semantic_retriever(
    train_samples: list[dict],
    *,
    query_mode: str = "normalized",
    context_mode: str = "full",
    analyzer: str = "word",
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.98,
    max_features: int = 40000,
    n_components: int = 128,
    random_state: int = 42,
) -> dict:
    corpus = build_training_corpus(
        train_samples,
        query_mode=query_mode,
        context_mode=context_mode,
    )
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    if matrix.shape[1] < 2:
        raise ValueError("Semantic retriever needs at least 2 TF-IDF features.")

    resolved_components = min(n_components, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=resolved_components, random_state=random_state)
    svd.fit(matrix)

    return {
        "model_type": "tfidf_svd_semantic_retriever",
        "vectorizer": vectorizer,
        "svd": svd,
        "training_corpus_size": len(corpus),
        "n_features": int(matrix.shape[1]),
        "n_components": int(resolved_components),
        "query_mode": query_mode,
        "context_mode": context_mode,
        "analyzer": analyzer,
        "ngram_range": [ngram_min, ngram_max],
        "min_df": int(min_df),
        "max_df": float(max_df),
        "max_features": int(max_features),
        "random_state": int(random_state),
    }


def save_semantic_retriever(model: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)


def load_semantic_retriever(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def encode_semantic_texts(texts: list[str], model: dict) -> np.ndarray:
    if not texts:
        return np.zeros((0, int(model["n_components"])), dtype=np.float32)
    matrix = model["vectorizer"].transform(texts)
    dense = model["svd"].transform(matrix)
    normalized = normalize(dense, norm="l2", copy=False)
    return normalized.astype(np.float32)


def build_semantic_score_cache(samples: list[dict], model: dict) -> tuple[dict, np.ndarray]:
    query_mode = str(model.get("query_mode", "normalized"))
    context_mode = str(model.get("context_mode", "full"))
    query_texts: list[str] = []
    query_seen: set[str] = set()
    context_texts: list[str] = []
    context_seen: set[str] = set()

    for sample in samples:
        query_text = semantic_query_text(sample["comment"], mode=query_mode)
        if query_text not in query_seen:
            query_seen.add(query_text)
            query_texts.append(query_text)
        for context in sample["contexts"]:
            context_text = semantic_context_text(context["text"], mode=context_mode)
            if context_text not in context_seen:
                context_seen.add(context_text)
                context_texts.append(context_text)

    query_vectors = encode_semantic_texts(query_texts, model)
    context_vectors = encode_semantic_texts(context_texts, model)
    query_map = {text: vector for text, vector in zip(query_texts, query_vectors, strict=True)}
    context_map = {text: vector for text, vector in zip(context_texts, context_vectors, strict=True)}

    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    group_sizes: list[int] = []
    score_blocks: list[np.ndarray] = []

    for sample in samples:
        query_vector = query_map[semantic_query_text(sample["comment"], mode=query_mode)]
        current_context_ids: list[str] = []
        current_vectors: list[np.ndarray] = []
        for context in sample["contexts"]:
            current_context_ids.append(context["context_id"])
            current_vectors.append(context_map[semantic_context_text(context["text"], mode=context_mode)])
        context_matrix = np.vstack(current_vectors).astype(np.float32) if current_vectors else np.zeros((0, query_vector.shape[0]), dtype=np.float32)
        score_blocks.append((context_matrix @ query_vector).astype(np.float32))
        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append(current_context_ids)
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))
        group_sizes.append(len(current_context_ids))

    scores = np.concatenate(score_blocks, axis=0) if score_blocks else np.zeros((0,), dtype=np.float32)
    dataset = {
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "group_sizes": group_sizes,
    }
    return dataset, scores
