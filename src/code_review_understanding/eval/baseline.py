from __future__ import annotations

import math
import random
import re
from collections import Counter

from code_review_understanding.data.io_utils import load_jsonl


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_SPLIT_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
CODE_FENCE_PATTERN = re.compile(r"```[A-Za-z0-9_+-]*")
INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
PATH_PATTERN = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+")
QUOTED_TEXT_PATTERN = re.compile(r'"([^"\n]{1,120})"')
DOUBLE_BACKTICK_PATTERN = re.compile(r"``([^`\n]{1,120})``")
SINGLE_QUOTED_TEXT_PATTERN = re.compile(r"'([^'\n]{1,120})'")
REPLACEMENT_PATTERN = re.compile(
    r"([\"'`]{1,2})([^\"'`\n]{1,120})\1\s*(?:--?>|=>)\s*([\"'`]{1,2})([^\"'`\n]{1,120})\3"
)
DEFAULT_RANKING_CONFIG = {
    "line_overlap_weight": 3.0,
    "lexical_overlap_weight": 1.0,
    "inline_code_weight": 1.5,
    "path_token_weight": 0.0,
    "path_exact_weight": 0.0,
    "inline_path_weight": 0.75,
    "inline_line_weight": 2.5,
    "word_tfidf_weight": 1.5,
    "char_tfidf_weight": 1.0,
    "quoted_span_weight": 3.0,
    "quoted_line_weight": 2.0,
}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def tokenize_retrieval(text: str) -> list[str]:
    raw_tokens = TOKEN_PATTERN.findall(text)
    expanded_tokens: list[str] = []
    for token in raw_tokens:
        lowered = token.lower()
        expanded_tokens.append(lowered)
        expanded_tokens.extend(
            part
            for part in split_identifier(token)
            if part and part != lowered
        )
    return expanded_tokens


def split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for chunk in token.replace("-", "_").split("_"):
        if not chunk:
            continue
        parts.extend(CAMEL_SPLIT_PATTERN.sub(" ", chunk).split())
    return [part.lower() for part in parts if part]


def clean_comment_text(text: str) -> str:
    return CODE_FENCE_PATTERN.sub("```", text.replace("\r", "").strip())


def normalize_line(line: str) -> str:
    line = line.strip()
    line = CODE_FENCE_PATTERN.sub("", line)
    line = re.sub(r"^[+\- ]+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip().lower()


def meaningful_lines(text: str) -> set[str]:
    return {normalized for line in text.splitlines() if (normalized := normalize_line(line))}


def lexical_overlap_ratio(query_tokens: list[str], doc_tokens: list[str]) -> float:
    query_vocab = set(query_tokens)
    if not query_vocab:
        return 0.0
    return len(query_vocab & set(doc_tokens)) / len(query_vocab)


def extract_inline_code_spans(text: str) -> list[str]:
    return [match.strip() for match in INLINE_CODE_PATTERN.findall(text) if match.strip()]


def looks_code_like_span(span: str) -> bool:
    if len(span.strip()) <= 1:
        return False
    return bool(re.search(r"[A-Z_./()\-]", span)) or " " not in span


def extract_quoted_spans(text: str) -> list[str]:
    spans = extract_inline_code_spans(text)
    spans.extend(match.strip() for match in DOUBLE_BACKTICK_PATTERN.findall(text) if match.strip())
    spans.extend(match.strip() for match in QUOTED_TEXT_PATTERN.findall(text) if match.strip())
    spans.extend(
        match.strip()
        for match in SINGLE_QUOTED_TEXT_PATTERN.findall(text)
        if match.strip() and looks_code_like_span(match.strip())
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for span in spans:
        if span not in seen:
            seen.add(span)
            deduped.append(span)
    return deduped


def extract_path_mentions(text: str) -> list[str]:
    return [match.lower() for match in PATH_PATTERN.findall(text)]


def extract_replacement_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in REPLACEMENT_PATTERN.finditer(text):
        old = match.group(2).strip()
        new = match.group(4).strip()
        if old and new:
            pairs.append((old, new))
    return pairs


def extract_context_path(text: str) -> str:
    match = re.search(r"^path:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_changed_lines(text: str) -> tuple[set[str], set[str]]:
    added_lines: set[str] = set()
    removed_lines: set[str] = set()
    for raw_line in text.splitlines():
        if raw_line.startswith(("path: ", "@@", "+++", "---")):
            continue
        if raw_line.startswith("+"):
            normalized = normalize_line(raw_line)
            if normalized:
                added_lines.add(normalized)
        elif raw_line.startswith("-"):
            normalized = normalize_line(raw_line)
            if normalized:
                removed_lines.add(normalized)
    return added_lines, removed_lines


def tokenize_from_lines(lines: set[str]) -> list[str]:
    if not lines:
        return []
    return tokenize_retrieval(" ".join(sorted(lines)))


def split_path_parts(path: str) -> list[str]:
    return [part.lower() for part in re.split(r"[/._-]+", path) if part]


def build_ranking_config(overrides: dict | None = None) -> dict[str, float]:
    config = dict(DEFAULT_RANKING_CONFIG)
    if overrides:
        config.update({key: float(value) for key, value in overrides.items()})
    return config


def tfidf_similarity_scores(
    query_text: str,
    context_texts: list[str],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
) -> list[float]:
    if not context_texts:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        lowercase=True,
    )
    matrix = vectorizer.fit_transform([query_text] + context_texts)
    return [float(score) for score in linear_kernel(matrix[0:1], matrix[1:]).ravel()]


def bm25_scores(documents: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not documents:
        return []

    doc_freq: Counter[str] = Counter()
    for doc in documents:
        for token in set(doc):
            doc_freq[token] += 1

    avgdl = sum(len(doc) for doc in documents) / len(documents)
    scores: list[float] = []
    for doc in documents:
        tf = Counter(doc)
        score = 0.0
        doc_len = len(doc)
        for token in query:
            if token not in tf:
                continue
            df = doc_freq[token]
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * (doc_len / avgdl if avgdl else 0.0))
            score += idf * ((freq * (k1 + 1)) / denom)
        scores.append(score)
    return scores


def build_context_feature_rows(sample: dict) -> list[dict]:
    contexts = sample["contexts"]
    context_texts = [context["text"] for context in contexts]
    corpus = [tokenize_retrieval(text) for text in context_texts]
    query_text = clean_comment_text(sample["comment"])
    query_tokens = tokenize_retrieval(query_text)
    query_lines = meaningful_lines(sample["comment"])
    inline_code_spans = extract_inline_code_spans(sample["comment"])
    inline_code_tokens = tokenize_retrieval(" ".join(inline_code_spans))
    quoted_spans = extract_quoted_spans(sample["comment"])
    normalized_quoted_spans = {normalize_line(span) for span in quoted_spans if normalize_line(span)}
    replacement_pairs = extract_replacement_pairs(sample["comment"])
    normalized_replacement_pairs = [
        (normalize_line(old), normalize_line(new))
        for old, new in replacement_pairs
        if normalize_line(old) and normalize_line(new)
    ]
    path_mentions = extract_path_mentions(sample["comment"])
    query_path_tokens = split_path_parts(" ".join(path_mentions))
    bm25 = bm25_scores(corpus, query_tokens)
    word_tfidf_scores = tfidf_similarity_scores(
        sample["comment"],
        context_texts,
        analyzer="word",
        ngram_range=(1, 2),
    )
    char_tfidf_scores = tfidf_similarity_scores(
        sample["comment"],
        context_texts,
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    context_line_sets = [meaningful_lines(text) for text in context_texts]
    changed_line_sets = [extract_changed_lines(text) for text in context_texts]
    context_paths = [extract_context_path(text) for text in context_texts]
    context_path_tokens = [split_path_parts(path) for path in context_paths]
    normalized_inline_spans = {
        normalized for span in inline_code_spans if (normalized := normalize_line(span))
    }

    feature_rows: list[dict] = []
    for context, context_tokens, bm25_score, context_lines, changed_lines, context_path, path_tokens, word_tfidf_score, char_tfidf_score in zip(
        contexts,
        corpus,
        bm25,
        context_line_sets,
        changed_line_sets,
        context_paths,
        context_path_tokens,
        word_tfidf_scores,
        char_tfidf_scores,
        strict=True,
    ):
        added_lines, removed_lines = changed_lines
        added_tokens = tokenize_from_lines(added_lines)
        removed_tokens = tokenize_from_lines(removed_lines)
        changed_tokens = tokenize_from_lines(added_lines | removed_lines)
        feature_rows.append(
            {
                "context_id": context["context_id"],
                "source": context.get("source", "unknown"),
                "text": context["text"],
                "features": {
                    "bm25": float(bm25_score),
                    "line_overlap": float(len(query_lines & context_lines)),
                    "lexical_overlap": lexical_overlap_ratio(query_tokens, context_tokens),
                    "inline_code_overlap": lexical_overlap_ratio(inline_code_tokens, context_tokens),
                    "path_token_overlap": lexical_overlap_ratio(query_path_tokens, path_tokens),
                    "path_exact_match": (
                        max(
                            1.0 if path_mention in context_path.lower() else 0.0
                            for path_mention in path_mentions
                        )
                        if path_mentions
                        else 0.0
                    ),
                    "inline_path_overlap": lexical_overlap_ratio(inline_code_tokens, path_tokens),
                    "inline_line_overlap": float(len(normalized_inline_spans & context_lines)),
                    "added_line_overlap": float(len(query_lines & added_lines)),
                    "removed_line_overlap": float(len(query_lines & removed_lines)),
                    "changed_token_overlap": lexical_overlap_ratio(query_tokens, changed_tokens),
                    "added_token_overlap": lexical_overlap_ratio(query_tokens, added_tokens),
                    "removed_token_overlap": lexical_overlap_ratio(query_tokens, removed_tokens),
                    "word_tfidf": float(word_tfidf_score),
                    "char_tfidf": float(char_tfidf_score),
                    "quoted_span_exact": float(
                        sum(1 for span in quoted_spans if span.lower() in context["text"].lower())
                    ),
                    "quoted_line_exact": float(len(normalized_quoted_spans & context_lines)),
                    "quoted_added_line_exact": float(len(normalized_quoted_spans & added_lines)),
                    "quoted_removed_line_exact": float(len(normalized_quoted_spans & removed_lines)),
                    "replacement_pair_match": float(
                        sum(
                            1
                            for old, new in normalized_replacement_pairs
                            if old in removed_lines and new in added_lines
                        )
                    ),
                    "replacement_old_removed": float(
                        sum(1 for old, _ in normalized_replacement_pairs if old in removed_lines)
                    ),
                    "replacement_new_added": float(
                        sum(1 for _, new in normalized_replacement_pairs if new in added_lines)
                    ),
                },
            }
        )
    return feature_rows


def score_context_features(feature_row: dict, ranking_config: dict | None = None) -> float:
    config = build_ranking_config(ranking_config)
    features = feature_row["features"]
    return float(
        features["bm25"]
        + (config["line_overlap_weight"] * features["line_overlap"])
        + (config["lexical_overlap_weight"] * features["lexical_overlap"])
        + (config["inline_code_weight"] * features["inline_code_overlap"])
        + (config["path_token_weight"] * features["path_token_overlap"])
        + (config["path_exact_weight"] * features["path_exact_match"])
        + (config["inline_path_weight"] * features["inline_path_overlap"])
        + (config["inline_line_weight"] * features["inline_line_overlap"])
        + (config["word_tfidf_weight"] * features["word_tfidf"])
        + (config["char_tfidf_weight"] * features["char_tfidf"])
        + (config["quoted_span_weight"] * features["quoted_span_exact"])
        + (config["quoted_line_weight"] * features["quoted_line_exact"])
    )


def rank_feature_rows(feature_rows: list[dict], ranking_config: dict | None = None) -> list[dict]:
    ranked = [
        {
            "context_id": row["context_id"],
            "source": row["source"],
            "text": row["text"],
            "score": score_context_features(row, ranking_config=ranking_config),
            "features": row["features"],
        }
        for row in feature_rows
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def rank_contexts(sample: dict, ranking_config: dict | None = None) -> list[dict]:
    return rank_feature_rows(
        build_context_feature_rows(sample),
        ranking_config=ranking_config,
    )


def evidence_metrics(samples: list[dict], top_k: int, ranking_config: dict | None = None) -> dict:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    ranked_predictions = []

    for sample in samples:
        ranked = rank_contexts(sample, ranking_config=ranking_config)
        gold = set(sample["gold_context_ids"])
        top_ids = [item["context_id"] for item in ranked[:top_k]]
        hit1 += int(bool(ranked) and ranked[0]["context_id"] in gold)
        hitk += int(any(context_id in gold for context_id in top_ids))

        reciprocal_rank = 0.0
        for index, item in enumerate(ranked, start=1):
            if item["context_id"] in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank

        ranked_predictions.append(
            {
                "sample_id": sample["sample_id"],
                "gold_context_ids": sample["gold_context_ids"],
                "ranked_context_ids": [item["context_id"] for item in ranked],
            }
        )

    total = len(samples) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
        "ranked_predictions": ranked_predictions,
    }


def sample_text(
    sample: dict,
    mode: str,
    top_k: int,
    rng: random.Random,
    ranking_config: dict | None = None,
) -> str:
    comment = sample["comment"]
    contexts = sample["contexts"]

    if mode == "none":
        context_text = ""
    elif mode == "full":
        context_text = "\n".join(context["text"] for context in contexts)
    elif mode == "random":
        chosen = rng.sample(contexts, k=min(top_k, len(contexts)))
        context_text = "\n".join(context["text"] for context in chosen)
    elif mode == "retrieved":
        chosen = rank_contexts(sample, ranking_config=ranking_config)[:top_k]
        context_text = "\n".join(context["text"] for context in chosen)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return f"comment: {comment}\ncontext: {context_text}".strip()


class MultinomialNaiveBayes:
    def __init__(self) -> None:
        self.label_counts: Counter[str] = Counter()
        self.token_counts_by_label: dict[str, Counter[str]] = {}
        self.total_tokens_by_label: Counter[str] = Counter()
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        for text, label in zip(texts, labels, strict=True):
            tokens = tokenize(text)
            self.label_counts[label] += 1
            self.token_counts_by_label.setdefault(label, Counter()).update(tokens)
            self.total_tokens_by_label[label] += len(tokens)
            self.vocab.update(tokens)

    def predict(self, texts: list[str]) -> list[str]:
        labels = sorted(self.label_counts)
        total_docs = sum(self.label_counts.values())
        vocab_size = max(len(self.vocab), 1)
        predictions: list[str] = []

        for text in texts:
            token_counts = Counter(tokenize(text))
            best_label = ""
            best_score = float("-inf")

            for label in labels:
                prior = math.log(self.label_counts[label] / total_docs)
                total_tokens = self.total_tokens_by_label[label]
                score = prior
                label_token_counts = self.token_counts_by_label[label]

                for token, freq in token_counts.items():
                    token_prob = (label_token_counts[token] + 1) / (total_tokens + vocab_size)
                    score += freq * math.log(token_prob)

                if score > best_score:
                    best_score = score
                    best_label = label

            predictions.append(best_label)

        return predictions


def classification_metrics(gold: list[str], pred: list[str]) -> dict:
    labels = sorted(set(gold) | set(pred))
    total = len(gold) or 1
    accuracy = sum(int(g == p) for g, p in zip(gold, pred, strict=True)) / total

    per_label_f1: dict[str, float] = {}
    weighted_sum = 0.0
    support_sum = 0
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        support = sum(1 for g in gold if g == label)
        per_label_f1[label] = f1
        weighted_sum += f1 * support
        support_sum += support

    macro_f1 = sum(per_label_f1.values()) / len(per_label_f1) if per_label_f1 else 0.0
    weighted_f1 = weighted_sum / support_sum if support_sum else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def build_xy(
    samples: list[dict],
    mode: str,
    top_k: int,
    seed: int,
    ranking_config: dict | None = None,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    features = [
        sample_text(sample, mode=mode, top_k=top_k, rng=rng, ranking_config=ranking_config)
        for sample in samples
    ]
    labels = [sample["intent"] for sample in samples]
    return features, labels


def train_intent_classifier(
    train_samples: list[dict],
    eval_samples: list[dict],
    mode: str,
    top_k: int,
    max_features: int,
    seed: int,
    ranking_config: dict | None = None,
) -> tuple[dict, list[str]]:
    x_train, y_train = build_xy(
        train_samples,
        mode=mode,
        top_k=top_k,
        seed=seed,
        ranking_config=ranking_config,
    )
    x_eval, y_eval = build_xy(
        eval_samples,
        mode=mode,
        top_k=top_k,
        seed=seed,
        ranking_config=ranking_config,
    )

    classifier = MultinomialNaiveBayes()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_eval)

    metrics = classification_metrics(y_eval, predictions)
    return metrics, predictions


def select_best_mode(
    train_samples: list[dict],
    dev_samples: list[dict],
    modes: list[str],
    top_k: int,
    max_features: int,
    seed: int,
    ranking_config: dict | None = None,
) -> tuple[str, dict]:
    best_mode = ""
    best_metrics: dict = {}
    best_score = -1.0

    for mode in modes:
        metrics, _ = train_intent_classifier(
            train_samples=train_samples,
            eval_samples=dev_samples,
            mode=mode,
            top_k=top_k,
            max_features=max_features,
            seed=seed,
            ranking_config=ranking_config,
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_mode = mode
            best_metrics = metrics

    return best_mode, best_metrics


def run_experiment(config: dict) -> dict:
    train_samples = load_jsonl(config["dataset"]["train_path"])
    dev_samples = load_jsonl(config["dataset"]["dev_path"])
    test_samples = load_jsonl(config["dataset"]["test_path"])

    top_k = int(config["retrieval"]["top_k"])
    seed = int(config["retrieval"]["seed"])
    evaluate_retrieval = bool(config["retrieval"].get("evaluate", True))
    ranking_config = config.get("ranking")
    modes = list(config["classification"]["modes"])
    max_features = int(config["classification"].get("max_features", 0))

    retrieval_dev = (
        evidence_metrics(dev_samples, top_k=top_k, ranking_config=ranking_config)
        if evaluate_retrieval
        else None
    )
    retrieval_test = (
        evidence_metrics(test_samples, top_k=top_k, ranking_config=ranking_config)
        if evaluate_retrieval
        else None
    )

    best_mode, dev_cls_metrics = select_best_mode(
        train_samples=train_samples,
        dev_samples=dev_samples,
        modes=modes,
        top_k=top_k,
        max_features=max_features,
        seed=seed,
        ranking_config=ranking_config,
    )

    test_cls_metrics, test_predictions = train_intent_classifier(
        train_samples=train_samples + dev_samples,
        eval_samples=test_samples,
        mode=best_mode,
        top_k=top_k,
        max_features=max_features,
        seed=seed,
        ranking_config=ranking_config,
    )

    predictions = []
    ranked_test = (
        {item["sample_id"]: item for item in retrieval_test["ranked_predictions"]}
        if retrieval_test is not None
        else {}
    )
    for sample, prediction in zip(test_samples, test_predictions, strict=True):
        ranked_context_ids = (
            ranked_test[sample["sample_id"]]["ranked_context_ids"]
            if sample["sample_id"] in ranked_test
            else []
        )
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "gold_intent": sample["intent"],
                "predicted_intent": prediction,
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[:top_k],
            }
        )

    intent_distribution = Counter(sample["intent"] for sample in train_samples + dev_samples + test_samples)

    result = {
        "metrics": {
            "dataset_sizes": {
                "train": len(train_samples),
                "dev": len(dev_samples),
                "test": len(test_samples),
            },
            "intent_distribution": dict(intent_distribution),
            "best_dev_mode": best_mode,
            "dev_classification": dev_cls_metrics,
            "test_classification": test_cls_metrics,
        },
        "predictions": predictions,
    }

    if retrieval_dev is not None and retrieval_test is not None:
        result["metrics"]["retrieval_dev"] = {
            "hit@1": retrieval_dev["hit@1"],
            f"hit@{top_k}": retrieval_dev[f"hit@{top_k}"],
            "mrr": retrieval_dev["mrr"],
        }
        result["metrics"]["retrieval_test"] = {
            "hit@1": retrieval_test["hit@1"],
            f"hit@{top_k}": retrieval_test[f"hit@{top_k}"],
            "mrr": retrieval_test["mrr"],
        }

    return result
