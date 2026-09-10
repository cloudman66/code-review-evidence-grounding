"""Run leakage-resistant domain/repository holdout experiments for SWE-CARE v2.

The older holdout scripts reuse semantic score caches fitted on the complete
training split.  That is convenient for a quick diagnostic, but it is not a
strict holdout because the held-out domain/repository contributes to the
TF-IDF vocabulary, IDF values, and SVD basis.  This entry point keeps the old
scripts and their reported numbers untouched and rebuilds every learned
component inside each fold:

* the word and character TF-IDF/SVD retrievers are fitted on non-target train;
* semantic and file-mean scores are recomputed for fold train/dev/target-test;
* the semantic char+file anchor ranker is fitted on the non-target fold;
* fuzzy and semantic-feedback scores are recomputed from fold artifacts; and
* the final learned ranker is fitted and evaluated on the target test split.

Only the v2 namespace is accepted.  Results are written below
``results_v2/holdout_validation_strict`` (or an explicitly supplied path that
still contains the ``results_v2`` namespace marker).  The script is designed
to keep score caches in memory; optional semantic model pickles are written
only inside the isolated fold directory when ``--save-semantic-models`` is
requested.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.eval.baseline import (
    evidence_metrics,
    extract_context_path,
)
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    fit_pointwise_logistic_ranker,
    linear_model_scores,
    load_feature_row_cache,
    save_linear_ranker_model,
)
from code_review_understanding.models.semantic_retrieval import (
    build_semantic_score_cache,
    fit_semantic_retriever,
    save_semantic_retriever,
)

# These two modules contain deterministic score builders.  Importing the
# functions rather than spawning their command-line entry points lets each
# fold keep all intermediate caches isolated in memory.
from scripts.experiments.cache_feedback_semantic_scores import (
    build_score_cache as build_feedback_score_cache,
)
from scripts.experiments.cache_fuzzy_comment_scores import (
    build_score_cache as build_fuzzy_score_cache,
)


DEFAULT_FINAL_CONFIG = PROJECT_ROOT / (
    "src/configs/"
    "swe_care_learned_ranker_context_slices_file_context_semantic_char_"
    "file_mean_fuzzy_len2_feedback_exact_cached_v2.yaml"
)
DEFAULT_ANCHOR_CONFIG = PROJECT_ROOT / "src/configs/swe_care_reproduction_anchor_v2.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results_v2/holdout_validation_strict"

V2_DATA_MARKER = "swe_care_grounding_v2"
V2_CACHE_MARKER = "cache_v2"
V2_RESULTS_MARKER = "results_v2"
LEGACY_PATH_MARKERS = (
    "data/processed/swe_care_grounding/",
    "data/cache/",
    "results/",
)

DEFAULT_SEMANTIC_CONFIG: dict[str, Any] = {
    "query_mode": "normalized",
    "context_mode": "full",
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 40000,
    "n_components": 128,
    "random_state": 42,
    "word": {"analyzer": "word", "ngram_min": 1, "ngram_max": 2},
    "char": {"analyzer": "char_wb", "ngram_min": 3, "ngram_max": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("domain", "repo"),
        required=True,
        help="Hold out a problem_domain or repository at a time.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_FINAL_CONFIG),
        help="v2 final-ranker YAML configuration.",
    )
    parser.add_argument(
        "--anchor-config",
        default=str(DEFAULT_ANCHOR_CONFIG),
        help="v2 semantic char+file anchor-ranker YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory; it must remain under the results_v2 namespace. "
            "By default, results_v2/holdout_validation_strict/<mode> is used."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Explicit target values. If omitted, eligible targets are selected by split counts.",
    )
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=0,
        help="Run only the first N eligible targets (sorted by target-test count).",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=None,
        help="Minimum pre-holdout train samples for an eligible target.",
    )
    parser.add_argument(
        "--min-test",
        type=int,
        default=None,
        help="Minimum target-test samples for an eligible target.",
    )
    parser.add_argument(
        "--save-semantic-models",
        action="store_true",
        help="Persist fold-fitted word/char retrievers inside each v2 fold directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing target output directory to be replaced.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate v2 inputs and print selected folds without fitting models or writing outputs.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def path_text(path: str | Path) -> str:
    return str(project_path(path)).replace("\\", "/")


def assert_not_legacy(path: str | Path) -> None:
    normalized = path_text(path)
    for marker in LEGACY_PATH_MARKERS:
        if marker in normalized:
            raise ValueError(f"legacy namespace is not allowed in strict v2 run: {path}")


def assert_namespace(path: str | Path, marker: str, *, label: str) -> None:
    assert_not_legacy(path)
    if marker not in path_text(path).split("/") and marker not in Path(path).name:
        raise ValueError(f"{label} must contain the {marker!r} namespace marker: {path}")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def validate_v2_config(config: dict[str, Any], *, config_path: Path, label: str) -> None:
    # Config files themselves should be visibly versioned, even though the
    # stronger checks below validate every data/cache/output path.
    if "v2" not in config_path.name.lower():
        raise ValueError(f"{label} config must be a v2 config: {config_path}")

    dataset = config.get("dataset", {})
    for split in ("train", "dev", "test"):
        value = dataset.get(f"{split}_path")
        if not value:
            raise ValueError(f"{label} config is missing dataset.{split}_path")
        assert_namespace(value, V2_DATA_MARKER, label=f"{label} dataset.{split}_path")

    feature_cache = config.get("feature_row_cache", {})
    for split in ("train", "dev", "test"):
        value = feature_cache.get(f"{split}_path")
        if not value:
            raise ValueError(f"{label} config is missing feature_row_cache.{split}_path")
        assert_namespace(value, V2_CACHE_MARKER, label=f"{label} feature_row_cache.{split}_path")

    for item in config.get("external_features", config.get("benchmarks_features", [])):
        for key in ("train_score_cache_path", "dev_score_cache_path", "test_score_cache_path"):
            value = item.get(key)
            if value:
                assert_namespace(value, V2_CACHE_MARKER, label=f"{label} {key}")

    output_dir = config.get("output", {}).get("dir")
    if output_dir:
        # The anchor model is an intermediate model under data/cache_v2,
        # whereas the final ranker is under results_v2.  Both are isolated
        # namespaces; neither may point at the legacy ``results/`` tree.
        assert_not_legacy(output_dir)
        output_text = path_text(output_dir)
        if V2_RESULTS_MARKER not in output_text and V2_CACHE_MARKER not in output_text:
            raise ValueError(
                f"{label} output.dir must use results_v2 or cache_v2 namespace: {output_dir}"
            )


def sample_attribute(sample: dict[str, Any], mode: str) -> str:
    metadata = sample.get("metadata", {}) or {}
    key = "problem_domain" if mode == "domain" else "repo"
    value = str(metadata.get(key, "unknown")).strip()
    return value or "unknown"


def validate_sample_ids(samples: list[dict[str, Any]], *, label: str) -> list[str]:
    ids = [str(sample.get("sample_id", "")) for sample in samples]
    if any(not sample_id for sample_id in ids):
        raise ValueError(f"{label} contains a sample without sample_id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate sample_id values")
    return ids


def load_split_samples(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        path = project_path(config["dataset"][f"{split}_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)
        validate_sample_ids(rows, label=f"{split} split")
        samples[split] = rows
    return samples


def check_split_group_disjointness(samples: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    group_sets = {
        split: {
            str(sample.get("metadata", {}).get("group_key", sample.get("sample_id", "")))
            for sample in rows
        }
        for split, rows in samples.items()
    }
    overlaps: dict[str, int] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{left}_vs_{right}"] = len(group_sets[left] & group_sets[right])
    if any(overlaps.values()):
        raise ValueError(f"v2 split group overlap detected: {overlaps}")
    return overlaps


def load_feature_rows_for_split(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    path = project_path(config["feature_row_cache"][f"{split}_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    rows_by_id = load_feature_row_cache(path)
    expected_ids = {sample["sample_id"] for sample in samples}
    if set(rows_by_id) != expected_ids:
        missing = sorted(expected_ids - set(rows_by_id))[:5]
        extra = sorted(set(rows_by_id) - expected_ids)[:5]
        raise ValueError(
            f"{split} feature cache sample_id mismatch; missing={missing}, extra={extra}"
        )
    for sample in samples:
        feature_rows = rows_by_id[sample["sample_id"]]
        expected_context_ids = [context["context_id"] for context in sample["contexts"]]
        observed_context_ids = [row.get("context_id") for row in feature_rows]
        if observed_context_ids != expected_context_ids:
            raise ValueError(
                f"{split} feature cache context ordering mismatch for sample_id={sample['sample_id']}"
            )
    return rows_by_id


def filter_feature_rows(
    rows_by_id: dict[str, list[dict[str, Any]]],
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        sample["sample_id"]: copy.deepcopy(rows_by_id[sample["sample_id"]])
        for sample in samples
    }


def validate_score_payload(
    samples: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    dataset = payload["dataset"]
    expected_sample_ids = [sample["sample_id"] for sample in samples]
    if dataset["sample_ids"] != expected_sample_ids:
        raise ValueError(f"{label} sample_ids are not aligned with the fold")
    expected_context_ids = [
        [context["context_id"] for context in sample["contexts"]] for sample in samples
    ]
    if dataset["context_ids_by_group"] != expected_context_ids:
        raise ValueError(f"{label} context ordering is not aligned with the fold")
    expected_group_sizes = [len(ids) for ids in expected_context_ids]
    if [int(value) for value in dataset["group_sizes"]] != expected_group_sizes:
        raise ValueError(f"{label} group_sizes are not aligned with the fold")
    expected_gold_context_ids = [
        list(sample.get("gold_context_ids", [])) for sample in samples
    ]
    if dataset.get("gold_context_ids_by_group") != expected_gold_context_ids:
        raise ValueError(f"{label} gold_context_ids are not aligned with the fold")
    scores = np.asarray(payload["scores"], dtype=np.float32)
    if len(scores) != sum(expected_group_sizes):
        raise ValueError(f"{label} score length does not match the fold contexts")


def make_payload(
    samples: list[dict[str, Any]],
    dataset: dict[str, Any],
    scores: np.ndarray,
    *,
    model_path: str,
    label: str,
) -> dict[str, Any]:
    payload = {
        "model_path": str(model_path),
        "dataset": dataset,
        "scores": np.asarray(scores, dtype=np.float32),
    }
    validate_score_payload(samples, payload, label=label)
    return payload


def payload_group_scores(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    dataset = payload["dataset"]
    scores = np.asarray(payload["scores"], dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for sample_id, group_size in zip(dataset["sample_ids"], dataset["group_sizes"], strict=True):
        next_offset = offset + int(group_size)
        result[sample_id] = scores[offset:next_offset]
        offset = next_offset
    return result


def augment_rows_from_payload(
    rows_by_id: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
    *,
    feature_name: str,
) -> dict[str, list[dict[str, Any]]]:
    score_map = payload_group_scores(payload)
    augmented = copy.deepcopy(rows_by_id)
    for sample_id, rows in augmented.items():
        if sample_id not in score_map:
            raise KeyError(f"Missing score payload sample_id={sample_id} for feature={feature_name}")
        scores = score_map[sample_id]
        if len(rows) != len(scores):
            raise ValueError(f"Feature/score context count mismatch for sample_id={sample_id}")
        for row, score in zip(rows, scores, strict=True):
            row["features"][feature_name] = float(score)
    return augmented


def aggregate_file_mean_payload(
    samples: list[dict[str, Any]],
    semantic_payload: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Aggregate a semantic score payload by file while preserving row order."""
    score_map = payload_group_scores(semantic_payload)
    aggregated: list[float] = []
    for sample in samples:
        sample_scores = score_map[sample["sample_id"]]
        by_path: dict[str, list[float]] = {}
        context_paths: list[str] = []
        for context, score in zip(sample["contexts"], sample_scores, strict=True):
            path = extract_context_path(context["text"])
            context_paths.append(path)
            by_path.setdefault(path, []).append(float(score))
        means = {path: sum(values) / len(values) for path, values in by_path.items()}
        aggregated.extend(means[path] for path in context_paths)

    payload = {
        "model_path": f"{semantic_payload['model_path']}::file_prior::mean::none::{label}",
        "dataset": semantic_payload["dataset"],
        "scores": np.asarray(aggregated, dtype=np.float32),
    }
    validate_score_payload(samples, payload, label=f"{label} file-mean semantic score")
    return payload


def semantic_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SEMANTIC_CONFIG)
    overrides = config.get("semantic", {}) or {}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key].update(value)
        else:
            settings[key] = value
    return settings


def fit_fold_semantic_models(
    fold_train: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "query_mode": str(settings["query_mode"]),
        "context_mode": str(settings["context_mode"]),
        "min_df": int(settings["min_df"]),
        "max_df": float(settings["max_df"]),
        "max_features": int(settings["max_features"]),
        "n_components": int(settings["n_components"]),
        "random_state": int(settings["random_state"]),
    }
    word_cfg = settings["word"]
    char_cfg = settings["char"]
    word_model = fit_semantic_retriever(
        fold_train,
        analyzer=str(word_cfg["analyzer"]),
        ngram_min=int(word_cfg["ngram_min"]),
        ngram_max=int(word_cfg["ngram_max"]),
        **common,
    )
    char_model = fit_semantic_retriever(
        fold_train,
        analyzer=str(char_cfg["analyzer"]),
        ngram_min=int(char_cfg["ngram_min"]),
        ngram_max=int(char_cfg["ngram_max"]),
        **common,
    )
    return word_model, char_model


def semantic_metadata(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: model.get(key)
        for key in (
            "model_type",
            "training_corpus_size",
            "n_features",
            "n_components",
            "query_mode",
            "context_mode",
            "analyzer",
            "ngram_range",
            "min_df",
            "max_df",
            "max_features",
            "random_state",
        )
    }


def build_semantic_payloads(
    samples_by_split: dict[str, list[dict[str, Any]]],
    *,
    word_model: dict[str, Any],
    char_model: dict[str, Any],
    target_label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    word_payloads: dict[str, dict[str, Any]] = {}
    char_payloads: dict[str, dict[str, Any]] = {}
    for split, rows in samples_by_split.items():
        word_dataset, word_scores = build_semantic_score_cache(rows, word_model)
        char_dataset, char_scores = build_semantic_score_cache(rows, char_model)
        word_payloads[split] = make_payload(
            rows,
            word_dataset,
            word_scores,
            model_path=f"strict_v2::{target_label}::word_tfidf_svd",
            label=f"{target_label} {split} word semantic",
        )
        char_payloads[split] = make_payload(
            rows,
            char_dataset,
            char_scores,
            model_path=f"strict_v2::{target_label}::char_tfidf_svd",
            label=f"{target_label} {split} char semantic",
        )
    return word_payloads, char_payloads


def build_anchor_payloads(
    samples_by_split: dict[str, list[dict[str, Any]]],
    rows_by_split: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    char_payloads: dict[str, dict[str, Any]],
    file_payloads: dict[str, dict[str, Any]],
    anchor_config: dict[str, Any],
    top_k: int,
    target_label: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    anchor_training_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, rows in samples_by_split.items():
        augmented = augment_rows_from_payload(
            rows_by_split[split],
            char_payloads[split],
            feature_name="semantic_char_score",
        )
        augmented = augment_rows_from_payload(
            augmented,
            file_payloads[split],
            feature_name="semantic_file_score",
        )
        anchor_training_rows[split] = augmented

    training = fit_pointwise_logistic_ranker(
        samples_by_split["train"],
        samples_by_split["dev"],
        top_k=top_k,
        training_config=anchor_config.get("training", {}),
        train_feature_rows_by_sample_id=anchor_training_rows["train"],
        dev_feature_rows_by_sample_id=anchor_training_rows["dev"],
    )
    anchor_model = training["model"]
    payloads: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    for split, rows in samples_by_split.items():
        dataset = build_labeled_ranking_dataset(
            rows,
            base_feature_names=anchor_model["base_feature_names"],
            feature_transform=anchor_model["feature_transform"],
            feature_rows_by_sample_id=anchor_training_rows[split],
        )
        scores = linear_model_scores(dataset["X"], anchor_model)
        payloads[split] = make_payload(
            rows,
            dataset,
            scores,
            model_path=f"strict_v2::{target_label}::semantic_char_file_anchor",
            label=f"{target_label} {split} anchor",
        )
        if split in ("dev", "test"):
            evaluated = evaluate_ranking_dataset(dataset, anchor_model, top_k=top_k)
            metrics[split] = {
                "hit@1": float(evaluated["hit@1"]),
                f"hit@{top_k}": float(evaluated[f"hit@{top_k}"]),
                "mrr": float(evaluated["mrr"]),
            }
    return anchor_model, anchor_training_rows, payloads, metrics


def build_fuzzy_payloads(
    samples_by_split: dict[str, list[dict[str, Any]]],
    *,
    target_label: str,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for split, rows in samples_by_split.items():
        dataset, scores = build_fuzzy_score_cache(
            rows,
            min_token_len=2,
            focus_mode="all",
        )
        payloads[split] = make_payload(
            rows,
            dataset,
            scores,
            model_path=f"strict_v2::{target_label}::fuzzy_comment_len2",
            label=f"{target_label} {split} fuzzy",
        )
    return payloads


def build_feedback_payloads(
    samples_by_split: dict[str, list[dict[str, Any]]],
    *,
    word_model: dict[str, Any],
    anchor_payloads: dict[str, dict[str, Any]],
    target_label: str,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for split, rows in samples_by_split.items():
        dataset, scores = build_feedback_score_cache(
            rows,
            semantic_model=word_model,
            base_group_scores=payload_group_scores(anchor_payloads[split]),
            feedback_top_k=1,
            max_feedback_terms=12,
            max_paths=2,
            diversify_by_file=False,
        )
        payloads[split] = make_payload(
            rows,
            dataset,
            scores,
            model_path=f"strict_v2::{target_label}::semantic_feedback_top1",
            label=f"{target_label} {split} feedback",
        )
    return payloads


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "unknown"


def select_targets(
    samples: dict[str, list[dict[str, Any]]],
    *,
    mode: str,
    explicit_targets: list[str] | None,
    min_train: int | None,
    min_test: int | None,
    limit_targets: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    resolved_min_train = int(min_train) if min_train is not None else (100 if mode == "domain" else 200)
    resolved_min_test = int(min_test) if min_test is not None else (30 if mode == "domain" else 40)
    train_counts = Counter(sample_attribute(sample, mode) for sample in samples["train"])
    test_counts = Counter(sample_attribute(sample, mode) for sample in samples["test"])
    dev_counts = Counter(sample_attribute(sample, mode) for sample in samples["dev"])
    counts = {
        value: {
            "train": int(train_counts.get(value, 0)),
            "dev": int(dev_counts.get(value, 0)),
            "test": int(test_count),
        }
        for value, test_count in test_counts.items()
    }

    if explicit_targets:
        missing = [target for target in explicit_targets if target not in test_counts]
        if missing:
            raise ValueError(f"Explicit targets absent from test split: {missing}")
        targets = list(dict.fromkeys(explicit_targets))
    else:
        targets = [
            value
            for value, test_count in test_counts.most_common()
            if test_count >= resolved_min_test and train_counts.get(value, 0) >= resolved_min_train
        ]

    if limit_targets > 0:
        targets = targets[:limit_targets]
    if not targets:
        raise ValueError(
            f"No eligible {mode} holdout targets; thresholds are train>={resolved_min_train}, "
            f"test>={resolved_min_test}."
        )
    return targets, counts


def lexical_summary(samples: list[dict[str, Any]], *, top_k: int) -> dict[str, float]:
    metrics = evidence_metrics(samples, top_k=top_k)
    return {
        "hit@1": float(metrics["hit@1"]),
        f"hit@{top_k}": float(metrics[f"hit@{top_k}"]),
        "mrr": float(metrics["mrr"]),
    }


def weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["test_samples"]) for row in rows) or 1
    return sum(float(row[key]) * int(row["test_samples"]) for row in rows) / total


def macro_average(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / (len(rows) or 1)


def aggregate_summary(rows: list[dict[str, Any]], *, mode: str, targets: list[str], top_k: int) -> dict[str, Any]:
    metric_keys = [
        "lexical_hit@1",
        f"lexical_hit@{top_k}",
        "lexical_mrr",
        "holdout_hit@1",
        f"holdout_hit@{top_k}",
        "holdout_mrr",
        "anchor_hit@1",
        f"anchor_hit@{top_k}",
        "anchor_mrr",
    ]
    return {
        "mode": mode,
        "targets": targets,
        "completed_targets": [row["target"] for row in rows],
        "target_rows": rows,
        "macro": {key: macro_average(rows, key) for key in metric_keys} if rows else {},
        "weighted": {key: weighted_average(rows, key) for key in metric_keys} if rows else {},
    }


def write_summary_files(
    output_dir: Path,
    *,
    aggregate: dict[str, Any],
    counts: dict[str, dict[str, int]],
    top_k: int,
) -> None:
    payload = dict(aggregate)
    payload["target_counts"] = counts
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Strict v2 {aggregate['mode']} holdout",
        "",
        "Every fold fits semantic TF-IDF/SVD models on non-target training samples only.",
        "",
        f"Targets: {', '.join(aggregate['targets'])}",
        "",
        "| target | train | dev | test | lexical H@1 | strict H@1 | lexical H@3 | strict H@3 | strict MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate["target_rows"]:
        lines.append(
            f"| {row['target']} | {row['train_samples']} | {row['dev_samples']} | {row['test_samples']} | "
            f"{row['lexical_hit@1']:.4f} | {row['holdout_hit@1']:.4f} | "
            f"{row[f'lexical_hit@{top_k}']:.4f} | {row[f'holdout_hit@{top_k}']:.4f} | "
            f"{row['holdout_mrr']:.4f} |"
        )
    if aggregate["target_rows"]:
        lines.extend(
            [
                "",
                "## Macro average",
                "",
                f"- lexical: H@1={aggregate['macro']['lexical_hit@1']:.4f}, "
                f"H@{top_k}={aggregate['macro'][f'lexical_hit@{top_k}']:.4f}, "
                f"MRR={aggregate['macro']['lexical_mrr']:.4f}",
                f"- strict holdout: H@1={aggregate['macro']['holdout_hit@1']:.4f}, "
                f"H@{top_k}={aggregate['macro'][f'holdout_hit@{top_k}']:.4f}, "
                f"MRR={aggregate['macro']['holdout_mrr']:.4f}",
            ]
        )
    # Keep report generation idempotent even if a caller supplies a summary
    # list assembled from a resumed run.  In particular, never emit the
    # target table header/separator more than once.  (The previous prototype
    # appended a second header when it merged an already-rendered table.)
    table_header = (
        "| target | train | dev | test | lexical H@1 | strict H@1 | "
        "lexical H@3 | strict H@3 | strict MRR |"
    )
    table_separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    deduped_lines: list[str] = []
    seen_header = False
    seen_separator = False
    for line in lines:
        if line == table_header:
            if seen_header:
                continue
            seen_header = True
        elif line == table_separator:
            if seen_separator:
                continue
            seen_separator = True
        deduped_lines.append(line)
    lines = deduped_lines
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions(path: Path, metrics: dict[str, Any], *, top_k: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in metrics["ranked_predictions"]:
            handle.write(
                json.dumps(
                    {
                        "sample_id": item["sample_id"],
                        "gold_context_ids": item["gold_context_ids"],
                        "predicted_context_ids_topk": item["ranked_context_ids"][:top_k],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_target_fold(
    target: str,
    *,
    mode: str,
    samples: dict[str, list[dict[str, Any]]],
    feature_rows_all: dict[str, dict[str, list[dict[str, Any]]]],
    final_config: dict[str, Any],
    anchor_config: dict[str, Any],
    output_dir: Path,
    semantic_cfg: dict[str, Any],
    top_k: int,
    save_semantic_models: bool,
    overwrite: bool,
) -> dict[str, Any]:
    fold_train = [sample for sample in samples["train"] if sample_attribute(sample, mode) != target]
    fold_dev = [sample for sample in samples["dev"] if sample_attribute(sample, mode) != target]
    target_test = [sample for sample in samples["test"] if sample_attribute(sample, mode) == target]
    if not fold_train or not fold_dev or not target_test:
        raise ValueError(
            f"Invalid {mode} fold {target!r}: train={len(fold_train)}, dev={len(fold_dev)}, "
            f"target_test={len(target_test)}"
        )

    fold_samples = {"train": fold_train, "dev": fold_dev, "test": target_test}
    fold_dir = output_dir / safe_slug(target)
    if fold_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Fold output already exists: {fold_dir}; pass --overwrite to replace it."
            )
        # Remove only this validated target directory.  The caller never
        # points the command at a legacy namespace.
        import shutil

        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = {
        split: filter_feature_rows(feature_rows_all[split], fold_samples[split])
        for split in ("train", "dev", "test")
    }

    # The only fitted text models in the fold are trained on non-target train.
    word_model, char_model = fit_fold_semantic_models(fold_train, settings=semantic_cfg)
    if save_semantic_models:
        save_semantic_retriever(word_model, fold_dir / "semantic_word_model.pkl")
        save_semantic_retriever(char_model, fold_dir / "semantic_char_model.pkl")

    word_payloads, char_payloads = build_semantic_payloads(
        fold_samples,
        word_model=word_model,
        char_model=char_model,
        target_label=target,
    )
    file_payloads = {
        split: aggregate_file_mean_payload(
            fold_samples[split],
            word_payloads[split],
            label=target,
        )
        for split in ("train", "dev", "test")
    }

    anchor_model, _, anchor_payloads, anchor_metrics = build_anchor_payloads(
        fold_samples,
        fold_rows,
        char_payloads=char_payloads,
        file_payloads=file_payloads,
        anchor_config=anchor_config,
        top_k=top_k,
        target_label=target,
    )

    fuzzy_payloads = build_fuzzy_payloads(fold_samples, target_label=target)
    feedback_payloads = build_feedback_payloads(
        fold_samples,
        word_model=word_model,
        anchor_payloads=anchor_payloads,
        target_label=target,
    )

    final_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "dev", "test"):
        rows = augment_rows_from_payload(
            fold_rows[split],
            anchor_payloads[split],
            feature_name="semantic_char_file_score",
        )
        rows = augment_rows_from_payload(
            rows,
            fuzzy_payloads[split],
            feature_name="fuzzy_comment_score",
        )
        rows = augment_rows_from_payload(
            rows,
            feedback_payloads[split],
            feature_name="semantic_feedback_score",
        )
        final_rows[split] = rows

    final_training = fit_pointwise_logistic_ranker(
        fold_train,
        fold_dev,
        top_k=top_k,
        training_config=final_config.get("training", {}),
        train_feature_rows_by_sample_id=final_rows["train"],
        dev_feature_rows_by_sample_id=final_rows["dev"],
    )
    final_model = final_training["model"]
    final_dev_dataset = final_training["dev_dataset"]
    final_test_dataset = build_labeled_ranking_dataset(
        target_test,
        base_feature_names=final_model["base_feature_names"],
        feature_transform=final_model["feature_transform"],
        feature_rows_by_sample_id=final_rows["test"],
    )
    final_dev_metrics = evaluate_ranking_dataset(final_dev_dataset, final_model, top_k=top_k)
    final_test_metrics = evaluate_ranking_dataset(final_test_dataset, final_model, top_k=top_k)
    lexical = lexical_summary(target_test, top_k=top_k)

    row: dict[str, Any] = {
        "target": target,
        "mode": mode,
        "train_samples": len(fold_train),
        "dev_samples": len(fold_dev),
        "test_samples": len(target_test),
        "excluded_train_samples": len(samples["train"]) - len(fold_train),
        "excluded_dev_samples": len(samples["dev"]) - len(fold_dev),
        "selected_C": float(final_model["search"]["selected_C"]),
        "lexical_hit@1": lexical["hit@1"],
        f"lexical_hit@{top_k}": lexical[f"hit@{top_k}"],
        "lexical_mrr": lexical["mrr"],
        "holdout_hit@1": float(final_test_metrics["hit@1"]),
        f"holdout_hit@{top_k}": float(final_test_metrics[f"hit@{top_k}"]),
        "holdout_mrr": float(final_test_metrics["mrr"]),
        "anchor_hit@1": float(anchor_metrics["test"]["hit@1"]),
        f"anchor_hit@{top_k}": float(anchor_metrics["test"][f"hit@{top_k}"]),
        "anchor_mrr": float(anchor_metrics["test"]["mrr"]),
        "dev_holdout_hit@1": float(final_dev_metrics["hit@1"]),
        f"dev_holdout_hit@{top_k}": float(final_dev_metrics[f"hit@{top_k}"]),
        "dev_holdout_mrr": float(final_dev_metrics["mrr"]),
        "delta_vs_lexical_hit@1": float(final_test_metrics["hit@1"] - lexical["hit@1"]),
        f"delta_vs_lexical_hit@{top_k}": float(
            final_test_metrics[f"hit@{top_k}"] - lexical[f"hit@{top_k}"]
        ),
        "delta_vs_lexical_mrr": float(final_test_metrics["mrr"] - lexical["mrr"]),
    }

    save_linear_ranker_model(final_model, fold_dir / "model.json")
    save_linear_ranker_model(anchor_model, fold_dir / "anchor_model.json")
    (fold_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": row,
                "anchor": anchor_metrics,
                "dev": {
                    "hit@1": float(final_dev_metrics["hit@1"]),
                    f"hit@{top_k}": float(final_dev_metrics[f"hit@{top_k}"]),
                    "mrr": float(final_dev_metrics["mrr"]),
                },
                "test": {
                    "hit@1": float(final_test_metrics["hit@1"]),
                    f"hit@{top_k}": float(final_test_metrics[f"hit@{top_k}"]),
                    "mrr": float(final_test_metrics["mrr"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_predictions(fold_dir / "predictions_test.jsonl", final_test_metrics, top_k=top_k)

    provenance = {
        "provenance_version": "strict_holdout_v2.1",
        "mode": mode,
        "target": target,
        "target_attribute": "problem_domain" if mode == "domain" else "repo",
        "namespace": {
            "processed": V2_DATA_MARKER,
            "cache": V2_CACHE_MARKER,
            "results": V2_RESULTS_MARKER,
        },
        "fit_sample_ids": {
            "ranker_train": [sample["sample_id"] for sample in fold_train],
            "ranker_dev": [sample["sample_id"] for sample in fold_dev],
            "semantic_train": [sample["sample_id"] for sample in fold_train],
        },
        "target_test_sample_ids": [sample["sample_id"] for sample in target_test],
        "semantic_models": {
            "word": semantic_metadata(word_model),
            "char": semantic_metadata(char_model),
        },
        "components": {
            "file_mean": "word semantic scores aggregated by exact context path",
            "anchor": "semantic_char_score + semantic_file_score",
            "fuzzy": "min_token_len=2, focus_mode=all",
            "feedback": "anchor top-1, max_feedback_terms=12, max_paths=2",
        },
        "alignment_checks": {
            "target_in_semantic_fit": any(
                sample_attribute(sample, mode) == target for sample in fold_train
            ),
            "target_in_ranker_fit": any(
                sample_attribute(sample, mode) == target for sample in fold_train
            ),
            "target_test_isolated": all(
                sample_attribute(sample, mode) == target for sample in target_test
            ),
        },
    }
    # The two booleans above should always be false/true respectively.  Keep a
    # hard assertion so a future refactor cannot silently reintroduce leakage.
    if provenance["alignment_checks"]["target_in_semantic_fit"]:
        raise AssertionError(f"Target {target!r} leaked into semantic fit samples")
    if provenance["alignment_checks"]["target_in_ranker_fit"]:
        raise AssertionError(f"Target {target!r} leaked into ranker fit samples")
    (fold_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return row


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    anchor_config_path = project_path(args.anchor_config)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / args.mode
    )
    assert_namespace(config_path, "v2", label="final config")
    assert_namespace(anchor_config_path, "v2", label="anchor config")
    assert_namespace(output_dir, V2_RESULTS_MARKER, label="output directory")

    final_config = load_yaml(config_path)
    anchor_config = load_yaml(anchor_config_path)
    validate_v2_config(final_config, config_path=config_path, label="final")
    validate_v2_config(anchor_config, config_path=anchor_config_path, label="anchor")

    samples = load_split_samples(final_config)
    overlap = check_split_group_disjointness(samples)
    targets, target_counts = select_targets(
        samples,
        mode=args.mode,
        explicit_targets=args.targets,
        min_train=args.min_train,
        min_test=args.min_test,
        limit_targets=int(args.limit_targets),
    )
    top_k = int(final_config.get("retrieval", {}).get("top_k", 3))
    semantic_cfg = semantic_settings(final_config)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "targets": targets,
                "target_counts": {target: target_counts[target] for target in targets},
                "split_group_overlap": overlap,
                "output_dir": str(output_dir),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows_all = {
        split: load_feature_rows_for_split(final_config, samples[split], split=split)
        for split in ("train", "dev", "test")
    }

    root_provenance = {
        "provenance_version": "strict_holdout_v2.1",
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "script_sha256": hash_file(Path(__file__)),
        "mode": args.mode,
        "targets": targets,
        "target_counts": {target: target_counts[target] for target in targets},
        "split_sizes": {split: len(rows) for split, rows in samples.items()},
        "split_group_overlap": overlap,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": hash_file(config_path),
        "anchor_config": str(anchor_config_path.relative_to(PROJECT_ROOT)),
        "anchor_config_sha256": hash_file(anchor_config_path),
        "namespace": {
            "processed": V2_DATA_MARKER,
            "cache": V2_CACHE_MARKER,
            "results": V2_RESULTS_MARKER,
        },
        "semantic_settings": semantic_cfg,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(root_provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] strict {args.mode} holdout: {target}")
        row = run_target_fold(
            target,
            mode=args.mode,
            samples=samples,
            feature_rows_all=feature_rows_all,
            final_config=final_config,
            anchor_config=anchor_config,
            output_dir=output_dir,
            semantic_cfg=semantic_cfg,
            top_k=top_k,
            save_semantic_models=bool(args.save_semantic_models),
            overwrite=bool(args.overwrite),
        )
        summary_rows.append(row)
        aggregate = aggregate_summary(summary_rows, mode=args.mode, targets=targets, top_k=top_k)
        write_summary_files(output_dir, aggregate=aggregate, counts=target_counts, top_k=top_k)
        print(json.dumps(row, ensure_ascii=False))

    aggregate = aggregate_summary(summary_rows, mode=args.mode, targets=targets, top_k=top_k)
    write_summary_files(output_dir, aggregate=aggregate, counts=target_counts, top_k=top_k)
    print(json.dumps(aggregate["weighted"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
