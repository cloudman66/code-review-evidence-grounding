"""Evaluate metadata slices on the leakage-safe SWE-CARE v2 test split.

This is the v2 replacement for the historical metadata-slice and
``Documentation Updates`` probe scripts.  It deliberately refuses legacy
``data/processed/swe_care_grounding``, ``data/cache`` and ``results`` paths,
checks that every prediction file has the exact v2 test sample order, and
keeps the output below ``results_v2``.

The public prediction files contain only a top-k prefix.  For systems with a
model/score-cache specification this driver reconstructs the complete
candidate ranking in memory and reports standard full-ranking MRR.  A system
with predictions but no complete-ranking source is still useful for Hit@1 and
Hit@3, but its MRR is reported as ``null`` and the truncated ``mrr_at_3`` is
kept separately as a diagnostic.

The defaults cover the current v2 artifacts:

* ``lexical`` — deterministic v2 lexical baseline;
* ``canonical_single`` — the v2 fuzzy+feedback learned ranker;
* ``intent`` — the v2 intent-aware learned ranker;
* ``intent_feedback`` — an explicitly labelled v2 exploratory combination;
* ``canonical_overall`` — an explicit ``missing`` placeholder.  The latter
  is not inferred from a router or from the intent+feedback ablation.

Examples::

    PYTHONPATH=src .venv/bin/python \
      scripts/experiments/run_metadata_slice_evaluation_v2.py

    # Add a top-k-only prediction file without claiming full-ranking MRR:
    PYTHONPATH=src .venv/bin/python \
      scripts/experiments/run_metadata_slice_evaluation_v2.py \
      --prediction router=results_v2/ablations/swe_care_multimodel_router/predictions_test.jsonl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.eval.baseline import build_context_feature_rows, rank_feature_rows
from code_review_understanding.eval.external_scores import (
    augment_feature_rows_with_external_scores,
    validate_feature_rows_have_features,
)
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    linear_model_scores,
    load_feature_row_cache,
    load_linear_ranker_model,
)


V2_DATA_MARKER = "swe_care_grounding_v2"
V2_CACHE_MARKER = "cache_v2"
V2_RESULTS_MARKER = "results_v2"
LEGACY_MARKERS = (
    "data/processed/swe_care_grounding/",
    "data/cache/",
    "results/",
)
DEFAULT_DATASET = ROOT / "data/processed/swe_care_grounding_v2/test.jsonl"
DEFAULT_FEATURE_CACHE = ROOT / "data/cache_v2/feature_rows/swe_care_test.jsonl.gz"
DEFAULT_LEXICAL_CONFIG = ROOT / "src/configs/swe_care_grounding_v2.yaml"
DEFAULT_OUTPUT = ROOT / "results_v2/diagnostics/metadata_slices_v2"
DEFAULT_TOP_K = 3

DEFAULT_SYSTEMS: dict[str, dict[str, Any]] = {
    "lexical": {
        "label": "Lexical baseline",
        "kind": "active",
        "prediction": ROOT / "results_v2/baselines/swe_care_grounding_baseline/predictions_test.jsonl",
        "ranking": {"type": "lexical"},
        "note": "Deterministic v2 lexical ranking rebuilt from the v2 candidate pools.",
    },
    "canonical_single": {
        "label": "Canonical single learned ranker",
        "kind": "active",
        "prediction": ROOT
        / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached/predictions_test.jsonl",
        "ranking": {
            "type": "model",
            "model": ROOT
            / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached/model.json",
            "feature_cache": DEFAULT_FEATURE_CACHE,
            "external": {
                "semantic_char_file_score": ROOT
                / "data/cache_v2/scores/swe_care_context_slices_semantic_char_file_mean_test_exact.json.gz",
                "fuzzy_comment_score": ROOT
                / "data/cache_v2/scores/swe_care_fuzzy_comment_len2_test_exact.json.gz",
                "semantic_feedback_score": ROOT
                / "data/cache_v2/scores/swe_care_semantic_feedback_top1_test_exact.json.gz",
            },
            "score_cache": ROOT / "data/cache_v2/scores/swe_care_canonical_ranker_test_exact.json.gz",
        },
        "note": "v2 fuzzy len-2 plus pseudo-relevance feedback ranker.",
    },
    "intent": {
        "label": "Intent-aware learned ranker",
        "kind": "active",
        "prediction": ROOT
        / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_intent_exact_cached/predictions_test.jsonl",
        "ranking": {
            "type": "model",
            "model": ROOT
            / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_intent_exact_cached/model.json",
            "feature_cache": DEFAULT_FEATURE_CACHE,
            "external": {
                "semantic_char_file_score": ROOT
                / "data/cache_v2/scores/swe_care_context_slices_semantic_char_file_mean_test_exact.json.gz",
                "fuzzy_comment_score": ROOT
                / "data/cache_v2/scores/swe_care_fuzzy_comment_len2_test_exact.json.gz",
                "semantic_intent_score": ROOT
                / "data/cache_v2/scores/swe_care_semantic_intent_expanded_test_exact.json.gz",
            },
        },
        "note": "v2 intent-expanded specialist; reported independently from the canonical single ranker.",
    },
    "intent_feedback": {
        "label": "Intent + feedback learned ranker",
        "kind": "exploratory",
        "prediction": ROOT
        / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_intent_feedback_exact_cached/predictions_test.jsonl",
        "ranking": {
            "type": "model",
            "model": ROOT
            / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_intent_feedback_exact_cached/model.json",
            "feature_cache": DEFAULT_FEATURE_CACHE,
            "external": {
                "semantic_char_file_score": ROOT
                / "data/cache_v2/scores/swe_care_context_slices_semantic_char_file_mean_test_exact.json.gz",
                "fuzzy_comment_score": ROOT
                / "data/cache_v2/scores/swe_care_fuzzy_comment_len2_test_exact.json.gz",
                "semantic_intent_score": ROOT
                / "data/cache_v2/scores/swe_care_semantic_intent_expanded_test_exact.json.gz",
                "semantic_feedback_score": ROOT
                / "data/cache_v2/scores/swe_care_semantic_feedback_top1_test_exact.json.gz",
            },
        },
        "note": "Available v2 ablation; not renamed as the final overall system.",
    },
    "canonical_overall": {
        "label": "Canonical overall system",
        "kind": "missing",
        "prediction": None,
        "ranking": None,
        "note": "No final v2 canonical-overall prediction artifact was identified; not inferred from a router.",
    },
}

EXT_TO_LANG = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "CSharp",
    ".cpp": "CPP",
    ".cc": "CPP",
    ".cxx": "CPP",
    ".c": "C",
    ".h": "C/C++",
    ".hpp": "CPP",
    ".scala": "Scala",
    ".kt": "Kotlin",
    ".rs": "Rust",
    ".swift": "Swift",
    ".sh": "Shell",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".md": "Markdown",
    ".rst": "RST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lexical-config", type=Path, default=DEFAULT_LEXICAL_CONFIG)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add/override a prediction file. Repeatable; defaults are the standard v2 systems.",
    )
    parser.add_argument(
        "--missing-system",
        action="append",
        default=[],
        metavar="NAME=REASON",
        help="Add an explicitly missing system placeholder (repeatable).",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Do not load the standard v2 system definitions; use only --prediction/--missing-system.",
    )
    return parser.parse_args()


def absolute(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def portable(path: Path | str) -> str:
    value = absolute(path).resolve()
    try:
        return value.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def assert_v2_path(path: Path | str, *, label: str, allow_source: bool = False) -> None:
    """Reject legacy namespace paths and require an explicit v2 marker."""
    text = portable(path).replace("\\", "/")
    for marker in LEGACY_MARKERS:
        # ``results_v2`` contains the string ``results/`` only when the path
        # is inspected naively; match path components rather than substrings.
        if marker in text:
            raise ValueError(f"{label} points to a legacy namespace: {path}")
    if allow_source and text.startswith("-"):
        return
    if not any(marker in text.split("/") or marker in Path(text).name for marker in (V2_DATA_MARKER, V2_CACHE_MARKER, V2_RESULTS_MARKER)):
        raise ValueError(f"{label} must visibly use a v2 namespace: {path}")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_if_file(path: Path | None) -> str | None:
    return hash_file(path) if path is not None and path.is_file() else None


def parse_name_path_specs(values: list[str], *, separator: str = "=") -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if separator not in value:
            raise ValueError(f"Expected NAME{separator}VALUE, got: {value}")
        name, payload = value.split(separator, 1)
        name = name.strip()
        payload = payload.strip()
        if not name or not payload:
            raise ValueError(f"Empty name or value in: {value}")
        result[name] = payload
    return result


def language_from_path(path: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]+)$", path or "")
    extension = match.group(1).lower() if match else ""
    return EXT_TO_LANG.get(extension, extension or "unknown")


def collect_slice_groups(dataset: list[dict], *, min_samples: int) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {
        "difficulty": defaultdict(list),
        "problem_domain": defaultdict(list),
        "language": defaultdict(list),
    }
    for sample in dataset:
        sample_id = str(sample["sample_id"])
        metadata = sample.get("metadata", {}) or {}
        grouped["difficulty"][str(metadata.get("difficulty", "unknown"))].append(sample_id)
        grouped["problem_domain"][str(metadata.get("problem_domain", "unknown"))].append(sample_id)
        grouped["language"][language_from_path(str(metadata.get("path", "")))].append(sample_id)
    return {
        axis: {
            label: sample_ids
            for label, sample_ids in groups.items()
            if len(sample_ids) >= min_samples
        }
        for axis, groups in grouped.items()
    }


def validate_dataset(dataset: list[dict], path: Path) -> list[str]:
    assert_v2_path(path, label="dataset")
    ids = [str(sample.get("sample_id", "")) for sample in dataset]
    if any(not sample_id for sample_id in ids):
        raise ValueError("v2 dataset contains a sample without sample_id")
    if len(set(ids)) != len(ids):
        raise ValueError("v2 dataset contains duplicate sample_id values")
    return ids


def load_predictions(path: Path, dataset: list[dict], *, top_k: int, name: str) -> tuple[list[dict], dict[str, Any]]:
    assert_v2_path(path, label=f"{name} predictions")
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    expected_ids = [str(sample["sample_id"]) for sample in dataset]
    observed_ids = [str(row.get("sample_id", "")) for row in rows]
    if observed_ids != expected_ids:
        raise ValueError(f"{name} prediction sample_id order/count does not match v2 test data")
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError(f"{name} predictions contain duplicate sample_id values")
    expected_by_id = {str(sample["sample_id"]): sample for sample in dataset}
    for row in rows:
        sample_id = str(row["sample_id"])
        sample = expected_by_id[sample_id]
        expected_gold = list(sample.get("gold_context_ids", []))
        if list(row.get("gold_context_ids", [])) != expected_gold:
            raise ValueError(f"{name} gold_context_ids mismatch for sample_id={sample_id}")
        predicted = row.get("predicted_context_ids_topk", [])
        if not isinstance(predicted, list):
            raise ValueError(f"{name} predicted_context_ids_topk is not a list for {sample_id}")
        if not predicted:
            raise ValueError(f"{name} has an empty prediction ranking for {sample_id}")
        if len(predicted) > top_k:
            raise ValueError(f"{name} has more than top_k={top_k} predictions for {sample_id}")
        context_ids = {str(context["context_id"]) for context in sample.get("contexts", [])}
        unknown = [str(context_id) for context_id in predicted if str(context_id) not in context_ids]
        if unknown:
            raise ValueError(f"{name} contains unknown context ids for {sample_id}: {unknown[:3]}")
        if len(set(predicted)) != len(predicted):
            raise ValueError(f"{name} contains duplicate predicted context ids for {sample_id}")
    return rows, {"path": portable(path), "sha256": hash_file(path), "rows": len(rows), "validated": True}


def load_ranking_config(path: Path) -> dict[str, float]:
    assert_v2_path(path, label="lexical config")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ranking = payload.get("ranking", {})
    return {str(key): float(value) for key, value in ranking.items()}


def ranking_from_score_cache(dataset: list[dict], path: Path, *, name: str) -> tuple[dict[str, list[str]], dict[str, Any]]:
    assert_v2_path(path, label=f"{name} score cache")
    cache = load_score_cache(path)
    cached = cache["dataset"]
    expected_ids = [str(sample["sample_id"]) for sample in dataset]
    expected_contexts = [
        [str(context["context_id"]) for context in sample.get("contexts", [])]
        for sample in dataset
    ]
    expected_gold = [list(sample.get("gold_context_ids", [])) for sample in dataset]
    expected_sizes = [len(contexts) for contexts in expected_contexts]
    observed_ids = [str(value) for value in cached.get("sample_ids", [])]
    observed_contexts = [[str(value) for value in group] for group in cached.get("context_ids_by_group", [])]
    observed_gold = [list(group) for group in cached.get("gold_context_ids_by_group", [])]
    observed_sizes = [int(value) for value in cached.get("group_sizes", [])]
    checks = {
        "sample_ids": observed_ids == expected_ids,
        "context_ids_by_group": observed_contexts == expected_contexts,
        "gold_context_ids_by_group": observed_gold == expected_gold,
        "group_sizes": observed_sizes == expected_sizes,
        "score_length": len(cache["scores"]) == sum(expected_sizes),
        "finite_scores": bool(np.isfinite(cache["scores"]).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"{name} score cache alignment failed: {checks}")
    rankings: dict[str, list[str]] = {}
    offset = 0
    for sample_id, context_ids, size in zip(expected_ids, expected_contexts, expected_sizes, strict=True):
        next_offset = offset + size
        scores = cache["scores"][offset:next_offset]
        order = np.argsort(-scores, kind="stable")
        rankings[sample_id] = [context_ids[int(index)] for index in order]
        offset = next_offset
    model_path_value = str(cache.get("model_path", "") or "")
    model_path = absolute(model_path_value) if model_path_value else None
    return rankings, {
        "type": "score_cache",
        "path": portable(path),
        "sha256": hash_file(path),
        "model": (
            {
                "path": portable(model_path),
                "sha256": hash_file(model_path),
            }
            if model_path is not None and model_path.is_file()
            else ("unresolved:" + model_path_value if model_path_value else None)
        ),
        "checks": checks,
        "complete": True,
    }


def ranking_from_lexical(dataset: list[dict], config_path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    ranking_config = load_ranking_config(config_path)
    rankings: dict[str, list[str]] = {}
    for sample in dataset:
        ranked = rank_feature_rows(build_context_feature_rows(sample), ranking_config=ranking_config)
        rankings[str(sample["sample_id"])] = [item["context_id"] for item in ranked]
    return rankings, {
        "type": "lexical_rebuilt",
        "config": portable(config_path),
        "config_sha256": hash_file(config_path),
        "complete": True,
    }


def ranking_from_model(
    dataset: list[dict],
    *,
    model_path: Path,
    feature_cache_path: Path,
    external: dict[str, Path],
    name: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    for path, label in ((model_path, "model"), (feature_cache_path, "feature cache")):
        assert_v2_path(path, label=f"{name} {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
    model = load_linear_ranker_model(model_path)
    feature_rows = load_feature_row_cache(feature_cache_path)
    expected_ids = [str(sample["sample_id"]) for sample in dataset]
    if list(feature_rows) != expected_ids:
        raise ValueError(f"{name} feature-row cache sample order does not match v2 test data")
    # Do not mutate the shared cache when multiple rankers are evaluated.
    feature_rows = {sample_id: copy.deepcopy(rows) for sample_id, rows in feature_rows.items()}
    external_meta: dict[str, Any] = {}
    for feature_name, score_path in external.items():
        score_path = absolute(score_path)
        assert_v2_path(score_path, label=f"{name} external score {feature_name}")
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        feature_rows = augment_feature_rows_with_external_scores(
            feature_rows,
            score_cache_path=score_path,
            feature_name=feature_name,
        )
        external_meta[feature_name] = {"path": portable(score_path), "sha256": hash_file(score_path)}
    validate_feature_rows_have_features(
        feature_rows,
        required_feature_names=list(model["base_feature_names"]),
    )
    ranking_dataset = build_labeled_ranking_dataset(
        dataset,
        base_feature_names=list(model["base_feature_names"]),
        feature_transform=str(model["feature_transform"]),
        feature_rows_by_sample_id=feature_rows,
    )
    scores = linear_model_scores(ranking_dataset["X"], model)
    rankings: dict[str, list[str]] = {}
    offset = 0
    for sample_id, context_ids, group_size in zip(
        ranking_dataset["sample_ids"],
        ranking_dataset["context_ids_by_group"],
        ranking_dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + int(group_size)
        order = np.argsort(-scores[offset:next_offset], kind="stable")
        rankings[str(sample_id)] = [str(context_ids[int(index)]) for index in order]
        offset = next_offset
    return rankings, {
        "type": "model_rebuilt",
        "model": portable(model_path),
        "model_sha256": hash_file(model_path),
        "feature_cache": portable(feature_cache_path),
        "feature_cache_sha256": hash_file(feature_cache_path),
        "feature_transform": model.get("feature_transform"),
        "base_feature_count": len(model.get("base_feature_names", [])),
        "external_scores": external_meta,
        "complete": True,
    }


def validate_complete_rankings(dataset: list[dict], rankings: dict[str, list[str]], *, name: str) -> None:
    for sample in dataset:
        sample_id = str(sample["sample_id"])
        expected = [str(context["context_id"]) for context in sample.get("contexts", [])]
        observed = list(rankings.get(sample_id, []))
        if len(observed) != len(expected) or set(observed) != set(expected) or len(set(observed)) != len(observed):
            raise ValueError(f"{name} complete ranking does not cover exactly the v2 candidates for {sample_id}")


def metric_values(sample: dict, ranked: list[str], *, top_k: int) -> dict[str, float]:
    gold = {str(value) for value in sample.get("gold_context_ids", [])}
    hit1 = float(bool(ranked) and ranked[0] in gold)
    hitk = float(any(value in gold for value in ranked[:top_k]))
    first_rank = 0
    for index, context_id in enumerate(ranked, start=1):
        if context_id in gold:
            first_rank = index
            break
    reciprocal = 1.0 / first_rank if first_rank else 0.0
    return {
        "hit@1": hit1,
        f"hit@{top_k}": hitk,
        "mrr": reciprocal,
        "mrr_at_3": reciprocal if first_rank and first_rank <= 3 else 0.0,
        "first_relevant_rank": float(first_rank),
    }


def summarize_system(
    dataset: list[dict],
    sample_ids: list[str],
    prediction_rows: list[dict],
    rankings: dict[str, list[str]] | None,
    *,
    top_k: int,
    name: str,
) -> dict[str, Any]:
    prediction_map = {str(row["sample_id"]): row for row in prediction_rows}
    values: dict[str, dict[str, float]] = {}
    for sample in dataset:
        sample_id = str(sample["sample_id"])
        if rankings is None:
            ranked = [str(value) for value in prediction_map[sample_id].get("predicted_context_ids_topk", [])]
        else:
            ranked = rankings[sample_id]
        values[sample_id] = metric_values(sample, ranked, top_k=top_k)

    def aggregate(ids: list[str]) -> dict[str, float | None]:
        if not ids:
            return {"hit@1": None, f"hit@{top_k}": None, "mrr": None, "mrr_at_3": None}
        result: dict[str, float | None] = {}
        for key in ("hit@1", f"hit@{top_k}", "mrr", "mrr_at_3"):
            result[key] = float(np.mean([values[sample_id][key] for sample_id in ids]))
        return result

    overall = aggregate(sample_ids)
    if rankings is None:
        overall["mrr"] = None
    return {
        "name": name,
        "samples": len(sample_ids),
        "complete_ranking": rankings is not None,
        "metrics": overall,
        "values": values,
    }


def render_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def build_axis_summary(
    dataset: list[dict],
    groups: dict[str, dict[str, list[str]]],
    systems: dict[str, dict[str, Any]],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    sample_map = {str(sample["sample_id"]): sample for sample in dataset}
    for axis, axis_groups in groups.items():
        summary[axis] = {}
        for label, sample_ids in sorted(axis_groups.items(), key=lambda item: (-len(item[1]), item[0])):
            payload: dict[str, Any] = {"samples": len(sample_ids), "systems": {}}
            for name, system in systems.items():
                if system.get("status") == "missing":
                    payload["systems"][name] = {"status": "missing", "reason": system.get("reason", "")}
                    rows.append(
                        {
                            "axis": axis,
                            "label": label,
                            "samples": len(sample_ids),
                            "system": name,
                            "status": "missing",
                            "hit@1": None,
                            f"hit@{top_k}": None,
                            "mrr": None,
                            "mrr_at_3": None,
                        }
                    )
                    continue
                slice_result = summarize_system(
                    dataset,
                    sample_ids,
                    system["prediction_rows"],
                    system.get("rankings"),
                    top_k=top_k,
                    name=name,
                )
                metrics = slice_result["metrics"]
                payload["systems"][name] = {
                    "status": system.get("status", "active"),
                    "complete_ranking": slice_result["complete_ranking"],
                    "metrics": metrics,
                }
                rows.append(
                    {
                        "axis": axis,
                        "label": label,
                        "samples": len(sample_ids),
                        "system": name,
                        "status": system.get("status", "active"),
                        "complete_ranking": slice_result["complete_ranking"],
                        "hit@1": metrics["hit@1"],
                        f"hit@{top_k}": metrics[f"hit@{top_k}"],
                        "mrr": metrics["mrr"],
                        "mrr_at_3": metrics["mrr_at_3"],
                    }
                )
            summary[axis][label] = payload
    return rows, summary


def build_documentation_probe(
    dataset: list[dict],
    systems: dict[str, dict[str, Any]],
    *,
    top_k: int,
    domain: str = "Documentation Updates",
) -> dict[str, Any]:
    sample_ids = [
        str(sample["sample_id"])
        for sample in dataset
        if str((sample.get("metadata", {}) or {}).get("problem_domain", "")) == domain
    ]
    rows: list[dict[str, Any]] = []
    for name, system in systems.items():
        if system.get("status") == "missing":
            rows.append(
                {
                    "name": name,
                    "label": system.get("label", name),
                    "kind": "missing",
                    "samples": len(sample_ids),
                    "complete_ranking": False,
                    "metrics": {"hit@1": None, f"hit@{top_k}": None, "mrr": None, "mrr_at_3": None},
                    "note": system.get("reason", ""),
                }
            )
            continue
        result = summarize_system(
            dataset,
            sample_ids,
            system["prediction_rows"],
            system.get("rankings"),
            top_k=top_k,
            name=name,
        )
        rows.append(
            {
                "name": name,
                "label": system.get("label", name),
                "kind": system.get("status", "active"),
                "samples": len(sample_ids),
                "complete_ranking": result["complete_ranking"],
                "metrics": result["metrics"],
                "note": system.get("note", ""),
            }
        )
    available = [row for row in rows if row["metrics"]["hit@1"] is not None]
    best = max(available, key=lambda row: float(row["metrics"]["hit@1"])) if available else None
    return {
        "domain": domain,
        "slice_size": len(sample_ids),
        "variants": rows,
        "best_hit@1_variant": best["name"] if best else None,
        "best_hit@1": best["metrics"]["hit@1"] if best else None,
        "mrr_definition": "full-ranking MRR when complete_ranking=true; otherwise null (mrr_at_3 is diagnostic only)",
    }


def write_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    doc_probe: dict[str, Any],
    systems: dict[str, dict[str, Any]],
    top_k: int,
    min_samples: int,
) -> None:
    lines = [
        "# Metadata Slice Evaluation (v2)",
        "",
        "All rows use `data/processed/swe_care_grounding_v2/test.jsonl` and v2 prediction artifacts.",
        "Full-ranking MRR is reported only when a complete ranking was rebuilt or validated; `mrr_at_3` is diagnostic only.",
        "",
    ]
    for axis in ("difficulty", "problem_domain", "language"):
        axis_rows = [row for row in rows if row["axis"] == axis]
        if not axis_rows:
            continue
        lines.extend(
            [
                f"## {axis}",
                "",
                f"Groups with at least `{min_samples}` samples. `—` denotes an explicitly missing or unavailable value.",
                "",
                f"| label | samples | "
                + " | ".join(f"{systems[name].get('label', name)} H@1" for name in systems)
                + " |",
                "| --- | ---: | " + " | ".join("---:" for _ in systems) + " |",
            ]
        )
        by_label: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in axis_rows:
            by_label[row["label"]][row["system"]] = row
        for label in sorted(by_label, key=lambda value: (-by_label[value][next(iter(by_label[value]))]["samples"], value)):
            sample_count = next(iter(by_label[label].values()))["samples"]
            metrics = [render_metric(by_label[label].get(name, {}).get("hit@1")) for name in systems]
            lines.append(f"| {label} | {sample_count} | " + " | ".join(metrics) + " |")
        lines.append("")

    lines.extend(
        [
            "## Documentation Updates probe",
            "",
            f"Slice size: `{doc_probe['slice_size']}` comments.",
            f"Best available Hit@1: `{doc_probe['best_hit@1_variant'] or 'none'}` at `{render_metric(doc_probe['best_hit@1'])}`.",
            "",
            "| variant | status | samples | complete ranking | Hit@1 | Hit@3 | full MRR | MRR@3 diagnostic | note |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for variant in doc_probe["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['label']} | {variant['kind']} | {variant['samples']} | "
            f"{'yes' if variant['complete_ranking'] else 'no'} | {render_metric(metrics['hit@1'])} | "
            f"{render_metric(metrics[f'hit@{top_k}'])} | {render_metric(metrics['mrr'])} | "
            f"{render_metric(metrics['mrr_at_3'])} | {variant['note']} |"
        )
    lines.extend(
        [
            "",
            "`canonical_overall` is intentionally shown as missing when no final v2 artifact is available; this report does not infer it from a router or another ablation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive")
    dataset_path = absolute(args.dataset)
    output_dir = absolute(args.output_dir)
    feature_cache_path = absolute(args.feature_cache)
    # The driver must never silently replace the historical ``results/``
    # diagnostics.  Re-running into its own ``results_v2`` directory is safe;
    # any other destination is rejected before an output directory is made.
    assert_v2_path(output_dir, label="output directory")
    dataset = load_jsonl(dataset_path)
    sample_ids = validate_dataset(dataset, dataset_path)
    if not dataset:
        raise ValueError("v2 test dataset is empty")

    systems: dict[str, dict[str, Any]] = {}
    if not args.no_defaults:
        systems = copy.deepcopy(DEFAULT_SYSTEMS)
    for name, path_text in parse_name_path_specs(args.prediction).items():
        path = absolute(path_text)
        assert_v2_path(path, label=f"{name} predictions")
        if name not in systems:
            systems[name] = {
                "label": name,
                "kind": "active",
                "prediction": path,
                "ranking": None,
                "note": "User-supplied v2 prediction file; no complete-ranking source was specified.",
            }
        else:
            systems[name]["prediction"] = path
            if systems[name].get("kind") == "missing":
                # An explicit prediction override turns a placeholder into a
                # real system; keep the distinction between a supplied
                # artifact and an inferred/guessed overall system explicit.
                systems[name]["kind"] = "active"
                systems[name]["note"] = "User-supplied v2 prediction override."
            # An override is intentionally top-k-only unless its default
            # complete-ranking specification remains valid.
    for name, reason in parse_name_path_specs(args.missing_system).items():
        systems[name] = {
            "label": name,
            "kind": "missing",
            "prediction": None,
            "ranking": None,
            "note": reason,
        }

    prediction_validation: dict[str, Any] = {}
    for name, system in systems.items():
        if system.get("kind") == "missing" or system.get("prediction") is None:
            system["status"] = "missing"
            system["reason"] = system.get("note", "missing")
            continue
        prediction_path = absolute(system["prediction"])
        rows, validation = load_predictions(prediction_path, dataset, top_k=args.top_k, name=name)
        system["prediction_rows"] = rows
        system["status"] = system.get("kind", "active")
        prediction_validation[name] = validation

        ranking_spec = system.get("ranking")
        rankings: dict[str, list[str]] | None = None
        ranking_meta: dict[str, Any] | None = None
        if isinstance(ranking_spec, dict) and ranking_spec.get("type") == "lexical":
            rankings, ranking_meta = ranking_from_lexical(dataset, absolute(args.lexical_config))
        elif isinstance(ranking_spec, dict) and ranking_spec.get("type") == "model":
            score_cache = ranking_spec.get("score_cache")
            if score_cache is not None and absolute(score_cache).is_file():
                rankings, ranking_meta = ranking_from_score_cache(
                    dataset,
                    absolute(score_cache),
                    name=name,
                )
                # A model rebuild is not required for the canonical cache,
                # but record its model/config metadata for provenance below.
            else:
                rankings, ranking_meta = ranking_from_model(
                    dataset,
                    model_path=absolute(ranking_spec["model"]),
                    feature_cache_path=absolute(ranking_spec.get("feature_cache", feature_cache_path)),
                    external={key: absolute(value) for key, value in ranking_spec.get("external", {}).items()},
                    name=name,
                )
        if rankings is not None:
            validate_complete_rankings(dataset, rankings, name=name)
            # Supplied top-k files must agree with the rebuilt complete rank.
            for row in rows:
                expected_prefix = rankings[str(row["sample_id"])][: args.top_k]
                if list(row.get("predicted_context_ids_topk", [])) != expected_prefix[: len(row.get("predicted_context_ids_topk", []))]:
                    raise ValueError(f"{name} prediction prefix is stale for sample_id={row['sample_id']}")
        system["rankings"] = rankings
        system["ranking_meta"] = ranking_meta

    groups = collect_slice_groups(dataset, min_samples=args.min_samples)
    rows, axis_summary = build_axis_summary(dataset, groups, systems, top_k=args.top_k)
    doc_probe = build_documentation_probe(dataset, systems, top_k=args.top_k)

    provenance_systems: dict[str, Any] = {}
    for name, system in systems.items():
        provenance_systems[name] = {
            "label": system.get("label", name),
            "status": system.get("status", "active"),
            "note": system.get("note", ""),
            "prediction": prediction_validation.get(name),
            "ranking": system.get("ranking_meta"),
        }
    provenance = {
        "provenance_version": "metadata_slice_v2.1",
        "script": portable(Path(__file__)),
        "script_sha256": hash_file(Path(__file__)),
        "dataset": portable(dataset_path),
        "dataset_sha256": hash_file(dataset_path),
        "feature_cache": portable(feature_cache_path),
        "feature_cache_sha256": hash_if_file(feature_cache_path),
        "namespace": {
            "processed": V2_DATA_MARKER,
            "cache": V2_CACHE_MARKER,
            "results": V2_RESULTS_MARKER,
        },
        "test_samples": len(dataset),
        "top_k": args.top_k,
        "min_samples": args.min_samples,
        "slice_group_counts": {axis: len(values) for axis, values in groups.items()},
        "mrr_definition": "mean reciprocal rank of the first gold context in the complete candidate ranking",
        "mrr_at_3_definition": "diagnostic only; reciprocal rank is set to zero when the first gold context is below rank 3",
        "systems": provenance_systems,
    }
    payload = {
        "dataset": provenance["dataset"],
        "dataset_sha256": provenance["dataset_sha256"],
        "test_samples": len(dataset),
        "slice_group_counts": provenance["slice_group_counts"],
        "rows": rows,
        "summary": axis_summary,
        "documentation_updates_probe": doc_probe,
        "systems": {
            name: {
                "label": system.get("label", name),
                "status": system.get("status", "active"),
                "complete_ranking": system.get("rankings") is not None,
                "ranking": system.get("ranking_meta"),
            }
            for name, system in systems.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "documentation_updates_probe.json").write_text(json.dumps(doc_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(
        output_dir / "summary.md",
        rows=rows,
        doc_probe=doc_probe,
        systems=systems,
        top_k=args.top_k,
        min_samples=args.min_samples,
    )

    print(json.dumps({
        "output_dir": portable(output_dir),
        "systems": list(systems),
        "test_samples": len(dataset),
        "slice_group_counts": provenance["slice_group_counts"],
        "documentation_updates_samples": doc_probe["slice_size"],
        "best_documentation_hit@1": doc_probe["best_hit@1_variant"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
