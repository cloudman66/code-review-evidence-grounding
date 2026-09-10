"""Cluster-aware statistical validation for the v2 test split.

The unit of resampling is the PR/group (``metadata.group_key``), rather than
individual comments.  This script intentionally compares only the lexical
baseline and the canonical learned ranker; router systems are out of scope.
Reported MRR is computed from a complete ranking of every candidate context.
The top-3 prefix is used only for Hit@3 and the separately labelled diagnostic
``MRR@3``; it must never be substituted for full-ranking MRR.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from code_review_understanding.eval.baseline import build_context_feature_rows, rank_feature_rows
from code_review_understanding.models.fusion import load_score_cache

DEFAULT_DATASET = ROOT / "data/processed/swe_care_grounding_v2/test.jsonl"
DEFAULT_LEXICAL_CONFIG = ROOT / "src/configs/swe_care_grounding_v2.yaml"
DEFAULT_LEXICAL = ROOT / "results_v2/baselines/swe_care_grounding_baseline/predictions_test.jsonl"
DEFAULT_CANONICAL = ROOT / "results_v2/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached/predictions_test.jsonl"
DEFAULT_CANONICAL_SCORE_CACHE = ROOT / "data/cache_v2/scores/swe_care_canonical_ranker_test_exact.json.gz"
DEFAULT_OUTPUT = ROOT / "results_v2/diagnostics/clustered_statistical_validation"
METRICS = ("hit@1", "hit@3", "mrr")


def display_path(path: Path) -> str:
    """Keep diagnostics portable by avoiding absolute local paths."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--lexical-config",
        type=Path,
        default=DEFAULT_LEXICAL_CONFIG,
        help="v2 lexical ranking config used to rebuild the complete baseline ranking.",
    )
    parser.add_argument("--lexical-predictions", type=Path, default=DEFAULT_LEXICAL)
    parser.add_argument("--canonical-predictions", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--canonical-score-cache",
        type=Path,
        default=DEFAULT_CANONICAL_SCORE_CACHE,
        help="Complete canonical score cache used to compute full-ranking MRR.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--randomization-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_prediction_file(
    path: Path,
    dataset: list[dict],
    rebuilt_predictions: list[dict],
    *,
    label: str,
) -> dict[str, object]:
    """Verify the recorded top-k file against the ranking rebuilt locally.

    The statistical script intentionally rebuilds complete rankings so that
    MRR is computed consistently.  The supplied prediction files are still
    useful provenance, but silently ignoring a stale file is misleading.  We
    therefore require its sample order, gold ids, and reported top-k prefix to
    agree with the locally reconstructed ranking.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    expected_ids = [sample["sample_id"] for sample in dataset]
    observed_ids = [row.get("sample_id") for row in rows]
    if observed_ids != expected_ids:
        raise ValueError(f"{label} prediction sample_ids do not align with v2 dataset")
    if len(rows) != len(rebuilt_predictions):
        raise ValueError(f"{label} prediction count does not match rebuilt ranking")
    for row, rebuilt in zip(rows, rebuilt_predictions, strict=True):
        if list(row.get("gold_context_ids", [])) != list(rebuilt["gold_context_ids"]):
            raise ValueError(f"{label} gold_context_ids do not align for {row['sample_id']}")
        reported = list(row.get("predicted_context_ids_topk", []))
        if not reported:
            raise ValueError(f"{label} has an empty top-k ranking for {row['sample_id']}")
        rebuilt_prefix = list(
            rebuilt.get("ranked_context_ids", rebuilt["predicted_context_ids_topk"])
        )[: len(reported)]
        if reported != rebuilt_prefix:
            raise ValueError(f"{label} top-k ranking is stale for {row['sample_id']}")
    return {"path": display_path(path), "rows": len(rows), "validated": True}


def load_ranking_config(path: Path) -> dict[str, float]:
    """Load the exact lexical weights used for the v2 baseline."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ranking = payload.get("ranking", {})
    return {str(key): float(value) for key, value in ranking.items()}


def full_lexical_predictions(
    dataset: list[dict], ranking_config: dict[str, float] | None = None
) -> list[dict]:
    """Build complete lexical rankings from each sample's candidate pool."""
    predictions: list[dict] = []
    for sample in dataset:
        ranked_ids = [
            item["context_id"]
            for item in rank_feature_rows(
                build_context_feature_rows(sample), ranking_config=ranking_config
            )
        ]
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "gold_context_ids": sample["gold_context_ids"],
                "ranked_context_ids": ranked_ids,
                "predicted_context_ids_topk": ranked_ids[:3],
            }
        )
    return predictions


def validate_canonical_cache(cache: dict, dataset: list[dict]) -> dict[str, object]:
    """Validate every dimension needed to recover a complete ranking.

    A top-k prediction file cannot establish full-ranking MRR on its own.  The
    score cache must therefore contain one finite score for every candidate,
    in the exact sample/context order of the v2 dataset.  This check prevents a
    stale, truncated, or partially aligned cache from silently producing a
    plausible-looking MRR.
    """
    cache_dataset = cache["dataset"]
    expected_ids = [sample["sample_id"] for sample in dataset]
    expected_context_ids = [
        [context["context_id"] for context in sample["contexts"]]
        for sample in dataset
    ]
    expected_gold_ids = [list(sample["gold_context_ids"]) for sample in dataset]
    expected_group_sizes = [len(context_ids) for context_ids in expected_context_ids]
    observed_group_sizes = [int(value) for value in cache_dataset["group_sizes"]]
    checks: dict[str, object] = {
        "sample_ids": cache_dataset["sample_ids"] == expected_ids,
        "context_ids_by_group": cache_dataset["context_ids_by_group"]
        == expected_context_ids,
        "gold_context_ids_by_group": cache_dataset["gold_context_ids_by_group"]
        == expected_gold_ids,
        "group_sizes": observed_group_sizes == expected_group_sizes,
        "score_length": len(cache["scores"]) == sum(expected_group_sizes),
        "finite_scores": bool(np.isfinite(cache["scores"]).all()),
        "nonempty_groups": all(value > 0 for value in observed_group_sizes),
    }
    if not all(bool(value) for value in checks.values()):
        raise ValueError(f"Canonical score cache alignment failed: {checks}")
    return checks


def full_canonical_predictions(
    dataset: list[dict], score_cache_path: Path, *, cache: dict | None = None
) -> list[dict]:
    """Recover complete canonical rankings from the aligned v2 score cache."""
    if cache is None:
        cache = load_score_cache(score_cache_path)
    validate_canonical_cache(cache, dataset)
    cache_dataset = cache["dataset"]

    predictions: list[dict] = []
    offset = 0
    for sample, context_ids, group_size in zip(
        dataset,
        cache_dataset["context_ids_by_group"],
        cache_dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + int(group_size)
        scores = cache["scores"][offset:next_offset]
        order = np.argsort(-scores, kind="stable")
        ranked_ids = [context_ids[index] for index in order]
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "gold_context_ids": sample["gold_context_ids"],
                "ranked_context_ids": ranked_ids,
                "predicted_context_ids_topk": ranked_ids[:3],
            }
        )
        offset = next_offset
    return predictions


def validate_complete_rankings(
    dataset: list[dict], predictions: list[dict], *, label: str
) -> dict[str, object]:
    """Ensure the internal ranking used for MRR contains every candidate once."""
    expected_ids = [sample["sample_id"] for sample in dataset]
    observed_ids = [row.get("sample_id") for row in predictions]
    if observed_ids != expected_ids:
        raise ValueError(f"{label} rebuilt ranking sample_ids do not align with v2 dataset")

    candidate_counts: list[int] = []
    for sample, row in zip(dataset, predictions, strict=True):
        expected_context_ids = [context["context_id"] for context in sample["contexts"]]
        ranked_ids = list(row.get("ranked_context_ids", []))
        if len(ranked_ids) != len(expected_context_ids):
            raise ValueError(
                f"{label} ranking is truncated for {sample['sample_id']}: "
                f"{len(ranked_ids)} of {len(expected_context_ids)} candidates"
            )
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError(f"{label} ranking contains duplicate context ids for {sample['sample_id']}")
        if set(ranked_ids) != set(expected_context_ids):
            raise ValueError(f"{label} ranking candidate ids do not match dataset for {sample['sample_id']}")
        if not set(sample["gold_context_ids"]).issubset(set(ranked_ids)):
            raise ValueError(f"{label} ranking omits a gold context for {sample['sample_id']}")
        candidate_counts.append(len(ranked_ids))

    return {
        "rows": len(predictions),
        "complete_rankings": True,
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "median": float(np.median(candidate_counts)) if candidate_counts else 0.0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "total": sum(candidate_counts),
        },
    }


def outcome(gold_ids: list[str], predicted_ids: list[str]) -> dict[str, float]:
    gold = set(gold_ids)
    hit1 = float(bool(predicted_ids) and predicted_ids[0] in gold)
    hit3 = float(any(context_id in gold for context_id in predicted_ids[:3]))
    first_relevant_rank: int | None = None
    for rank, context_id in enumerate(predicted_ids, 1):
        if context_id in gold:
            first_relevant_rank = rank
            break
    reciprocal = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
    truncated_reciprocal = (
        reciprocal if first_relevant_rank is not None and first_relevant_rank <= 3 else 0.0
    )
    return {
        "hit@1": hit1,
        "hit@3": hit3,
        # This is the reported MRR: reciprocal rank over the complete ranking.
        "mrr": reciprocal,
        # Diagnostic only; never use this as the reported MRR.
        "mrr_at_3": truncated_reciprocal,
        "first_relevant_rank": float(first_relevant_rank or 0),
    }


def grouped_values(dataset: list[dict], predictions: list[dict]) -> dict[str, dict[str, list[float]]]:
    prediction_map = {row["sample_id"]: row for row in predictions}
    grouped_metrics = (*METRICS, "mrr_at_3", "first_relevant_rank")
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in grouped_metrics}
    )
    for sample in dataset:
        sample_id = sample["sample_id"]
        if sample_id not in prediction_map:
            raise KeyError(f"Missing prediction for sample_id={sample_id}")
        metadata = sample.get("metadata", {})
        group_key = str(metadata.get("group_key") or metadata.get("instance_id") or sample_id)
        ranking = prediction_map[sample_id].get("ranked_context_ids")
        if ranking is None:
            raise ValueError(f"Prediction for {sample_id} has no complete ranked_context_ids")
        values = outcome(sample["gold_context_ids"], list(ranking))
        for metric in (*METRICS, "mrr_at_3", "first_relevant_rank"):
            groups[group_key][metric].append(values[metric])
    return groups


def ranking_diagnostics(
    dataset: list[dict], groups: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    """Summarize complete-vs-truncated reciprocal rank without conflation."""
    candidate_counts = [len(sample.get("contexts", [])) for sample in dataset]
    full_values = [
        value
        for group in groups.values()
        for value in group["mrr"]
    ]
    truncated_values = [
        value
        for group in groups.values()
        for value in group["mrr_at_3"]
    ]
    ranks = [
        int(value)
        for group in groups.values()
        for value in group["first_relevant_rank"]
    ]
    beyond_three = sum(rank > 3 for rank in ranks)
    no_relevant = sum(rank == 0 for rank in ranks)
    return {
        "mrr_definition": "mean reciprocal rank of the first gold context in the complete candidate ranking",
        "truncated_mrr_at_3_definition": "diagnostic only: reciprocal rank is set to zero when the first gold context is below rank 3",
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "median": float(np.median(candidate_counts)) if candidate_counts else 0.0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "total": sum(candidate_counts),
        },
        "samples_with_first_relevant_rank_gt_3": beyond_three,
        "samples_without_relevant_candidate": no_relevant,
        "full_mrr": float(np.mean(full_values)) if full_values else 0.0,
        "truncated_mrr_at_3": float(np.mean(truncated_values)) if truncated_values else 0.0,
        "full_minus_truncated": (
            float(np.mean(full_values) - np.mean(truncated_values))
            if full_values
            else 0.0
        ),
    }


def cluster_bootstrap(
    group_values: list[list[float]], rng: np.random.Generator, samples: int
) -> dict[str, float]:
    """Cluster-resample PRs while retaining the comment-level (micro) estimand.

    For each resample, PRs are sampled with replacement and the ratio of the
    resampled metric sum to the resampled comment count is reported.  Thus a
    PR remains the resampling unit while the point estimate remains the usual
    per-comment MRR/Hit mean.
    """
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not group_values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    sums = np.asarray([sum(values) for values in group_values], dtype=np.float64)
    counts = np.asarray([len(values) for values in group_values], dtype=np.float64)
    indices = rng.integers(0, len(group_values), size=(samples, len(group_values)))
    boot = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "mean": float(sums.sum() / counts.sum()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def sign_flip_pvalue(group_deltas: list[list[float]], rng: np.random.Generator, samples: int) -> float:
    """Two-sided PR-level sign-flip test for the comment-level mean delta."""
    if samples <= 0:
        raise ValueError("randomization sample count must be positive")
    if not group_deltas:
        return 1.0
    sums = np.asarray([sum(values) for values in group_deltas], dtype=np.float64)
    total = sum(len(values) for values in group_deltas)
    observed = abs(float(sums.sum() / total))
    if observed == 0.0:
        return 1.0
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(samples, len(sums)))
    randomized = (signs * sums[None, :]).sum(axis=1) / total
    return float((np.count_nonzero(np.abs(randomized) >= observed) + 1) / (samples + 1))


def fmt(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.4f} [{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"


def main() -> None:
    args = parse_args()
    dataset = load_jsonl(args.dataset)
    lexical_config = load_ranking_config(args.lexical_config)
    # The public prediction files intentionally contain only top-k ids.  Use
    # the aligned full score cache for canonical MRR and rebuild the lexical
    # full ranking from the candidate features, so MRR has one consistent
    # definition across systems.
    lexical_predictions = full_lexical_predictions(dataset, ranking_config=lexical_config)
    canonical_cache = load_score_cache(args.canonical_score_cache)
    canonical_cache_validation = validate_canonical_cache(canonical_cache, dataset)
    canonical_predictions = full_canonical_predictions(
        dataset, args.canonical_score_cache, cache=canonical_cache
    )
    lexical_ranking_validation = validate_complete_rankings(
        dataset, lexical_predictions, label="lexical"
    )
    canonical_ranking_validation = validate_complete_rankings(
        dataset, canonical_predictions, label="canonical"
    )
    lexical_file_validation = validate_prediction_file(
        args.lexical_predictions,
        dataset,
        lexical_predictions,
        label="lexical",
    )
    canonical_file_validation = validate_prediction_file(
        args.canonical_predictions,
        dataset,
        canonical_predictions,
        label="canonical",
    )
    lexical = grouped_values(dataset, lexical_predictions)
    canonical = grouped_values(dataset, canonical_predictions)
    group_keys = sorted(lexical)
    if set(group_keys) != set(canonical):
        raise ValueError("Lexical and canonical group sets differ")
    rng_seed = int(args.seed)
    systems = {}
    for name, groups in (("lexical", lexical), ("canonical", canonical)):
        systems[name] = {"groups": len(group_keys), "samples": len(dataset), "metrics": {}}
        for offset, metric in enumerate(METRICS):
            values = [groups[group][metric] for group in group_keys]
            systems[name]["metrics"][metric] = cluster_bootstrap(
                values, np.random.default_rng(rng_seed + offset + (100 if name == "canonical" else 0)), args.bootstrap_samples
            )

    comparison = {"left": "canonical", "right": "lexical", "groups": len(group_keys), "metrics": {}}
    for offset, metric in enumerate(METRICS):
        deltas = [
            [left - right for left, right in zip(canonical[group][metric], lexical[group][metric], strict=True)]
            for group in group_keys
        ]
        ci = cluster_bootstrap(deltas, np.random.default_rng(rng_seed + 300 + offset), args.bootstrap_samples)
        comparison["metrics"][metric] = {
            "delta": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "p_value": sign_flip_pvalue(deltas, np.random.default_rng(rng_seed + 600 + offset), args.randomization_samples),
        }

    payload = {
        "dataset": display_path(args.dataset),
        "lexical_config": display_path(args.lexical_config),
        "lexical_predictions": display_path(args.lexical_predictions),
        "canonical_predictions": display_path(args.canonical_predictions),
        "canonical_score_cache": display_path(args.canonical_score_cache),
        "prediction_file_validation": {
            "lexical": lexical_file_validation,
            "canonical": canonical_file_validation,
        },
        "complete_ranking_validation": {
            "lexical": lexical_ranking_validation,
            "canonical": canonical_ranking_validation,
            "canonical_score_cache": canonical_cache_validation,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "randomization_samples": args.randomization_samples,
        "seed": args.seed,
        "cluster_unit": "metadata.group_key (PR-level)",
        "estimand": "comment-level (micro) mean; PRs are the cluster/resampling unit",
        "bootstrap_estimator": "sum(metric over resampled PR comments) / count(resampled PR comments)",
        "sign_flip_test": "two-sided PR-level sign flip of each group's summed paired delta",
        "mrr_definition": "standard reciprocal rank computed from the complete candidate ranking; top-3 is used only for Hit@3",
        "groups": len(group_keys),
        "samples": len(dataset),
        "ranking_diagnostics": {
            "lexical": ranking_diagnostics(dataset, lexical),
            "canonical": ranking_diagnostics(dataset, canonical),
            "note": "The reported MRR and its cluster confidence interval use the complete ranking. truncated_mrr_at_3 is a separate diagnostic and is not substituted for MRR.",
        },
        "systems": systems,
        "comparison": comparison,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Clustered Statistical Validation (v2)", "", 
        "Resampling unit: PR/group (`metadata.group_key`). Router systems are not included.", "",
        "| system | Hit@1 | Hit@3 | MRR (full ranking) |", "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (("lexical", "Lexical baseline"), ("canonical", "Canonical learned ranker")):
        metrics = systems[name]["metrics"]
        lines.append(f"| {label} | {fmt(metrics['hit@1'])} | {fmt(metrics['hit@3'])} | {fmt(metrics['mrr'])} |")
    lines += [
        "",
        "MRR is the reciprocal rank of the first gold context in the **complete candidate ranking**. "
        "The top-3 prefix is used only for Hit@3.",
        "",
        "## Complete-ranking audit (truncated MRR@3 is diagnostic only)",
        "",
        "| system | full MRR | truncated MRR@3 | full − truncated | first relevant rank >3 | no relevant candidate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (("lexical", "Lexical baseline"), ("canonical", "Canonical learned ranker")):
        diagnostic = payload["ranking_diagnostics"][name]
        lines.append(
            f"| {label} | {diagnostic['full_mrr']:.4f} | "
            f"{diagnostic['truncated_mrr_at_3']:.4f} | "
            f"{diagnostic['full_minus_truncated']:+.4f} | "
            f"{diagnostic['samples_with_first_relevant_rank_gt_3']} | "
            f"{diagnostic['samples_without_relevant_candidate']} |"
        )
    lines += ["", "## Canonical − lexical (cluster sign-flip test)", "", "| metric | delta | 95% CI | p-value |", "| --- | ---: | ---: | ---: |"]
    for metric in METRICS:
        row = comparison["metrics"][metric]
        lines.append(f"| {metric} | {row['delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['p_value']:.4f} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
