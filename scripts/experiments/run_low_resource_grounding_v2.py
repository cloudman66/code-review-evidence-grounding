"""Run the leakage-safe SWE-CARE v2 low-resource grounding pilot.

This entry point is deliberately separate from the historical
``run_low_resource_grounding.py`` script.  Every input path is read from the
v2 ranker configuration and every output is required to live below
``results_v2``.  The pilot keeps the v2 dev/test splits fixed, samples only the
PR-isolated v2 training split, and records enough alignment/provenance checks
to make a partial or stale run fail loudly.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.eval.external_scores import (
    augment_feature_rows_with_external_scores,
    validate_feature_rows_have_features,
)
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    fit_pointwise_logistic_ranker,
    load_feature_row_cache,
)


DEFAULT_CONFIG = ROOT / (
    "src/configs/"
    "swe_care_learned_ranker_context_slices_file_context_semantic_char_"
    "file_mean_fuzzy_len2_feedback_exact_cached_v2.yaml"
)
DEFAULT_OUTPUT = ROOT / "results_v2/ablations/low_resource_grounding"
DEFAULT_LEXICAL_METRICS = ROOT / "results_v2/baselines/swe_care_grounding_baseline/metrics.json"
DEFAULT_FULL_SINGLE_METRICS = ROOT / (
    "results_v2/ablations/"
    "swe_care_learned_ranker_context_slices_file_context_semantic_char_"
    "file_mean_fuzzy_len2_feedback_exact_cached/metrics.json"
)
DEFAULT_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
DEFAULT_SEEDS = (7, 13, 23)

LEGACY_MARKERS = (
    "data/processed/swe_care_grounding/",
    "data/cache/",
    "results/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lexical-metrics", type=Path, default=DEFAULT_LEXICAL_METRICS)
    parser.add_argument("--full-single-metrics", type=Path, default=DEFAULT_FULL_SINGLE_METRICS)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Training fractions to evaluate (default: 0.10 0.25 0.50 1.00).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Sampling seeds; the full fraction uses only the first seed.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def assert_v2_path(path: Path, *, label: str, required_marker: str | None = None) -> None:
    """Reject accidental reads/writes through the historical namespace."""
    normalized = str(path.resolve()).replace("\\", "/")
    for marker in LEGACY_MARKERS:
        if marker in normalized:
            raise ValueError(f"{label} points to a legacy namespace: {path}")
    if required_marker and required_marker not in normalized:
        raise ValueError(f"{label} is not explicitly v2-scoped: {path}")


def require_file(path: Path, *, label: str, marker: str | None = None) -> None:
    assert_v2_path(path, label=label, required_marker=marker)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_repo(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) or {}
    return str(metadata.get("repo") or "unknown")


def sample_group(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) or {}
    return str(metadata.get("group_key") or metadata.get("instance_id") or "")


def validate_samples(samples: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{label} has missing or duplicate sample_id values")
    candidate_total = 0
    candidate_counts: list[int] = []
    for sample in samples:
        contexts = sample.get("contexts", [])
        context_ids = [str(context.get("context_id", "")) for context in contexts]
        if not context_ids or any(not value for value in context_ids):
            raise ValueError(f"{label} has an empty/invalid candidate pool for {sample['sample_id']}")
        if len(set(context_ids)) != len(context_ids):
            raise ValueError(f"{label} has duplicate context ids for {sample['sample_id']}")
        gold_ids = [str(value) for value in sample.get("gold_context_ids", [])]
        if not gold_ids or not set(gold_ids).issubset(set(context_ids)):
            raise ValueError(f"{label} gold contexts are not contained in the candidate pool for {sample['sample_id']}")
        group = sample_group(sample)
        if not group:
            raise ValueError(f"{label} is missing metadata.group_key/instance_id for {sample['sample_id']}")
        candidate_total += len(context_ids)
        candidate_counts.append(len(context_ids))
    return {
        "samples": len(samples),
        "sample_ids": sample_ids,
        "sample_id_digest": digest_json(sample_ids),
        "groups": len({sample_group(sample) for sample in samples}),
        "candidate_total": candidate_total,
        "candidate_count": {
            "min": min(candidate_counts),
            "median": sorted(candidate_counts)[len(candidate_counts) // 2],
            "max": max(candidate_counts),
        },
    }


def validate_split_isolation(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    group_sets = {
        split: {sample_group(sample) for sample in samples}
        for split, samples in splits.items()
    }
    overlaps = {
        f"{left}_vs_{right}": len(group_sets[left] & group_sets[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(value != 0 for value in overlaps.values()):
        raise ValueError(f"v2 PR/group isolation failed: {overlaps}")
    return {"group_overlap_counts": overlaps}


def validate_feature_cache(
    path: Path,
    samples: list[dict[str, Any]],
    *,
    label: str,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    require_file(path, label=label, marker="cache_v2")
    records = load_feature_row_cache(path)
    expected_ids = [sample["sample_id"] for sample in samples]
    observed_ids = list(records)
    if observed_ids != expected_ids:
        raise ValueError(f"{label} sample order does not align with its dataset")
    candidate_total = 0
    for sample in samples:
        rows = records[sample["sample_id"]]
        expected_context_ids = [context["context_id"] for context in sample["contexts"]]
        observed_context_ids = [row.get("context_id") for row in rows]
        if observed_context_ids != expected_context_ids:
            raise ValueError(f"{label} candidate order does not align for {sample['sample_id']}")
        if any("features" not in row or "text" not in row for row in rows):
            raise ValueError(f"{label} has malformed feature rows for {sample['sample_id']}")
        candidate_total += len(rows)
    return records, {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "samples": len(records),
        "candidate_total": candidate_total,
        "sample_id_digest": digest_json(observed_ids),
    }


def validate_score_cache(
    path: Path,
    samples: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    require_file(path, label=label, marker="cache_v2")
    cache = load_score_cache(path)
    dataset = cache["dataset"]
    expected_ids = [sample["sample_id"] for sample in samples]
    expected_context_ids = [
        [context["context_id"] for context in sample["contexts"]]
        for sample in samples
    ]
    expected_gold = [list(sample["gold_context_ids"]) for sample in samples]
    expected_sizes = [len(value) for value in expected_context_ids]
    checks = {
        "sample_ids": dataset["sample_ids"] == expected_ids,
        "context_ids_by_group": dataset["context_ids_by_group"] == expected_context_ids,
        "gold_context_ids_by_group": dataset["gold_context_ids_by_group"] == expected_gold,
        "group_sizes": [int(value) for value in dataset["group_sizes"]] == expected_sizes,
        "score_length": len(cache["scores"]) == sum(expected_sizes),
        "finite_scores": bool(np.isfinite(cache["scores"]).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"{label} is not aligned: {checks}")
    return {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "checks": checks,
        "samples": len(expected_ids),
        "candidate_total": sum(expected_sizes),
    }


def filter_feature_cache(
    feature_rows_by_sample_id: dict[str, list[dict]],
    sample_ids: set[str],
) -> dict[str, list[dict]]:
    return {
        sample_id: copy.deepcopy(rows)
        for sample_id, rows in feature_rows_by_sample_id.items()
        if sample_id in sample_ids
    }


def stratified_repo_sample(
    samples: list[dict[str, Any]], *, fraction: float, seed: int
) -> list[dict[str, Any]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 0.999:
        return list(samples)
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample_repo(sample)].append(sample)
    selected: list[dict[str, Any]] = []
    for repo_samples in grouped.values():
        items = list(repo_samples)
        rng.shuffle(items)
        keep = max(1, int(round(len(items) * fraction)))
        selected.extend(items[: min(keep, len(items))])
    rng.shuffle(selected)
    return selected


def load_reference_metrics(path: Path, *, label: str) -> dict[str, float]:
    require_file(path, label=label, marker="results_v2")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("retrieval_test", payload)
    return {
        "hit@1": float(metrics["hit@1"]),
        "hit@3": float(metrics["hit@3"]),
        "mrr": float(metrics["mrr"]),
    }


def mean(values: list[float]) -> float:
    return sum(values) / (len(values) or 1)


def stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def fraction_tag(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "p")


def validate_complete_prediction(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_ids = [sample["sample_id"] for sample in samples]
    if [row.get("sample_id") for row in predictions] != expected_ids:
        raise ValueError("test prediction sample order does not align with v2 test split")
    total = 0
    counts: list[int] = []
    for sample, prediction in zip(samples, predictions, strict=True):
        expected = {context["context_id"] for context in sample["contexts"]}
        ranked = list(prediction.get("ranked_context_ids", []))
        if len(ranked) != len(expected) or set(ranked) != expected or len(set(ranked)) != len(ranked):
            raise ValueError(f"incomplete/duplicate test ranking for {sample['sample_id']}")
        if not set(sample["gold_context_ids"]).issubset(set(ranked)):
            raise ValueError(f"test ranking omits gold context for {sample['sample_id']}")
        total += len(ranked)
        counts.append(len(ranked))
    return {
        "samples": len(predictions),
        "candidate_total": total,
        "candidate_count": {
            "min": min(counts),
            "median": sorted(counts)[len(counts) // 2],
            "max": max(counts),
        },
        "complete_rankings": True,
    }


def render_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    lexical: dict[str, float],
    full_single: dict[str, float],
) -> None:
    lines = [
        "# SWE-CARE v2 low-resource grounding pilot",
        "",
        "The PR-isolated v2 training split is sampled by repository; v2 dev/test are fixed.",
        "The full-ranking MRR is computed over every candidate context.",
        "",
        f"- Lexical baseline: Hit@1={lexical['hit@1']:.4f}, Hit@3={lexical['hit@3']:.4f}, MRR={lexical['mrr']:.4f}",
        f"- Full single learned ranker: Hit@1={full_single['hit@1']:.4f}, Hit@3={full_single['hit@3']:.4f}, MRR={full_single['mrr']:.4f}",
        "",
        "## Aggregated summary",
        "",
        "| train fraction | seeds | avg train samples | Hit@1 | Hit@3 | MRR | delta vs lexical H@1 | gap vs full H@1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['fraction']:.2f} | {row['runs']} | {row['avg_train_samples']:.1f} | "
            f"{row['hit@1_mean']:.4f} ± {row['hit@1_std']:.4f} | "
            f"{row['hit@3_mean']:.4f} ± {row['hit@3_std']:.4f} | "
            f"{row['mrr_mean']:.4f} ± {row['mrr_std']:.4f} | "
            f"{row['delta_vs_lexical_hit@1_mean']:+.4f} | "
            f"{row['gap_vs_full_single_hit@1_mean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-run results",
            "",
            "| fraction | seed | train samples | selected C | Hit@1 | Hit@3 | MRR |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['fraction']:.2f} | {row['seed']} | {row['train_samples']} | {row['selected_C']:.2f} | "
            f"{row['hit@1']:.4f} | {row['hit@3']:.4f} | {row['mrr']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    lexical_metrics_path = resolve_path(args.lexical_metrics)
    full_single_metrics_path = resolve_path(args.full_single_metrics)

    require_file(config_path, label="v2 ranker config", marker="v2")
    assert_v2_path(output_dir, label="v2 output directory", required_marker="results_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    dataset_cfg = config.get("dataset", {})
    feature_cfg = config.get("feature_row_cache", {})
    split_paths = {
        split: resolve_path(dataset_cfg[f"{split}_path"])
        for split in ("train", "dev", "test")
    }
    for split, path in split_paths.items():
        require_file(path, label=f"v2 {split} dataset", marker="swe_care_grounding_v2")

    splits = {split: load_jsonl(path) for split, path in split_paths.items()}
    split_validation = {
        split: validate_samples(samples, label=f"v2 {split}")
        for split, samples in splits.items()
    }
    isolation_validation = validate_split_isolation(splits)

    feature_paths = {
        split: resolve_path(feature_cfg[f"{split}_path"])
        for split in ("train", "dev", "test")
    }
    feature_rows: dict[str, dict[str, list[dict]]] = {}
    feature_validation: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        feature_rows[split], feature_validation[split] = validate_feature_cache(
            feature_paths[split], splits[split], label=f"v2 {split} feature-row cache"
        )

    external_specs: list[dict[str, Any]] = []
    for item in config.get("external_features", config.get("benchmarks_features", [])):
        name = str(item["name"])
        spec = {"name": name}
        for split in ("train", "dev", "test"):
            key = f"{split}_score_cache_path"
            if key not in item:
                raise ValueError(f"external feature {name} lacks {key}")
            path = resolve_path(item[key])
            require_file(path, label=f"{name} {split} score cache", marker="cache_v2")
            spec[f"{split}_path"] = path
        external_specs.append(spec)

    # Validate full cache alignment before mutation.  The external score
    # attachment below then verifies the same order against feature rows.
    external_validation: dict[str, dict[str, Any]] = {}
    for spec in external_specs:
        for split in ("train", "dev", "test"):
            external_validation[f"{spec['name']}::{split}"] = validate_score_cache(
                spec[f"{split}_path"], splits[split], label=f"{spec['name']} {split} score cache"
            )

    augmented_rows: dict[str, dict[str, list[dict]]] = {}
    for split in ("train", "dev", "test"):
        current = filter_feature_cache(
            feature_rows[split], {sample["sample_id"] for sample in splits[split]}
        )
        for spec in external_specs:
            current = augment_feature_rows_with_external_scores(
                current,
                score_cache_path=spec[f"{split}_path"],
                feature_name=spec["name"],
            )
        augmented_rows[split] = current

    model_feature_names = []
    for item in external_specs:
        model_feature_names.append(item["name"])
    validate_feature_rows_have_features(
        augmented_rows["train"],
        required_feature_names=model_feature_names,
    )
    validate_feature_rows_have_features(
        augmented_rows["dev"],
        required_feature_names=model_feature_names,
    )
    validate_feature_rows_have_features(
        augmented_rows["test"],
        required_feature_names=model_feature_names,
    )

    lexical_metrics = load_reference_metrics(lexical_metrics_path, label="v2 lexical metrics")
    full_single_metrics = load_reference_metrics(
        full_single_metrics_path, label="v2 full single-ranker metrics"
    )
    if split_validation["test"]["samples"] != int(
        json.loads(full_single_metrics_path.read_text(encoding="utf-8")).get("dataset_sizes", {}).get("test", split_validation["test"]["samples"])
    ):
        raise ValueError("full single-ranker reference metrics are not for the v2 test split")

    fractions = [float(value) for value in args.fractions]
    seeds = [int(value) for value in args.seeds]
    if not fractions or not seeds:
        raise ValueError("at least one fraction and one seed are required")
    if any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError(f"fractions must be in (0, 1], got {fractions}")

    run_rows: list[dict[str, Any]] = []
    run_provenance: list[dict[str, Any]] = []
    for fraction in fractions:
        fraction_seeds = [seeds[0]] if fraction >= 0.999 else seeds
        for seed in fraction_seeds:
            sampled_train = stratified_repo_sample(
                splits["train"], fraction=fraction, seed=seed
            )
            sampled_ids = [sample["sample_id"] for sample in sampled_train]
            sampled_id_set = set(sampled_ids)
            if len(sampled_ids) != len(sampled_id_set):
                raise ValueError(f"sampled train ids are duplicated for fraction={fraction}, seed={seed}")
            if not sampled_id_set.issubset(set(split_validation["train"]["sample_ids"])):
                raise ValueError("sampled train ids are not a subset of v2 train")
            sampled_groups = {sample_group(sample) for sample in sampled_train}
            if sampled_groups & {sample_group(sample) for sample in splits["dev"]}:
                raise ValueError("sampled train group overlaps v2 dev")
            if sampled_groups & {sample_group(sample) for sample in splits["test"]}:
                raise ValueError("sampled train group overlaps v2 test")

            # Keep the sampled cache in the same order as ``sampled_train``.
            # The source cache is ordered by the full v2 train split, whereas
            # repository-stratified sampling shuffles the selected rows.
            sampled_train_rows = {
                sample_id: copy.deepcopy(feature_rows["train"][sample_id])
                for sample_id in sampled_ids
            }
            for spec in external_specs:
                sampled_train_rows = augment_feature_rows_with_external_scores(
                    sampled_train_rows,
                    score_cache_path=spec["train_path"],
                    feature_name=spec["name"],
                )
            expected_sampled_ids = sampled_ids
            if list(sampled_train_rows) != expected_sampled_ids:
                raise ValueError("sampled train feature-row order changed unexpectedly")

            training_results = fit_pointwise_logistic_ranker(
                sampled_train,
                splits["dev"],
                top_k=int(config["retrieval"]["top_k"]),
                training_config=config["training"],
                train_feature_rows_by_sample_id=sampled_train_rows,
                dev_feature_rows_by_sample_id=augmented_rows["dev"],
            )
            model = training_results["model"]
            test_dataset = build_labeled_ranking_dataset(
                splits["test"],
                base_feature_names=model["base_feature_names"],
                feature_transform=model["feature_transform"],
                feature_rows_by_sample_id=augmented_rows["test"],
            )
            expected_test_ids = split_validation["test"]["sample_ids"]
            if test_dataset["sample_ids"] != expected_test_ids:
                raise ValueError("test feature matrix sample order does not align")
            test_metrics = evaluate_ranking_dataset(
                test_dataset,
                model,
                top_k=int(config["retrieval"]["top_k"]),
            )
            ranking_validation = validate_complete_prediction(
                splits["test"], test_metrics["ranked_predictions"]
            )

            run_name = f"fraction_{fraction_tag(fraction)}_seed_{seed}"
            run_dir = output_dir / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_payload = {
                "fraction": fraction,
                "seed": seed,
                "train_samples": len(sampled_train),
                "dev_samples": len(splits["dev"]),
                "test_samples": len(splits["test"]),
                "selected_C": float(model["search"]["selected_C"]),
                "best_dev_metrics": model["best_dev_metrics"],
                "retrieval_test": {
                    "hit@1": float(test_metrics["hit@1"]),
                    "hit@3": float(test_metrics[f"hit@{int(config['retrieval']['top_k'])}"]),
                    "mrr": float(test_metrics["mrr"]),
                },
                "ranking_validation": ranking_validation,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "model.json").write_text(
                json.dumps(model, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "search_results.json").write_text(
                json.dumps(training_results["search_results"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with (run_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as handle:
                for prediction in test_metrics["ranked_predictions"]:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            row = {
                "fraction": fraction,
                "seed": seed,
                "train_samples": len(sampled_train),
                "selected_C": float(model["search"]["selected_C"]),
                "hit@1": float(test_metrics["hit@1"]),
                "hit@3": float(test_metrics[f"hit@{int(config['retrieval']['top_k'])}"]),
                "mrr": float(test_metrics["mrr"]),
                "delta_vs_lexical_hit@1": float(test_metrics["hit@1"] - lexical_metrics["hit@1"]),
                "gap_vs_full_single_hit@1": float(test_metrics["hit@1"] - full_single_metrics["hit@1"]),
                "run_dir": portable_path(run_dir),
            }
            run_rows.append(row)
            run_provenance.append(
                {
                    "run": run_name,
                    "fraction": fraction,
                    "seed": seed,
                    "sampled_train": {
                        "samples": len(sampled_train),
                        "sample_id_digest": digest_json(sampled_ids),
                        "group_count": len(sampled_groups),
                        "repo_counts": dict(sorted(Counter(sample_repo(sample) for sample in sampled_train).items())),
                    },
                    "alignment_checks": {
                        "sampled_train_subset_of_v2_train": True,
                        "sampled_train_group_overlap_dev": 0,
                        "sampled_train_group_overlap_test": 0,
                        "sampled_train_feature_order": True,
                        "test_sample_order": True,
                        "test_complete_ranking": ranking_validation,
                    },
                    "model": {
                        "feature_transform": model["feature_transform"],
                        "base_feature_names": model["base_feature_names"],
                        "selected_C": float(model["search"]["selected_C"]),
                    },
                }
            )
            print(json.dumps(row, ensure_ascii=False))

    summary_rows: list[dict[str, Any]] = []
    by_fraction: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_fraction[float(row["fraction"])].append(row)
    for fraction in sorted(by_fraction):
        rows = by_fraction[fraction]
        summary_rows.append(
            {
                "fraction": fraction,
                "runs": len(rows),
                "avg_train_samples": mean([float(row["train_samples"]) for row in rows]),
                "hit@1_mean": mean([row["hit@1"] for row in rows]),
                "hit@1_std": stdev([row["hit@1"] for row in rows]),
                "hit@3_mean": mean([row["hit@3"] for row in rows]),
                "hit@3_std": stdev([row["hit@3"] for row in rows]),
                "mrr_mean": mean([row["mrr"] for row in rows]),
                "mrr_std": stdev([row["mrr"] for row in rows]),
                "delta_vs_lexical_hit@1_mean": mean([row["delta_vs_lexical_hit@1"] for row in rows]),
                "gap_vs_full_single_hit@1_mean": mean([row["gap_vs_full_single_hit@1"] for row in rows]),
            }
        )

    summary_payload = {
        "pilot_version": "swe_care_low_resource_v2",
        "config": portable_path(config_path),
        "dataset_paths": {split: portable_path(path) for split, path in split_paths.items()},
        "feature_row_paths": {split: portable_path(path) for split, path in feature_paths.items()},
        "external_features": [
            {
                "name": spec["name"],
                "paths": {split: portable_path(spec[f"{split}_path"]) for split in ("train", "dev", "test")},
            }
            for spec in external_specs
        ],
        "fractions": fractions,
        "seeds": seeds,
        "seed_policy": "fractions >= 0.999 use only the first supplied seed; other fractions use all supplied seeds",
        "sampling": "repository-stratified random sampling of the v2 train split",
        "lexical": lexical_metrics,
        "full_single": full_single_metrics,
        "split_validation": split_validation,
        "split_isolation_validation": isolation_validation,
        "feature_cache_validation": feature_validation,
        "external_cache_validation": external_validation,
        "runs": run_rows,
        "summary": summary_rows,
        "provenance": run_provenance,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "pilot_version": summary_payload["pilot_version"],
                "config": summary_payload["config"],
                "dataset_paths": summary_payload["dataset_paths"],
                "feature_row_paths": summary_payload["feature_row_paths"],
                "external_cache_validation": external_validation,
                "split_isolation_validation": isolation_validation,
                "runs": run_provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    render_markdown(
        output_dir / "summary.md",
        rows=run_rows,
        summary=summary_rows,
        lexical=lexical_metrics,
        full_single=full_single_metrics,
    )
    print(json.dumps({"output_dir": portable_path(output_dir), "runs": len(run_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
