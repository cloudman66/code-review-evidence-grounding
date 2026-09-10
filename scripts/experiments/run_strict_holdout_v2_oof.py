"""Run an OOF-stacked variant of the strict SWE-CARE v2 holdout.

The regular strict-holdout runner keeps the target domain/repository out of
all fitted components, but its anchor and feedback features for outer-fold
training samples are generated in-sample.  This entry point keeps the same
outer-fold semantic fit and evaluation protocol while generating anchor
scores for outer-fold training samples with group-aware inner cross-fitting.
The resulting OOF anchor scores are then used to construct the training
feedback feature.  Dev/test scores continue to come from an anchor fitted on
the complete outer training fold.

Outputs are deliberately written below a separate ``results_v2`` directory;
the default strict runner and its results are not modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    fit_pointwise_logistic_ranker,
    linear_model_scores,
    load_feature_row_cache,
    save_linear_ranker_model,
)

from scripts.experiments.run_strict_holdout_v2 import (
    DEFAULT_ANCHOR_CONFIG,
    DEFAULT_FINAL_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    V2_CACHE_MARKER,
    V2_DATA_MARKER,
    V2_RESULTS_MARKER,
    aggregate_file_mean_payload,
    augment_rows_from_payload,
    build_anchor_payloads,
    build_feedback_payloads,
    build_fuzzy_payloads,
    build_semantic_payloads,
    check_split_group_disjointness,
    filter_feature_rows,
    hash_file,
    lexical_summary,
    load_split_samples,
    load_yaml,
    make_payload,
    payload_group_scores,
    project_path,
    safe_slug,
    sample_attribute,
    semantic_settings,
    select_targets,
    validate_v2_config,
    write_predictions,
    write_summary_files,
    fit_fold_semantic_models,
)


DEFAULT_OUTPUT_ROOT_OOF = PROJECT_ROOT / "results_v2/holdout_validation_strict_oof"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("domain", "repo"), required=True)
    parser.add_argument("--config", default=str(DEFAULT_FINAL_CONFIG))
    parser.add_argument("--anchor-config", default=str(DEFAULT_ANCHOR_CONFIG))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--limit-targets", type=int, default=0)
    parser.add_argument("--min-train", type=int, default=None)
    parser.add_argument("--min-test", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-semantic-models", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def group_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) or {}
    return str(metadata.get("group_key", sample.get("sample_id", "")))


def group_aware_folds(
    samples: list[dict[str, Any]], *, n_splits: int, seed: int
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Return deterministic folds without splitting a PR group."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(group_key(sample), []).append(sample)
    if len(grouped) < 2:
        raise ValueError("OOF anchor requires at least two distinct PR groups")
    n_splits = min(int(n_splits), len(grouped))
    if n_splits < 2:
        raise ValueError("OOF anchor requires n_splits >= 2")

    # Greedily balance record counts while retaining deterministic tie breaks.
    groups = list(grouped.items())
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    buckets: list[list[tuple[str, list[dict[str, Any]]]]] = [[] for _ in range(n_splits)]
    bucket_sizes = [0] * n_splits
    for key, rows in groups:
        bucket = min(range(n_splits), key=lambda index: (bucket_sizes[index], index))
        buckets[bucket].append((key, rows))
        bucket_sizes[bucket] += len(rows)

    folds: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for bucket in buckets:
        validation_keys = {key for key, _ in bucket}
        validation = [sample for key, rows in groups for sample in rows if key in validation_keys]
        training = [sample for key, rows in groups for sample in rows if key not in validation_keys]
        if not validation or not training:
            raise ValueError("OOF group partition produced an empty train/validation fold")
        folds.append((training, validation))
    return folds


def filter_rows(rows_by_id: dict[str, list[dict[str, Any]]], samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {sample["sample_id"]: copy.deepcopy(rows_by_id[sample["sample_id"]]) for sample in samples}


def score_payload_from_model(
    samples: list[dict[str, Any]],
    feature_rows: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
    *,
    scores_by_sample_id: dict[str, np.ndarray] | None = None,
    model_path: str,
    label: str,
) -> dict[str, Any]:
    dataset = build_labeled_ranking_dataset(
        samples,
        base_feature_names=model["base_feature_names"],
        feature_transform=model["feature_transform"],
        feature_rows_by_sample_id=feature_rows,
    )
    if scores_by_sample_id is None:
        scores = linear_model_scores(dataset["X"], model)
    else:
        blocks = [np.asarray(scores_by_sample_id[sample["sample_id"]], dtype=np.float32) for sample in samples]
        scores = np.concatenate(blocks, axis=0) if blocks else np.zeros((0,), dtype=np.float32)
    return make_payload(samples, dataset, scores, model_path=model_path, label=label)


def build_oof_anchor_train_payload(
    fold_train: list[dict[str, Any]],
    anchor_training_rows: dict[str, list[dict[str, Any]]],
    *,
    anchor_config: dict[str, Any],
    selected_c: float,
    top_k: int,
    target_label: str,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit inner group-aware anchors and assemble one OOF score payload."""
    folds = group_aware_folds(fold_train, n_splits=n_splits, seed=seed)
    scores_by_id: dict[str, np.ndarray] = {}
    fold_diagnostics: list[dict[str, Any]] = []
    schema_model: dict[str, Any] | None = None
    inner_training_config = copy.deepcopy(anchor_config.get("training", {}))
    inner_training_config["candidate_c_values"] = [float(selected_c)]

    for index, (inner_train, inner_val) in enumerate(folds):
        # C has already been selected by the full outer anchor on outer dev.
        # The inner validation groups are excluded from coefficient fitting;
        # their labels are used only for diagnostics after scoring.
        training = fit_pointwise_logistic_ranker(
            inner_train,
            inner_val,
            top_k=top_k,
            training_config=inner_training_config,
            train_feature_rows_by_sample_id=filter_rows(anchor_training_rows, inner_train),
            dev_feature_rows_by_sample_id=filter_rows(anchor_training_rows, inner_val),
        )
        model = training["model"]
        if schema_model is None:
            schema_model = model
        val_dataset = build_labeled_ranking_dataset(
            inner_val,
            base_feature_names=model["base_feature_names"],
            feature_transform=model["feature_transform"],
            feature_rows_by_sample_id=filter_rows(anchor_training_rows, inner_val),
        )
        val_scores = linear_model_scores(val_dataset["X"], model)
        offset = 0
        for sample, size in zip(inner_val, val_dataset["group_sizes"], strict=True):
            next_offset = offset + int(size)
            sample_id = sample["sample_id"]
            if sample_id in scores_by_id:
                raise AssertionError(f"Duplicate OOF score assignment: {sample_id}")
            scores_by_id[sample_id] = np.asarray(val_scores[offset:next_offset], dtype=np.float32)
            offset = next_offset
        val_metrics = evaluate_ranking_dataset(val_dataset, model, top_k=top_k)
        fold_diagnostics.append(
            {
                "fold": index,
                "inner_train_samples": len(inner_train),
                "inner_validation_samples": len(inner_val),
                "inner_train_groups": len({group_key(sample) for sample in inner_train}),
                "inner_validation_groups": len({group_key(sample) for sample in inner_val}),
                "selected_C": float(model["search"]["selected_C"]),
                "validation_hit@1": float(val_metrics["hit@1"]),
                f"validation_hit@{top_k}": float(val_metrics[f"hit@{top_k}"]),
                "validation_mrr": float(val_metrics["mrr"]),
            }
        )

    expected_ids = {sample["sample_id"] for sample in fold_train}
    if set(scores_by_id) != expected_ids:
        raise AssertionError(
            f"OOF score IDs mismatch; missing={sorted(expected_ids - set(scores_by_id))[:3]}, "
            f"extra={sorted(set(scores_by_id) - expected_ids)[:3]}"
        )
    if schema_model is None:
        raise AssertionError("No OOF anchor model was fitted")
    # The first inner model supplies only the feature schema/transform.  The
    # score vectors themselves are exclusively generated by the corresponding
    # model that did not train on each validation group's samples.
    payload = score_payload_from_model(
        fold_train,
        anchor_training_rows,
        schema_model,
        scores_by_sample_id=scores_by_id,
        model_path=f"strict_v2::{target_label}::semantic_char_file_anchor::group_oof_{n_splits}",
        label=f"{target_label} train OOF anchor",
    )
    diagnostics = {
        "n_splits": n_splits,
        "seed": seed,
        "folds": fold_diagnostics,
        "sample_count": len(scores_by_id),
        "group_count": len({group_key(sample) for sample in fold_train}),
    }
    return payload, diagnostics


def run_target_fold_oof(
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
    inner_folds: int,
    seed: int,
    save_semantic_models: bool,
    overwrite: bool,
) -> dict[str, Any]:
    fold_train = [sample for sample in samples["train"] if sample_attribute(sample, mode) != target]
    fold_dev = [sample for sample in samples["dev"] if sample_attribute(sample, mode) != target]
    target_test = [sample for sample in samples["test"] if sample_attribute(sample, mode) == target]
    if not fold_train or not fold_dev or not target_test:
        raise ValueError(f"Invalid {mode} fold {target!r}")
    fold_samples = {"train": fold_train, "dev": fold_dev, "test": target_test}
    fold_dir = output_dir / safe_slug(target)
    if fold_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Fold output already exists: {fold_dir}")
        import shutil

        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = {
        split: filter_feature_rows(feature_rows_all[split], fold_samples[split])
        for split in ("train", "dev", "test")
    }

    word_model, char_model = fit_fold_semantic_models(fold_train, settings=semantic_cfg)
    if save_semantic_models:
        from code_review_understanding.models.semantic_retrieval import save_semantic_retriever

        save_semantic_retriever(word_model, fold_dir / "semantic_word_model.pkl")
        save_semantic_retriever(char_model, fold_dir / "semantic_char_model.pkl")
    word_payloads, char_payloads = build_semantic_payloads(
        fold_samples, word_model=word_model, char_model=char_model, target_label=target
    )
    file_payloads = {
        split: aggregate_file_mean_payload(fold_samples[split], word_payloads[split], label=target)
        for split in ("train", "dev", "test")
    }

    full_anchor_model, full_anchor_rows, full_anchor_payloads, anchor_metrics = build_anchor_payloads(
        fold_samples,
        fold_rows,
        char_payloads=char_payloads,
        file_payloads=file_payloads,
        anchor_config=anchor_config,
        top_k=top_k,
        target_label=target,
    )
    oof_train_payload, oof_diagnostics = build_oof_anchor_train_payload(
        fold_train,
        full_anchor_rows["train"],
        anchor_config=anchor_config,
        selected_c=float(full_anchor_model["search"]["selected_C"]),
        top_k=top_k,
        target_label=target,
        n_splits=inner_folds,
        seed=seed,
    )
    mixed_anchor_payloads = {
        "train": oof_train_payload,
        "dev": full_anchor_payloads["dev"],
        "test": full_anchor_payloads["test"],
    }
    fuzzy_payloads = build_fuzzy_payloads(fold_samples, target_label=target)
    feedback_payloads = build_feedback_payloads(
        fold_samples,
        word_model=word_model,
        anchor_payloads=mixed_anchor_payloads,
        target_label=target,
    )

    final_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "dev", "test"):
        rows = augment_rows_from_payload(
            fold_rows[split], mixed_anchor_payloads[split], feature_name="semantic_char_file_score"
        )
        rows = augment_rows_from_payload(rows, fuzzy_payloads[split], feature_name="fuzzy_comment_score")
        rows = augment_rows_from_payload(rows, feedback_payloads[split], feature_name="semantic_feedback_score")
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
    final_dev_metrics = evaluate_ranking_dataset(final_training["dev_dataset"], final_model, top_k=top_k)
    final_test_dataset = build_labeled_ranking_dataset(
        target_test,
        base_feature_names=final_model["base_feature_names"],
        feature_transform=final_model["feature_transform"],
        feature_rows_by_sample_id=final_rows["test"],
    )
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
        f"delta_vs_lexical_hit@{top_k}": float(final_test_metrics[f"hit@{top_k}"] - lexical[f"hit@{top_k}"]),
        "delta_vs_lexical_mrr": float(final_test_metrics["mrr"] - lexical["mrr"]),
    }
    save_linear_ranker_model(final_model, fold_dir / "model.json")
    save_linear_ranker_model(full_anchor_model, fold_dir / "anchor_model.json")
    (fold_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": row,
                "anchor": anchor_metrics,
                "oof_anchor": oof_diagnostics,
                "dev": {"hit@1": float(final_dev_metrics["hit@1"]), f"hit@{top_k}": float(final_dev_metrics[f"hit@{top_k}"]), "mrr": float(final_dev_metrics["mrr"])},
                "test": {"hit@1": float(final_test_metrics["hit@1"]), f"hit@{top_k}": float(final_test_metrics[f"hit@{top_k}"]), "mrr": float(final_test_metrics["mrr"])},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_predictions(fold_dir / "predictions_test.jsonl", final_test_metrics, top_k=top_k)
    provenance = {
        "provenance_version": "strict_holdout_v2.2_oof_anchor",
        "mode": mode,
        "target": target,
        "target_attribute": "problem_domain" if mode == "domain" else "repo",
        "namespace": {"processed": V2_DATA_MARKER, "cache": V2_CACHE_MARKER, "results": V2_RESULTS_MARKER},
        "fit_sample_ids": {
            "ranker_train": [sample["sample_id"] for sample in fold_train],
            "ranker_dev": [sample["sample_id"] for sample in fold_dev],
            "semantic_train": [sample["sample_id"] for sample in fold_train],
        },
        "target_test_sample_ids": [sample["sample_id"] for sample in target_test],
        "anchor_training_feature_mode": "group_aware_oof",
        "semantic_training_feature_mode": "outer_train_fitted_not_inner_cross_fitted",
        "anchor_oof": oof_diagnostics,
        "components": {
            "file_mean": "word semantic scores aggregated by exact context path",
            "anchor": "semantic_char_score + semantic_file_score",
            "fuzzy": "min_token_len=2, focus_mode=all",
            "feedback": "train uses OOF anchor top-1; dev/test use full outer-train anchor top-1",
        },
        "alignment_checks": {
            "target_in_semantic_fit": any(sample_attribute(sample, mode) == target for sample in fold_train),
            "target_in_ranker_fit": any(sample_attribute(sample, mode) == target for sample in fold_train),
            "target_test_isolated": all(sample_attribute(sample, mode) == target for sample in target_test),
        },
    }
    if provenance["alignment_checks"]["target_in_semantic_fit"] or provenance["alignment_checks"]["target_in_ranker_fit"]:
        raise AssertionError(f"Target {target!r} leaked into OOF outer-fold fit")
    (fold_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> None:
    args = parse_args()
    if int(args.inner_folds) < 2:
        raise SystemExit("--inner-folds must be >= 2")
    config_path = project_path(args.config)
    anchor_config_path = project_path(args.anchor_config)
    output_dir = project_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT_OOF / args.mode
    if V2_RESULTS_MARKER not in str(output_dir).replace("\\", "/").split("/"):
        raise ValueError(f"OOF output must remain in results_v2 namespace: {output_dir}")
    validate_v2_config(load_yaml(config_path), config_path=config_path, label="final")
    validate_v2_config(load_yaml(anchor_config_path), config_path=anchor_config_path, label="anchor")
    final_config = load_yaml(config_path)
    anchor_config = load_yaml(anchor_config_path)
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
    print(json.dumps({"mode": args.mode, "targets": targets, "target_counts": {target: target_counts[target] for target in targets}, "split_group_overlap": overlap, "output_dir": str(output_dir), "inner_folds": int(args.inner_folds), "dry_run": bool(args.dry_run)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    feature_rows_all = {
        split: load_feature_row_cache(final_config["feature_row_cache"][f"{split}_path"])
        for split in ("train", "dev", "test")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    root_provenance = {
        "provenance_version": "strict_holdout_v2.2_oof_anchor",
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
        "oof_protocol": {
            "inner_folds": int(args.inner_folds),
            "seed": int(args.seed),
            "group_field": "metadata.group_key",
            "anchor_hyperparameter_selection": "outer dev",
            "semantic_features": "outer-train fitted; not inner cross-fitted",
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(root_provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] OOF strict {args.mode} holdout: {target}")
        summary_rows.append(
            run_target_fold_oof(
                target,
                mode=args.mode,
                samples=samples,
                feature_rows_all=feature_rows_all,
                final_config=final_config,
                anchor_config=anchor_config,
                output_dir=output_dir,
                semantic_cfg=semantic_settings(final_config),
                top_k=top_k,
                inner_folds=int(args.inner_folds),
                seed=int(args.seed),
                save_semantic_models=bool(args.save_semantic_models),
                overwrite=bool(args.overwrite),
            )
        )
        aggregate = {
            "mode": args.mode,
            "targets": targets,
            "completed_targets": [row["target"] for row in summary_rows],
            "target_rows": summary_rows,
            "macro": {key: sum(float(row[key]) for row in summary_rows) / len(summary_rows) for key in summary_rows[0] if key.startswith(("lexical_", "holdout_", "anchor_"))},
            "weighted": {},
        }
        total = sum(int(row["test_samples"]) for row in summary_rows) or 1
        for key in aggregate["macro"]:
            aggregate["weighted"][key] = sum(float(row[key]) * int(row["test_samples"]) for row in summary_rows) / total
        write_summary_files(output_dir, aggregate=aggregate, counts=target_counts, top_k=top_k)
        print(json.dumps(summary_rows[-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
