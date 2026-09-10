"""Validate the files produced by :mod:`run_strict_holdout_v2`.

This is a read-only checker.  It verifies the v2 namespace, split/group
alignment, per-fold sample IDs and predictions, model dimensions, provenance
exclusion checks, and the strict-holdout summary format.  It never fits a
model, rewrites a result, or touches the legacy namespace.

Examples::

    .venv/bin/python scripts/experiments/validate_strict_holdout_v2.py \
        --mode domain
    .venv/bin/python scripts/experiments/validate_strict_holdout_v2.py \
        --mode repo --allow-incomplete
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.experiments.run_strict_holdout_v2 import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_FINAL_CONFIG,
    V2_RESULTS_MARKER,
    assert_namespace,
    load_split_samples,
    load_yaml,
    project_path,
    safe_slug,
    sample_attribute,
    validate_v2_config,
)


REQUIRED_ROOT_FILES = ("provenance.json", "summary.json", "summary.md")
REQUIRED_FOLD_FILES = (
    "anchor_model.json",
    "metrics.json",
    "model.json",
    "predictions_test.jsonl",
    "provenance.json",
)
SUMMARY_HEADER = (
    "| target | train | dev | test | lexical H@1 | strict H@1 | "
    "lexical H@3 | strict H@3 | strict MRR |"
)
SUMMARY_SEPARATOR = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("domain", "repo"),
        default="",
        help="Holdout mode. If omitted, read it from the output provenance.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Strict holdout output directory; defaults to results_v2/.../<mode>.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_FINAL_CONFIG),
        help="v2 final-ranker YAML used to locate the processed splits.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report missing/in-progress folds as warnings and exit successfully.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def check_model_shape(model: dict[str, Any], *, label: str, errors: list[str]) -> None:
    fields = ("feature_names", "coefficients", "scaler_mean", "scaler_scale")
    missing = [field for field in fields if field not in model]
    if missing:
        errors.append(f"{label}: missing model fields {missing}")
        return
    lengths = {field: len(model[field]) for field in fields}
    if len(set(lengths.values())) != 1:
        errors.append(f"{label}: model vector lengths differ: {lengths}")
    for field in ("coefficients", "scaler_mean", "scaler_scale"):
        try:
            values = [float(value) for value in model[field]]
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: non-numeric {field}: {exc}")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{label}: non-finite values in {field}")


def check_prediction_rows(
    rows: list[dict[str, Any]],
    expected_samples: list[dict[str, Any]],
    *,
    top_k: int,
    label: str,
    errors: list[str],
) -> None:
    expected_by_id = {sample["sample_id"]: sample for sample in expected_samples}
    observed_ids = [str(row.get("sample_id", "")) for row in rows]
    expected_ids = [sample["sample_id"] for sample in expected_samples]
    if observed_ids != expected_ids:
        errors.append(f"{label}: prediction sample_id order/count mismatch")
    if len(set(observed_ids)) != len(observed_ids):
        errors.append(f"{label}: duplicate prediction sample_id")

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        expected = expected_by_id.get(sample_id)
        if expected is None:
            errors.append(f"{label}: unexpected prediction sample_id={sample_id}")
            continue
        gold = list(row.get("gold_context_ids", []))
        if gold != list(expected.get("gold_context_ids", [])):
            errors.append(f"{label}: gold_context_ids mismatch for sample_id={sample_id}")
        predicted = row.get("predicted_context_ids_topk", [])
        if not isinstance(predicted, list):
            errors.append(f"{label}: predicted_context_ids_topk is not a list for sample_id={sample_id}")
            continue
        if len(predicted) > top_k:
            errors.append(f"{label}: more than top_k predictions for sample_id={sample_id}")
        context_ids = {context["context_id"] for context in expected.get("contexts", [])}
        invalid = [context_id for context_id in predicted if context_id not in context_ids]
        if invalid:
            errors.append(
                f"{label}: prediction contains unknown context IDs for sample_id={sample_id}: {invalid[:3]}"
            )


def check_summary_markdown(path: Path, *, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    header_count = text.count(SUMMARY_HEADER)
    separator_count = text.count(SUMMARY_SEPARATOR)
    if header_count != 1:
        errors.append(f"{path}: expected exactly one table header, found {header_count}")
    if separator_count != 1:
        errors.append(f"{path}: expected exactly one table separator, found {separator_count}")
    # A repeated table header was previously easy to miss when a run resumed;
    # check line-wise too, so a future formatting change cannot silently emit
    # the same table twice.
    header_lines = [line for line in text.splitlines() if line.strip() == SUMMARY_HEADER]
    if len(header_lines) != 1:
        errors.append(f"{path}: duplicate or malformed target table header")


def check_fold(
    fold_dir: Path,
    *,
    target: str,
    mode: str,
    samples: dict[str, list[dict[str, Any]]],
    top_k: int,
    errors: list[str],
) -> bool:
    if not fold_dir.is_dir():
        errors.append(f"missing fold directory: {fold_dir}")
        return False
    missing = [name for name in REQUIRED_FOLD_FILES if not (fold_dir / name).is_file()]
    if missing:
        errors.append(f"{fold_dir}: missing required files {missing}")
        return False

    fold_train = [sample for sample in samples["train"] if sample_attribute(sample, mode) != target]
    fold_dev = [sample for sample in samples["dev"] if sample_attribute(sample, mode) != target]
    target_test = [sample for sample in samples["test"] if sample_attribute(sample, mode) == target]
    expected_train_ids = [sample["sample_id"] for sample in fold_train]
    expected_dev_ids = [sample["sample_id"] for sample in fold_dev]
    expected_test_ids = [sample["sample_id"] for sample in target_test]

    try:
        fold_provenance = read_json(fold_dir / "provenance.json")
        metrics = read_json(fold_dir / "metrics.json")
        model = read_json(fold_dir / "model.json")
        anchor_model = read_json(fold_dir / "anchor_model.json")
        predictions = read_jsonl(fold_dir / "predictions_test.jsonl")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"{fold_dir}: cannot read required JSON artifacts: {exc}")
        return False

    if fold_provenance.get("mode") != mode or fold_provenance.get("target") != target:
        errors.append(f"{fold_dir}: provenance mode/target mismatch")
    namespace = fold_provenance.get("namespace", {})
    if namespace.get("processed") != "swe_care_grounding_v2" or namespace.get("cache") != "cache_v2":
        errors.append(f"{fold_dir}: provenance namespace is not v2")

    fit_ids = fold_provenance.get("fit_sample_ids", {})
    if fit_ids.get("ranker_train") != expected_train_ids:
        errors.append(f"{fold_dir}: ranker_train sample IDs do not match filtered fold")
    if fit_ids.get("ranker_dev") != expected_dev_ids:
        errors.append(f"{fold_dir}: ranker_dev sample IDs do not match filtered fold")
    if fit_ids.get("semantic_train") != expected_train_ids:
        errors.append(f"{fold_dir}: semantic_train sample IDs do not match filtered fold")
    if any(sample_attribute(sample, mode) == target for sample in fold_train):
        errors.append(f"{fold_dir}: target appears in ranker/semantic training samples")
    if fold_provenance.get("target_test_sample_ids") != expected_test_ids:
        errors.append(f"{fold_dir}: target_test sample IDs do not match v2 test split")

    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    expected_counts = {
        "train_samples": len(fold_train),
        "dev_samples": len(fold_dev),
        "test_samples": len(target_test),
    }
    for key, expected_value in expected_counts.items():
        if summary.get(key) != expected_value:
            errors.append(
                f"{fold_dir}: metrics summary {key}={summary.get(key)!r}, expected {expected_value}"
            )

    check_model_shape(model, label=f"{fold_dir}/model.json", errors=errors)
    check_model_shape(anchor_model, label=f"{fold_dir}/anchor_model.json", errors=errors)
    check_prediction_rows(
        predictions,
        target_test,
        top_k=top_k,
        label=f"{fold_dir}/predictions_test.jsonl",
        errors=errors,
    )
    return True


def main() -> int:
    args = parse_args()
    mode = args.mode or ""
    output_dir = project_path(args.output_dir) if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / mode if mode else None
    )
    errors: list[str] = []
    warnings: list[str] = []

    if output_dir is None:
        # If no mode is given, inspect the two standard mode roots without
        # writing anything.  An explicit mode remains preferable for CI.
        candidates = [DEFAULT_OUTPUT_ROOT / candidate for candidate in ("domain", "repo")]
        existing = [candidate for candidate in candidates if (candidate / "provenance.json").is_file()]
        if len(existing) != 1:
            errors.append(
                "Specify --mode (or --output-dir); cannot infer a unique strict holdout output directory."
            )
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 1
        output_dir = existing[0]

    assert_namespace(output_dir, V2_RESULTS_MARKER, label="output directory")
    if not output_dir.is_dir():
        errors.append(f"missing output directory: {output_dir}")
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    missing_root = [name for name in REQUIRED_ROOT_FILES if not (output_dir / name).is_file()]
    if missing_root:
        errors.append(f"missing root output files: {missing_root}")
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    try:
        root_provenance = read_json(output_dir / "provenance.json")
        aggregate = read_json(output_dir / "summary.json")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read root output JSON: {exc}")
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    inferred_mode = str(root_provenance.get("mode", ""))
    if not mode:
        mode = inferred_mode
    if mode not in {"domain", "repo"}:
        errors.append(f"invalid or missing holdout mode: {mode!r}")
    if inferred_mode and inferred_mode != mode:
        errors.append(f"root provenance mode={inferred_mode!r} differs from requested mode={mode!r}")

    config_path = project_path(args.config)
    try:
        config = load_yaml(config_path)
        validate_v2_config(config, config_path=config_path, label="final")
        samples = load_split_samples(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"cannot load validated v2 config/splits: {exc}")
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    if mode not in {"domain", "repo"}:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    top_k = int(config.get("retrieval", {}).get("top_k", 3))
    targets = list(root_provenance.get("targets", aggregate.get("targets", [])))
    if list(aggregate.get("targets", [])) != targets:
        errors.append("root provenance and summary target lists differ")
    if aggregate.get("mode") != mode:
        errors.append("summary mode differs from requested mode")
    check_summary_markdown(output_dir / "summary.md", errors=errors)

    completed_targets: list[str] = []
    for target in targets:
        fold_dir = output_dir / safe_slug(target)
        if not fold_dir.is_dir() or any(not (fold_dir / name).is_file() for name in REQUIRED_FOLD_FILES):
            message = f"incomplete fold: {target} ({fold_dir})"
            if args.allow_incomplete:
                warnings.append(message)
                continue
            errors.append(message)
            continue
        if check_fold(
            fold_dir,
            target=target,
            mode=mode,
            samples=samples,
            top_k=top_k,
            errors=errors,
        ):
            completed_targets.append(target)

    aggregate_completed = list(aggregate.get("completed_targets", []))
    if aggregate_completed != completed_targets:
        if args.allow_incomplete and set(aggregate_completed).issubset(set(completed_targets)):
            warnings.append("summary completed_targets differs while --allow-incomplete is active")
        else:
            errors.append(
                f"summary completed_targets={aggregate_completed!r}, observed={completed_targets!r}"
            )

    result = {
        "ok": not errors,
        "mode": mode,
        "output_dir": str(output_dir),
        "targets": targets,
        "completed_targets": completed_targets,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
