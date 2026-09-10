"""Audit split, annotation, and score-cache isolation for SWE-CARE v2.

The audit deliberately reports only aggregate counts.  It verifies the PR-level
split invariant that prevents candidate-pool leakage, checks that exact
comment+gold duplicates are absent across splits, and validates any supplied
score caches against the corresponding ordered dataset.  Repeated context text
across independent PRs is reported separately rather than treated as a split
overlap: the context identifiers are local to each sample and such reuse is a
property of the benchmark, not a shared PR candidate pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
from typing import Any

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.fusion import load_score_cache


DEFAULT_DATA = ROOT / "data/processed/swe_care_grounding_v2"
DEFAULT_CACHE = ROOT / "data/cache_v2/scores"
DEFAULT_OUTPUT = ROOT / "results_v2/diagnostics/v2_isolation_audit"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--score-caches",
        nargs="*",
        default=[],
        help="Optional score-cache paths to validate against their split datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def group_key(sample: dict[str, Any]) -> str:
    return str(sample.get("metadata", {}).get("group_key", ""))


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_splits(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    splits = {split: load_jsonl(data_dir / f"{split}.jsonl") for split in SPLITS}
    for split, rows in splits.items():
        ids = [str(row.get("sample_id", "")) for row in rows]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{split} contains missing or duplicate sample_id values")
        for row in rows:
            contexts = row.get("contexts", [])
            context_ids = [str(context.get("context_id", "")) for context in contexts]
            if not context_ids or len(set(context_ids)) != len(context_ids):
                raise ValueError(f"{split} has invalid context ids for {row.get('sample_id')}")
            if not set(row.get("gold_context_ids", [])) <= set(context_ids):
                raise ValueError(f"{split} has gold context outside candidate pool for {row.get('sample_id')}")
    return splits


def cache_alignment(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    payload = load_score_cache(path)
    dataset = payload["dataset"]
    expected_ids = [row["sample_id"] for row in samples]
    expected_contexts = [[context["context_id"] for context in row["contexts"]] for row in samples]
    expected_gold = [list(row["gold_context_ids"]) for row in samples]
    checks = {
        "sample_ids": dataset["sample_ids"] == expected_ids,
        "context_ids_by_group": dataset["context_ids_by_group"] == expected_contexts,
        "gold_context_ids_by_group": dataset["gold_context_ids_by_group"] == expected_gold,
        "group_sizes": [int(value) for value in dataset["group_sizes"]]
        == [len(value) for value in expected_contexts],
        "score_length": len(payload["scores"]) == sum(len(value) for value in expected_contexts),
    }
    if not all(checks.values()):
        raise ValueError(f"Score cache alignment failed: {path}: {checks}")
    return checks


def main() -> None:
    args = parse_args()
    splits = load_splits(args.data_dir)
    group_sets = {split: {group_key(row) for row in rows} for split, rows in splits.items()}
    split_overlaps = {
        f"{left}_vs_{right}": len(group_sets[left] & group_sets[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }

    fingerprints = {
        split: {
            digest({"comment": row.get("comment", ""), "gold_context_ids": row.get("gold_context_ids", [])})
            for row in rows
        }
        for split, rows in splits.items()
    }
    exact_duplicate_counts = {
        f"{left}_vs_{right}": len(fingerprints[left] & fingerprints[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    context_text_fingerprints = {
        split: {
            digest({"path": context.get("path", ""), "text": context.get("text", "")})
            for row in rows
            for context in row.get("contexts", [])
        }
        for split, rows in splits.items()
    }
    context_text_overlap_counts = {
        f"{left}_vs_{right}": len(context_text_fingerprints[left] & context_text_fingerprints[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }

    cache_results: dict[str, Any] = {}
    for raw_path in args.score_caches:
        candidate_path = Path(raw_path)
        path = candidate_path if candidate_path.is_absolute() else ROOT / candidate_path
        name = path.name
        split = next((candidate for candidate in SPLITS if f"_{candidate}_" in name), None)
        if split is None:
            raise ValueError(f"Cannot infer split from cache name: {path}")
        cache_results[rel(path)] = {"split": split, "checks": cache_alignment(path, splits[split])}

    payload = {
        "audit_version": "v2_isolation_audit.1",
        "data_dir": rel(args.data_dir),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "group_counts": {split: len(values) for split, values in group_sets.items()},
        "split_group_overlap_counts": split_overlaps,
        "exact_comment_gold_duplicate_counts": exact_duplicate_counts,
        "context_text_overlap_counts": context_text_overlap_counts,
        "cache_alignment": cache_results,
        "pass": all(value == 0 for value in split_overlaps.values())
        and all(value == 0 for value in exact_duplicate_counts.values())
        and all(all(checks.values()) for item in cache_results.values() for checks in [item["checks"]]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SWE-CARE v2 isolation audit",
        "",
        f"- Verdict: **{'PASS' if payload['pass'] else 'FAIL'}**",
        f"- Split sizes: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}",
        f"- PR/group overlaps: {split_overlaps}",
        f"- Exact comment+gold duplicates across splits: {exact_duplicate_counts}",
        f"- Repeated context text across independent splits (reported, not a PR overlap): {context_text_overlap_counts}",
        f"- Validated score caches: {len(cache_results)}",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
