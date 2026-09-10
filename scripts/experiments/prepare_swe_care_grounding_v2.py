"""Prepare a leakage-resistant SWE-CARE grounding dataset.

The original preparation script split individual review comments at random.  In
SWE-CARE, however, comments from one pull request share the same candidate pool,
so the unit of splitting must be the repository/PR pair.  This script keeps the
official test PRs out of train/dev, resolves review comments to diff hunks using
the hunk content and line metadata, and records every unresolved annotation
instead of silently selecting the first hunk in a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?(?: @@ ?(.*))?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-parquet", default="third_party/benchmarks/swe_care/dev.parquet")
    parser.add_argument("--test-parquet", default="third_party/benchmarks/swe_care/test.parquet")
    parser.add_argument("--output-dir", default="data/processed/swe_care_grounding_v2")
    parser.add_argument("--language", default="Python")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--min-contexts", type=int, default=2)
    return parser.parse_args()


def clean_comment_text(text: str) -> str:
    text = text.replace("\r", "").strip()
    text = re.split(r"\n@author:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"^@\w+:\s*", "", text)
    return text.strip()


def normalize_path(path: str | None) -> str:
    value = (path or "").strip().replace("\\", "/")
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return value


def parse_hunk_header(header: str) -> tuple[int, int, int, int] | None:
    match = HUNK_HEADER_RE.match(header.strip())
    if not match:
        return None
    old_start = int(match.group(1))
    # An explicitly supplied zero count is meaningful for insertions/deletions
    # (e.g. ``@@ -10,0 +10,3 @@``).  Using ``value or 1`` here silently
    # converted zero to one and could make line-coordinate matching select a
    # hunk that does not contain the annotated line.
    old_count = int(match.group(2)) if match.group(2) is not None else 1
    new_start = int(match.group(3))
    new_count = int(match.group(4)) if match.group(4) is not None else 1
    return old_start, old_count, new_start, new_count


def _new_path_from_diff_header(line: str) -> str | None:
    # ``diff --git a/foo b/foo`` is only a fallback; ``+++ b/foo`` below is
    # authoritative for normal modifications and handles quoted paths better.
    parts = line.split()
    if len(parts) >= 4:
        return normalize_path(parts[-1])
    return None


def split_patch_to_hunks(patch: str) -> list[dict[str, Any]]:
    """Parse a unified diff into hunk records with old/new line ranges."""
    contexts: list[dict[str, Any]] = []
    current_file: str | None = None
    pending_file: str | None = None
    current: dict[str, Any] | None = None
    per_file_index: Counter[str] = Counter()

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        per_file_index[current["path"]] += 1
        current["context_id"] = f"{current['path']}::hunk_{per_file_index[current['path']]}"
        current["source"] = "diff_hunk"
        lines = current.pop("lines")
        current["text"] = f"path: {current['path']}\n" + "\n".join(
            [current["header"], *lines]
        ).strip()
        contexts.append(current)
        current = None

    for raw_line in patch.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if raw_line.startswith("diff --git "):
            flush()
            pending_file = _new_path_from_diff_header(raw_line)
            current_file = pending_file
            continue
        # ``--- ``/``+++ `` are file headers only outside an active hunk.
        # Within a hunk they are legitimate diff content (for example a
        # deleted/added line whose source text starts with three dashes or
        # pluses) and must be retained verbatim for matching/features.
        if current is None and raw_line.startswith("--- "):
            # For deleted files ``+++ /dev/null`` carries no usable path; keep
            # the old-side path so review annotations can still be located.
            marker_path = raw_line[4:].strip()
            if marker_path != "/dev/null":
                current_file = normalize_path(marker_path)
            continue
        if current is None and raw_line.startswith("+++ "):
            marker_path = raw_line[4:].strip()
            if marker_path != "/dev/null":
                current_file = normalize_path(marker_path)
            continue
        if raw_line.startswith("@@ "):
            flush()
            parsed = parse_hunk_header(raw_line)
            if parsed is None:
                # Keep malformed headers visible to the matcher as unresolved.
                parsed = (0, 0, 0, 0)
            current = {
                "path": current_file or pending_file or "unknown",
                "header": raw_line,
                "old_start": parsed[0],
                "old_count": parsed[1],
                "new_start": parsed[2],
                "new_count": parsed[3],
                "lines": [],
            }
            continue
        if current is not None:
            current["lines"].append(raw_line)

    flush()
    return contexts


def _normalized_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.replace("\r", "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("@@ "):
            continue
        if stripped.lower().startswith("@user") or stripped.lower().startswith("@author"):
            continue
        # Diff prefixes are not semantic content for matching purposes.
        if stripped[:1] in {"+", "-", " "}:
            stripped = stripped[1:].strip()
        if stripped:
            lines.append(stripped)
    return lines


def _line_in_range(line: int | None, start: int, count: int) -> bool:
    if line is None or count <= 0:
        return False
    return start <= int(line) < start + count


def _candidate_content_score(needle: str, candidate: dict[str, Any]) -> float:
    needle_lines = set(_normalized_lines(needle))
    candidate_lines = set(_normalized_lines(candidate["header"] + "\n" + candidate["text"]))
    if not needle_lines or not candidate_lines:
        return 0.0
    return len(needle_lines & candidate_lines) / float(len(needle_lines))


def match_gold_context_ids(
    contexts: list[dict[str, Any]], review: dict[str, Any]
) -> tuple[list[str], str, str | None]:
    """Return a unique hunk match, or an explicit unresolved reason."""
    path = normalize_path(review.get("path"))
    needle = (review.get("diff_hunk") or "").strip()
    by_path = [context for context in contexts if normalize_path(context["path"]) == path]
    if not by_path:
        return [], "unresolved", "path_not_found"

    # 1. Full diff-hunk content (strongest evidence).
    normalized_needle = "\n".join(line.rstrip() for line in needle.replace("\r", "").splitlines()).strip()
    exact = [
        context
        for context in by_path
        if normalized_needle
        and normalized_needle
        in "\n".join(line.rstrip() for line in context["text"].splitlines()).strip()
    ]
    if len(exact) == 1:
        return [exact[0]["context_id"]], "diff_hunk_content", None
    if len(exact) > 1:
        return [], "unresolved", "ambiguous_diff_hunk_content"

    # 2. Header coordinates.  Review hunks can contain more context than the
    # commit patch, so compare the exact path/header before using line ranges.
    header = needle.splitlines()[0].strip() if needle else ""
    exact_header = [context for context in by_path if context["header"].strip() == header]
    if len(exact_header) == 1:
        return [exact_header[0]["context_id"]], "diff_hunk_header", None
    if len(exact_header) > 1:
        return [], "unresolved", "ambiguous_diff_hunk_header"

    # 3. New/old line metadata from the annotation.
    new_line = review.get("line")
    old_line = review.get("original_line")
    old_start_line = review.get("original_start_line")
    for label, line, field in (
        ("new_line", new_line, "new_start"),
        ("old_line", old_line, "old_start"),
        ("old_start_line", old_start_line, "old_start"),
    ):
        if line is None:
            continue
        matching = [
            context
            for context in by_path
            if _line_in_range(line, context[field], context["new_count" if field == "new_start" else "old_count"])
        ]
        if len(matching) == 1:
            return [matching[0]["context_id"]], label, None
        if len(matching) > 1:
            return [], "unresolved", f"ambiguous_{label}"

    # 4. Conservative content overlap for expanded/contracted review hunks.
    scored = [(float(_candidate_content_score(needle, context)), context) for context in by_path]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and scored[0][0] >= 0.5:
        best_score = scored[0][0]
        tied = [context for score, context in scored if abs(score - best_score) < 1e-9]
        if len(tied) == 1 and (len(scored) == 1 or best_score - scored[1][0] >= 0.1):
            return [tied[0]["context_id"]], "content_overlap", None
        return [], "unresolved", "ambiguous_content_overlap"

    return [], "unresolved", "no_unique_hunk_match"


def review_group_key(row: dict[str, Any]) -> str:
    repo = str(row.get("repo") or "").strip()
    pull_number = row.get("pull_number")
    if repo and pull_number is not None:
        return f"{repo}::pr_{int(pull_number)}"
    return f"instance::{row.get('instance_id', '')}"


def _record_from_review(
    row: dict[str, Any],
    review: dict[str, Any],
    comment_index: int,
    contexts: list[dict[str, Any]],
    source_split: str,
    group_key: str,
    min_contexts: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    comment_text = clean_comment_text(review.get("text") or "")
    sample_id = f"{row['instance_id']}::comment_{comment_index}"
    if not comment_text:
        return None, None, "empty_comment"
    if len(contexts) < min_contexts:
        return (
            None,
            {
                "sample_id": sample_id,
                "source_split": source_split,
                "group_key": group_key,
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "pull_number": row.get("pull_number"),
                "head_commit": (row.get("commit_to_review") or {}).get("head_commit"),
                "path": review.get("path"),
                "line": review.get("line"),
                "original_line": review.get("original_line"),
                "original_start_line": review.get("original_start_line"),
                "diff_hunk": review.get("diff_hunk") or "",
                "candidate_context_ids": [context["context_id"] for context in contexts],
                "reason": "fewer_than_min_contexts",
                "matcher_stage": "candidate_pool_filter",
            },
            "unresolved",
        )

    gold_ids, method, reason = match_gold_context_ids(contexts, review)
    if not gold_ids:
        unresolved = {
            "sample_id": sample_id,
            "source_split": source_split,
            "group_key": group_key,
            "instance_id": row.get("instance_id"),
            "repo": row.get("repo"),
            "pull_number": row.get("pull_number"),
            "head_commit": (row.get("commit_to_review") or {}).get("head_commit"),
            "path": review.get("path"),
            "line": review.get("line"),
            "original_line": review.get("original_line"),
            "original_start_line": review.get("original_start_line"),
            "diff_hunk": review.get("diff_hunk") or "",
            "candidate_context_ids": [context["context_id"] for context in contexts],
            "reason": reason or "unknown",
            "matcher_stage": method,
        }
        return None, unresolved, "unresolved"

    public_contexts = [
        {"context_id": context["context_id"], "source": context["source"], "text": context["text"]}
        for context in contexts
    ]
    record = {
        "sample_id": sample_id,
        "comment": comment_text,
        "intent": "review_comment",
        "gold_context_ids": gold_ids,
        "contexts": public_contexts,
        "metadata": {
            "repo": row["repo"],
            "pull_number": row.get("pull_number"),
            "instance_id": row.get("instance_id"),
            "group_key": group_key,
            "head_commit": (row.get("commit_to_review") or {}).get("head_commit"),
            "difficulty": row["metadata"]["difficulty"],
            "problem_domain": row["metadata"]["problem_domain"],
            "path": review.get("path"),
            "gold_match_method": method,
        },
    }
    return record, None, method


def load_grounding_records(
    parquet_path: str,
    language: str,
    source_split: str,
    excluded_group_keys: set[str],
    min_contexts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    table = pq.read_table(
        parquet_path,
        columns=[
            "instance_id",
            "repo",
            "pull_number",
            "language",
            "commit_to_review",
            "reference_review_comments",
            "metadata",
        ],
    )
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in table.to_pylist():
        if row["language"] != language:
            stats["non_target_language_rows"] += 1
            continue
        group_key = review_group_key(row)
        stats["source_rows"] += 1
        if group_key in excluded_group_keys:
            stats["excluded_official_test_overlap_rows"] += 1
            continue
        patch = row["commit_to_review"]["patch_to_review"]
        contexts = split_patch_to_hunks(patch)
        if len(contexts) < min_contexts:
            stats["rows_fewer_than_min_contexts"] += 1
        for idx, review in enumerate(row["reference_review_comments"]):
            record, unresolved_row, method = _record_from_review(
                row, review, idx, contexts, source_split, group_key, min_contexts
            )
            if record is not None:
                records.append(record)
                stats[f"matched_{method}"] += 1
            elif unresolved_row is not None:
                unresolved.append(unresolved_row)
                stats[f"unresolved_{unresolved_row['reason']}"] += 1
            else:
                stats[method] += 1
    stats["groups"] = len({record["metadata"]["group_key"] for record in records})
    return records, unresolved, dict(stats)


def split_groups(
    records: list[dict[str, Any]], dev_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["metadata"]["group_key"]].append(record)
    groups = list(grouped.items())
    random.Random(seed).shuffle(groups)
    target = max(1, int(round(len(records) * dev_fraction))) if records else 0
    dev_groups: list[tuple[str, list[dict[str, Any]]]] = []
    dev_count = 0
    remaining = list(groups)
    # A nearest-fit strategy can select one very large PR (dozens of comments)
    # and make the dev candidate-pool distribution badly unrepresentative.
    # Sequential accumulation keeps the split deterministic while avoiding that
    # pathological bias; the final size may differ from the target by one group.
    while remaining and (dev_count < target or not dev_groups):
        group = remaining.pop(0)
        dev_groups.append(group)
        dev_count += len(group[1])
    dev_keys = {key for key, _ in dev_groups}
    train = [record for key, rows in groups if key not in dev_keys for record in rows]
    dev = [record for key, rows in dev_groups for record in rows]
    return train, dev, {
        "group_count": len(groups),
        "train_group_count": len(groups) - len(dev_groups),
        "dev_group_count": len(dev_groups),
        "target_dev_records": target,
        "actual_dev_records": len(dev),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.dev_fraction < 1.0:
        raise SystemExit("--dev-fraction must be between 0 and 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read official test PR keys first.  The dev parquet contains these same
    # instances, so exclusion must happen before train/dev splitting.
    test_table = pq.read_table(args.test_parquet, columns=["repo", "pull_number", "instance_id"])
    official_test_group_keys = {
        review_group_key(row) for row in test_table.to_pylist() if row.get("repo")
    }

    test_records, test_unresolved, test_stats = load_grounding_records(
        args.test_parquet,
        args.language,
        "official_test",
        excluded_group_keys=set(),
        min_contexts=args.min_contexts,
    )
    dev_source_records, dev_unresolved, dev_stats = load_grounding_records(
        args.dev_parquet,
        args.language,
        "dev_source",
        excluded_group_keys=official_test_group_keys,
        min_contexts=args.min_contexts,
    )
    train_records, dev_records, split_stats = split_groups(
        dev_source_records, dev_fraction=args.dev_fraction, seed=args.seed
    )

    train_groups = {record["metadata"]["group_key"] for record in train_records}
    dev_groups = {record["metadata"]["group_key"] for record in dev_records}
    test_groups = {record["metadata"]["group_key"] for record in test_records}
    overlap = {
        "train_dev": sorted(train_groups & dev_groups),
        "train_test": sorted(train_groups & test_groups),
        "dev_test": sorted(dev_groups & test_groups),
    }

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "dev.jsonl", dev_records)
    write_jsonl(output_dir / "test.jsonl", test_records)
    unresolved = dev_unresolved + test_unresolved
    write_jsonl(output_dir / "unresolved.jsonl", unresolved)

    summary = {
        "format_version": "swe_care_grounding_v2",
        "source": {
            "dev_parquet": str(args.dev_parquet),
            "dev_parquet_sha256": sha256(Path(args.dev_parquet)),
            "test_parquet": str(args.test_parquet),
            "test_parquet_sha256": sha256(Path(args.test_parquet)),
            "language": args.language,
            "seed": args.seed,
            "dev_fraction": args.dev_fraction,
            "min_contexts": args.min_contexts,
        },
        "records": {"train": len(train_records), "dev": len(dev_records), "test": len(test_records)},
        "groups": {
            "official_test": len(official_test_group_keys),
            "train": len(train_groups),
            "dev": len(dev_groups),
            "test": len(test_groups),
            "overlap_counts": {key: len(value) for key, value in overlap.items()},
        },
        "excluded_official_test_overlap_rows": dev_stats.get("excluded_official_test_overlap_rows", 0),
        "unresolved": {
            "total": len(unresolved),
            "dev_source": len(dev_unresolved),
            "official_test": len(test_unresolved),
            "reasons": dict(Counter(item["reason"] for item in unresolved)),
        },
        "matcher": {
            "dev_source": dev_stats,
            "official_test": test_stats,
        },
        "split": split_stats,
        # Keep the positive, unambiguous field name as the canonical check.
        # ``leakage_check`` in an earlier draft used ``True`` to mean "passed",
        # which is easy to misread as evidence that leakage exists.  Retain a
        # backwards-compatible alias while making the interpretation explicit.
        "leakage_check": {
            "no_overlap": {key: len(value) == 0 for key, value in overlap.items()},
            "overlap_counts": {key: len(value) for key, value in overlap.items()},
        },
        "unresolved_file": "unresolved.jsonl",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
