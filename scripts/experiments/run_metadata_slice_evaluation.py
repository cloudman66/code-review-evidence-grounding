from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/swe_care_grounding/test.jsonl",
    )
    parser.add_argument(
        "--lexical-predictions",
        default="results/diagnostics/swe_care_statistical_validation/predictions/lexical.jsonl",
    )
    parser.add_argument(
        "--single-predictions",
        default="results/diagnostics/swe_care_statistical_validation/predictions/canonical_single.jsonl",
    )
    parser.add_argument(
        "--overall-predictions",
        default="results/diagnostics/swe_care_statistical_validation/predictions/canonical_overall.jsonl",
    )
    parser.add_argument(
        "--intent-predictions",
        default="results/diagnostics/swe_care_statistical_validation/predictions/intent_hit3.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations/swe_care_metadata_slices",
    )
    parser.add_argument("--min-samples", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def language_from_path(path: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]+)$", path or "")
    ext = match.group(1).lower() if match else ""
    return EXT_TO_LANG.get(ext, ext or "unknown")


def prediction_metrics(rows: list[dict], gold_by_sample_id: dict[str, list[str]]) -> dict[str, float]:
    hit1 = 0
    hit3 = 0
    mrr = 0.0
    total = len(rows) or 1
    for row in rows:
        gold = set(row.get("gold_context_ids", gold_by_sample_id[row["sample_id"]]))
        ranked = row["predicted_context_ids_topk"]
        hit1 += int(bool(ranked) and ranked[0] in gold)
        hit3 += int(any(context_id in gold for context_id in ranked[:3]))
        rr = 0.0
        for index, context_id in enumerate(ranked, start=1):
            if context_id in gold:
                rr = 1.0 / index
                break
        mrr += rr
    return {"hit@1": hit1 / total, "hit@3": hit3 / total, "mrr": mrr / total}


def collect_slice_groups(dataset_rows: list[dict], *, min_samples: int) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {
        "difficulty": defaultdict(list),
        "problem_domain": defaultdict(list),
        "language": defaultdict(list),
    }
    for row in dataset_rows:
        sample_id = row["sample_id"]
        grouped["difficulty"][row["metadata"]["difficulty"]].append(sample_id)
        grouped["problem_domain"][row["metadata"]["problem_domain"]].append(sample_id)
        grouped["language"][language_from_path(row["metadata"]["path"])].append(sample_id)
    filtered: dict[str, dict[str, list[str]]] = {}
    for axis, axis_groups in grouped.items():
        filtered[axis] = {
            label: sample_ids
            for label, sample_ids in axis_groups.items()
            if len(sample_ids) >= min_samples
        }
    return filtered


def build_slice_tables(
    dataset_rows: list[dict],
    predictions_by_name: dict[str, dict[str, dict]],
    *,
    min_samples: int,
) -> tuple[list[dict], dict[str, dict]]:
    gold_by_sample_id = {row["sample_id"]: row["gold_context_ids"] for row in dataset_rows}
    slice_groups = collect_slice_groups(dataset_rows, min_samples=min_samples)
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for axis, groups in slice_groups.items():
        summary[axis] = {}
        for label, sample_ids in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            label_summary = {"samples": len(sample_ids), "systems": {}}
            for system_name, prediction_map in predictions_by_name.items():
                system_rows = [prediction_map[sample_id] for sample_id in sample_ids]
                metrics = prediction_metrics(system_rows, gold_by_sample_id)
                label_summary["systems"][system_name] = metrics
                rows.append(
                    {
                        "axis": axis,
                        "label": label,
                        "samples": len(sample_ids),
                        "system": system_name,
                        "hit@1": metrics["hit@1"],
                        "hit@3": metrics["hit@3"],
                        "mrr": metrics["mrr"],
                    }
                )
            summary[axis][label] = label_summary
    return rows, summary


def write_markdown(path: Path, rows: list[dict], summary: dict[str, dict]) -> None:
    lines = ["# Metadata Slice Evaluation", ""]
    for axis, groups in summary.items():
        lines.append(f"## {axis}")
        lines.append("")
        lines.append("| label | samples | lexical H@1 | single H@1 | overall H@1 | intent H@3 | lexical MRR | overall MRR |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label, payload in sorted(groups.items(), key=lambda item: (-item[1]["samples"], item[0])):
            lexical = payload["systems"]["lexical"]
            single = payload["systems"]["canonical_single"]
            overall = payload["systems"]["canonical_overall"]
            intent = payload["systems"]["intent_hit3"]
            lines.append(
                f"| {label} | {payload['samples']} | {lexical['hit@1']:.4f} | {single['hit@1']:.4f} | "
                f"{overall['hit@1']:.4f} | {intent['hit@3']:.4f} | {lexical['mrr']:.4f} | {overall['mrr']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = load_jsonl(Path(args.dataset))
    predictions_by_name = {
        "lexical": {row["sample_id"]: row for row in load_jsonl(Path(args.lexical_predictions))},
        "canonical_single": {row["sample_id"]: row for row in load_jsonl(Path(args.single_predictions))},
        "canonical_overall": {row["sample_id"]: row for row in load_jsonl(Path(args.overall_predictions))},
        "intent_hit3": {row["sample_id"]: row for row in load_jsonl(Path(args.intent_predictions))},
    }

    rows, summary = build_slice_tables(
        dataset_rows,
        predictions_by_name,
        min_samples=args.min_samples,
    )

    (output_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "summary.md", rows, summary)

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "axes": list(summary.keys()),
                "counts": {axis: len(groups) for axis, groups in summary.items()},
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
