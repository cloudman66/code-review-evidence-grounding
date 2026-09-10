from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-metrics", required=True, help="Path to baseline metrics.json.")
    parser.add_argument("--new-metrics", required=True, help="Path to new-run metrics.json.")
    parser.add_argument("--base-analysis", required=True, help="Path to baseline analysis.json.")
    parser.add_argument("--new-analysis", required=True, help="Path to new-run analysis.json.")
    parser.add_argument("--output-json", required=True, help="Path to output comparison JSON.")
    parser.add_argument("--output-md", required=True, help="Path to output comparison markdown.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of slice improvements/regressions to keep.")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_test_metrics(metrics: dict) -> dict[str, float]:
    if "retrieval_test" in metrics:
        retrieval = metrics["retrieval_test"]
    elif "test" in metrics:
        retrieval = metrics["test"]
    elif "best" in metrics and "test" in metrics["best"]:
        retrieval = metrics["best"]["test"]
    else:
        raise KeyError("Unable to locate test metrics in input JSON")
    return {
        "hit@1": float(retrieval["hit@1"]),
        "hit@3": float(retrieval["hit@3"]),
        "mrr": float(retrieval["mrr"]),
    }


def flatten_slices(analysis: dict) -> dict[str, dict]:
    flattened: dict[str, dict] = {}
    for group_name, group in analysis["slices"].items():
        for key, metrics in group.items():
            flattened[f"{group_name}::{key}"] = metrics
    return flattened


def format_signed(value: float) -> str:
    return f"{value:+.4f}"


def build_comparison(base_metrics: dict, new_metrics: dict, base_analysis: dict, new_analysis: dict) -> dict:
    base_test = extract_test_metrics(base_metrics)
    new_test = extract_test_metrics(new_metrics)

    overall = {
        metric: {
            "base": base_test[metric],
            "new": new_test[metric],
            "delta": new_test[metric] - base_test[metric],
        }
        for metric in ("hit@1", "hit@3", "mrr")
    }

    base_slices = flatten_slices(base_analysis)
    new_slices = flatten_slices(new_analysis)

    slice_deltas = []
    for slice_name, new_slice in new_slices.items():
        base_slice = base_slices.get(slice_name)
        if base_slice is None:
            continue
        slice_deltas.append(
            {
                "slice": slice_name,
                "samples": int(new_slice["samples"]),
                "base_hit@1": float(base_slice["hit@1"]),
                "new_hit@1": float(new_slice["hit@1"]),
                "delta_hit@1": float(new_slice["hit@1"]) - float(base_slice["hit@1"]),
                "base_hit@3": float(base_slice["hit@3"]),
                "new_hit@3": float(new_slice["hit@3"]),
                "delta_hit@3": float(new_slice["hit@3"]) - float(base_slice["hit@3"]),
            }
        )

    return {
        "overall": overall,
        "slice_deltas": slice_deltas,
    }


def build_report(comparison: dict, *, top_k: int) -> str:
    lines: list[str] = []
    overall = comparison["overall"]
    lines.append("# Grounding Run Comparison")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | base | new | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for metric in ("hit@1", "hit@3", "mrr"):
        item = overall[metric]
        lines.append(
            f"| {metric} | {item['base']:.4f} | {item['new']:.4f} | {format_signed(item['delta'])} |"
        )
    lines.append("")

    slice_deltas = comparison["slice_deltas"]
    best_hit1 = sorted(slice_deltas, key=lambda row: (row["delta_hit@1"], row["delta_hit@3"]), reverse=True)[:top_k]
    worst_hit1 = sorted(slice_deltas, key=lambda row: (row["delta_hit@1"], row["delta_hit@3"]))[:top_k]
    best_hit3 = sorted(slice_deltas, key=lambda row: (row["delta_hit@3"], row["delta_hit@1"]), reverse=True)[:top_k]

    def add_slice_section(title: str, rows: list[dict], metric_key: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| slice | samples | new Hit@1 | delta Hit@1 | new Hit@3 | delta Hit@3 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['slice']} | {row['samples']} | {row['new_hit@1']:.4f} | {format_signed(row['delta_hit@1'])} | "
                f"{row['new_hit@3']:.4f} | {format_signed(row['delta_hit@3'])} |"
            )
        lines.append("")

    add_slice_section("Top Hit@1 Improvements", best_hit1, "delta_hit@1")
    add_slice_section("Top Hit@1 Regressions", worst_hit1, "delta_hit@1")
    add_slice_section("Top Hit@3 Improvements", best_hit3, "delta_hit@3")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    comparison = build_comparison(
        load_json(args.base_metrics),
        load_json(args.new_metrics),
        load_json(args.base_analysis),
        load_json(args.new_analysis),
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_report(comparison, top_k=args.top_k), encoding="utf-8")

    print(json.dumps(comparison["overall"], ensure_ascii=False, indent=2))
    print(f"Saved JSON comparison to {output_json}")
    print(f"Saved Markdown report to {output_md}")


if __name__ == "__main__":
    main()
