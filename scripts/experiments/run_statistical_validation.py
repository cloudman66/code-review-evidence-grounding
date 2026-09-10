from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import rank_contexts
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/swe_care_grounding/test.jsonl",
    )
    parser.add_argument(
        "--lexical-config",
        default="src/configs/swe_care_grounding.yaml",
    )
    parser.add_argument(
        "--single-cache",
        default="data/cache/scores/swe_care_single_best_fuzzy_len2_feedback_test_scores.json.gz",
    )
    parser.add_argument(
        "--router-json",
        default="data/cache/models/swe_care_multimodel_router_word_char_file_mean_char_file_mean_exact/router.json",
    )
    parser.add_argument(
        "--router-word-cache",
        default="data/cache/scores/swe_care_context_slices_semantic_test_exact.json.gz",
    )
    parser.add_argument(
        "--router-char-cache",
        default="data/cache/scores/swe_care_context_slices_semantic_char_test_exact.json.gz",
    )
    parser.add_argument(
        "--router-file-mean-cache",
        default="data/cache/scores/swe_care_context_slices_semantic_file_mean_test_exact.json.gz",
    )
    parser.add_argument(
        "--router-char-file-mean-cache",
        default="data/cache/scores/swe_care_context_slices_semantic_char_file_mean_test_exact.json.gz",
    )
    parser.add_argument(
        "--overall-aux-cache",
        default="data/cache/scores/swe_care_single_best_fuzzy_len2_feedback_test_scores.json.gz",
    )
    parser.add_argument(
        "--intent-aux-cache",
        default="data/cache/scores/swe_care_single_best_intent_test_scores.json.gz",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--randomization-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="results/diagnostics/swe_care_statistical_validation",
    )
    return parser.parse_args()


def load_ranking_config(path: Path) -> dict[str, float]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in config.get("ranking", {}).items()}


def rank_context_ids_from_cache_group(group: dict) -> list[str]:
    scores = np.asarray(group["scores"], dtype=np.float32)
    ranked_indices = np.argsort(-scores, kind="stable")
    return [group["context_ids"][index] for index in ranked_indices]


def cache_to_group_map(path: Path) -> dict[str, dict]:
    payload = load_score_cache(path)
    dataset = payload["dataset"]
    group_map: dict[str, dict] = {}
    offset = 0
    for sample_id, context_ids, gold_context_ids, group_size in zip(
        dataset["sample_ids"],
        dataset["context_ids_by_group"],
        dataset["gold_context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + int(group_size)
        group_map[sample_id] = {
            "context_ids": list(context_ids),
            "gold_context_ids": list(gold_context_ids),
            "scores": payload["scores"][offset:next_offset],
        }
        offset = next_offset
    return group_map


def sample_outcomes_from_ranked_ids(sample_id: str, gold_context_ids: list[str], ranked_context_ids: list[str]) -> dict:
    gold = set(gold_context_ids)
    hit1 = 1.0 if ranked_context_ids and ranked_context_ids[0] in gold else 0.0
    hit3 = 1.0 if any(context_id in gold for context_id in ranked_context_ids[:3]) else 0.0
    reciprocal_rank = 0.0
    for index, context_id in enumerate(ranked_context_ids, start=1):
        if context_id in gold:
            reciprocal_rank = 1.0 / index
            break
    return {
        "sample_id": sample_id,
        "hit@1": hit1,
        "hit@3": hit3,
        "mrr": reciprocal_rank,
    }


def build_lexical_rows(samples: list[dict], ranking_config: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        ranked = rank_contexts(sample, ranking_config=ranking_config)
        ranked_context_ids = [item["context_id"] for item in ranked]
        row = sample_outcomes_from_ranked_ids(
            sample["sample_id"],
            sample["gold_context_ids"],
            ranked_context_ids,
        )
        row["predicted_context_ids_topk"] = ranked_context_ids[:3]
        rows.append(row)
    return rows


def build_rows_from_cache_map(samples: list[dict], cache_map: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        group = cache_map[sample["sample_id"]]
        ranked_context_ids = rank_context_ids_from_cache_group(group)
        row = sample_outcomes_from_ranked_ids(
            sample["sample_id"],
            sample["gold_context_ids"],
            ranked_context_ids,
        )
        row["predicted_context_ids_topk"] = ranked_context_ids[:3]
        rows.append(row)
    return rows


def build_router_rows(
    samples: list[dict],
    router_payload: dict,
    base_group_maps: dict[str, dict[str, dict]],
) -> list[dict]:
    prediction_map = {row["sample_id"]: row for row in router_payload["test"]["predictions"]}
    rows: list[dict] = []
    for sample in samples:
        prediction = prediction_map[sample["sample_id"]]
        selected_model = prediction["selected_model"]
        base_group = base_group_maps[selected_model][sample["sample_id"]]
        ranked_context_ids = rank_context_ids_from_cache_group(base_group)
        row = sample_outcomes_from_ranked_ids(
            sample["sample_id"],
            sample["gold_context_ids"],
            ranked_context_ids,
        )
        row["predicted_context_ids_topk"] = ranked_context_ids[:3]
        row["selected_model"] = selected_model
        rows.append(row)
    return rows


def merge_base_and_aux_rankings(base_ranked: list[str], aux_ranked: list[str], *, keep_base_top1: bool) -> list[str]:
    merged: list[str] = []
    if keep_base_top1 and base_ranked:
        merged.append(base_ranked[0])
    for source in (aux_ranked, base_ranked):
        for context_id in source:
            if context_id not in merged:
                merged.append(context_id)
    return merged


def build_global_hybrid_rows(
    samples: list[dict],
    router_payload: dict,
    base_group_maps: dict[str, dict[str, dict]],
    aux_group_map: dict[str, dict],
) -> list[dict]:
    prediction_map = {row["sample_id"]: row for row in router_payload["test"]["predictions"]}
    rows: list[dict] = []
    for sample in samples:
        prediction = prediction_map[sample["sample_id"]]
        selected_model = prediction["selected_model"]
        base_group = base_group_maps[selected_model][sample["sample_id"]]
        aux_group = aux_group_map[sample["sample_id"]]
        base_ranked = rank_context_ids_from_cache_group(base_group)
        aux_ranked = rank_context_ids_from_cache_group(aux_group)
        merged = merge_base_and_aux_rankings(base_ranked, aux_ranked, keep_base_top1=True)
        row = sample_outcomes_from_ranked_ids(
            sample["sample_id"],
            sample["gold_context_ids"],
            merged,
        )
        row["predicted_context_ids_topk"] = merged[:3]
        row["selected_model"] = selected_model
        rows.append(row)
    return rows


def outcome_matrix(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "hit@1": np.asarray([float(row["hit@1"]) for row in rows], dtype=np.float32),
        "hit@3": np.asarray([float(row["hit@3"]) for row in rows], dtype=np.float32),
        "mrr": np.asarray([float(row["mrr"]) for row in rows], dtype=np.float32),
    }


def bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, samples: int) -> dict[str, float]:
    n = int(values.shape[0])
    indices = rng.integers(0, n, size=(samples, n))
    boot = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def paired_delta_ci(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    n = int(deltas.shape[0])
    indices = rng.integers(0, n, size=(samples, n))
    boot = deltas[indices].mean(axis=1)
    observed = float(deltas.mean())
    return {
        "delta": observed,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def paired_randomization_pvalue(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> float:
    observed = abs(float(deltas.mean()))
    if observed == 0.0:
        return 1.0
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(samples, deltas.shape[0]))
    shuffled = (signs * deltas[None, :]).mean(axis=1)
    return float((np.sum(np.abs(shuffled) >= observed) + 1) / (samples + 1))


def build_system_summary(
    name: str,
    label: str,
    rows: list[dict],
    *,
    rng_seed: int,
    bootstrap_samples: int,
) -> dict:
    outcomes = outcome_matrix(rows)
    summary = {
        "name": name,
        "label": label,
        "samples": len(rows),
        "metrics": {},
    }
    for offset, metric_name in enumerate(("hit@1", "hit@3", "mrr")):
        metric_rng = np.random.default_rng(rng_seed + offset)
        summary["metrics"][metric_name] = bootstrap_ci(
            outcomes[metric_name],
            rng=metric_rng,
            samples=bootstrap_samples,
        )
    return summary


def build_comparison_summary(
    left_name: str,
    right_name: str,
    left_rows: list[dict],
    right_rows: list[dict],
    *,
    rng_seed: int,
    bootstrap_samples: int,
    randomization_samples: int,
) -> dict:
    left_outcomes = outcome_matrix(left_rows)
    right_outcomes = outcome_matrix(right_rows)
    summary = {
        "left": left_name,
        "right": right_name,
        "metrics": {},
    }
    for offset, metric_name in enumerate(("hit@1", "hit@3", "mrr")):
        deltas = left_outcomes[metric_name] - right_outcomes[metric_name]
        delta_rng = np.random.default_rng(rng_seed + (offset * 7))
        p_rng = np.random.default_rng(rng_seed + (offset * 7) + 1000)
        summary["metrics"][metric_name] = {
            **paired_delta_ci(
                deltas,
                rng=delta_rng,
                samples=bootstrap_samples,
            ),
            "p_value": paired_randomization_pvalue(
                deltas,
                rng=p_rng,
                samples=randomization_samples,
            ),
        }
    return summary


def format_metric_row(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.4f} [{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"


def format_delta_row(metric: dict[str, float]) -> str:
    return (
        f"{metric['delta']:+.4f} "
        f"[{metric['ci_low']:+.4f}, {metric['ci_high']:+.4f}], "
        f"p={metric['p_value']:.4f}"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown_report(
    path: Path,
    *,
    system_summaries: list[dict],
    comparison_summaries: list[dict],
) -> None:
    lines = [
        "# Statistical Validation",
        "",
        "## Systems",
        "",
        "| system | Hit@1 | Hit@3 | MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in system_summaries:
        lines.append(
            f"| {row['label']} | "
            f"{format_metric_row(row['metrics']['hit@1'])} | "
            f"{format_metric_row(row['metrics']['hit@3'])} | "
            f"{format_metric_row(row['metrics']['mrr'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| comparison | Hit@1 delta | Hit@3 delta | MRR delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_summaries:
        label = f"{row['left']} vs {row['right']}"
        lines.append(
            f"| {label} | "
            f"{format_delta_row(row['metrics']['hit@1'])} | "
            f"{format_delta_row(row['metrics']['hit@3'])} | "
            f"{format_delta_row(row['metrics']['mrr'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.dataset)
    ranking_config = load_ranking_config(Path(args.lexical_config))

    router_payload = json.loads(Path(args.router_json).read_text(encoding="utf-8"))
    base_group_maps = {
        "word": cache_to_group_map(Path(args.router_word_cache)),
        "char": cache_to_group_map(Path(args.router_char_cache)),
        "file_mean": cache_to_group_map(Path(args.router_file_mean_cache)),
        "char_file_mean": cache_to_group_map(Path(args.router_char_file_mean_cache)),
    }
    single_cache_map = cache_to_group_map(Path(args.single_cache))
    overall_aux_group_map = cache_to_group_map(Path(args.overall_aux_cache))
    intent_aux_group_map = cache_to_group_map(Path(args.intent_aux_cache))

    lexical_rows = build_lexical_rows(samples, ranking_config)
    single_rows = build_rows_from_cache_map(samples, single_cache_map)
    router_rows = build_router_rows(samples, router_payload, base_group_maps)
    overall_rows = build_global_hybrid_rows(samples, router_payload, base_group_maps, overall_aux_group_map)
    intent_rows = build_global_hybrid_rows(samples, router_payload, base_group_maps, intent_aux_group_map)

    system_specs = [
        ("lexical", "Lexical baseline", lexical_rows),
        ("canonical_single", "Single learned ranker", single_rows),
        ("base_router", "Router-only", router_rows),
        ("canonical_overall", "Overall system", overall_rows),
        ("intent_hit3", "Intent top-3 specialist", intent_rows),
    ]
    system_summaries = [
        build_system_summary(
            name=name,
            label=label,
            rows=rows,
            rng_seed=args.seed + (index * 100),
            bootstrap_samples=args.bootstrap_samples,
        )
        for index, (name, label, rows) in enumerate(system_specs)
    ]

    comparison_specs = [
        ("canonical_single", "lexical", single_rows, lexical_rows),
        ("base_router", "lexical", router_rows, lexical_rows),
        ("canonical_overall", "lexical", overall_rows, lexical_rows),
        ("canonical_overall", "canonical_single", overall_rows, single_rows),
        ("intent_hit3", "canonical_overall", intent_rows, overall_rows),
    ]
    comparison_summaries = [
        build_comparison_summary(
            left_name=left,
            right_name=right,
            left_rows=left_rows,
            right_rows=right_rows,
            rng_seed=args.seed + 500 + (index * 100),
            bootstrap_samples=args.bootstrap_samples,
            randomization_samples=args.randomization_samples,
        )
        for index, (left, right, left_rows, right_rows) in enumerate(comparison_specs)
    ]

    payload = {
        "dataset": str(args.dataset),
        "bootstrap_samples": args.bootstrap_samples,
        "randomization_samples": args.randomization_samples,
        "systems": system_summaries,
        "comparisons": comparison_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "summary.md",
        system_summaries=system_summaries,
        comparison_summaries=comparison_summaries,
    )

    write_jsonl(output_dir / "predictions" / "lexical.jsonl", lexical_rows)
    write_jsonl(output_dir / "predictions" / "canonical_single.jsonl", single_rows)
    write_jsonl(output_dir / "predictions" / "base_router.jsonl", router_rows)
    write_jsonl(output_dir / "predictions" / "canonical_overall.jsonl", overall_rows)
    write_jsonl(output_dir / "predictions" / "intent_hit3.jsonl", intent_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
