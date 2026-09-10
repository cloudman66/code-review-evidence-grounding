from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.external_scores import augment_feature_rows_with_external_scores
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    fit_pointwise_logistic_ranker,
    load_feature_row_cache,
)


ROOT = PROJECT_ROOT
CONFIG_PATH = (
    ROOT
    / "src/configs/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached.yaml"
)
CACHE_ROOT = ROOT / "data/cache"
OUTPUT_DIR = ROOT / "results/holdout_validation/swe_care_low_resource_generalization"
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SEEDS = [7, 13, 23]


def remap_cache_path(path_str: str, *, root_dir: Path) -> Path:
    return root_dir / Path(path_str).name


def sample_repo(sample: dict) -> str:
    return str(sample.get("metadata", {}).get("repo", "unknown"))


def stratified_repo_sample(samples: list[dict], *, fraction: float, seed: int) -> list[dict]:
    if fraction >= 0.999:
        return list(samples)

    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample_repo(sample)].append(sample)

    selected: list[dict] = []
    for repo, repo_samples in grouped.items():
        items = list(repo_samples)
        rng.shuffle(items)
        keep = max(1, int(round(len(items) * fraction)))
        keep = min(keep, len(items))
        selected.extend(items[:keep])
    rng.shuffle(selected)
    return selected


def filter_feature_cache(
    feature_rows_by_sample_id: dict[str, list[dict]],
    sample_ids: set[str],
) -> dict[str, list[dict]]:
    return {
        sample_id: copy.deepcopy(rows)
        for sample_id, rows in feature_rows_by_sample_id.items()
        if sample_id in sample_ids
    }


def load_reference_metrics(path: Path) -> dict[str, float]:
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
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(variance)


def render_markdown(path: Path, *, rows: list[dict], summary: list[dict], lexical: dict[str, float], full_single: dict[str, float]) -> None:
    lines = [
        "# Low-resource grounding pilot",
        "",
        "This pilot downsamples the SWE-CARE training split while keeping the dev/test splits fixed.",
        "Sampling is stratified by repository so that small fractions do not collapse to a few dominant repos.",
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
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    feature_cache_root = CACHE_ROOT / "feature_rows"
    score_cache_root = CACHE_ROOT / "scores"

    train_samples = load_jsonl(ROOT / config["dataset"]["train_path"])
    dev_samples = load_jsonl(ROOT / config["dataset"]["dev_path"])
    test_samples = load_jsonl(ROOT / config["dataset"]["test_path"])
    top_k = int(config["retrieval"]["top_k"])

    train_cache_all = load_feature_row_cache(remap_cache_path(config["feature_row_cache"]["train_path"], root_dir=feature_cache_root))
    dev_cache_all = load_feature_row_cache(remap_cache_path(config["feature_row_cache"]["dev_path"], root_dir=feature_cache_root))
    test_cache_all = load_feature_row_cache(remap_cache_path(config["feature_row_cache"]["test_path"], root_dir=feature_cache_root))

    external_feature_specs = [
        {
            "name": str(item["name"]),
            "train_path": remap_cache_path(str(item["train_score_cache_path"]), root_dir=score_cache_root),
            "dev_path": remap_cache_path(str(item["dev_score_cache_path"]), root_dir=score_cache_root),
            "test_path": remap_cache_path(str(item["test_score_cache_path"]), root_dir=score_cache_root),
        }
        for item in config.get("external_features", config.get("benchmarks_features", []))
    ]

    lexical_metrics = load_reference_metrics(ROOT / "results/baselines/swe_care_grounding_baseline/metrics.json")
    full_single_metrics = load_reference_metrics(
        ROOT
        / "results/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached/metrics.json"
    )

    dev_ids = {sample["sample_id"] for sample in dev_samples}
    test_ids = {sample["sample_id"] for sample in test_samples}
    dev_cache = filter_feature_cache(dev_cache_all, dev_ids)
    test_cache = filter_feature_cache(test_cache_all, test_ids)
    for item in external_feature_specs:
        dev_cache = augment_feature_rows_with_external_scores(
            dev_cache,
            score_cache_path=item["dev_path"],
            feature_name=item["name"],
        )
        test_cache = augment_feature_rows_with_external_scores(
            test_cache,
            score_cache_path=item["test_path"],
            feature_name=item["name"],
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []

    for fraction in FRACTIONS:
        fraction_seeds = [SEEDS[0]] if fraction >= 0.999 else SEEDS
        for seed in fraction_seeds:
            sampled_train = stratified_repo_sample(train_samples, fraction=fraction, seed=seed)
            train_ids = {sample["sample_id"] for sample in sampled_train}
            train_cache = filter_feature_cache(train_cache_all, train_ids)
            for item in external_feature_specs:
                train_cache = augment_feature_rows_with_external_scores(
                    train_cache,
                    score_cache_path=item["train_path"],
                    feature_name=item["name"],
                )

            training_results = fit_pointwise_logistic_ranker(
                sampled_train,
                dev_samples,
                top_k=top_k,
                training_config=config["training"],
                train_feature_rows_by_sample_id=train_cache,
                dev_feature_rows_by_sample_id=dev_cache,
            )
            model = training_results["model"]
            test_dataset = build_labeled_ranking_dataset(
                test_samples,
                base_feature_names=model["base_feature_names"],
                feature_transform=model["feature_transform"],
                feature_rows_by_sample_id=test_cache,
            )
            test_metrics = evaluate_ranking_dataset(test_dataset, model, top_k=top_k)
            run_rows.append(
                {
                    "fraction": fraction,
                    "seed": seed,
                    "train_samples": len(sampled_train),
                    "selected_C": float(model["search"]["selected_C"]),
                    "hit@1": float(test_metrics["hit@1"]),
                    "hit@3": float(test_metrics[f"hit@{top_k}"]),
                    "mrr": float(test_metrics["mrr"]),
                    "delta_vs_lexical_hit@1": float(test_metrics["hit@1"] - lexical_metrics["hit@1"]),
                    "gap_vs_full_single_hit@1": float(test_metrics["hit@1"] - full_single_metrics["hit@1"]),
                }
            )
            print(
                json.dumps(
                    {
                        "fraction": fraction,
                        "seed": seed,
                        "train_samples": len(sampled_train),
                        "selected_C": model["search"]["selected_C"],
                        "hit@1": test_metrics["hit@1"],
                        "hit@3": test_metrics[f"hit@{top_k}"],
                        "mrr": test_metrics["mrr"],
                    },
                    ensure_ascii=False,
                )
            )

    summary_rows: list[dict] = []
    by_fraction: dict[float, list[dict]] = defaultdict(list)
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
        "fractions": FRACTIONS,
        "seeds": SEEDS,
        "lexical": lexical_metrics,
        "full_single": full_single_metrics,
        "runs": run_rows,
        "summary": summary_rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_markdown(
        OUTPUT_DIR / "summary.md",
        rows=run_rows,
        summary=summary_rows,
        lexical=lexical_metrics,
        full_single=full_single_metrics,
    )
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": len(run_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
