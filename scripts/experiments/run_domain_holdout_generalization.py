from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
import sys
import warnings

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
    load_linear_ranker_model,
)


ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / "src/configs/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached.yaml"
CURRENT_MODEL_PATH = (
    ROOT
    / "results/ablations/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached/model.json"
)
OUTPUT_DIR = ROOT / "results/holdout_validation/swe_care_domain_holdout_generalization"
TOP_K = 3
MIN_TRAIN = 100
MIN_TEST = 30


def sample_domain(sample: dict) -> str:
    return str(sample.get("metadata", {}).get("problem_domain", "unknown"))


def compute_retrieval_metrics(ranked_predictions: list[dict], gold_map: dict[str, list[str]], *, top_k: int) -> dict[str, float]:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    for row in ranked_predictions:
        gold = set(gold_map[row["sample_id"]])
        ranked_context_ids = row["ranked_context_ids"]
        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))
        for index, context_id in enumerate(ranked_context_ids):
            if context_id in gold:
                mrr += 1.0 / float(index + 1)
                break
    total = len(ranked_predictions) or 1
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
    }


def lexical_predictions(samples: list[dict]) -> list[dict]:
    from code_review_understanding.eval.baseline import build_context_feature_rows
    from code_review_understanding.eval.baseline import rank_feature_rows

    rows: list[dict] = []
    for sample in samples:
        ranked = rank_feature_rows(build_context_feature_rows(sample))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "gold_context_ids": sample["gold_context_ids"],
                "ranked_context_ids": [item["context_id"] for item in ranked],
            }
        )
    return rows


def filter_samples(samples: list[dict], domain: str, *, keep_domain: bool) -> list[dict]:
    return [sample for sample in samples if (sample_domain(sample) == domain) is keep_domain]


def filter_feature_cache(feature_rows_by_sample_id: dict[str, list[dict]], sample_ids: set[str]) -> dict[str, list[dict]]:
    return {
        sample_id: copy.deepcopy(rows)
        for sample_id, rows in feature_rows_by_sample_id.items()
        if sample_id in sample_ids
    }


def augment_cache_subset(feature_cache: dict[str, list[dict]], feature_specs: list[dict]) -> dict[str, list[dict]]:
    augmented = feature_cache
    for item in feature_specs:
        augmented = augment_feature_rows_with_external_scores(
            augmented,
            score_cache_path=str(item["path"]),
            feature_name=item["name"],
        )
    return augmented


def choose_domains(train_samples: list[dict], test_samples: list[dict]) -> list[str]:
    train_counts = Counter(sample_domain(sample) for sample in train_samples)
    test_counts = Counter(sample_domain(sample) for sample in test_samples)
    return [
        domain
        for domain, count in test_counts.most_common()
        if count >= MIN_TEST and train_counts.get(domain, 0) >= MIN_TRAIN
    ]


def weighted_average(rows: list[dict], key: str) -> float:
    total_weight = sum(int(row["test_samples"]) for row in rows) or 1
    return sum(float(row[key]) * int(row["test_samples"]) for row in rows) / total_weight


def macro_average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / (len(rows) or 1)


def build_aggregate(domains: list[str], summary_rows: list[dict]) -> dict:
    return {
        "problem_domains": domains,
        "completed_problem_domains": [row["problem_domain"] for row in summary_rows],
        "domain_rows": summary_rows,
        "macro": {
            "lexical_hit@1": macro_average(summary_rows, "lexical_hit@1"),
            "lexical_hit@3": macro_average(summary_rows, "lexical_hit@3"),
            "lexical_mrr": macro_average(summary_rows, "lexical_mrr"),
            "holdout_hit@1": macro_average(summary_rows, "holdout_hit@1"),
            "holdout_hit@3": macro_average(summary_rows, "holdout_hit@3"),
            "holdout_mrr": macro_average(summary_rows, "holdout_mrr"),
            "full_single_hit@1": macro_average(summary_rows, "full_single_hit@1"),
            "full_single_hit@3": macro_average(summary_rows, "full_single_hit@3"),
            "full_single_mrr": macro_average(summary_rows, "full_single_mrr"),
        },
        "weighted": {
            "lexical_hit@1": weighted_average(summary_rows, "lexical_hit@1"),
            "lexical_hit@3": weighted_average(summary_rows, "lexical_hit@3"),
            "lexical_mrr": weighted_average(summary_rows, "lexical_mrr"),
            "holdout_hit@1": weighted_average(summary_rows, "holdout_hit@1"),
            "holdout_hit@3": weighted_average(summary_rows, "holdout_hit@3"),
            "holdout_mrr": weighted_average(summary_rows, "holdout_mrr"),
            "full_single_hit@1": weighted_average(summary_rows, "full_single_hit@1"),
            "full_single_hit@3": weighted_average(summary_rows, "full_single_hit@3"),
            "full_single_mrr": weighted_average(summary_rows, "full_single_mrr"),
        },
    }


def write_aggregate(output_dir: Path, domains: list[str], summary_rows: list[dict]) -> dict:
    aggregate = build_aggregate(domains, summary_rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Problem-Domain Holdout Generalization",
        "",
        "Cross-domain holdout experiment for the single learned ranker.",
        "",
        f"Domains: {', '.join(domains)}",
        f"Completed: {', '.join(aggregate['completed_problem_domains']) or '(none)'}",
        "",
        "| problem domain | test | lexical H@1 | holdout H@1 | full-single H@1 | lexical H@3 | holdout H@3 | full-single H@3 | lexical MRR | holdout MRR | full-single MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {problem_domain} | {test_samples} | {lexical_hit@1:.4f} | {holdout_hit@1:.4f} | {full_single_hit@1:.4f} | {lexical_hit@3:.4f} | {holdout_hit@3:.4f} | {full_single_hit@3:.4f} | {lexical_mrr:.4f} | {holdout_mrr:.4f} | {full_single_mrr:.4f} |".format(
                **row
            )
        )
    if summary_rows:
        lines.extend(
            [
                "",
                "## Weighted Average",
                "",
                "- lexical: `Hit@1={lexical_hit@1:.4f} / Hit@3={lexical_hit@3:.4f} / MRR={lexical_mrr:.4f}`".format(
                    **aggregate["weighted"]
                ),
                "- holdout: `Hit@1={holdout_hit@1:.4f} / Hit@3={holdout_hit@3:.4f} / MRR={holdout_mrr:.4f}`".format(
                    **aggregate["weighted"]
                ),
                "- full_single: `Hit@1={full_single_hit@1:.4f} / Hit@3={full_single_hit@3:.4f} / MRR={full_single_mrr:.4f}`".format(
                    **aggregate["weighted"]
                ),
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
    )
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_samples = load_jsonl(config["dataset"]["train_path"])
    dev_samples = load_jsonl(config["dataset"]["dev_path"])
    test_samples = load_jsonl(config["dataset"]["test_path"])
    domains = choose_domains(train_samples, test_samples)
    print(f"Selected domains: {domains}")

    feature_cache_cfg = config["feature_row_cache"]
    train_cache_all = load_feature_row_cache(feature_cache_cfg["train_path"])
    dev_cache_all = load_feature_row_cache(feature_cache_cfg["dev_path"])
    test_cache_all = load_feature_row_cache(feature_cache_cfg["test_path"])

    external_feature_specs = [
        {
            "name": str(item["name"]),
            "train": Path(item["train_score_cache_path"]),
            "dev": Path(item["dev_score_cache_path"]),
            "test": Path(item["test_score_cache_path"]),
        }
        for item in config.get("external_features", config.get("benchmarks_features", []))
    ]

    current_model = load_linear_ranker_model(CURRENT_MODEL_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    gold_map_full = {sample["sample_id"]: sample["gold_context_ids"] for sample in test_samples}

    for domain in domains:
        print(f"[domain-holdout] running {domain}")
        safe_name = (
            domain.replace("/", "_")
            .replace(" ", "_")
            .replace("&", "and")
            .replace(",", "")
        )
        domain_dir = OUTPUT_DIR / safe_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        domain_train = filter_samples(train_samples, domain, keep_domain=False)
        domain_dev = filter_samples(dev_samples, domain, keep_domain=False)
        domain_test = filter_samples(test_samples, domain, keep_domain=True)

        train_ids = {sample["sample_id"] for sample in domain_train}
        dev_ids = {sample["sample_id"] for sample in domain_dev}
        test_ids = {sample["sample_id"] for sample in domain_test}

        domain_train_cache = filter_feature_cache(train_cache_all, train_ids)
        domain_dev_cache = filter_feature_cache(dev_cache_all, dev_ids)
        domain_test_cache = filter_feature_cache(test_cache_all, test_ids)

        augment_cache_subset(domain_train_cache, [{"name": item["name"], "path": item["train"]} for item in external_feature_specs])
        augment_cache_subset(domain_dev_cache, [{"name": item["name"], "path": item["dev"]} for item in external_feature_specs])
        augment_cache_subset(domain_test_cache, [{"name": item["name"], "path": item["test"]} for item in external_feature_specs])

        training_results = fit_pointwise_logistic_ranker(
            domain_train,
            domain_dev,
            top_k=TOP_K,
            training_config=config["training"],
            train_feature_rows_by_sample_id=domain_train_cache,
            dev_feature_rows_by_sample_id=domain_dev_cache,
        )
        holdout_model = training_results["model"]
        holdout_test_dataset = build_labeled_ranking_dataset(
            domain_test,
            base_feature_names=holdout_model["base_feature_names"],
            feature_transform=holdout_model["feature_transform"],
            feature_rows_by_sample_id=domain_test_cache,
        )
        holdout_metrics = evaluate_ranking_dataset(holdout_test_dataset, holdout_model, top_k=TOP_K)

        in_domain_dataset = build_labeled_ranking_dataset(
            domain_test,
            base_feature_names=current_model["base_feature_names"],
            feature_transform=current_model["feature_transform"],
            feature_rows_by_sample_id=domain_test_cache,
        )
        in_domain_metrics = evaluate_ranking_dataset(in_domain_dataset, current_model, top_k=TOP_K)

        lexical_ranked_predictions = lexical_predictions(domain_test)
        lexical_metrics = compute_retrieval_metrics(
            lexical_ranked_predictions,
            {sample["sample_id"]: sample["gold_context_ids"] for sample in domain_test},
            top_k=TOP_K,
        )

        row = {
            "problem_domain": domain,
            "train_samples": len(domain_train),
            "dev_samples": len(domain_dev),
            "test_samples": len(domain_test),
            "selected_C": holdout_model["search"]["selected_C"],
            "lexical_hit@1": lexical_metrics["hit@1"],
            "lexical_hit@3": lexical_metrics["hit@3"],
            "lexical_mrr": lexical_metrics["mrr"],
            "holdout_hit@1": holdout_metrics["hit@1"],
            "holdout_hit@3": holdout_metrics["hit@3"],
            "holdout_mrr": holdout_metrics["mrr"],
            "full_single_hit@1": in_domain_metrics["hit@1"],
            "full_single_hit@3": in_domain_metrics["hit@3"],
            "full_single_mrr": in_domain_metrics["mrr"],
            "delta_vs_lexical_hit@1": holdout_metrics["hit@1"] - lexical_metrics["hit@1"],
            "delta_vs_lexical_hit@3": holdout_metrics["hit@3"] - lexical_metrics["hit@3"],
            "delta_vs_lexical_mrr": holdout_metrics["mrr"] - lexical_metrics["mrr"],
            "generalization_gap_hit@1": in_domain_metrics["hit@1"] - holdout_metrics["hit@1"],
            "generalization_gap_hit@3": in_domain_metrics["hit@3"] - holdout_metrics["hit@3"],
            "generalization_gap_mrr": in_domain_metrics["mrr"] - holdout_metrics["mrr"],
        }
        summary_rows.append(row)

        with (domain_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)
        with (domain_dir / "holdout_predictions_test.jsonl").open("w", encoding="utf-8") as handle:
            for item in holdout_metrics["ranked_predictions"]:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": item["sample_id"],
                            "gold_context_ids": gold_map_full[item["sample_id"]],
                            "predicted_context_ids_topk": item["ranked_context_ids"][:TOP_K],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        aggregate = write_aggregate(OUTPUT_DIR, domains, summary_rows)
        print(
            f"[domain-holdout] {domain} lexical={lexical_metrics['hit@1']:.4f}/{lexical_metrics['hit@3']:.4f}/{lexical_metrics['mrr']:.4f} "
            f"holdout={holdout_metrics['hit@1']:.4f}/{holdout_metrics['hit@3']:.4f}/{holdout_metrics['mrr']:.4f}"
        )
    aggregate = write_aggregate(OUTPUT_DIR, domains, summary_rows)
    print(json.dumps(aggregate["weighted"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
