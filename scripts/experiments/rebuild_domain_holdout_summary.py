from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.data.io_utils import load_jsonl


OUTPUT_DIR = PROJECT_ROOT / "results/holdout_validation/swe_care_domain_holdout_generalization"
CONFIG_PATH = PROJECT_ROOT / "src/configs/swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached.yaml"
MIN_TRAIN = 100
MIN_TEST = 30


def sample_domain(sample: dict) -> str:
    return str(sample.get("metadata", {}).get("problem_domain", "unknown"))


def choose_domains(train_samples: list[dict], test_samples: list[dict]) -> list[str]:
    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    for sample in train_samples:
        domain = sample_domain(sample)
        train_counts[domain] = train_counts.get(domain, 0) + 1
    for sample in test_samples:
        domain = sample_domain(sample)
        test_counts[domain] = test_counts.get(domain, 0) + 1
    return [
        domain
        for domain, count in sorted(test_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= MIN_TEST and train_counts.get(domain, 0) >= MIN_TRAIN
    ]


def weighted_average(rows: list[dict], key: str) -> float:
    total_weight = sum(int(row["test_samples"]) for row in rows) or 1
    return sum(float(row[key]) * int(row["test_samples"]) for row in rows) / total_weight


def macro_average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / (len(rows) or 1)


def safe_name(domain: str) -> str:
    return domain.replace("/", "_").replace(" ", "_").replace("&", "and").replace(",", "")


def build_aggregate(all_domains: list[str], summary_rows: list[dict]) -> dict:
    return {
        "problem_domains": all_domains,
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


def write_markdown(path: Path, aggregate: dict) -> None:
    rows = aggregate["domain_rows"]
    lines = [
        "# Problem-Domain Holdout Generalization",
        "",
        "Cross-domain holdout experiment for the single learned ranker.",
        "",
        f"Domains: {', '.join(aggregate['problem_domains'])}",
        f"Completed: {', '.join(aggregate['completed_problem_domains']) or '(none)'}",
        "",
        "| problem domain | test | lexical H@1 | holdout H@1 | full-single H@1 | lexical H@3 | holdout H@3 | full-single H@3 | lexical MRR | holdout MRR | full-single MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {problem_domain} | {test_samples} | {lexical_hit@1:.4f} | {holdout_hit@1:.4f} | {full_single_hit@1:.4f} | {lexical_hit@3:.4f} | {holdout_hit@3:.4f} | {full_single_hit@3:.4f} | {lexical_mrr:.4f} | {holdout_mrr:.4f} | {full_single_mrr:.4f} |".format(
                **row
            )
        )
    if rows:
        weighted = aggregate["weighted"]
        lines.extend(
            [
                "",
                "## Weighted Average",
                "",
                "- lexical: `Hit@1={lexical_hit@1:.4f} / Hit@3={lexical_hit@3:.4f} / MRR={lexical_mrr:.4f}`".format(
                    **weighted
                ),
                "- holdout: `Hit@1={holdout_hit@1:.4f} / Hit@3={holdout_hit@3:.4f} / MRR={holdout_mrr:.4f}`".format(
                    **weighted
                ),
                "- full_single: `Hit@1={full_single_hit@1:.4f} / Hit@3={full_single_hit@3:.4f} / MRR={full_single_mrr:.4f}`".format(
                    **weighted
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    train_samples = load_jsonl(PROJECT_ROOT / cfg["dataset"]["train_path"])
    test_samples = load_jsonl(PROJECT_ROOT / cfg["dataset"]["test_path"])
    all_domains = choose_domains(train_samples, test_samples)

    rows: list[dict] = []
    for domain in all_domains:
        summary_path = OUTPUT_DIR / safe_name(domain) / "summary.json"
        if summary_path.exists():
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate = build_aggregate(all_domains, rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(OUTPUT_DIR / "summary.md", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
