from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import extract_quoted_spans
from code_review_understanding.models.fusion import load_score_cache
from code_review_understanding.data.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-json", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--dev-base-caches", required=True, help="Comma-separated model=path specs.")
    parser.add_argument("--test-base-caches", required=True, help="Comma-separated model=path specs.")
    parser.add_argument("--dev-aux-cache-a", required=True)
    parser.add_argument("--test-aux-cache-a", required=True)
    parser.add_argument("--dev-aux-cache-b", required=True)
    parser.add_argument("--test-aux-cache-b", required=True)
    parser.add_argument(
        "--policies",
        default="global,gt40,quoted_no,mention_yes,gt40_or_quoted_no,gt40_or_mention,gt40_or_quoted_no_or_mention",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def parse_cache_specs(spec: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid cache spec: {item}")
        name, path = item.split("=", 1)
        mapping[name.strip()] = Path(path.strip())
    return mapping


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
        next_offset = offset + group_size
        group_map[sample_id] = {
            "context_ids": list(context_ids),
            "gold_context_ids": list(gold_context_ids),
            "scores": payload["scores"][offset:next_offset],
        }
        offset = next_offset
    return group_map


def rank_contexts(context_ids: list[str], scores: list[float]) -> list[str]:
    ranked_indices = sorted(range(len(context_ids)), key=lambda index: (-scores[index], index))
    return [context_ids[index] for index in ranked_indices]


def build_policy(name: str):
    if name == "global":
        return lambda sample: True
    if name == "gt40":
        return lambda sample: len(sample["contexts"]) > 40
    if name == "quoted_no":
        return lambda sample: not bool(extract_quoted_spans(sample["comment"]))
    if name == "mention_yes":
        return lambda sample: "@" in sample["comment"]
    if name == "gt40_or_quoted_no":
        return lambda sample: len(sample["contexts"]) > 40 or not bool(extract_quoted_spans(sample["comment"]))
    if name == "gt40_or_mention":
        return lambda sample: len(sample["contexts"]) > 40 or "@" in sample["comment"]
    if name == "gt40_or_quoted_no_or_mention":
        return lambda sample: (
            len(sample["contexts"]) > 40
            or not bool(extract_quoted_spans(sample["comment"]))
            or "@" in sample["comment"]
        )
    raise ValueError(f"Unsupported policy: {name}")


def evaluate_rankings(rankings: dict[str, list[str]], gold_map: dict[str, list[str]]) -> dict[str, float]:
    hit1 = 0
    hit3 = 0
    mrr = 0.0
    for sample_id, ranked_context_ids in rankings.items():
        gold = set(gold_map[sample_id])
        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hit3 += int(any(context_id in gold for context_id in ranked_context_ids[:3]))
        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_context_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank
    total = len(rankings) or 1
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "mrr": mrr / total,
    }


def merge_rankings(base_ranked: list[str], aux_a_ranked: list[str], aux_b_ranked: list[str]) -> list[str]:
    merged = [base_ranked[0]]
    for source in (aux_a_ranked, aux_b_ranked, base_ranked):
        for context_id in source:
            if context_id not in merged:
                merged.append(context_id)
    return merged


def run_split(
    *,
    samples_by_id: dict[str, dict],
    prediction_rows: list[dict],
    base_group_maps: dict[str, dict[str, dict]],
    aux_group_map_a: dict[str, dict],
    aux_group_map_b: dict[str, dict],
    policy_name: str,
) -> tuple[dict[str, float], list[dict], int]:
    policy = build_policy(policy_name)
    rankings: dict[str, list[str]] = {}
    gold_map: dict[str, list[str]] = {}
    predictions: list[dict] = []
    applied = 0

    for row in prediction_rows:
        sample_id = row["sample_id"]
        selected_model = row["selected_model"]
        sample = samples_by_id[sample_id]
        base_group = base_group_maps[selected_model][sample_id]
        aux_group_a = aux_group_map_a[sample_id]
        aux_group_b = aux_group_map_b[sample_id]
        base_ranked = rank_contexts(base_group["context_ids"], base_group["scores"])
        aux_a_ranked = rank_contexts(aux_group_a["context_ids"], aux_group_a["scores"])
        aux_b_ranked = rank_contexts(aux_group_b["context_ids"], aux_group_b["scores"])

        if policy(sample):
            applied += 1
            merged = merge_rankings(base_ranked, aux_a_ranked, aux_b_ranked)
        else:
            merged = base_ranked

        rankings[sample_id] = merged
        gold_map[sample_id] = list(base_group["gold_context_ids"])
        predictions.append(
            {
                "sample_id": sample_id,
                "comment": row["comment"],
                "gold_context_ids": list(base_group["gold_context_ids"]),
                "predicted_context_ids_topk": merged[:3],
                "selected_model": selected_model,
                "policy_applied": policy(sample),
            }
        )

    return evaluate_rankings(rankings, gold_map), predictions, applied


def build_report(results: list[dict], best: dict) -> str:
    lines = ["# Selective Top-K Hybrid Search (Two Aux)", "", "## Best", ""]
    lines.append(f"- policy: `{best['policy']}`")
    lines.append(f"- dev: `{best['dev']['hit@1']:.4f} / {best['dev']['hit@3']:.4f} / {best['dev']['mrr']:.4f}`")
    lines.append(f"- test: `{best['test']['hit@1']:.4f} / {best['test']['hit@3']:.4f} / {best['test']['mrr']:.4f}`")
    lines.append(f"- applied: `dev={best['applied_dev']} / test={best['applied_test']}`")
    lines.append("")
    lines.append("## All Policies")
    lines.append("")
    lines.append("| policy | dev Hit@1 | dev Hit@3 | dev MRR | test Hit@1 | test Hit@3 | test MRR | applied dev | applied test |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        lines.append(
            f"| {row['policy']} | {row['dev']['hit@1']:.4f} | {row['dev']['hit@3']:.4f} | {row['dev']['mrr']:.4f} | "
            f"{row['test']['hit@1']:.4f} | {row['test']['hit@3']:.4f} | {row['test']['mrr']:.4f} | "
            f"{row['applied_dev']} | {row['applied_test']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    router_payload = json.loads(Path(args.router_json).read_text(encoding="utf-8"))
    dev_samples_by_id = {sample["sample_id"]: sample for sample in load_jsonl(args.dev_dataset)}
    test_samples_by_id = {sample["sample_id"]: sample for sample in load_jsonl(args.test_dataset)}

    dev_base_group_maps = {
        name: cache_to_group_map(path)
        for name, path in parse_cache_specs(args.dev_base_caches).items()
    }
    test_base_group_maps = {
        name: cache_to_group_map(path)
        for name, path in parse_cache_specs(args.test_base_caches).items()
    }
    dev_aux_group_map_a = cache_to_group_map(Path(args.dev_aux_cache_a))
    test_aux_group_map_a = cache_to_group_map(Path(args.test_aux_cache_a))
    dev_aux_group_map_b = cache_to_group_map(Path(args.dev_aux_cache_b))
    test_aux_group_map_b = cache_to_group_map(Path(args.test_aux_cache_b))

    results: list[dict] = []
    best: dict | None = None
    best_dev_predictions: list[dict] = []
    best_test_predictions: list[dict] = []

    for policy_name in [item.strip() for item in args.policies.split(",") if item.strip()]:
        dev_metrics, dev_predictions, applied_dev = run_split(
            samples_by_id=dev_samples_by_id,
            prediction_rows=router_payload["dev"]["predictions"],
            base_group_maps=dev_base_group_maps,
            aux_group_map_a=dev_aux_group_map_a,
            aux_group_map_b=dev_aux_group_map_b,
            policy_name=policy_name,
        )
        test_metrics, test_predictions, applied_test = run_split(
            samples_by_id=test_samples_by_id,
            prediction_rows=router_payload["test"]["predictions"],
            base_group_maps=test_base_group_maps,
            aux_group_map_a=test_aux_group_map_a,
            aux_group_map_b=test_aux_group_map_b,
            policy_name=policy_name,
        )
        row = {
            "policy": policy_name,
            "dev": dev_metrics,
            "test": test_metrics,
            "applied_dev": applied_dev,
            "applied_test": applied_test,
        }
        results.append(row)
        if best is None or (
            dev_metrics["hit@1"],
            dev_metrics["mrr"],
            dev_metrics["hit@3"],
        ) > (
            best["dev"]["hit@1"],
            best["dev"]["mrr"],
            best["dev"]["hit@3"],
        ):
            best = row
            best_dev_predictions = dev_predictions
            best_test_predictions = test_predictions

    if best is None:
        raise RuntimeError("No policies evaluated.")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"best": best, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_report(results, best), encoding="utf-8")

    with (output_json.parent / "predictions_dev.jsonl").open("w", encoding="utf-8") as handle:
        for row in best_dev_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_json.parent / "predictions_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in best_test_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_payload = {
        "router": {"policy": best["policy"]},
        "retrieval_dev": best["dev"],
        "retrieval_test": best["test"],
    }
    (output_json.parent / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"best": best}, ensure_ascii=False, indent=2))
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


if __name__ == "__main__":
    main()
