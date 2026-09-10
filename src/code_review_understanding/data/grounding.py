from __future__ import annotations

from code_review_understanding.eval.baseline import evidence_metrics
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.learning import (
    evidence_metrics_with_model,
    load_linear_ranker_model,
)


def run_grounding_experiment(config: dict) -> dict:
    train_path = config["dataset"].get("train_path")
    dev_path = config["dataset"]["dev_path"]
    test_path = config["dataset"]["test_path"]
    top_k = int(config["retrieval"]["top_k"])
    ranking_config = config.get("ranking")
    learned_ranker_path = config.get("learned_ranker_path")
    learned_model = load_linear_ranker_model(learned_ranker_path) if learned_ranker_path else None

    train_samples = load_jsonl(train_path) if train_path else []
    dev_samples = load_jsonl(dev_path)
    test_samples = load_jsonl(test_path)

    if learned_model:
        dev_metrics = evidence_metrics_with_model(dev_samples, learned_model, top_k=top_k)
        test_metrics = evidence_metrics_with_model(test_samples, learned_model, top_k=top_k)
    else:
        dev_metrics = evidence_metrics(dev_samples, top_k=top_k, ranking_config=ranking_config)
        test_metrics = evidence_metrics(test_samples, top_k=top_k, ranking_config=ranking_config)

    predictions = []
    ranked_test = {item["sample_id"]: item for item in test_metrics["ranked_predictions"]}
    for sample in test_samples:
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_test[sample["sample_id"]]["ranked_context_ids"][:top_k],
            }
        )

    return {
        "metrics": {
            "dataset_sizes": {
                "train": len(train_samples),
                "dev": len(dev_samples),
                "test": len(test_samples),
            },
            "retrieval_dev": {
                "hit@1": dev_metrics["hit@1"],
                f"hit@{top_k}": dev_metrics[f"hit@{top_k}"],
                "mrr": dev_metrics["mrr"],
            },
            "retrieval_test": {
                "hit@1": test_metrics["hit@1"],
                f"hit@{top_k}": test_metrics[f"hit@{top_k}"],
                "mrr": test_metrics["mrr"],
            },
        },
        "predictions": predictions,
    }
