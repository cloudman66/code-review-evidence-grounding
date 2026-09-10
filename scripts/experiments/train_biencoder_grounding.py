from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from code_review_understanding.models.fusion import write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.semantic_retrieval import (
    semantic_context_text,
    semantic_query_text,
)


BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=str(PROJECT_ROOT / "data" / "processed" / "swe_care_grounding" / "train.jsonl"),
    )
    parser.add_argument(
        "--dev",
        default=str(PROJECT_ROOT / "data" / "processed" / "swe_care_grounding" / "dev.jsonl"),
    )
    parser.add_argument(
        "--test",
        default=str(PROJECT_ROOT / "data" / "processed" / "swe_care_grounding" / "test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "swe_care_biencoder_finetuned"),
    )
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--model-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--query-mode", default="expanded")
    parser.add_argument("--context-mode", default="full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    return parser.parse_args()


def choose_device(preferred: str) -> str:
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def limit_samples(samples: list[dict], limit: int) -> list[dict]:
    if limit > 0:
        return samples[:limit]
    return samples


def build_training_examples(
    samples: list[dict],
    *,
    query_mode: str,
    context_mode: str,
) -> list[InputExample]:
    examples: list[InputExample] = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        context_map = {
            context["context_id"]: semantic_context_text(context["text"], mode=context_mode)
            for context in sample["contexts"]
        }
        query_text = semantic_query_text(sample["comment"], mode=query_mode)
        for gold_context_id in sample["gold_context_ids"]:
            context_text = context_map.get(gold_context_id)
            if not context_text:
                continue
            key = (query_text, context_text)
            if key in seen:
                continue
            seen.add(key)
            examples.append(InputExample(texts=[query_text, context_text]))
    return examples


def ranking_metrics(
    predicted_context_ids_by_sample: dict[str, list[str]],
    gold_by_sample: dict[str, set[str]],
    *,
    top_k: int,
) -> dict[str, float]:
    hit1 = 0
    hitk = 0
    mrr = 0.0
    total = len(predicted_context_ids_by_sample) or 1
    for sample_id, ranked_context_ids in predicted_context_ids_by_sample.items():
        gold = gold_by_sample[sample_id]
        hit1 += int(bool(ranked_context_ids) and ranked_context_ids[0] in gold)
        hitk += int(any(context_id in gold for context_id in ranked_context_ids[:top_k]))
        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_context_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank
    return {
        "hit@1": hit1 / total,
        f"hit@{top_k}": hitk / total,
        "mrr": mrr / total,
    }


def encode_unique_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    label: str,
) -> dict[str, np.ndarray]:
    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    if not deduped:
        return {}
    print(
        json.dumps(
            {
                "stage": "encode_unique_texts",
                "label": label,
                "count": len(deduped),
                "batch_size": batch_size,
            },
            ensure_ascii=False,
        )
    )
    vectors = model.encode(
        deduped,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {
        text: np.asarray(vector, dtype=np.float32)
        for text, vector in zip(deduped, vectors, strict=True)
    }


def build_biencoder_score_cache(
    samples: list[dict],
    *,
    model: SentenceTransformer,
    query_mode: str,
    context_mode: str,
    batch_size: int,
) -> tuple[dict, np.ndarray]:
    query_texts: list[str] = []
    context_texts: list[str] = []
    for sample in samples:
        query_texts.append(semantic_query_text(sample["comment"], mode=query_mode))
        for context in sample["contexts"]:
            context_texts.append(semantic_context_text(context["text"], mode=context_mode))

    query_map = encode_unique_texts(
        model,
        query_texts,
        batch_size=batch_size,
        label="queries",
    )
    context_map = encode_unique_texts(
        model,
        context_texts,
        batch_size=batch_size,
        label="contexts",
    )

    sample_ids: list[str] = []
    context_ids_by_group: list[list[str]] = []
    gold_context_ids_by_group: list[list[str]] = []
    group_sizes: list[int] = []
    score_blocks: list[np.ndarray] = []

    for sample in samples:
        query_text = semantic_query_text(sample["comment"], mode=query_mode)
        query_vector = query_map[query_text]

        current_context_ids: list[str] = []
        current_vectors: list[np.ndarray] = []
        for context in sample["contexts"]:
            context_text = semantic_context_text(context["text"], mode=context_mode)
            current_context_ids.append(context["context_id"])
            current_vectors.append(context_map[context_text])

        if current_vectors:
            context_matrix = np.vstack(current_vectors).astype(np.float32)
            scores = (context_matrix @ query_vector).astype(np.float32)
        else:
            scores = np.zeros((0,), dtype=np.float32)

        score_blocks.append(scores)
        sample_ids.append(sample["sample_id"])
        context_ids_by_group.append(current_context_ids)
        gold_context_ids_by_group.append(list(sample["gold_context_ids"]))
        group_sizes.append(len(current_context_ids))

    dataset = {
        "sample_ids": sample_ids,
        "context_ids_by_group": context_ids_by_group,
        "gold_context_ids_by_group": gold_context_ids_by_group,
        "group_sizes": group_sizes,
    }
    all_scores = (
        np.concatenate(score_blocks, axis=0) if score_blocks else np.zeros((0,), dtype=np.float32)
    )
    return dataset, all_scores


def evaluate_samples(
    samples: list[dict],
    *,
    dataset: dict,
    scores: np.ndarray,
    top_k: int,
) -> tuple[dict[str, float], list[dict]]:
    predicted_context_ids_by_sample: dict[str, list[str]] = {}
    gold_by_sample: dict[str, set[str]] = {}
    predictions: list[dict] = []
    offset = 0
    for sample, context_ids, group_size in zip(
        samples,
        dataset["context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + int(group_size)
        group_scores = scores[offset:next_offset]
        ranked_indices = sorted(
            range(int(group_size)),
            key=lambda index: (float(group_scores[index]), -index),
            reverse=True,
        )
        ranked_context_ids = [context_ids[index] for index in ranked_indices]
        predicted_context_ids_by_sample[sample["sample_id"]] = ranked_context_ids
        gold_by_sample[sample["sample_id"]] = set(sample["gold_context_ids"])
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "comment": sample["comment"],
                "gold_context_ids": sample["gold_context_ids"],
                "predicted_context_ids_topk": ranked_context_ids[:top_k],
            }
        )
        offset = next_offset
    return ranking_metrics(predicted_context_ids_by_sample, gold_by_sample, top_k=top_k), predictions


def better_dev_metrics(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    candidate_key = (candidate["hit@1"], candidate["mrr"], candidate["hit@3"])
    incumbent_key = (incumbent["hit@1"], incumbent["mrr"], incumbent["hit@3"])
    return candidate_key > incumbent_key


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "model_best"
    dev_cache_path = output_dir / "dev_score_cache.json.gz"
    test_cache_path = output_dir / "test_score_cache.json.gz"

    train_samples = limit_samples(load_jsonl(args.train), int(args.train_limit))
    dev_samples = limit_samples(load_jsonl(args.dev), int(args.dev_limit))
    test_samples = limit_samples(load_jsonl(args.test), int(args.test_limit))

    train_examples = build_training_examples(
        train_samples,
        query_mode=args.query_mode,
        context_mode=args.context_mode,
    )
    if not train_examples:
        raise ValueError("No positive training pairs were built.")

    device = choose_device(args.device)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        revision=(args.model_revision or None),
    )
    model.max_seq_length = int(args.max_seq_length)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=int(args.batch_size),
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(train_dataloader) * float(args.warmup_ratio)))

    history: list[dict] = []
    best_dev: dict[str, float] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        print(
            json.dumps(
                {"stage": "train_epoch_start", "epoch": epoch},
                ensure_ascii=False,
            )
        )
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            scheduler="WarmupLinear",
            warmup_steps=warmup_steps,
            optimizer_params={"lr": float(args.learning_rate)},
            output_path=None,
            save_best_model=False,
            show_progress_bar=False,
        )

        print(
            json.dumps(
                {"stage": "dev_scoring_start", "epoch": epoch},
                ensure_ascii=False,
            )
        )
        dev_dataset, dev_scores = build_biencoder_score_cache(
            dev_samples,
            model=model,
            query_mode=args.query_mode,
            context_mode=args.context_mode,
            batch_size=int(args.encode_batch_size),
        )
        dev_metrics, _ = evaluate_samples(
            dev_samples,
            dataset=dev_dataset,
            scores=dev_scores,
            top_k=int(args.top_k),
        )
        epoch_record = {"epoch": epoch, "retrieval_dev": dev_metrics}
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if better_dev_metrics(dev_metrics, best_dev):
            best_dev = dev_metrics
            model.save(str(best_model_dir))

    best_model = SentenceTransformer(str(best_model_dir), device=device)
    best_model.max_seq_length = int(args.max_seq_length)

    print(json.dumps({"stage": "final_dev_scoring_start"}, ensure_ascii=False))
    dev_dataset, dev_scores = build_biencoder_score_cache(
        dev_samples,
        model=best_model,
        query_mode=args.query_mode,
        context_mode=args.context_mode,
        batch_size=int(args.encode_batch_size),
    )
    print(json.dumps({"stage": "final_test_scoring_start"}, ensure_ascii=False))
    test_dataset, test_scores = build_biencoder_score_cache(
        test_samples,
        model=best_model,
        query_mode=args.query_mode,
        context_mode=args.context_mode,
        batch_size=int(args.encode_batch_size),
    )

    dev_metrics, dev_predictions = evaluate_samples(
        dev_samples,
        dataset=dev_dataset,
        scores=dev_scores,
        top_k=int(args.top_k),
    )
    test_metrics, test_predictions = evaluate_samples(
        test_samples,
        dataset=test_dataset,
        scores=test_scores,
        top_k=int(args.top_k),
    )

    write_score_cache(
        dev_cache_path,
        model_path=(
            f"biencoder_finetuned::{args.model_name}"
            f"::query_mode={args.query_mode}"
            f"::context_mode={args.context_mode}"
            f"::epochs={args.epochs}"
        ),
        dataset=dev_dataset,
        scores=dev_scores,
    )
    write_score_cache(
        test_cache_path,
        model_path=(
            f"biencoder_finetuned::{args.model_name}"
            f"::query_mode={args.query_mode}"
            f"::context_mode={args.context_mode}"
            f"::epochs={args.epochs}"
        ),
        dataset=test_dataset,
        scores=test_scores,
    )

    metrics_payload = {
        "model_path": str(best_model_dir),
        "config": {
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "query_mode": args.query_mode,
            "context_mode": args.context_mode,
            "batch_size": int(args.batch_size),
            "encode_batch_size": int(args.encode_batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "max_seq_length": int(args.max_seq_length),
            "device": device,
            "seed": int(args.seed),
        },
        "dataset_sizes": {
            "train": len(train_samples),
            "dev": len(dev_samples),
            "test": len(test_samples),
            "train_pairs": len(train_examples),
        },
        "retrieval_dev": dev_metrics,
        "retrieval_test": test_metrics,
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(output_dir / "training_history.json", history)
    save_jsonl(output_dir / "predictions_dev.jsonl", dev_predictions)
    save_jsonl(output_dir / "predictions_test.jsonl", test_predictions)

    report_lines = [
        "# Fine-tuned Bi-encoder Baseline",
        "",
        f"- model: `{args.model_name}`",
        f"- device: `{device}`",
        f"- query mode: `{args.query_mode}`",
        f"- context mode: `{args.context_mode}`",
        f"- train pairs: `{len(train_examples)}`",
        "",
        "## Dev",
        "",
        f"- hit@1: {dev_metrics['hit@1']:.4f}",
        f"- hit@{args.top_k}: {dev_metrics[f'hit@{args.top_k}']:.4f}",
        f"- mrr: {dev_metrics['mrr']:.4f}",
        "",
        "## Test",
        "",
        f"- hit@1: {test_metrics['hit@1']:.4f}",
        f"- hit@{args.top_k}: {test_metrics[f'hit@{args.top_k}']:.4f}",
        f"- mrr: {test_metrics['mrr']:.4f}",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
