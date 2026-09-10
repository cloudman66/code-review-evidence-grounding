from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sentence_transformers import SentenceTransformer
import yaml

from code_review_understanding.eval.baseline import evidence_metrics
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.eval.external_scores import augment_feature_rows_with_external_scores
from code_review_understanding.models.learning import (
    build_labeled_ranking_dataset,
    evaluate_ranking_dataset,
    load_feature_row_cache,
    load_linear_ranker_model,
)
from code_review_understanding.models.semantic_retrieval import (
    semantic_context_text,
    semantic_query_text,
)


TEST_PATH = PROJECT_ROOT / "data" / "processed" / "swe_care_grounding" / "test.jsonl"
TOP_K = 3
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "swe_care_runtime_footprint"
LEARNED_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "ablations"
    / "swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached"
    / "model.json"
)
LEARNED_CONFIG_PATH = (
    PROJECT_ROOT
    / "src"
    / "configs"
    / "swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached.yaml"
)
BIENCODER_METRICS_PATH = (
    PROJECT_ROOT / "results" / "baselines" / "swe_care_biencoder_finetuned_minilm" / "metrics.json"
)
RESULTS_ARCHIVE_DIR = PROJECT_ROOT / "artifacts" / "legacy" / "experiment_results_20260412" / "dirs"


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def dir_size_mb(path: Path) -> float:
    total = 0
    for subpath in path.rglob("*"):
        if subpath.is_file():
            total += subpath.stat().st_size
    return total / (1024 * 1024)


def resolve_results_path(path: Path) -> Path:
    if path.exists():
        return path
    relative = path.relative_to(PROJECT_ROOT / "results")
    archived = RESULTS_ARCHIVE_DIR / relative
    if archived.exists():
        return archived
    raise FileNotFoundError(path)


def dataset_stats(samples: list[dict]) -> dict[str, float]:
    context_counts = [len(sample["contexts"]) for sample in samples]
    total_contexts = sum(context_counts)
    return {
        "samples": len(samples),
        "total_contexts": total_contexts,
        "avg_contexts_per_sample": total_contexts / max(len(samples), 1),
    }


def run_lexical(samples: list[dict]) -> dict[str, float]:
    started = time.perf_counter()
    metrics = evidence_metrics(samples, top_k=TOP_K)
    elapsed = time.perf_counter() - started
    return {
        "hit@1": metrics["hit@1"],
        "hit@3": metrics["hit@3"],
        "mrr": metrics["mrr"],
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / max(len(samples), 1),
        "artifact_size_mb": 0.0,
    }


def run_learned_ranker(samples: list[dict]) -> dict[str, float]:
    model = load_linear_ranker_model(LEARNED_MODEL_PATH)
    config = yaml.safe_load(LEARNED_CONFIG_PATH.read_text(encoding="utf-8"))
    started = time.perf_counter()
    test_feature_cache = load_feature_row_cache(
        resolve_results_path(PROJECT_ROOT / config["feature_row_cache"]["test_path"])
    )
    for item in config.get("external_features", []):
        test_feature_cache = augment_feature_rows_with_external_scores(
            test_feature_cache,
            score_cache_path=resolve_results_path(PROJECT_ROOT / item["test_score_cache_path"]),
            feature_name=str(item["name"]),
        )
    dataset = build_labeled_ranking_dataset(
        samples,
        base_feature_names=model["base_feature_names"],
        feature_transform=model["feature_transform"],
        feature_rows_by_sample_id=test_feature_cache,
    )
    metrics = evaluate_ranking_dataset(dataset, model, top_k=TOP_K)
    elapsed = time.perf_counter() - started
    return {
        "hit@1": metrics["hit@1"],
        "hit@3": metrics["hit@3"],
        "mrr": metrics["mrr"],
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / max(len(samples), 1),
        "artifact_size_mb": file_size_mb(LEARNED_MODEL_PATH),
    }


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> dict[str, object]:
    unique = list(dict.fromkeys(texts))
    started = time.perf_counter()
    vectors = model.encode(
        unique,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started
    return {
        "count": len(unique),
        "elapsed_seconds": elapsed,
        "vectors": {text: vector for text, vector in zip(unique, vectors, strict=True)},
    }


def run_biencoder(samples: list[dict]) -> dict[str, float]:
    metrics_payload = json.loads(BIENCODER_METRICS_PATH.read_text(encoding="utf-8"))
    model_dir = Path(metrics_payload["model_path"])
    model = SentenceTransformer(str(model_dir), device="cpu")
    query_mode = metrics_payload["config"]["query_mode"]
    context_mode = metrics_payload["config"]["context_mode"]
    batch_size = int(metrics_payload["config"]["encode_batch_size"])

    query_texts = [semantic_query_text(sample["comment"], mode=query_mode) for sample in samples]
    context_texts = [
        semantic_context_text(context["text"], mode=context_mode)
        for sample in samples
        for context in sample["contexts"]
    ]

    query_result = encode_texts(model, query_texts, batch_size=batch_size)
    context_result = encode_texts(model, context_texts, batch_size=batch_size)
    scoring_started = time.perf_counter()
    hit1 = 0
    hit3 = 0
    mrr = 0.0
    for sample in samples:
        q = semantic_query_text(sample["comment"], mode=query_mode)
        qvec = query_result["vectors"][q]
        scored = []
        for context in sample["contexts"]:
            ctext = semantic_context_text(context["text"], mode=context_mode)
            score = float((qvec * context_result["vectors"][ctext]).sum())
            scored.append((context["context_id"], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [context_id for context_id, _ in scored]
        gold = set(sample["gold_context_ids"])
        hit1 += int(bool(ranked_ids) and ranked_ids[0] in gold)
        hit3 += int(any(context_id in gold for context_id in ranked_ids[:TOP_K]))
        reciprocal_rank = 0.0
        for index, context_id in enumerate(ranked_ids, start=1):
            if context_id in gold:
                reciprocal_rank = 1.0 / index
                break
        mrr += reciprocal_rank
    scoring_elapsed = time.perf_counter() - scoring_started
    total_elapsed = query_result["elapsed_seconds"] + context_result["elapsed_seconds"] + scoring_elapsed
    return {
        "hit@1": hit1 / max(len(samples), 1),
        "hit@3": hit3 / max(len(samples), 1),
        "mrr": mrr / max(len(samples), 1),
        "elapsed_seconds": total_elapsed,
        "seconds_per_sample": total_elapsed / max(len(samples), 1),
        "artifact_size_mb": dir_size_mb(model_dir),
        "query_encoding_seconds": query_result["elapsed_seconds"],
        "context_encoding_seconds": context_result["elapsed_seconds"],
        "scoring_seconds": scoring_elapsed,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(TEST_PATH)
    payload = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "dataset": dataset_stats(samples),
        "systems": {
            "lexical_baseline": run_lexical(samples),
            "canonical_single_model": run_learned_ranker(samples),
            "fine_tuned_biencoder_cpu": run_biencoder(samples),
        },
    }
    (OUTPUT_DIR / "runtime_footprint.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [
        ("Lexical baseline", payload["systems"]["lexical_baseline"]),
        ("Single learned ranker", payload["systems"]["canonical_single_model"]),
        ("Fine-tuned bi-encoder (CPU)", payload["systems"]["fine_tuned_biencoder_cpu"]),
    ]
    lines = [
        "# Runtime and Footprint",
        "",
        f"- test samples: `{payload['dataset']['samples']}`",
        f"- total contexts: `{payload['dataset']['total_contexts']}`",
        f"- average candidate hunks/sample: `{payload['dataset']['avg_contexts_per_sample']:.2f}`",
        f"- platform: `{payload['environment']['platform']}`",
        f"- python: `{payload['environment']['python']}`",
        "",
        "| System | Hit@1 | Hit@3 | MRR | Runtime (s) | ms/sample | Artifact size (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in rows:
        lines.append(
            "| "
            + f"{name} | {metrics['hit@1']:.4f} | {metrics['hit@3']:.4f} | {metrics['mrr']:.4f} | "
            + f"{metrics['elapsed_seconds']:.2f} | {metrics['seconds_per_sample'] * 1000:.2f} | {metrics['artifact_size_mb']:.2f} |"
        )
    biencoder = payload["systems"]["fine_tuned_biencoder_cpu"]
    lines.extend(
        [
            "",
            "Lexical and single learned ranker runtimes are measured against the prepared SWE-CARE candidate sets. The single learned ranker timing includes cache loading, external-score attachment, feature-matrix construction, and final ranking.",
            "",
            "## Fine-tuned Bi-encoder Breakdown",
            "",
            f"- query encoding: `{biencoder['query_encoding_seconds']:.2f}` s",
            f"- context encoding: `{biencoder['context_encoding_seconds']:.2f}` s",
            f"- scoring and ranking: `{biencoder['scoring_seconds']:.2f}` s",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
