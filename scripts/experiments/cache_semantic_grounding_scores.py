from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.models.fusion import write_score_cache
from code_review_understanding.data.io_utils import load_jsonl
from code_review_understanding.models.semantic_retrieval import (
    build_semantic_score_cache,
    fit_semantic_retriever,
    load_semantic_retriever,
    save_semantic_retriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training jsonl used to fit the semantic retriever.")
    parser.add_argument("--dataset", required=True, help="Dataset jsonl to encode into a score cache.")
    parser.add_argument("--output", required=True, help="Output score cache (.json or .json.gz).")
    parser.add_argument("--model-output", default="", help="Optional path to save the fitted semantic model.")
    parser.add_argument("--model-input", default="", help="Optional path to a previously saved semantic model.")
    parser.add_argument("--query-mode", default="normalized")
    parser.add_argument("--context-mode", default="full")
    parser.add_argument("--analyzer", default="word")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_input:
        model = load_semantic_retriever(args.model_input)
        model_path = args.model_input
    else:
        train_samples = load_jsonl(args.train)
        model = fit_semantic_retriever(
            train_samples,
            query_mode=args.query_mode,
            context_mode=args.context_mode,
            analyzer=args.analyzer,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            n_components=args.n_components,
            random_state=args.random_state,
        )
        model_path = args.model_output or ""
        if args.model_output:
            save_semantic_retriever(model, args.model_output)

    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]

    dataset, scores = build_semantic_score_cache(samples, model)
    output_path = Path(args.output)
    write_score_cache(
        output_path,
        model_path=model_path,
        dataset=dataset,
        scores=scores,
    )

    print(
        {
            "samples": len(dataset["sample_ids"]),
            "contexts": len(scores),
            "training_corpus_size": int(model["training_corpus_size"]),
            "n_features": int(model["n_features"]),
            "n_components": int(model["n_components"]),
            "query_mode": str(model.get("query_mode", "normalized")),
            "context_mode": str(model.get("context_mode", "full")),
            "output": str(output_path),
            "model_path": model_path,
        }
    )


if __name__ == "__main__":
    main()
