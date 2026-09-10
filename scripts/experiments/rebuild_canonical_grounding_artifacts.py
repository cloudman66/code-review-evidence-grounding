from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "experiments"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the processed SWE-CARE splits and every derived cache required by "
            "the canonical learned-ranker configuration."
        )
    )
    parser.add_argument(
        "--dev-parquet",
        default="third_party/benchmarks/swe_care/dev.parquet",
    )
    parser.add_argument(
        "--test-parquet",
        default="third_party/benchmarks/swe_care/test.parquet",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for each documented subprocess.",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Reuse existing processed train/dev/test files after checking that they exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete ordered command sequence without running it.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def script(name: str) -> str:
    return relative(SCRIPT_DIR / name)


def command_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python = args.python
    processed = Path("data/processed/swe_care_grounding")
    feature_root = Path("data/cache/feature_rows")
    score_root = Path("data/cache/scores")
    model_root = Path("data/cache/models")
    word_model = model_root / "swe_care_semantic_word_exact.pkl"
    char_model = model_root / "swe_care_semantic_char_exact.pkl"
    anchor_model = model_root / "swe_care_reproduction_anchor/model.json"

    commands: list[tuple[str, list[str]]] = []
    if not args.skip_data_prep:
        commands.append(
            (
                "prepare deterministic train/dev/test splits",
                [
                    python,
                    script("prepare_swe_care_grounding.py"),
                    "--dev-parquet",
                    args.dev_parquet,
                    "--test-parquet",
                    args.test_parquet,
                    "--output-dir",
                    processed.as_posix(),
                    "--language",
                    "Python",
                    "--seed",
                    "42",
                ],
            )
        )

    for split in SPLITS:
        commands.append(
            (
                f"build {split} feature rows",
                [
                    python,
                    script("cache_feature_rows.py"),
                    "--dataset",
                    (processed / f"{split}.jsonl").as_posix(),
                    "--output",
                    (feature_root / f"swe_care_{split}.jsonl.gz").as_posix(),
                ],
            )
        )

    for analyzer, model_path, suffix, ngram_min, ngram_max in (
        ("word", word_model, "", "1", "2"),
        ("char_wb", char_model, "_char", "3", "5"),
    ):
        for split in SPLITS:
            semantic_command = [
                python,
                script("cache_semantic_grounding_scores.py"),
                "--train",
                (processed / "train.jsonl").as_posix(),
                "--dataset",
                (processed / f"{split}.jsonl").as_posix(),
                "--output",
                (score_root / f"swe_care_context_slices_semantic{suffix}_{split}_exact.json.gz").as_posix(),
                "--query-mode",
                "normalized",
                "--context-mode",
                "full",
                "--analyzer",
                analyzer,
                "--ngram-min",
                ngram_min,
                "--ngram-max",
                ngram_max,
                "--min-df",
                "2",
                "--max-df",
                "0.98",
                "--max-features",
                "40000",
                "--n-components",
                "128",
                "--random-state",
                "42",
            ]
            if split == "train":
                semantic_command.extend(["--model-output", model_path.as_posix()])
            else:
                semantic_command.extend(["--model-input", model_path.as_posix()])
            commands.append((f"build {split} {analyzer} semantic scores", semantic_command))

    for split in SPLITS:
        commands.append(
            (
                f"aggregate {split} word-semantic scores by file mean",
                [
                    python,
                    script("cache_file_prior_scores.py"),
                    "--dataset",
                    (processed / f"{split}.jsonl").as_posix(),
                    "--input-score-cache",
                    (score_root / f"swe_care_context_slices_semantic_{split}_exact.json.gz").as_posix(),
                    "--output",
                    (score_root / f"swe_care_context_slices_semantic_file_mean_{split}_exact.json.gz").as_posix(),
                    "--agg",
                    "mean",
                    "--normalize",
                    "none",
                ],
            )
        )

    commands.append(
        (
            "train the semantic char+file anchor",
            [
                python,
                script("train_learned_grounding_ranker.py"),
                "--config",
                "src/configs/swe_care_reproduction_anchor.yaml",
            ],
        )
    )

    for split in SPLITS:
        commands.append(
            (
                f"cache {split} semantic char+file anchor scores",
                [
                    python,
                    script("cache_grounding_scores.py"),
                    "--model-path",
                    anchor_model.as_posix(),
                    "--dataset",
                    (processed / f"{split}.jsonl").as_posix(),
                    "--feature-row-cache",
                    (feature_root / f"swe_care_{split}.jsonl.gz").as_posix(),
                    "--external-score",
                    (
                        "semantic_char_score="
                        + (score_root / f"swe_care_context_slices_semantic_char_{split}_exact.json.gz").as_posix()
                    ),
                    "--external-score",
                    (
                        "semantic_file_score="
                        + (score_root / f"swe_care_context_slices_semantic_file_mean_{split}_exact.json.gz").as_posix()
                    ),
                    "--output",
                    (score_root / f"swe_care_context_slices_semantic_char_file_mean_{split}_exact.json.gz").as_posix(),
                ],
            )
        )
        commands.append(
            (
                f"cache {split} fuzzy-comment scores",
                [
                    python,
                    script("cache_fuzzy_comment_scores.py"),
                    "--dataset",
                    (processed / f"{split}.jsonl").as_posix(),
                    "--output",
                    (score_root / f"swe_care_fuzzy_comment_len2_{split}_exact.json.gz").as_posix(),
                    "--min-token-len",
                    "2",
                    "--focus-mode",
                    "all",
                ],
            )
        )
        commands.append(
            (
                f"cache {split} top-1 feedback scores",
                [
                    python,
                    script("cache_feedback_semantic_scores.py"),
                    "--dataset",
                    (processed / f"{split}.jsonl").as_posix(),
                    "--semantic-model",
                    word_model.as_posix(),
                    "--base-score-cache",
                    (score_root / f"swe_care_context_slices_semantic_char_file_mean_{split}_exact.json.gz").as_posix(),
                    "--output",
                    (score_root / f"swe_care_semantic_feedback_top1_{split}_exact.json.gz").as_posix(),
                    "--feedback-top-k",
                    "1",
                    "--max-feedback-terms",
                    "12",
                    "--max-paths",
                    "2",
                ],
            )
        )

    commands.append(
        (
            "train and evaluate the canonical learned ranker",
            [
                python,
                script("train_learned_grounding_ranker.py"),
                "--config",
                (
                    "src/configs/"
                    "swe_care_learned_ranker_context_slices_file_context_semantic_char_"
                    "file_mean_fuzzy_len2_feedback_exact_cached.yaml"
                ),
            ],
        )
    )
    return commands


def main() -> None:
    args = parse_args()
    plan = command_plan(args)
    if args.skip_data_prep and not args.dry_run:
        missing = [
            Path("data/processed/swe_care_grounding") / f"{split}.jsonl"
            for split in SPLITS
            if not (PROJECT_ROOT / "data/processed/swe_care_grounding" / f"{split}.jsonl").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing processed splits: {', '.join(map(str, missing))}")

    for index, (label, command) in enumerate(plan, start=1):
        print(f"[{index:02d}/{len(plan):02d}] {label}")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
