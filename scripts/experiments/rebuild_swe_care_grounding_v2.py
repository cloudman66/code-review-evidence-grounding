"""Rebuild all canonical SWE-CARE grounding artifacts in an isolated v2 namespace.

The original rebuild driver writes to ``data/processed/swe_care_grounding``,
``data/cache`` and ``results``.  This entry point deliberately keeps those
artifacts untouched: every generated dataset, cache, model and result is under
the corresponding ``*_v2`` namespace.  Semantic retrievers are fitted only on
the v2 training split and are then reused for v2 dev/test scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "experiments"
SPLITS = ("train", "dev", "test")
DEFAULT_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python") if (
    PROJECT_ROOT / ".venv" / "bin" / "python"
).is_file() else sys.executable

PROCESSED_ROOT = Path("data/processed/swe_care_grounding_v2")
FEATURE_ROOT = Path("data/cache_v2/feature_rows")
SCORE_ROOT = Path("data/cache_v2/scores")
MODEL_ROOT = Path("data/cache_v2/models")
ANCHOR_MODEL = MODEL_ROOT / "swe_care_reproduction_anchor" / "model.json"
WORD_MODEL = MODEL_ROOT / "swe_care_semantic_word_exact.pkl"
CHAR_MODEL = MODEL_ROOT / "swe_care_semantic_char_exact.pkl"

ANCHOR_CONFIG = "src/configs/swe_care_reproduction_anchor_v2.yaml"
FINAL_CONFIG = (
    "src/configs/"
    "swe_care_learned_ranker_context_slices_file_context_semantic_char_"
    "file_mean_fuzzy_len2_feedback_exact_cached_v2.yaml"
)

# These are guards against accidentally routing a command through the legacy
# artifact namespace.  They intentionally use path-component boundaries so
# ``data/cache_v2`` and ``results_v2`` remain allowed.
LEGACY_PREFIXES = (
    "data/processed/swe_care_grounding/",
    "data/cache/",
    "results/",
)
LEGACY_FILENAMES = {
    "prepare_swe_care_grounding.py",
    "swe_care_reproduction_anchor.yaml",
    "swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-parquet",
        default="third_party/benchmarks/swe_care/dev.parquet",
        help="SWE-CARE dev parquet used as the train/dev source.",
    )
    parser.add_argument(
        "--test-parquet",
        default="third_party/benchmarks/swe_care/test.parquet",
        help="Official SWE-CARE test parquet.",
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help=(
            "Python executable used for each subprocess (defaults to the project's "
            ".venv when present)."
        ),
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Reuse existing v2 train/dev/test JSONL files after checking they exist.",
    )
    parser.add_argument(
        "--skip-final-ranker",
        action="store_true",
        help="Stop after generating all input caches and the anchor model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered v2 command plan without executing or creating outputs.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    """Return a project-relative POSIX path for an internally generated path."""
    if path.is_absolute():
        return path.relative_to(PROJECT_ROOT).as_posix()
    return path.as_posix()


def script(name: str) -> str:
    return relative(SCRIPT_DIR / name)


def _contains_legacy_path(token: str) -> bool:
    normalized = str(token).replace("\\", "/")
    # Check every suffix so absolute paths under this project are guarded too.
    components = normalized.split("/")
    for index in range(len(components)):
        suffix = "/".join(components[index:])
        if any(suffix == prefix.rstrip("/") or suffix.startswith(prefix) for prefix in LEGACY_PREFIXES):
            return True
    return False


def assert_v2_command(command: list[str]) -> None:
    """Reject a command that could read/write legacy artifacts or configs."""
    for token in command:
        if _contains_legacy_path(token):
            raise AssertionError(f"legacy artifact path in v2 command: {token}")
        if Path(token).name in LEGACY_FILENAMES:
            raise AssertionError(f"legacy file in v2 command: {token}")
    if not any("v2" in token.lower() for token in command):
        raise AssertionError(f"v2 command has no v2 namespace marker: {shlex.join(command)}")


def semantic_command(
    *,
    python: str,
    analyzer: str,
    model_path: Path,
    suffix: str,
    ngram_min: str,
    ngram_max: str,
    split: str,
) -> list[str]:
    command = [
        python,
        script("cache_semantic_grounding_scores.py"),
        "--train",
        relative(PROCESSED_ROOT / "train.jsonl"),
        "--dataset",
        relative(PROCESSED_ROOT / f"{split}.jsonl"),
        "--output",
        relative(SCORE_ROOT / f"swe_care_context_slices_semantic{suffix}_{split}_exact.json.gz"),
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
        # Only this command fits and persists a model.  It is fitted from the
        # v2 train split above; dev/test commands load exactly this artifact.
        command.extend(["--model-output", relative(model_path)])
    else:
        command.extend(["--model-input", relative(model_path)])
    return command


def command_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python = args.python
    commands: list[tuple[str, list[str]]] = []

    if not args.skip_data_prep:
        commands.append(
            (
                "prepare isolated v2 train/dev/test splits",
                [
                    python,
                    script("prepare_swe_care_grounding_v2.py"),
                    "--dev-parquet",
                    args.dev_parquet,
                    "--test-parquet",
                    args.test_parquet,
                    "--output-dir",
                    relative(PROCESSED_ROOT),
                    "--language",
                    "Python",
                    "--seed",
                    "42",
                    "--dev-fraction",
                    "0.1",
                ],
            )
        )

    for split in SPLITS:
        commands.append(
            (
                f"build v2 {split} feature rows",
                [
                    python,
                    script("cache_feature_rows.py"),
                    "--dataset",
                    relative(PROCESSED_ROOT / f"{split}.jsonl"),
                    "--output",
                    relative(FEATURE_ROOT / f"swe_care_{split}.jsonl.gz"),
                ],
            )
        )

    # Fit each semantic representation exactly once on v2 train, then use the
    # saved v2 model for dev/test.  The --train argument remains explicit on
    # all three calls for provenance and alignment checks in the cache writer.
    for analyzer, model_path, suffix, ngram_min, ngram_max in (
        ("word", WORD_MODEL, "", "1", "2"),
        ("char_wb", CHAR_MODEL, "_char", "3", "5"),
    ):
        for split in SPLITS:
            commands.append(
                (
                    f"build v2 {split} {analyzer} semantic scores",
                    semantic_command(
                        python=python,
                        analyzer=analyzer,
                        model_path=model_path,
                        suffix=suffix,
                        ngram_min=ngram_min,
                        ngram_max=ngram_max,
                        split=split,
                    ),
                )
            )

    for split in SPLITS:
        commands.append(
            (
                f"aggregate v2 {split} word-semantic scores by file mean",
                [
                    python,
                    script("cache_file_prior_scores.py"),
                    "--dataset",
                    relative(PROCESSED_ROOT / f"{split}.jsonl"),
                    "--input-score-cache",
                    relative(SCORE_ROOT / f"swe_care_context_slices_semantic_{split}_exact.json.gz"),
                    "--output",
                    relative(SCORE_ROOT / f"swe_care_context_slices_semantic_file_mean_{split}_exact.json.gz"),
                    "--agg",
                    "mean",
                    "--normalize",
                    "none",
                ],
            )
        )

    commands.append(
        (
            "train isolated v2 semantic char+file anchor",
            [
                python,
                script("train_learned_grounding_ranker.py"),
                "--config",
                ANCHOR_CONFIG,
            ],
        )
    )

    for split in SPLITS:
        commands.append(
            (
                f"cache v2 {split} semantic char+file anchor scores",
                [
                    python,
                    script("cache_grounding_scores.py"),
                    "--model-path",
                    relative(ANCHOR_MODEL),
                    "--dataset",
                    relative(PROCESSED_ROOT / f"{split}.jsonl"),
                    "--feature-row-cache",
                    relative(FEATURE_ROOT / f"swe_care_{split}.jsonl.gz"),
                    "--external-score",
                    "semantic_char_score="
                    + relative(SCORE_ROOT / f"swe_care_context_slices_semantic_char_{split}_exact.json.gz"),
                    "--external-score",
                    "semantic_file_score="
                    + relative(SCORE_ROOT / f"swe_care_context_slices_semantic_file_mean_{split}_exact.json.gz"),
                    "--output",
                    relative(SCORE_ROOT / f"swe_care_context_slices_semantic_char_file_mean_{split}_exact.json.gz"),
                ],
            )
        )
        commands.append(
            (
                f"cache v2 {split} fuzzy-comment scores",
                [
                    python,
                    script("cache_fuzzy_comment_scores.py"),
                    "--dataset",
                    relative(PROCESSED_ROOT / f"{split}.jsonl"),
                    "--output",
                    relative(SCORE_ROOT / f"swe_care_fuzzy_comment_len2_{split}_exact.json.gz"),
                    "--min-token-len",
                    "2",
                    "--focus-mode",
                    "all",
                ],
            )
        )
        commands.append(
            (
                f"cache v2 {split} top-1 feedback semantic scores",
                [
                    python,
                    script("cache_feedback_semantic_scores.py"),
                    "--dataset",
                    relative(PROCESSED_ROOT / f"{split}.jsonl"),
                    "--semantic-model",
                    relative(WORD_MODEL),
                    "--base-score-cache",
                    relative(SCORE_ROOT / f"swe_care_context_slices_semantic_char_file_mean_{split}_exact.json.gz"),
                    "--output",
                    relative(SCORE_ROOT / f"swe_care_semantic_feedback_top1_{split}_exact.json.gz"),
                    "--feedback-top-k",
                    "1",
                    "--max-feedback-terms",
                    "12",
                    "--max-paths",
                    "2",
                ],
            )
        )

    if not args.skip_final_ranker:
        commands.append(
            (
                "train and evaluate isolated v2 canonical learned ranker",
                [
                    python,
                    script("train_learned_grounding_ranker.py"),
                    "--config",
                    FINAL_CONFIG,
                ],
            )
        )

    for _, command in commands:
        assert_v2_command(command)
    return commands


def check_processed_inputs() -> None:
    missing = [
        PROCESSED_ROOT / f"{split}.jsonl"
        for split in SPLITS
        if not (PROJECT_ROOT / PROCESSED_ROOT / f"{split}.jsonl").is_file()
    ]
    if missing:
        joined = ", ".join(relative(path) for path in missing)
        raise FileNotFoundError(f"Missing v2 processed splits: {joined}")


def main() -> None:
    args = parse_args()
    plan = command_plan(args)

    if args.skip_data_prep and not args.dry_run:
        check_processed_inputs()

    for index, (label, command) in enumerate(plan, start=1):
        print(f"[{index:02d}/{len(plan):02d}] {label}")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
