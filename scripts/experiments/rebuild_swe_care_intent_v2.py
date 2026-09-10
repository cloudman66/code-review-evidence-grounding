"""Build leakage-safe intent-expanded semantic artifacts for SWE-CARE v2.

The legacy intent configurations point to the v1/``data/cache`` namespace and
were never valid for the PR-isolated v2 split.  This driver fits one
``intent_expanded`` word TF-IDF/SVD model on v2 train only, encodes v2 train,
dev, and test with that same model, and then trains the two v2 intent ablation
rankers.  Context expansion is deliberately ``full``: the semantic retriever
does not support an ``expanded`` context mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "experiments"
PYTHON_DEFAULT = ROOT / ".venv" / "bin" / "python"
PYTHON = str(PYTHON_DEFAULT if PYTHON_DEFAULT.is_file() else sys.executable)
DATA = Path("data/processed/swe_care_grounding_v2")
CACHE = Path("data/cache_v2/scores")
MODEL = Path("data/cache_v2/models/swe_care_semantic_intent_expanded_exact.pkl")
CONFIGS = (
    "src/configs/swe_care_learned_ranker_context_slices_file_context_semantic_char_"
    "file_mean_fuzzy_len2_intent_exact_cached_v2.yaml",
    "src/configs/swe_care_learned_ranker_context_slices_file_context_semantic_char_"
    "file_mean_fuzzy_len2_intent_feedback_exact_cached_v2.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-rankers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.as_posix()


def guard(command: list[str]) -> None:
    text = " ".join(command).replace("\\", "/")
    if "data/cache/" in text or "data/processed/swe_care_grounding/" in text or "results/" in text:
        raise AssertionError(f"legacy namespace in command: {shlex.join(command)}")
    if "v2" not in text.lower():
        raise AssertionError(f"command is not visibly v2-scoped: {shlex.join(command)}")


def command_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if not args.skip_cache:
        for split in ("train", "dev", "test"):
            command = [
                args.python,
                str(SCRIPT_DIR / "cache_semantic_grounding_scores.py"),
                "--train",
                rel(DATA / "train.jsonl"),
                "--dataset",
                rel(DATA / f"{split}.jsonl"),
                "--output",
                rel(CACHE / f"swe_care_semantic_intent_expanded_{split}_exact.json.gz"),
                "--query-mode",
                "intent_expanded",
                "--context-mode",
                "full",
                "--analyzer",
                "word",
                "--ngram-min",
                "1",
                "--ngram-max",
                "2",
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
                command.extend(["--model-output", rel(MODEL)])
            else:
                command.extend(["--model-input", rel(MODEL)])
            commands.append((f"build v2 intent semantic {split} cache", command))

    if not args.skip_rankers:
        for config in CONFIGS:
            commands.append((f"train v2 intent ranker: {Path(config).stem}", [args.python, str(ROOT / "scripts/experiments/train_learned_grounding_ranker.py"), "--config", config]))

    for _, command in commands:
        guard(command)
    return commands


def main() -> None:
    args = parse_args()
    for index, (label, command) in enumerate(command_plan(args), start=1):
        print(f"[{index}] {label}")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
