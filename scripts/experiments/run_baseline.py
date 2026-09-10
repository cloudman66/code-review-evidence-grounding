from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.eval.baseline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results = run_experiment(config)
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results["metrics"], handle, ensure_ascii=False, indent=2)

    predictions_path = output_dir / "predictions_test.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in results["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(json.dumps(results["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
