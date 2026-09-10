from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_review_understanding.models.fusion import load_score_cache, write_score_cache


HUNK_PATTERN = re.compile(r"^(?P<path>.+)::hunk_(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", required=True, help="Input score cache (.json or .json.gz).")
    parser.add_argument("--output", required=True, help="Output score cache (.json or .json.gz).")
    parser.add_argument(
        "--distance-penalty",
        type=float,
        default=0.25,
        help="Subtractive penalty per hunk distance within the same file.",
    )
    return parser.parse_args()


def parse_context_id(context_id: str) -> tuple[str, int] | None:
    match = HUNK_PATTERN.match(context_id)
    if not match:
        return None
    return match.group("path"), int(match.group("index"))


def build_neighbor_scores(cache: dict, *, distance_penalty: float) -> np.ndarray:
    dataset = cache["dataset"]
    scores = cache["scores"]
    neighbor_scores = np.zeros_like(scores, dtype=np.float32)
    offset = 0

    for context_ids, group_size in zip(
        dataset["context_ids_by_group"],
        dataset["group_sizes"],
        strict=True,
    ):
        next_offset = offset + group_size
        group_scores = scores[offset:next_offset]
        parsed = [parse_context_id(context_id) for context_id in context_ids]

        file_to_entries: dict[str, list[tuple[int, int]]] = {}
        for local_index, parsed_item in enumerate(parsed):
            if parsed_item is None:
                continue
            path, hunk_index = parsed_item
            file_to_entries.setdefault(path, []).append((local_index, hunk_index))

        group_neighbor_scores = np.array(group_scores, copy=True)
        for entries in file_to_entries.values():
            for local_index, hunk_index in entries:
                best_score = float(group_scores[local_index])
                for other_local_index, other_hunk_index in entries:
                    distance = abs(hunk_index - other_hunk_index)
                    candidate = float(group_scores[other_local_index]) - (distance_penalty * distance)
                    if candidate > best_score:
                        best_score = candidate
                group_neighbor_scores[local_index] = best_score

        neighbor_scores[offset:next_offset] = group_neighbor_scores.astype(np.float32)
        offset = next_offset

    return neighbor_scores


def main() -> None:
    args = parse_args()
    base_cache = load_score_cache(Path(args.base_cache))
    neighbor_scores = build_neighbor_scores(base_cache, distance_penalty=float(args.distance_penalty))
    write_score_cache(
        Path(args.output),
        model_path=f"{base_cache['model_path']}::neighbor_penalty={args.distance_penalty}",
        dataset=base_cache["dataset"],
        scores=neighbor_scores,
    )
    print(
        {
            "samples": len(base_cache["dataset"]["sample_ids"]),
            "contexts": len(neighbor_scores),
            "distance_penalty": float(args.distance_penalty),
            "output": args.output,
        }
    )


if __name__ == "__main__":
    main()
