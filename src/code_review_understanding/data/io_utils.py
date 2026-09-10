from __future__ import annotations

import gzip
import json
from pathlib import Path


def open_maybe_gzip(path: str | Path, mode: str):
    resolved = Path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, mode, encoding="utf-8")
    return resolved.open(mode, encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open_maybe_gzip(resolved, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
