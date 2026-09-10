from __future__ import annotations

import json
from pathlib import Path


DATASET = {
    "train": [
        {
            "sample_id": "train-001",
            "comment": "This may throw when config is missing.",
            "intent": "correctness",
            "gold_context_ids": ["ctx-2"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "if config is None: config = {}"},
                {"context_id": "ctx-2", "source": "function", "text": "timeout = config['timeout']"},
                {"context_id": "ctx-3", "source": "file", "text": "logger.info('config loaded')"},
            ],
        },
        {
            "sample_id": "train-002",
            "comment": "This loop is repeated in two places. Consider extracting a helper.",
            "intent": "maintainability",
            "gold_context_ids": ["ctx-1"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "for item in items: normalized.append(clean(item))"},
                {"context_id": "ctx-2", "source": "function", "text": "def build_payload(items): body = []"},
                {"context_id": "ctx-3", "source": "file", "text": "class PayloadBuilder:"},
            ],
        },
        {
            "sample_id": "train-003",
            "comment": "Please add a test for the empty input case.",
            "intent": "testing",
            "gold_context_ids": ["ctx-3"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "def tokenize(text): return text.split()"},
                {"context_id": "ctx-2", "source": "function", "text": "def test_tokenize_basic(): assert tokenize('a b') == ['a', 'b']"},
                {"context_id": "ctx-3", "source": "file", "text": "tests/test_tokenizer.py"},
            ],
        },
        {
            "sample_id": "train-004",
            "comment": "This query is likely to become slow on large datasets.",
            "intent": "performance",
            "gold_context_ids": ["ctx-2"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "users = session.query(User).all()"},
                {"context_id": "ctx-2", "source": "function", "text": "for user in session.query(User).all(): send_email(user)"},
                {"context_id": "ctx-3", "source": "file", "text": "EMAIL_BATCH_SIZE = 100"},
            ],
        },
        {
            "sample_id": "train-005",
            "comment": "Can you rename this variable to make the intent clearer?",
            "intent": "readability",
            "gold_context_ids": ["ctx-1"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "tmp = compute_score(user, weights)"},
                {"context_id": "ctx-2", "source": "function", "text": "def compute_score(user, weights):"},
                {"context_id": "ctx-3", "source": "file", "text": "SCORE_LIMIT = 0.9"},
            ],
        },
        {
            "sample_id": "train-006",
            "comment": "This validation belongs before saving the file.",
            "intent": "correctness",
            "gold_context_ids": ["ctx-2"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "storage.save(upload)"},
                {"context_id": "ctx-2", "source": "function", "text": "if not is_valid(upload): raise ValueError('invalid upload')"},
                {"context_id": "ctx-3", "source": "file", "text": "def store_upload(upload):"},
            ],
        },
        {
            "sample_id": "train-007",
            "comment": "This helper should be private if it is only used in this module.",
            "intent": "maintainability",
            "gold_context_ids": ["ctx-2"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "def collect_metrics(records):"},
                {"context_id": "ctx-2", "source": "function", "text": "def build_stat_map(records):"},
                {"context_id": "ctx-3", "source": "file", "text": "from .stat_helpers import build_stat_map"},
            ],
        },
        {
            "sample_id": "train-008",
            "comment": "We need a regression test for the timezone conversion bug.",
            "intent": "testing",
            "gold_context_ids": ["ctx-3"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "return dt.astimezone(UTC)"},
                {"context_id": "ctx-2", "source": "function", "text": "def normalize_timezone(dt):"},
                {"context_id": "ctx-3", "source": "file", "text": "tests/test_timezones.py"},
            ],
        },
    ],
    "dev": [
        {
            "sample_id": "dev-001",
            "comment": "This should short-circuit when the token list is empty.",
            "intent": "correctness",
            "gold_context_ids": ["ctx-1"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "if not tokens: return []"},
                {"context_id": "ctx-2", "source": "function", "text": "result = tokens[0]"},
                {"context_id": "ctx-3", "source": "file", "text": "def merge_tokens(tokens):"},
            ],
        },
        {
            "sample_id": "dev-002",
            "comment": "This nested loop will be expensive for large batches.",
            "intent": "performance",
            "gold_context_ids": ["ctx-2"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "items = load_items()"},
                {"context_id": "ctx-2", "source": "function", "text": "for row in rows:\n    for item in items:\n        match(row, item)"},
                {"context_id": "ctx-3", "source": "file", "text": "MAX_BATCH = 500"},
            ],
        },
    ],
    "test": [
        {
            "sample_id": "test-001",
            "comment": "Could you add coverage for the invalid header case?",
            "intent": "testing",
            "gold_context_ids": ["ctx-3"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "def parse_header(header):"},
                {"context_id": "ctx-2", "source": "function", "text": "raise HeaderError('invalid header')"},
                {"context_id": "ctx-3", "source": "file", "text": "tests/test_headers.py"},
            ],
        },
        {
            "sample_id": "test-002",
            "comment": "The name here is too vague and hurts readability.",
            "intent": "readability",
            "gold_context_ids": ["ctx-1"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "val = normalize(record)"},
                {"context_id": "ctx-2", "source": "function", "text": "def normalize(record):"},
                {"context_id": "ctx-3", "source": "file", "text": "DEFAULT_VALUE = 1"},
            ],
        },
        {
            "sample_id": "test-003",
            "comment": "This belongs in a helper instead of being duplicated again.",
            "intent": "maintainability",
            "gold_context_ids": ["ctx-2"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "payload.append(clean(name))"},
                {"context_id": "ctx-2", "source": "function", "text": "def build_user_payload(names): for name in names: payload.append(clean(name))"},
                {"context_id": "ctx-3", "source": "file", "text": "class UserSerializer:"},
            ],
        },
        {
            "sample_id": "test-004",
            "comment": "Fetching all rows first may hurt performance.",
            "intent": "performance",
            "gold_context_ids": ["ctx-1"],
            "contexts": [
                {"context_id": "ctx-1", "source": "diff", "text": "rows = repository.find_all()"},
                {"context_id": "ctx-2", "source": "function", "text": "for row in rows: publish(row)"},
                {"context_id": "ctx-3", "source": "file", "text": "PUBLISH_LIMIT = 200"},
            ],
        },
    ],
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in DATASET.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
    print(f"Wrote toy dataset to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
