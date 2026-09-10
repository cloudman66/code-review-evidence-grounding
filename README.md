# Code Review Evidence Grounding

Release: `v1.1.0`
Release date: `2026-09-10`
Versioned release: https://github.com/cloudman66/code-review-evidence-grounding/releases/tag/v1.1.0

`v1.1.0` is the current public artifact; earlier experimental snapshots are excluded from the
public distribution. This release contains the parser-corrected, pull-request-isolated results.

This repository contains code and aggregate result files for reproducible experiments on
repository-level code review evidence grounding.
The primary data source is SWE-CARE, the dataset associated with the CodeFuse-CR-Bench paper;
these names refer to one benchmark. The CR-classification-ESEM23 assets support auxiliary intent
analysis. Exact source revisions, download URLs, and source-file SHA-256 checksums are recorded
in `data/dataset_sources.md`.

## Included

- Source code for lexical, dense, learned-ranker, routing, holdout, and diagnostic experiments.
- Experiment configuration files under `src/configs/`.
- Aggregate result CSV files under `results/csv/`.
- Aggregate holdout, slice, diagnostic, and OOF-sensitivity JSON files under `results/aggregates/`.
  The OOF file reports only macro and test-size-weighted results for six selected domain and six
  selected repository holdouts; it excludes prediction and sample-level files.
- Scripts for rebuilding processed datasets and all canonical caches from official source files.
- The parser-fixed v2 trained learned-ranker model under `results/ablations/`.
- Citation and data-availability metadata in `CITATION.cff`, `.zenodo.json`, and `DATA_AVAILABILITY.md`.

## Not Included

The repository does not redistribute third-party benchmark files. Download them from their official
sources under the applicable upstream terms and place them under the paths documented in
`data/dataset_sources.md`. Derived candidate-pool manifests and prediction files that may expose
review comments, repository paths, or code snippets are also excluded.

Score caches and the fine-tuned bi-encoder package and weights are excluded. These derived files are
rebuilt locally with the documented commands. The trained lightweight learned-ranker artifact is
included.

## Setup

```bash
uv sync --frozen
```

## Rebuild Data Locally

```bash
uv run python scripts/experiments/prepare_swe_care_grounding_v2.py \
  --dev-parquet third_party/benchmarks/swe_care/dev.parquet \
  --test-parquet third_party/benchmarks/swe_care/test.parquet \
  --output-dir data/processed/swe_care_grounding_v2

uv run python scripts/experiments/prepare_cr_intent_dataset.py \
  --xlsx-path third_party/benchmarks/cr_classification_assets/labeled_dataset.xlsx \
  --attributes-csv third_party/benchmarks/cr_classification_assets/code_attributes.csv \
  --output-dir data/processed/cr_intent
```

## Rebuild the Corrected v2 Learned Ranker

The v2 ordered driver rebuilds the corrected processed splits, feature-row caches, train-fitted word
and character semantic models, all dependent v2 score caches, and the final learned-ranker model.
It never writes to the legacy `data/processed/swe_care_grounding/`, `data/cache/`, or `results/`
trees:

```bash
uv run python scripts/experiments/rebuild_swe_care_grounding_v2.py
uv run python scripts/experiments/rebuild_swe_care_grounding_v2.py --dry-run
```

The dry run prints every fixed parameter and subprocess without executing the experiment. Generated
processed files and caches remain local because they can contain third-party text or repository paths.
The previous `prepare_swe_care_grounding.py` and `rebuild_canonical_grounding_artifacts.py` entry
points are retained as legacy commands for reproducing the submitted manuscript's historical artifacts.

## Rebuild the Fine-Tuned Bi-Encoder Baseline

The reported MiniLM baseline used `sentence-transformers/all-MiniLM-L6-v2` at revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` with this locked environment and full command:

```bash
uv sync --frozen --extra biencoder
uv run --extra biencoder python scripts/experiments/train_biencoder_grounding.py   --model-name sentence-transformers/all-MiniLM-L6-v2   --model-revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf   --query-mode expanded --context-mode full   --batch-size 32 --encode-batch-size 64   --epochs 2 --learning-rate 2e-5 --warmup-ratio 0.1   --max-seq-length 256 --top-k 3 --seed 13   --output-dir results_v2/swe_care_biencoder_finetuned
```

The downloaded base model and generated fine-tuned weights are not bundled.

## Rebuild the Off-the-Shelf Dense Baseline

The reported off-the-shelf BGE rows use `BAAI/bge-small-en-v1.5` with full hunk contexts and
`query_mode=normalized` or `query_mode=expanded`. The cache command writes the aligned v2 score
cache under `data/cache_v2/scores/`:

```bash
uv run python scripts/experiments/cache_dense_grounding_scores.py \
  --dataset data/processed/swe_care_grounding_v2/test.jsonl \
  --output data/cache_v2/scores/dense_bge_normalized_test.json.gz \
  --model-name BAAI/bge-small-en-v1.5 --query-mode normalized --context-mode full \
  --batch-size 64 --threads 8

uv run python scripts/experiments/cache_dense_grounding_scores.py \
  --dataset data/processed/swe_care_grounding_v2/test.jsonl \
  --output data/cache_v2/scores/dense_bge_expanded_test.json.gz \
  --model-name BAAI/bge-small-en-v1.5 --query-mode expanded --context-mode full \
  --batch-size 64 --threads 8
```

## Reproduce Core Checks

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests
uv run python scripts/experiments/validate_dataset.py --path data/processed/swe_care_grounding_v2/test.jsonl
uv run python scripts/experiments/run_grounding_baseline.py --config src/configs/swe_care_grounding_v2.yaml
```

The corrected v2 ranker command is the final driver step after its complete v2 cache dependency chain
has been generated. The aggregate machine-readable result files in this release are generated from
the parser-fixed v2 summaries and diagnostics; pending or exploratory systems are not promoted to
the primary result table.

## Citation and Availability

Use `CITATION.cff` for software citation metadata. `DATA_AVAILABILITY.md` summarizes which
materials are available here and which third-party data must be obtained from official sources.

## License

The authors' own source code, experiment configuration files, scripts, aggregate result CSV files,
and trained learned-ranker model in this repository are released under the MIT License. See
`LICENSE`.

The license does not cover third-party benchmark data, processed sample-level manifests, review
comments, repository code snippets, the fine-tuned bi-encoder package or weights, or files owned by
third parties. See `THIRD_PARTY_NOTICES.md`.
