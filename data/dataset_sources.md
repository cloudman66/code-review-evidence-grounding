# Dataset Sources (v2 / parser-fixed)

## Main Benchmark Source: SWE-CARE / CodeFuse-CR-Bench

- Official dataset source: `https://huggingface.co/datasets/inclusionAI/SWE-CARE`
- Official project repository: `https://github.com/inclusionAI/SWE-CARE`
- The dataset name SWE-CARE and the paper/project name CodeFuse-CR-Bench refer to the same upstream benchmark; CodeFuse-CR-Bench is not a second independent benchmark.
- Exact dataset revision used: `3b3a625ef26bd497a3a13485c0dc9ece537f24d0`
- License: Apache License 2.0, as declared by the official dataset card and project repository
- Upstream files at this revision: [`data/dev-00000-of-00001.parquet`](https://huggingface.co/datasets/inclusionAI/SWE-CARE/resolve/3b3a625ef26bd497a3a13485c0dc9ece537f24d0/data/dev-00000-of-00001.parquet) and [`data/test-00000-of-00001.parquet`](https://huggingface.co/datasets/inclusionAI/SWE-CARE/resolve/3b3a625ef26bd497a3a13485c0dc9ece537f24d0/data/test-00000-of-00001.parquet)
- Local source files: `third_party/benchmarks/swe_care/dev.parquet` and `third_party/benchmarks/swe_care/test.parquet`
- Local source SHA-256: `dev.parquet` = `39bcd8ff2f833e5aae381a26d110f52999d1ac60a7715c0d5604e19b558f5e18`; `test.parquet` = `be45381d2579e74d64abee7aedb1438843a83cd0a22a29d9bb8e141c35d26b07`
- Upstream dataset citation: Guo, H., Zheng, X., Liao, Z., Yu, H., Di, P., Zhang, Z., and Dai, H.-N. (2025). *CodeFuse-CR-Bench: A Comprehensiveness-aware Benchmark for End-to-End Code Review Evaluation in Python Projects*. arXiv:2509.14856. DOI: `10.48550/arXiv.2509.14856`.
- v2 processed splits: `train.jsonl` (`9,623`), `dev.jsonl` (`1,070`), `test.jsonl` (`1,222`)
- Split policy: assign whole repository/PR groups to a split; exclude official test PR keys from
  the development source; conservatively match review comments to hunks; write non-unique or
  insufficient candidate annotations to `unresolved.jsonl` instead of guessing.
- Rebuild entry point: `uv run python scripts/experiments/rebuild_swe_care_grounding_v2.py`

## Active Auxiliary Dataset: CR-classification-ESEM23 Assets

- Official project repository: `https://github.com/WSU-SEAL/CR-classification-ESEM23`
- Exact repository revision used: `d29e2a8c3cef9e1b608fe7bc62071ed8b5fd7d45`
- License: GNU General Public License v3.0
- Upstream files: [`dataset/code_attributes.csv`](https://raw.githubusercontent.com/WSU-SEAL/CR-classification-ESEM23/d29e2a8c3cef9e1b608fe7bc62071ed8b5fd7d45/dataset/code_attributes.csv) and [`dataset/labeled_dataset.xlsx`](https://raw.githubusercontent.com/WSU-SEAL/CR-classification-ESEM23/d29e2a8c3cef9e1b608fe7bc62071ed8b5fd7d45/dataset/labeled_dataset.xlsx)
- Local source SHA-256: `code_attributes.csv` = `0b1ba0a240315f008ca93f10fd3eb62480099aeac88f1186c7076779dc98b818`; `labeled_dataset.xlsx` = `1740fe1ebd2a32453a88aa774a8645288d5dfd8339f4f2cee28bd5305b0db82f`
- Processed splits under `data/processed/cr_intent/` are not redistributed in this public artifact.

## Related But Not Directly Evaluated

CodeReviewQA is used as related benchmark context only; this workflow evaluates repository-level
candidate-hunk ranking with Hit@k and MRR.

## Validation

- Dataset readability: `uv run python scripts/experiments/validate_dataset.py --path data/processed/swe_care_grounding_v2/<split>.jsonl`
- Leakage check: confirm `data/processed/swe_care_grounding_v2/summary.json` reports zero PR overlap
  between train/dev/test.
