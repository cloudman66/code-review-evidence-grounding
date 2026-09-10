# Data and Code Availability

This repository provides code, experiment scripts, tests, locked environment files, and aggregate
machine-readable results for repository-level code review evidence grounding experiments.

## Available in This Repository

- Source code under `src/`.
- Experiment entry points under `scripts/experiments/`.
- A smoke test under `tests/`.
- Aggregate result CSV files under `results/csv/` and holdout, slice, diagnostic, and OOF-sensitivity
  JSON files under `results/aggregates/`. `oof_anchor_sensitivity.json` contains only macro and
  test-size-weighted results for six selected domain and six selected repository holdouts; it does
  not contain predictions or sample-level records.
- Environment files: `pyproject.toml` and `uv.lock`.
- Dataset provenance and the isolated v2 rebuild driver under `data/` and `scripts/experiments/`.
- The v2 rebuild namespace is `data/processed/swe_care_grounding_v2/`, `data/cache_v2/`, and
  `results_v2/`; the public aggregate files under `results/` are generated from v2 sources.
- The parser-fixed v2 trained learned-ranker model under `results/ablations/`.

## Third-Party Data

The repository does not redistribute third-party benchmark source files. Obtain those files from
their official sources under the applicable upstream terms and place them under the paths documented
in `data/dataset_sources.md`. Processed candidate-pool files, sample-level predictions, and score
caches are excluded because they may expose review comments, repository paths, or code snippets from
external projects. The supplied driver rebuilds them locally. The parser-fixed v2 learned-ranker
`model.json` is included. The fine-tuned bi-encoder package and weights are not included; its locked
optional environment, exact base-model revision, and full run command are recorded in `README.md`.

## Versioned Release

The `v1.1.0` source code, experiment scripts, locked environment and configurations,
aggregate research data, and fitted learned-ranker artifact are available in the versioned GitHub
release at `https://github.com/cloudman66/code-review-evidence-grounding/releases/tag/v1.1.0` (release date: 2026-09-10). The identical ZIP can also
be supplied through the submission system as reviewer-accessible supplementary software.
The release URL identifies the cited version. No DOI is claimed for this artifact.

The repository also contains dataset-provenance documentation and an ordered v2 driver for rebuilding
the corrected processed splits and all v2 caches after the required benchmark files have been obtained.
The v2 TF-IDF/SVD models are fitted only on the v2 training split and then reused for v2 dev/test
scoring. The release does not publish the superseded PR-overlapping result namespace.
SWE-CARE and CodeFuse-CR-Bench denote the same upstream benchmark; the data card at the fixed
revision cites the CodeFuse-CR-Bench paper. The auxiliary CR-classification-ESEM23 assets support
intent analysis and do not constitute an independent grounding benchmark.
Third-party benchmark source files must be obtained from
their official sources under the applicable upstream terms and are not redistributed by the authors.
Derived files that may expose review comments, repository paths, or code snippets are also not
redistributed; they can be rebuilt locally with the supplied driver. The fine-tuned bi-encoder
package and weights are not included, while its locked optional environment and exact rebuild
command are provided.
