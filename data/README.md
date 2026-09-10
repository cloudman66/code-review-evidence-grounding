# Data Format and Corrected v2 Namespace

The corrected parser-fixed SWE-CARE splits live under `data/processed/swe_care_grounding_v2/` and
use JSON Lines with one sample per line. The reported split sizes are train `9,623`, dev `1,070`,
and test `1,222`; comments from the same pull request are never split across train/dev/test, and
official test pull requests are excluded from training.

This public artifact does not redistribute processed JSONL files because they contain third-party
review comments and code snippets. Provenance and rebuild commands are documented in
`data/dataset_sources.md`. Local v2 caches under `data/cache_v2/` and outputs under `results_v2/`
should be rebuilt with the supplied drivers.
