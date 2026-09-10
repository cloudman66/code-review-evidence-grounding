# Experiment Configs

This directory keeps YAML configurations for reproducible experiment runs.

## Active experiment configs

- `swe_care_grounding_v2.yaml` and
  `swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached_v2.yaml`:
  leakage-resistant v2 dataset/ranker configs. They use the isolated
  `data/processed/swe_care_grounding_v2/`, `data/cache_v2/`, and `results_v2/`
  namespaces; use `scripts/experiments/rebuild_swe_care_grounding_v2.py` to
  rebuild the canonical v2 artifacts.
- `swe_care_grounding.yaml`: lexical-diff baseline.
- `swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_feedback_exact_cached.yaml`: canonical single-model ranker.
- `swe_care_external_feature_search_feedback_exact.yaml` and `swe_care_external_feature_search_intent_exact.yaml`: auxiliary feature-search configs used to produce the canonical feedback and intent branches.
- `swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_intent_exact_cached_v2.yaml` and `swe_care_learned_ranker_context_slices_file_context_semantic_char_file_mean_fuzzy_len2_intent_feedback_exact_cached_v2.yaml`: v2-only intent ablations. They require intent semantic caches generated with `query_mode=intent_expanded` and `context_mode=full`; the legacy intent configs must not be reused.
- `cr_intent.yaml`: intent-classification side dataset config retained for provenance.

## Historical configs

The remaining `swe_care_*` configs document intermediate ablations and search runs. Their `output.dir` fields may still point to the historical flat `results/swe_care_*` layout used before the project was reorganized. The active result artifacts are now grouped under:

- `results/baselines/`
- `results/ablations/`
- `results/diagnostics/`
- `results/holdout_validation/`
- `results/summaries/`

Large score caches and sample-level prediction files are not part of the public artifact. Rebuild
them locally under `data/cache_v2/` for v2 runs (or under `data/cache/` only when deliberately
reproducing the legacy manuscript namespace). Do not mix cache namespaces within one run.
