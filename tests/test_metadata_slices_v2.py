from __future__ import annotations

import unittest

from scripts.experiments.run_metadata_slice_evaluation_v2 import (
    assert_v2_path,
    metric_values,
    summarize_system,
)


class MetadataSlicesV2Test(unittest.TestCase):
    def test_metric_values_keep_full_mrr_separate_from_mrr_at_3(self) -> None:
        sample = {"gold_context_ids": ["gold"]}
        metrics = metric_values(
            sample,
            ["noise-1", "noise-2", "noise-3", "gold"],
            top_k=3,
        )
        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["hit@3"], 0.0)
        self.assertEqual(metrics["mrr"], 0.25)
        self.assertEqual(metrics["mrr_at_3"], 0.0)

    def test_top_k_only_system_does_not_claim_full_mrr(self) -> None:
        dataset = [
            {
                "sample_id": "sample-1",
                "gold_context_ids": ["gold"],
                "contexts": [
                    {"context_id": "noise"},
                    {"context_id": "gold"},
                ],
            }
        ]
        predictions = [
            {
                "sample_id": "sample-1",
                "gold_context_ids": ["gold"],
                "predicted_context_ids_topk": ["noise", "gold"],
            }
        ]
        result = summarize_system(
            dataset,
            ["sample-1"],
            predictions,
            None,
            top_k=3,
            name="top_k_only",
        )
        self.assertFalse(result["complete_ranking"])
        self.assertIsNone(result["metrics"]["mrr"])
        self.assertEqual(result["metrics"]["mrr_at_3"], 0.5)

    def test_namespace_guard_accepts_v2_and_rejects_legacy_results(self) -> None:
        assert_v2_path("results_v2/diagnostics/example", label="test output")
        with self.assertRaises(ValueError):
            assert_v2_path("results/diagnostics/example", label="test output")


if __name__ == "__main__":
    unittest.main()
