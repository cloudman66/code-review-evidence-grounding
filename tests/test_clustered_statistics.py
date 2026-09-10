from __future__ import annotations

import unittest

import numpy as np

from scripts.experiments.run_clustered_v2_statistical_validation import (
    cluster_bootstrap,
    outcome,
    validate_complete_rankings,
)


class ClusteredStatisticsTest(unittest.TestCase):
    def test_outcome_uses_complete_ranking_for_mrr(self) -> None:
        values = outcome(["gold"], ["noise-1", "noise-2", "noise-3", "gold"])
        self.assertEqual(values["hit@1"], 0.0)
        self.assertEqual(values["hit@3"], 0.0)
        self.assertEqual(values["mrr"], 0.25)
        self.assertEqual(values["mrr_at_3"], 0.0)
        self.assertEqual(values["first_relevant_rank"], 4.0)

    def test_complete_ranking_validation_rejects_top_k_only_rows(self) -> None:
        dataset = [
            {
                "sample_id": "s1",
                "gold_context_ids": ["c4"],
                "contexts": [
                    {"context_id": "c1"},
                    {"context_id": "c2"},
                    {"context_id": "c3"},
                    {"context_id": "c4"},
                ],
            }
        ]
        truncated = [{"sample_id": "s1", "ranked_context_ids": ["c1", "c2", "c3"]}]
        with self.assertRaises(ValueError):
            validate_complete_rankings(dataset, truncated, label="test")

    def test_cluster_bootstrap_preserves_micro_estimand(self) -> None:
        values = [[1.0, 0.0], [0.5]]
        summary = cluster_bootstrap(values, np.random.default_rng(7), samples=200)
        self.assertEqual(summary["mean"], 0.5)
        self.assertLessEqual(summary["ci_low"], summary["mean"])
        self.assertGreaterEqual(summary["ci_high"], summary["mean"])


if __name__ == "__main__":
    unittest.main()
