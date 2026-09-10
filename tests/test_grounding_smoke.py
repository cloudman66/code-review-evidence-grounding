from __future__ import annotations

import unittest

from code_review_understanding.eval.baseline import evidence_metrics, rank_contexts
from scripts.experiments.prepare_swe_care_grounding_v2 import (
    parse_hunk_header,
    split_patch_to_hunks,
)


class GroundingSmokeTest(unittest.TestCase):
    def test_hunk_header_preserves_explicit_zero_counts(self) -> None:
        self.assertEqual(
            parse_hunk_header("@@ -10,0 +10,3 @@"),
            (10, 0, 10, 3),
        )
        self.assertEqual(
            parse_hunk_header("@@ -10 +10,0 @@"),
            (10, 1, 10, 0),
        )

    def test_lexical_ranker_places_matching_hunk_first(self) -> None:
        sample = {
            "sample_id": "toy-1",
            "comment": "Please guard `parse_user_id` when user_id is missing.",
            "gold_context_ids": ["ctx-match"],
            "contexts": [
                {
                    "context_id": "ctx-noise",
                    "source": "diff_hunk",
                    "text": (
                        "path: app/cache.py\n"
                        "@@ -1,2 +1,3 @@\n"
                        "+def refresh_cache(items):\n"
                        "+    return list(items)\n"
                    ),
                },
                {
                    "context_id": "ctx-match",
                    "source": "diff_hunk",
                    "text": (
                        "path: app/auth.py\n"
                        "@@ -10,6 +10,8 @@\n"
                        "+def parse_user_id(user_id):\n"
                        "+    if user_id is None:\n"
                        "+        return None\n"
                    ),
                },
            ],
        }

        ranked = rank_contexts(sample)
        metrics = evidence_metrics([sample], top_k=1)

        self.assertEqual(ranked[0]["context_id"], "ctx-match")
        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_file_markers_inside_hunk_are_preserved_as_content(self) -> None:
        patch = "\n".join(
            [
                "diff --git a/example.txt b/example.txt",
                "--- a/example.txt",
                "+++ b/example.txt",
                "@@ -1,2 +1,2 @@",
                "---- old value",
                "+--- new value",
                "@@ -10,1 +10,1 @@",
                "-old tail",
                "+new tail",
            ]
        )
        contexts = split_patch_to_hunks(patch)
        self.assertEqual(len(contexts), 2)
        self.assertIn("---- old value", contexts[0]["text"])
        self.assertIn("+--- new value", contexts[0]["text"])
        self.assertEqual(contexts[0]["path"], "example.txt")
        self.assertEqual(contexts[1]["path"], "example.txt")


if __name__ == "__main__":
    unittest.main()
