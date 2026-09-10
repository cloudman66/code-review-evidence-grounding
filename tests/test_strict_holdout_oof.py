from __future__ import annotations

import unittest

from scripts.experiments.run_strict_holdout_v2_oof import group_aware_folds


class StrictHoldoutOOFTest(unittest.TestCase):
    def test_group_aware_folds_cover_each_sample_once(self) -> None:
        samples = [
            {
                "sample_id": f"sample-{group}-{index}",
                "metadata": {"group_key": f"repo::pr_{group}"},
            }
            for group, size in enumerate((3, 2, 2, 1, 1, 1))
            for index in range(size)
        ]
        folds = group_aware_folds(samples, n_splits=3, seed=42)
        validation_ids: list[str] = []
        for training, validation in folds:
            train_groups = {sample["metadata"]["group_key"] for sample in training}
            validation_groups = {sample["metadata"]["group_key"] for sample in validation}
            self.assertFalse(train_groups & validation_groups)
            validation_ids.extend(sample["sample_id"] for sample in validation)

        self.assertEqual(sorted(validation_ids), sorted(sample["sample_id"] for sample in samples))
        self.assertEqual(len(validation_ids), len(set(validation_ids)))

    def test_fold_assignment_is_deterministic(self) -> None:
        samples = [
            {"sample_id": f"sample-{index}", "metadata": {"group_key": f"group-{index}"}}
            for index in range(12)
        ]
        first = group_aware_folds(samples, n_splits=5, seed=7)
        second = group_aware_folds(samples, n_splits=5, seed=7)
        self.assertEqual(
            [[sample["sample_id"] for sample in validation] for _, validation in first],
            [[sample["sample_id"] for sample in validation] for _, validation in second],
        )


if __name__ == "__main__":
    unittest.main()
