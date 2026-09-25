from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicalalign_ad.modeling import (  # noqa: E402
    LogisticL2,
    MedianStandardizer,
    choose_youden_threshold,
    roc_auc,
    stratified_holdout_indices,
)


class ModelingTests(unittest.TestCase):
    def test_auc_known_ordering(self):
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(roc_auc(y, probabilities), 0.75)

    def test_stratified_holdout_is_disjoint_and_deterministic(self):
        y = np.array([0] * 70 + [1] * 30)
        train_a, test_a = stratified_holdout_indices(y, 0.2, 42)
        train_b, test_b = stratified_holdout_indices(y, 0.2, 42)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(test_a, test_b))
        self.assertEqual(len(set(train_a) & set(test_a)), 0)
        self.assertEqual(y[test_a].sum(), 6)
        self.assertEqual(len(test_a), 20)

    def test_preprocessor_uses_training_median(self):
        train = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
        test = pd.DataFrame({"x": [100.0, np.nan]})
        processor = MedianStandardizer().fit(train, ["x"])
        transformed = processor.transform(test)
        self.assertAlmostEqual(processor.medians[0], 1.5)
        self.assertAlmostEqual(transformed[1, 0], 0.0)
        self.assertEqual(transformed[1, 1], 1.0)

    def test_logistic_learns_separable_signal(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticL2(penalty=0.01).fit(x, y)
        probabilities = model.predict_proba(x)
        self.assertGreater(roc_auc(y, probabilities), 0.99)

    def test_youden_threshold_uses_training_predictions(self):
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        threshold = choose_youden_threshold(y, probabilities)
        self.assertGreater(threshold, 0.4)
        self.assertLessEqual(threshold, 0.6)


if __name__ == "__main__":
    unittest.main()