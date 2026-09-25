from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicalalign_ad.cohort import (  # noqa: E402
    CohortRule,
    build_cohort,
    clean_diagnosis_rows,
    harmonize_current_diagnosis,
)


class CohortTests(unittest.TestCase):
    def test_dxchange_mapping(self):
        frame = pd.DataFrame(
            {
                "DIAGNOSIS": [None, None, None],
                "DXCHANGE": [2, 5, 7],
                "DXCURREN": [None, None, None],
            }
        )
        self.assertEqual(harmonize_current_diagnosis(frame).tolist(), [2.0, 3.0, 1.0])

    def test_future_dates_are_removed(self):
        frame = pd.DataFrame(
            {
                "RID": [1, 1],
                "VISCODE2": ["bl", "m36"],
                "EXAMDATE": ["2020-01-01", "2109-01-01"],
                "DIAGNOSIS": [2, 2],
                "DXCHANGE": [None, None],
                "DXCURREN": [None, None],
            }
        )
        cleaned, audit = clean_diagnosis_rows(frame, "2000-01-01", "2026-09-20")
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(audit["future_date_rows"], 1)

    def test_converter_and_stable_labels(self):
        frame = pd.DataFrame(
            {
                "RID": [1, 1, 2, 2, 3, 3],
                "visit_code": ["bl", "m12", "bl", "m36", "bl", "m36"],
                "exam_date": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-12-31",
                        "2020-01-01",
                        "2022-12-31",
                        "2020-01-01",
                        "2022-12-31",
                    ]
                ),
                "current_diagnosis": [2, 3, 2, 2, 2, 1],
            }
        )
        rule = CohortRule(
            name="test",
            baseline_visit_codes=("bl",),
            baseline_diagnosis_code=2,
            ad_diagnosis_code=3,
            conversion_horizon_days=1095,
            stable_followup_min_days=1005,
            stable_diagnosis_codes=(1, 2),
        )
        cohort, audit = build_cohort(frame, rule)
        labels = dict(zip(cohort["RID"], cohort["mci_conversion_label_36m"]))
        self.assertEqual(labels, {1: 1, 2: 0, 3: 0})
        self.assertEqual(audit["converters"], 1)
        self.assertEqual(audit["stable"], 2)


if __name__ == "__main__":
    unittest.main()
