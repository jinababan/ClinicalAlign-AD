from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicalalign_ad.modalities import (  # noqa: E402
    prepare_dti,
    prepare_vitals,
    select_nearest_baseline,
)


class ModalityTests(unittest.TestCase):
    def setUp(self):
        self.cohort = pd.DataFrame(
            {
                "RID": [1],
                "baseline_date": ["2020-01-10"],
                "outcome_date": ["2020-03-01"],
            }
        )

    def test_nearest_prefers_prebaseline_on_equal_distance(self):
        observations = pd.DataFrame(
            {"RID": [1, 1], "EXAMDATE": ["2020-01-05", "2020-01-15"], "value": [1, 2]}
        )
        selected = select_nearest_baseline(self.cohort, observations, "EXAMDATE", 30, 30)
        self.assertEqual(selected.iloc[0]["value"], 1)
        self.assertEqual(selected.iloc[0]["days_from_baseline"], -5)

    def test_observations_at_or_after_outcome_are_excluded(self):
        observations = pd.DataFrame(
            {"RID": [1, 1], "EXAMDATE": ["2020-02-20", "2020-03-01"], "value": [1, 2]}
        )
        selected = select_nearest_baseline(self.cohort, observations, "EXAMDATE", 365, 365)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["value"], 1)

    def test_dti_excludes_severe_qc_and_uses_complete_v1(self):
        robust = pd.DataFrame(
            {
                "RID": [1],
                "EXAMDATE": ["2020-01-10"],
                "STATUS": ["complete"],
                "QC": ["severe artifacts"],
                "VERSION": [3],
                "FA_A": [0.5],
                "MD_A": [0.001],
                "AD_A": [0.0012],
                "RD_A": [0.0008],
            }
        )
        version1 = pd.DataFrame(
            {
                "RID": [1],
                "EXAMDATE": ["2020-01-11"],
                "STATUS": ["Complete"],
                "VERSION": [1],
                "FA_A": [0.4],
                "MD_A": [0.0011],
                "AD_A": [0.0013],
                "RD_A": [0.0009],
            }
        )
        selected, _ = prepare_dti(
            self.cohort,
            robust,
            version1,
            365,
            90,
            excluded_qc_terms=["severe artifacts"],
        )
        self.assertEqual(selected.iloc[0]["dti_source"], "mean_v1")
        self.assertAlmostEqual(selected.iloc[0]["dti_fa_mean"], 0.4)

    def test_vitals_can_use_screening_height_for_baseline_bmi(self):
        vitals = pd.DataFrame(
            {
                "RID": [1, 1],
                "VISDATE": ["2020-01-01", "2020-01-10"],
                "VSWEIGHT": [None, 176],
                "VSWTUNIT": [None, 1],
                "VSHEIGHT": [70, -4],
                "VSHTUNIT": [1, -4],
                "VSBPSYS": [120, 122],
                "VSBPDIA": [70, 72],
                "VSPULSE": [60, 62],
                "HAS_QC_ERROR": [0, 0],
            }
        )
        selected = prepare_vitals(self.cohort, vitals, 180, 90)
        self.assertAlmostEqual(selected.iloc[0]["height_cm"], 177.8)
        self.assertAlmostEqual(selected.iloc[0]["weight_kg"], 79.83225712)
        self.assertAlmostEqual(selected.iloc[0]["bmi"], 25.2531, places=3)


if __name__ == "__main__":
    unittest.main()