"""Cohort construction and audit utilities.

The functions in this module deliberately separate diagnosis harmonization,
date cleaning, and outcome labeling so that each decision can be audited and
unit tested. Participant-level outputs are intended for local, access-controlled
use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import pandas as pd


DXCHANGE_TO_CURRENT = {
    1: 1,  # stable cognitively normal
    2: 2,  # stable MCI
    3: 3,  # stable AD
    4: 2,  # cognitively normal to MCI
    5: 3,  # MCI to AD
    6: 3,  # cognitively normal to AD
    7: 1,  # MCI to cognitively normal
    8: 2,  # AD to MCI
    9: 1,  # AD to cognitively normal
}


@dataclass(frozen=True)
class CohortRule:
    """Explicit rule for assigning 36-month outcomes."""

    name: str
    baseline_visit_codes: tuple[str, ...]
    baseline_diagnosis_code: int
    ad_diagnosis_code: int
    conversion_horizon_days: int
    stable_followup_min_days: int
    stable_diagnosis_codes: tuple[int, ...]


def read_csv_member(zip_path: Path, member: str) -> pd.DataFrame:
    """Read a CSV directly from a ZIP archive without extracting raw data."""

    with ZipFile(zip_path) as archive:
        return pd.read_csv(BytesIO(archive.read(member)), low_memory=False)


def harmonize_current_diagnosis(frame: pd.DataFrame) -> pd.Series:
    """Return harmonized current diagnosis: 1=CN, 2=MCI, 3=AD."""

    diagnosis = pd.to_numeric(frame.get("DIAGNOSIS"), errors="coerce")
    dxchange = pd.to_numeric(frame.get("DXCHANGE"), errors="coerce")
    diagnosis = diagnosis.fillna(dxchange.map(DXCHANGE_TO_CURRENT))
    legacy = pd.to_numeric(frame.get("DXCURREN"), errors="coerce")
    return diagnosis.fillna(legacy)


def clean_diagnosis_rows(
    frame: pd.DataFrame,
    minimum_valid_date: str,
    data_cutoff: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse dates, remove invalid/future records, and harmonize diagnoses."""

    required = {"RID", "VISCODE2", "EXAMDATE"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Diagnosis table is missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["exam_date"] = pd.to_datetime(out["EXAMDATE"], errors="coerce")
    out["current_diagnosis"] = harmonize_current_diagnosis(out)

    minimum = pd.Timestamp(minimum_valid_date)
    cutoff = pd.Timestamp(data_cutoff)
    audit = {
        "raw_rows": int(len(out)),
        "missing_exam_date_rows": int(out["exam_date"].isna().sum()),
        "pre_minimum_date_rows": int((out["exam_date"] < minimum).sum()),
        "future_date_rows": int((out["exam_date"] > cutoff).sum()),
        "missing_harmonized_diagnosis_rows": int(out["current_diagnosis"].isna().sum()),
    }

    valid_date = out["exam_date"].between(minimum, cutoff, inclusive="both")
    out = out.loc[valid_date & out["current_diagnosis"].notna()].copy()
    out["RID"] = pd.to_numeric(out["RID"], errors="raise").astype(int)
    out["current_diagnosis"] = out["current_diagnosis"].astype(int)
    out["visit_code"] = out["VISCODE2"].astype(str).str.strip().str.lower()
    audit["retained_rows"] = int(len(out))
    audit["retained_participants"] = int(out["RID"].nunique())
    return out, audit


def build_cohort(frame: pd.DataFrame, rule: CohortRule) -> tuple[pd.DataFrame, dict[str, int]]:
    """Construct a participant-level cohort using an explicit labeling rule."""

    records: list[dict[str, object]] = []
    baseline_candidates = 0
    excluded_no_outcome = 0

    ordered = frame.sort_values(["RID", "exam_date"])
    for rid, participant in ordered.groupby("RID", sort=False):
        baseline = participant.loc[
            participant["visit_code"].isin(rule.baseline_visit_codes)
            & participant["current_diagnosis"].eq(rule.baseline_diagnosis_code)
        ]
        if baseline.empty:
            continue

        baseline_candidates += 1
        baseline_row = baseline.iloc[0]
        baseline_date = baseline_row["exam_date"]
        future = participant.loc[participant["exam_date"] > baseline_date].copy()
        future["days_from_baseline"] = (future["exam_date"] - baseline_date).dt.days

        conversions = future.loc[
            future["current_diagnosis"].eq(rule.ad_diagnosis_code)
            & future["days_from_baseline"].le(rule.conversion_horizon_days)
        ]
        stable_followup = future.loc[
            future["current_diagnosis"].isin(rule.stable_diagnosis_codes)
            & future["days_from_baseline"].ge(rule.stable_followup_min_days)
        ]

        if not conversions.empty:
            label = 1
            outcome = "converter"
            outcome_date = conversions.iloc[0]["exam_date"]
        elif not stable_followup.empty:
            label = 0
            outcome = "stable"
            outcome_date = stable_followup.iloc[0]["exam_date"]
        else:
            excluded_no_outcome += 1
            continue

        records.append(
            {
                "RID": int(rid),
                "baseline_date": baseline_date.date().isoformat(),
                "outcome_date": outcome_date.date().isoformat(),
                "mci_conversion_label_36m": label,
                "outcome": outcome,
                "cohort_rule": rule.name,
            }
        )

    cohort = pd.DataFrame.from_records(records)
    converters = int(cohort["mci_conversion_label_36m"].sum()) if len(cohort) else 0
    audit = {
        "baseline_mci_candidates": int(baseline_candidates),
        "excluded_without_qualifying_outcome": int(excluded_no_outcome),
        "cohort_total": int(len(cohort)),
        "converters": converters,
        "stable": int(len(cohort) - converters),
    }
    return cohort, audit


def participant_coverage(cohort: pd.DataFrame, frames: Iterable[pd.DataFrame]) -> int:
    """Count cohort participants represented in every supplied frame."""

    sets = []
    for frame in frames:
        if "RID" not in frame.columns:
            raise ValueError("Coverage frame does not contain RID")
        sets.append(set(pd.to_numeric(frame["RID"], errors="coerce").dropna().astype(int)))
    if not sets:
        return 0
    return int(cohort["RID"].isin(set.intersection(*sets)).sum())
