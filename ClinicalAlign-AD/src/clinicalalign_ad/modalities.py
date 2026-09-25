"""Baseline multimodal feature assembly with explicit temporal and QC rules.

All functions operate on local participant-level data. Public outputs should be
limited to aggregate audit tables produced by ``scripts/02_build_multimodal_dataset.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


NEURO_EXAM_FIELDS = (
    "NXVISUAL",
    "NXAUDITO",
    "NXTREMOR",
    "NXCONSCI",
    "NXNERVE",
    "NXMOTOR",
    "NXFINGER",
    "NXHEEL",
    "NXSENSOR",
    "NXTENDON",
    "NXPLANTA",
    "NXGAIT",
    "NXOTHER",
)

MEDICAL_HISTORY_FIELDS = (
    "MHPSYCH",
    "MH2NEURL",
    "MH3HEAD",
    "MH4CARD",
    "MH5RESP",
    "MH6HEPAT",
    "MH7DERM",
    "MH8MUSCL",
    "MH9ENDO",
    "MH10GAST",
    "MH11HEMA",
    "MH12RENA",
    "MH13ALLE",
    "MH14ALCH",
    "MH15DRUG",
    "MH16SMOK",
    "MH17MALI",
    "MH18SURG",
    "MH19OTHR",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _clean_rid(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["RID"] = pd.to_numeric(out["RID"], errors="coerce")
    out = out.loc[out["RID"].notna()].copy()
    out["RID"] = out["RID"].astype(int)
    return out


def select_nearest_baseline(
    cohort: pd.DataFrame,
    observations: pd.DataFrame,
    date_column: str,
    days_before: int,
    days_after: int,
    priority_column: str | None = None,
) -> pd.DataFrame:
    """Select one observation per participant without crossing the outcome date.

    Absolute distance from baseline is the primary criterion. Equidistant
    pre-baseline observations are preferred to post-baseline observations, then
    the optional numeric source priority is used as a deterministic tie-breaker.
    """

    _require_columns(cohort, ["RID", "baseline_date", "outcome_date"], "cohort")
    _require_columns(observations, ["RID", date_column], "observations")

    base = _clean_rid(cohort[["RID", "baseline_date", "outcome_date"]])
    base["baseline_date"] = pd.to_datetime(base["baseline_date"], errors="coerce")
    base["outcome_date"] = pd.to_datetime(base["outcome_date"], errors="coerce")

    obs = _clean_rid(observations)
    obs["observation_date"] = pd.to_datetime(obs[date_column], errors="coerce")
    obs = obs.loc[obs["observation_date"].notna()].copy()
    joined = obs.merge(base, on="RID", how="inner", validate="many_to_one")
    joined["days_from_baseline"] = (
        joined["observation_date"] - joined["baseline_date"]
    ).dt.days

    in_window = joined["days_from_baseline"].between(
        -int(days_before), int(days_after), inclusive="both"
    )
    before_outcome = joined["outcome_date"].isna() | (
        joined["observation_date"] < joined["outcome_date"]
    )
    joined = joined.loc[in_window & before_outcome].copy()
    if joined.empty:
        return joined

    joined["_absolute_days"] = joined["days_from_baseline"].abs()
    joined["_post_baseline"] = joined["days_from_baseline"].gt(0)
    if priority_column is None:
        joined["_source_priority"] = 0
    else:
        joined["_source_priority"] = pd.to_numeric(
            joined[priority_column], errors="coerce"
        ).fillna(999)

    joined = joined.sort_values(
        [
            "RID",
            "_absolute_days",
            "_post_baseline",
            "_source_priority",
            "observation_date",
        ],
        ascending=[True, True, True, True, False],
        kind="mergesort",
    )
    return joined.drop_duplicates("RID", keep="first").drop(
        columns=["_absolute_days", "_post_baseline", "_source_priority"]
    )


def prepare_demographics(cohort: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """Create one static demographic record per cohort participant."""

    _require_columns(
        demographics,
        ["RID", "PTGENDER", "PTDOBMM", "PTDOBYY", "PTEDUCAT", "VISCODE2"],
        "PTDEMOG",
    )
    demo = _clean_rid(demographics)
    for field in ["PTGENDER", "PTDOBMM", "PTDOBYY", "PTEDUCAT"]:
        demo[field] = pd.to_numeric(demo[field], errors="coerce")
        demo.loc[demo[field].lt(0), field] = np.nan

    demo["_completeness"] = demo[
        ["PTGENDER", "PTDOBMM", "PTDOBYY", "PTEDUCAT"]
    ].notna().sum(axis=1)
    visit = demo["VISCODE2"].astype(str).str.lower()
    demo["_visit_priority"] = np.select(
        [visit.eq("bl"), visit.isin(["sc", "f", "init"])], [0, 1], default=2
    )
    demo["_record_date"] = pd.to_datetime(demo.get("USERDATE"), errors="coerce")
    demo = demo.sort_values(
        ["RID", "_completeness", "_visit_priority", "_record_date"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).drop_duplicates("RID", keep="first")

    base = cohort[["RID", "baseline_date"]].copy()
    base["baseline_date"] = pd.to_datetime(base["baseline_date"], errors="coerce")
    selected = base.merge(demo, on="RID", how="left", validate="one_to_one")
    birthday_not_reached = selected["baseline_date"].dt.month.lt(selected["PTDOBMM"])
    selected["age_at_baseline"] = (
        selected["baseline_date"].dt.year
        - selected["PTDOBYY"]
        - birthday_not_reached.fillna(False).astype(int)
    )
    selected.loc[~selected["age_at_baseline"].between(40, 110), "age_at_baseline"] = np.nan
    selected["sex"] = selected["PTGENDER"].map({1: "male", 2: "female"})
    selected["education_years"] = selected["PTEDUCAT"].where(
        selected["PTEDUCAT"].between(0, 30)
    )
    return selected[["RID", "age_at_baseline", "sex", "education_years"]]


def _row_binary_summary(frame: pd.DataFrame, fields: Iterable[str]) -> pd.Series:
    available = [field for field in fields if field in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    yes = values.eq(1).any(axis=1)
    known = values.isin([0, 1]).any(axis=1)
    return pd.Series(np.where(yes, 1.0, np.where(known, 0.0, np.nan)), index=frame.index)


def prepare_family_history(
    fhq: pd.DataFrame,
    recfhq: pd.DataFrame,
    parents: pd.DataFrame,
    siblings: pd.DataFrame,
) -> pd.DataFrame:
    """Harmonize legacy and ADNI3/4 parent/sibling family-history tables."""

    specifications = [
        (fhq, ["FHQMOM", "FHQDAD"], ["FHQMOMAD", "FHQDADAD"], "FHQ"),
        (recfhq, ["FHQSIB"], ["FHQSIBAD"], "RECFHQ"),
        (parents, ["MOTHDEM", "FATHDEM"], ["MOTHAD", "FATHAD"], "FAMXHPAR"),
        (siblings, ["SIBDEMENT"], ["SIBAD"], "FAMXHSIB"),
    ]
    records = []
    for frame, dementia_fields, ad_fields, source in specifications:
        _require_columns(frame, ["RID"], source)
        part = _clean_rid(frame)
        part["family_history_dementia"] = _row_binary_summary(part, dementia_fields)
        part["family_history_ad"] = _row_binary_summary(part, ad_fields)
        part["family_history_source"] = source
        records.append(
            part[["RID", "family_history_dementia", "family_history_ad", "family_history_source"]]
        )

    combined = pd.concat(records, ignore_index=True)

    def reduce_binary(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.eq(1).any():
            return 1.0
        if numeric.eq(0).any():
            return 0.0
        return np.nan

    summary = combined.groupby("RID", as_index=False).agg(
        family_history_dementia=("family_history_dementia", reduce_binary),
        family_history_ad=("family_history_ad", reduce_binary),
        family_history_source_count=("family_history_source", "nunique"),
    )
    summary["family_history_any"] = summary[
        ["family_history_dementia", "family_history_ad"]
    ].max(axis=1, skipna=True)
    summary.loc[
        summary[["family_history_dementia", "family_history_ad"]].isna().all(axis=1),
        "family_history_any",
    ] = np.nan
    return summary


def _exclude_qc_errors(frame: pd.DataFrame) -> pd.DataFrame:
    if "HAS_QC_ERROR" not in frame.columns:
        return frame.copy()
    qc_error = pd.to_numeric(frame["HAS_QC_ERROR"], errors="coerce").eq(1)
    return frame.loc[~qc_error].copy()


def prepare_vitals(
    cohort: pd.DataFrame, vitals: pd.DataFrame, days_before: int, days_after: int
) -> pd.DataFrame:
    """Select baseline vital signs and harmonize units."""

    fields = [
        "RID",
        "VISDATE",
        "VSWEIGHT",
        "VSWTUNIT",
        "VSHEIGHT",
        "VSHTUNIT",
        "VSBPSYS",
        "VSBPDIA",
        "VSPULSE",
    ]
    _require_columns(vitals, fields, "VITALS")
    eligible = _exclude_qc_errors(vitals)
    selected = select_nearest_baseline(
        cohort, eligible, "VISDATE", days_before, days_after
    )
    if selected.empty:
        return pd.DataFrame(columns=["RID", "vitals_available"])

    valid_weight = eligible.copy()
    valid_weight["_weight"] = pd.to_numeric(valid_weight["VSWEIGHT"], errors="coerce")
    valid_weight["_weight_unit"] = pd.to_numeric(valid_weight["VSWTUNIT"], errors="coerce")
    valid_weight = valid_weight.loc[
        valid_weight["_weight"].gt(0) & valid_weight["_weight_unit"].isin([1, 2])
    ].copy()
    weight_selected = select_nearest_baseline(
        cohort, valid_weight, "VISDATE", days_before, days_after
    )
    if not weight_selected.empty:
        weight_selected["weight_kg"] = np.where(
            weight_selected["_weight_unit"].eq(1),
            weight_selected["_weight"] * 0.45359237,
            weight_selected["_weight"],
        )
        selected = selected.drop(columns=["weight_kg"], errors="ignore").merge(
            weight_selected[["RID", "weight_kg"]], on="RID", how="left", validate="one_to_one"
        )
    else:
        selected["weight_kg"] = np.nan

    valid_height = eligible.copy()
    valid_height["_height"] = pd.to_numeric(valid_height["VSHEIGHT"], errors="coerce")
    valid_height["_height_unit"] = pd.to_numeric(valid_height["VSHTUNIT"], errors="coerce")
    valid_height = valid_height.loc[
        valid_height["_height"].gt(0) & valid_height["_height_unit"].isin([1, 2])
    ].copy()
    height_selected = select_nearest_baseline(
        cohort, valid_height, "VISDATE", days_before, days_after
    )
    if not height_selected.empty:
        height_selected["height_cm"] = np.where(
            height_selected["_height_unit"].eq(1),
            height_selected["_height"] * 2.54,
            height_selected["_height"],
        )
        selected = selected.drop(columns=["height_cm"], errors="ignore").merge(
            height_selected[["RID", "height_cm"]], on="RID", how="left", validate="one_to_one"
        )
    else:
        selected["height_cm"] = np.nan

    selected.loc[~selected["weight_kg"].between(25, 300), "weight_kg"] = np.nan
    selected.loc[~selected["height_cm"].between(100, 230), "height_cm"] = np.nan
    selected["bmi"] = selected["weight_kg"] / (selected["height_cm"] / 100.0) ** 2
    selected.loc[~selected["bmi"].between(10, 80), "bmi"] = np.nan
    selected["systolic_bp"] = pd.to_numeric(selected["VSBPSYS"], errors="coerce")
    selected["diastolic_bp"] = pd.to_numeric(selected["VSBPDIA"], errors="coerce")
    selected["pulse_bpm"] = pd.to_numeric(selected["VSPULSE"], errors="coerce")
    selected.loc[~selected["systolic_bp"].between(60, 260), "systolic_bp"] = np.nan
    selected.loc[~selected["diastolic_bp"].between(30, 150), "diastolic_bp"] = np.nan
    selected.loc[~selected["pulse_bpm"].between(30, 220), "pulse_bpm"] = np.nan
    selected["vitals_available"] = 1
    return selected[
        [
            "RID",
            "observation_date",
            "days_from_baseline",
            "weight_kg",
            "height_cm",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "pulse_bpm",
            "vitals_available",
        ]
    ].rename(
        columns={
            "observation_date": "vitals_date",
            "days_from_baseline": "vitals_days_from_baseline",
        }
    )


def prepare_neuro_exam(
    cohort: pd.DataFrame, neuro: pd.DataFrame, days_before: int, days_after: int
) -> pd.DataFrame:
    """Select the closest baseline neurological examination."""

    _require_columns(neuro, ["RID", "VISDATE", *NEURO_EXAM_FIELDS], "NEUROEXM")
    selected = select_nearest_baseline(
        cohort, _exclude_qc_errors(neuro), "VISDATE", days_before, days_after
    )
    if selected.empty:
        return pd.DataFrame(columns=["RID", "neuro_exam_available"])
    values = selected[list(NEURO_EXAM_FIELDS)].apply(pd.to_numeric, errors="coerce")
    selected["neuro_abnormal_count"] = values.eq(2).sum(axis=1)
    selected["neuro_exam_ineligible"] = pd.to_numeric(
        selected.get("NXABNORM"), errors="coerce"
    ).eq(2).astype(int)
    selected["neuro_exam_available"] = 1
    return selected[
        [
            "RID",
            "observation_date",
            "days_from_baseline",
            "neuro_abnormal_count",
            "neuro_exam_ineligible",
            "neuro_exam_available",
        ]
    ].rename(
        columns={
            "observation_date": "neuro_exam_date",
            "days_from_baseline": "neuro_exam_days_from_baseline",
        }
    )


def prepare_medical_history(
    cohort: pd.DataFrame,
    medical_history: pd.DataFrame,
    initial_health: pd.DataFrame,
    days_before: int,
    days_after: int,
) -> pd.DataFrame:
    """Combine legacy checklist and ADNI3/4 condition-list summaries."""

    _require_columns(medical_history, ["RID", "VISDATE", *MEDICAL_HISTORY_FIELDS], "MEDHIST")
    old = select_nearest_baseline(
        cohort, medical_history, "VISDATE", days_before, days_after
    )
    if not old.empty:
        values = old[list(MEDICAL_HISTORY_FIELDS)].apply(pd.to_numeric, errors="coerce")
        old["medical_history_condition_count"] = values.eq(1).sum(axis=1)
        old["history_neurologic"] = pd.to_numeric(old["MH2NEURL"], errors="coerce")
        old["history_cardiovascular"] = pd.to_numeric(old["MH4CARD"], errors="coerce")
        old["history_endocrine_metabolic"] = pd.to_numeric(old["MH9ENDO"], errors="coerce")
        old["history_smoking"] = pd.to_numeric(old["MH16SMOK"], errors="coerce")
        old["medical_history_source"] = "MEDHIST"

    _require_columns(
        initial_health,
        ["RID", "VISDATE", "IHSYMPTOM", "IHPRESENT", "IHONGOING"],
        "INITHEALTH",
    )
    initial = _exclude_qc_errors(_clean_rid(initial_health))
    initial["VISDATE"] = pd.to_datetime(initial["VISDATE"], errors="coerce")
    present = pd.to_numeric(initial["IHPRESENT"], errors="coerce").eq(1) | pd.to_numeric(
        initial["IHONGOING"], errors="coerce"
    ).eq(1)
    initial["_present"] = present.astype(int)
    initial["_category"] = pd.to_numeric(initial["IHSYMPTOM"], errors="coerce")

    grouped_records = []
    for (rid, visit_date), group in initial.groupby(["RID", "VISDATE"], dropna=False):
        active = group.loc[group["_present"].eq(1), "_category"]
        grouped_records.append(
            {
                "RID": rid,
                "VISDATE": visit_date,
                "medical_history_condition_count": int(group["_present"].sum()),
                "history_neurologic": int(active.eq(2).any()),
                "history_cardiovascular": int(active.eq(4).any()),
                "history_endocrine_metabolic": int(active.eq(9).any()),
                "history_smoking": int(active.eq(14).any()),
                "medical_history_source": "INITHEALTH",
            }
        )
    initial_summary = pd.DataFrame.from_records(grouped_records)
    if initial_summary.empty:
        new = initial_summary
    else:
        new = select_nearest_baseline(
            cohort, initial_summary, "VISDATE", days_before, days_after
        )

    common = [
        "RID",
        "observation_date",
        "days_from_baseline",
        "medical_history_condition_count",
        "history_neurologic",
        "history_cardiovascular",
        "history_endocrine_metabolic",
        "history_smoking",
        "medical_history_source",
    ]
    pieces = []
    if not old.empty:
        old["_source_priority"] = 0
        pieces.append(old[common + ["_source_priority"]])
    if not new.empty:
        new["_source_priority"] = 1
        pieces.append(new[common + ["_source_priority"]])
    if not pieces:
        return pd.DataFrame(columns=["RID", "medical_history_available"])

    combined = pd.concat(pieces, ignore_index=True)
    combined["_absolute_days"] = combined["days_from_baseline"].abs()
    combined = combined.sort_values(
        ["RID", "_absolute_days", "_source_priority"], kind="mergesort"
    ).drop_duplicates("RID", keep="first")
    combined["medical_history_available"] = 1
    return combined[
        [
            "RID",
            "observation_date",
            "days_from_baseline",
            "medical_history_condition_count",
            "history_neurologic",
            "history_cardiovascular",
            "history_endocrine_metabolic",
            "history_smoking",
            "medical_history_source",
            "medical_history_available",
        ]
    ].rename(
        columns={
            "observation_date": "medical_history_date",
            "days_from_baseline": "medical_history_days_from_baseline",
        }
    )


def prepare_nfl(
    cohort: pd.DataFrame, nfl: pd.DataFrame, days_before: int, days_after: int
) -> pd.DataFrame:
    """Select the closest valid plasma NfL measurement."""

    _require_columns(nfl, ["RID", "EXAMDATE", "PLASMA_NFL"], "plasma NfL")
    work = nfl.copy()
    work["PLASMA_NFL"] = pd.to_numeric(work["PLASMA_NFL"], errors="coerce")
    work = work.loc[work["PLASMA_NFL"].gt(0)].copy()
    selected = select_nearest_baseline(
        cohort, work, "EXAMDATE", days_before, days_after
    )
    if selected.empty:
        return pd.DataFrame(columns=["RID", "nfl_available"])
    selected["nfl_available"] = 1
    return selected[
        ["RID", "observation_date", "days_from_baseline", "PLASMA_NFL", "nfl_available"]
    ].rename(
        columns={
            "observation_date": "nfl_date",
            "days_from_baseline": "nfl_days_from_baseline",
            "PLASMA_NFL": "plasma_nfl",
        }
    )


def _dti_scan_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for metric in ["FA", "MD", "AD", "RD"]:
        columns = [column for column in out.columns if column.startswith(f"{metric}_")]
        values = out[columns].apply(pd.to_numeric, errors="coerce")
        out[f"dti_{metric.lower()}_mean"] = values.mean(axis=1, skipna=True)
        out[f"dti_{metric.lower()}_std"] = values.std(axis=1, skipna=True)
        out[f"dti_{metric.lower()}_n"] = values.notna().sum(axis=1)
    return out


def prepare_dti(
    cohort: pd.DataFrame,
    robust_mean: pd.DataFrame,
    version1: pd.DataFrame,
    days_before: int,
    days_after: int,
    excluded_qc_terms: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine robust V2/V3 DTI summaries with complete legacy V1 scans."""

    _require_columns(robust_mean, ["RID", "EXAMDATE", "STATUS", "QC", "VERSION"], "DTI robust mean")
    _require_columns(version1, ["RID", "EXAMDATE", "STATUS", "VERSION"], "DTI V1")

    robust = robust_mean.copy()
    robust_status = robust["STATUS"].astype(str).str.strip().str.lower()
    qc_text = robust["QC"].fillna("").astype(str).str.strip().str.lower()
    bad_qc = pd.Series(False, index=robust.index)
    for term in excluded_qc_terms:
        bad_qc |= qc_text.str.contains(str(term).lower(), regex=False)
    robust = robust.loc[robust_status.eq("complete") & ~bad_qc].copy()
    robust["dti_source"] = "robust_mean_v2_v3"
    robust["dti_source_priority"] = 0
    robust["dti_qc_warning"] = robust["QC"].notna().astype(int)
    robust["dti_qc_text"] = robust["QC"]

    v1 = version1.loc[
        version1["STATUS"].astype(str).str.strip().str.lower().eq("complete")
    ].copy()
    v1["dti_source"] = "mean_v1"
    v1["dti_source_priority"] = 1
    v1["dti_qc_warning"] = 0
    v1["dti_qc_text"] = np.nan

    combined = pd.concat([_dti_scan_summaries(robust), _dti_scan_summaries(v1)], ignore_index=True)
    selected = select_nearest_baseline(
        cohort,
        combined,
        "EXAMDATE",
        days_before,
        days_after,
        priority_column="dti_source_priority",
    )
    if selected.empty:
        return pd.DataFrame(columns=["RID", "dti_available"]), combined
    selected["dti_available"] = 1
    output = [
        "RID",
        "observation_date",
        "days_from_baseline",
        "dti_source",
        "VERSION",
        "MANUFACTURER",
        "DISTORTION_CORRECTION",
        "dti_qc_warning",
        "dti_qc_text",
        "dti_available",
    ]
    for metric in ["fa", "md", "ad", "rd"]:
        output.extend([f"dti_{metric}_mean", f"dti_{metric}_std", f"dti_{metric}_n"])
    for column in output:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[output].rename(
        columns={
            "observation_date": "dti_date",
            "days_from_baseline": "dti_days_from_baseline",
            "VERSION": "dti_version",
            "MANUFACTURER": "dti_manufacturer",
            "DISTORTION_CORRECTION": "dti_distortion_correction",
        }
    ), combined


def prepare_dti_standard_mean(
    cohort: pd.DataFrame,
    standard_mean: pd.DataFrame,
    days_before: int,
    days_after: int,
    excluded_qc_terms: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare V2/V3 standard-mean DTI summaries for sensitivity analysis."""

    _require_columns(
        standard_mean,
        ["RID", "EXAMDATE", "STATUS", "QC", "VERSION"],
        "DTI standard mean",
    )
    work = standard_mean.copy()
    status = work["STATUS"].astype(str).str.strip().str.lower()
    qc_text = work["QC"].fillna("").astype(str).str.strip().str.lower()
    bad_qc = pd.Series(False, index=work.index)
    for term in excluded_qc_terms:
        bad_qc |= qc_text.str.contains(str(term).lower(), regex=False)
    work = work.loc[status.eq("complete") & ~bad_qc].copy()
    work = _dti_scan_summaries(work)
    selected = select_nearest_baseline(
        cohort, work, "EXAMDATE", days_before, days_after
    )
    if selected.empty:
        return pd.DataFrame(columns=["RID", "dti_standard_mean_available"]), work

    selected["dti_standard_mean_available"] = 1
    selected["dti_standard_qc_warning"] = selected["QC"].notna().astype(int)
    output = [
        "RID",
        "observation_date",
        "days_from_baseline",
        "VERSION",
        "MANUFACTURER",
        "DISTORTION_CORRECTION",
        "dti_standard_qc_warning",
        "QC",
        "dti_standard_mean_available",
    ]
    for metric in ["fa", "md", "ad", "rd"]:
        output.extend([f"dti_{metric}_mean", f"dti_{metric}_std", f"dti_{metric}_n"])
    for column in output:
        if column not in selected.columns:
            selected[column] = np.nan
    renamed = selected[output].rename(
        columns={
            "observation_date": "dti_standard_date",
            "days_from_baseline": "dti_standard_days_from_baseline",
            "VERSION": "dti_standard_version",
            "MANUFACTURER": "dti_standard_manufacturer",
            "DISTORTION_CORRECTION": "dti_standard_distortion_correction",
            "QC": "dti_standard_qc_text",
            **{
                f"dti_{metric}_{stat}": f"dti_standard_{metric}_{stat}"
                for metric in ["fa", "md", "ad", "rd"]
                for stat in ["mean", "std", "n"]
            },
        }
    )
    return renamed, work


def prepare_freesurfer(
    cohort: pd.DataFrame,
    freesurfer: pd.DataFrame,
    days_before: int,
    days_after: int,
    field_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select QC-eligible FreeSurfer 7 baseline features."""

    required = ["RID", "EXAMDATE", "STATUS", "FSVER", "OVERALLQC", *field_map.values()]
    _require_columns(freesurfer, required, "UCSFFSX7")
    work = freesurfer.copy()
    overall = work["OVERALLQC"].fillna("").astype(str).str.strip().str.lower()
    work = work.loc[~overall.eq("fail")].copy()
    overall = work["OVERALLQC"].fillna("").astype(str).str.strip().str.lower()

    for output_name, source_name in field_map.items():
        work[output_name] = pd.to_numeric(work[source_name], errors="coerce")

    hippo_fail = work.get("HIPPOQC", pd.Series(index=work.index, dtype=object)).fillna("").astype(str).str.lower().eq("fail")
    temporal_fail = work.get("TEMPQC", pd.Series(index=work.index, dtype=object)).fillna("").astype(str).str.lower().eq("fail")
    ventricle_fail = work.get("VENTQC", pd.Series(index=work.index, dtype=object)).fillna("").astype(str).str.lower().eq("fail")

    hippo_affected = [
        name
        for name in field_map
        if any(term in name for term in ["amygdala", "entorhinal", "fusiform"])
    ]
    temporal_affected = [
        name
        for name in field_map
        if any(term in name for term in ["fusiform", "middle_temporal", "inferior_temporal"])
    ]
    ventricle_affected = [name for name in field_map if "ventricle" in name]
    work.loc[hippo_fail, hippo_affected] = np.nan
    work.loc[temporal_fail, temporal_affected] = np.nan
    work.loc[ventricle_fail, ventricle_affected] = np.nan

    hippocampus_only = overall.eq("hippocampus only")
    non_hippocampal = [name for name in field_map if "hippocampus" not in name]
    work.loc[hippocampus_only, non_hippocampal] = np.nan

    selected = select_nearest_baseline(
        cohort, work, "EXAMDATE", days_before, days_after
    )
    if selected.empty:
        return pd.DataFrame(columns=["RID", "freesurfer_available"]), work

    selected["fs_hippocampus_volume"] = selected[
        ["fs_left_hippocampus_volume", "fs_right_hippocampus_volume"]
    ].sum(axis=1, min_count=1)
    selected["fs_hippocampus_to_icv"] = selected["fs_hippocampus_volume"] / selected["fs_icv"]
    selected["fs_amygdala_volume"] = selected[
        ["fs_left_amygdala_volume", "fs_right_amygdala_volume"]
    ].sum(axis=1, min_count=1)
    selected["fs_lateral_ventricle_volume"] = selected[
        ["fs_left_lateral_ventricle_volume", "fs_right_lateral_ventricle_volume"]
    ].sum(axis=1, min_count=1)
    for region in ["entorhinal", "fusiform", "middle_temporal", "inferior_temporal"]:
        selected[f"fs_{region}_thickness"] = selected[
            [f"fs_left_{region}_thickness", f"fs_right_{region}_thickness"]
        ].mean(axis=1, skipna=True)
    selected["freesurfer_available"] = 1

    metadata = [
        "RID",
        "observation_date",
        "days_from_baseline",
        "FSVER",
        "FIELD_STRENGTH",
        "STATUS",
        "OVERALLQC",
        "TEMPQC",
        "VENTQC",
        "HIPPOQC",
        "freesurfer_available",
    ]
    features = list(field_map) + [
        "fs_hippocampus_volume",
        "fs_hippocampus_to_icv",
        "fs_amygdala_volume",
        "fs_lateral_ventricle_volume",
        "fs_entorhinal_thickness",
        "fs_fusiform_thickness",
        "fs_middle_temporal_thickness",
        "fs_inferior_temporal_thickness",
    ]
    for column in metadata:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[metadata + features].rename(
        columns={
            "observation_date": "freesurfer_date",
            "days_from_baseline": "freesurfer_days_from_baseline",
            "FSVER": "freesurfer_version",
            "FIELD_STRENGTH": "freesurfer_field_strength",
            "STATUS": "freesurfer_status",
            "OVERALLQC": "freesurfer_overall_qc",
            "TEMPQC": "freesurfer_temporal_qc",
            "VENTQC": "freesurfer_ventricle_qc",
            "HIPPOQC": "freesurfer_hippocampus_qc",
        }
    ), work


def merge_one_to_one(cohort: pd.DataFrame, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Left-join participant-level feature frames with one-to-one validation."""

    result = cohort.copy()
    for frame in frames:
        if frame.empty:
            continue
        if frame["RID"].duplicated().any():
            raise ValueError("Feature frame contains duplicate RID values")
        result = result.merge(frame, on="RID", how="left", validate="one_to_one")
    return result