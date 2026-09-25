#!/usr/bin/env python3
"""Build the baseline multimodal analysis table and public QC summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicalalign_ad.cohort import read_csv_member  # noqa: E402
from clinicalalign_ad.modalities import (  # noqa: E402
    merge_one_to_one,
    prepare_demographics,
    prepare_dti,
    prepare_dti_standard_mean,
    prepare_family_history,
    prepare_freesurfer,
    prepare_medical_history,
    prepare_neuro_exam,
    prepare_nfl,
    prepare_vitals,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/modalities.json")
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "data/derived/candidate_non_ad_followup.csv"
    )
    parser.add_argument(
        "--characteristics-zip",
        type=Path,
        default=ROOT / "prism-uploads/Subject_Characteristics.zip",
    )
    parser.add_argument("--nfl-zip", type=Path, default=ROOT / "prism-uploads/NFL.zip")
    parser.add_argument("--vitals", type=Path, default=ROOT / "prism-uploads/VITALS_18Sep2026.csv")
    parser.add_argument("--neuro", type=Path, default=ROOT / "prism-uploads/NEUROEXM_18Sep2026.csv")
    parser.add_argument("--medical-history", type=Path, default=ROOT / "prism-uploads/MEDHIST_18Sep2026.csv")
    parser.add_argument("--initial-health", type=Path, default=ROOT / "prism-uploads/INITHEALTH_18Sep2026.csv")
    parser.add_argument("--dti-v1", type=Path, default=ROOT / "prism-uploads/ADNI_DTIROI_V1_18Sep2026.csv")
    parser.add_argument("--dti-robust", type=Path, default=ROOT / "prism-uploads/DTIROI_ROBUSTMEAN_18Sep2026.csv")
    parser.add_argument("--dti-mean", type=Path, default=ROOT / "prism-uploads/DTIROI_MEAN_18Sep2026.csv")
    parser.add_argument("--freesurfer", type=Path, default=ROOT / "prism-uploads/UCSFFSX7_18Sep2026.csv")
    parser.add_argument("--private-output", type=Path, default=ROOT / "data/derived/multimodal_cohort.csv")
    parser.add_argument("--public-output", type=Path, default=ROOT / "results/modalities")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    windows = config["windows"]
    cohort = pd.read_csv(args.cohort)

    demographics = read_csv_member(args.characteristics_zip, "PTDEMOG.csv")
    family = prepare_family_history(
        read_csv_member(args.characteristics_zip, "FHQ.csv"),
        read_csv_member(args.characteristics_zip, "RECFHQ.csv"),
        read_csv_member(args.characteristics_zip, "FAMXHPAR.csv"),
        read_csv_member(args.characteristics_zip, "FAMXHSIB.csv"),
    )
    demo = prepare_demographics(cohort, demographics)

    vitals_raw = pd.read_csv(args.vitals, low_memory=False)
    neuro_raw = pd.read_csv(args.neuro, low_memory=False)
    medical_raw = pd.read_csv(args.medical_history, low_memory=False)
    initial_raw = pd.read_csv(args.initial_health, low_memory=False)
    clinical_window = windows["clinical"]
    vitals = prepare_vitals(cohort, vitals_raw, **clinical_window)
    neuro = prepare_neuro_exam(cohort, neuro_raw, **clinical_window)
    medical = prepare_medical_history(
        cohort, medical_raw, initial_raw, **clinical_window
    )

    nfl_raw = read_csv_member(args.nfl_zip, "ADNI_BLENNOWPLASMANFLLONG_10_03_18.csv")
    nfl = prepare_nfl(cohort, nfl_raw, **windows["nfl"])

    dti_raw = pd.read_csv(args.dti_robust, low_memory=False)
    dti_v1_raw = pd.read_csv(args.dti_v1, low_memory=False)
    dti, dti_qc_pool = prepare_dti(
        cohort,
        dti_raw,
        dti_v1_raw,
        excluded_qc_terms=config["dti_excluded_qc_terms"],
        **windows["dti"],
    )
    dti_mean_raw = pd.read_csv(args.dti_mean, low_memory=False)
    dti_standard, dti_standard_qc_pool = prepare_dti_standard_mean(
        cohort,
        dti_mean_raw,
        excluded_qc_terms=config["dti_excluded_qc_terms"],
        **windows["dti"],
    )

    freesurfer_raw = pd.read_csv(args.freesurfer, low_memory=False)
    freesurfer, freesurfer_qc_pool = prepare_freesurfer(
        cohort,
        freesurfer_raw,
        field_map=config["freesurfer_fields"],
        **windows["freesurfer"],
    )

    merged = merge_one_to_one(
        cohort, [demo, family, vitals, neuro, medical, nfl, dti, dti_standard, freesurfer]
    )
    availability = {
        "demographics": merged[["age_at_baseline", "sex", "education_years"]].notna().any(axis=1),
        "family_history": merged["family_history_any"].notna(),
        "vitals": merged.get("vitals_available", 0).fillna(0).eq(1),
        "neurological_exam": merged.get("neuro_exam_available", 0).fillna(0).eq(1),
        "medical_history": merged.get("medical_history_available", 0).fillna(0).eq(1),
        "plasma_nfl": merged.get("nfl_available", 0).fillna(0).eq(1),
        "dti": merged.get("dti_available", 0).fillna(0).eq(1),
        "dti_standard_mean": merged.get("dti_standard_mean_available", 0).fillna(0).eq(1),
        "freesurfer": merged.get("freesurfer_available", 0).fillna(0).eq(1),
    }
    for modality, flag in availability.items():
        merged[f"has_{modality}"] = flag.astype(int)

    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.private_output, index=False)

    coverage_records = []
    labels = merged["mci_conversion_label_36m"].eq(1)
    for modality, flag in availability.items():
        coverage_records.append(
            {
                "modality": modality,
                "available_n": int(flag.sum()),
                "available_percent": round(100 * flag.mean(), 1),
                "converter_n": int((flag & labels).sum()),
                "stable_n": int((flag & ~labels).sum()),
            }
        )
    pd.DataFrame(coverage_records).to_csv(
        args.public_output / "modality_coverage.csv", index=False
    )

    flag_columns = [f"has_{name}" for name in availability]
    overlap = (
        merged.groupby(flag_columns, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    overlap.to_csv(args.public_output / "modality_overlap.csv", index=False)

    protected = {
        "RID",
        "baseline_date",
        "outcome_date",
        "mci_conversion_label_36m",
        "outcome",
        "cohort_rule",
    }
    missingness = pd.DataFrame(
        [
            {
                "feature": column,
                "non_missing_n": int(merged[column].notna().sum()),
                "missing_n": int(merged[column].isna().sum()),
                "missing_percent": round(100 * merged[column].isna().mean(), 1),
            }
            for column in merged.columns
            if column not in protected
        ]
    ).sort_values(["missing_percent", "feature"], ascending=[False, True])
    missingness.to_csv(args.public_output / "feature_missingness.csv", index=False)

    dti_qc = pd.concat(
        [
            dti_raw.assign(method="robust_mean"),
            dti_mean_raw.assign(method="standard_mean"),
        ],
        ignore_index=True,
    )
    dti_qc = (
        dti_qc.assign(
            QC=dti_qc["QC"].fillna("none"),
            STATUS=dti_qc["STATUS"].fillna("missing"),
        )
        .groupby(["method", "STATUS", "QC"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    dti_qc.to_csv(args.public_output / "dti_qc_summary.csv", index=False)

    fs_qc = (
        freesurfer_raw.assign(
            OVERALLQC=freesurfer_raw["OVERALLQC"].fillna("not recorded"),
            STATUS=freesurfer_raw["STATUS"].fillna("missing"),
        )
        .groupby(["STATUS", "OVERALLQC"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    fs_qc.to_csv(args.public_output / "freesurfer_qc_summary.csv", index=False)

    input_paths = [
        args.characteristics_zip,
        args.nfl_zip,
        args.vitals,
        args.neuro,
        args.medical_history,
        args.initial_health,
        args.dti_v1,
        args.dti_robust,
        args.dti_mean,
        args.freesurfer,
    ]
    audit = {
        "config": config,
        "cohort_n": int(len(merged)),
        "duplicate_rid_n": int(merged["RID"].duplicated().sum()),
        "post_outcome_selected_measurements_n": int(
            sum(
                (
                    pd.to_datetime(merged[column], errors="coerce")
                    >= pd.to_datetime(merged["outcome_date"], errors="coerce")
                ).sum()
                for column in merged.columns
                if column.endswith("_date") and column not in {"baseline_date", "outcome_date"}
            )
        ),
        "eligible_dti_scan_pool_n": int(len(dti_qc_pool)),
        "eligible_dti_standard_mean_scan_pool_n": int(len(dti_standard_qc_pool)),
        "eligible_freesurfer_scan_pool_n": int(len(freesurfer_qc_pool)),
        "input_manifest": {path.name: sha256(path) for path in input_paths},
    }
    (args.public_output / "modality_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(coverage_records).to_string(index=False))
    print(json.dumps({key: audit[key] for key in ["cohort_n", "duplicate_rid_n", "post_outcome_selected_measurements_n"]}, indent=2))


if __name__ == "__main__":
    main()