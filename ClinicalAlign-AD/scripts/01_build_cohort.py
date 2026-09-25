#!/usr/bin/env python3
"""Build and audit candidate 36-month MCI-to-AD cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicalalign_ad.cohort import (  # noqa: E402
    CohortRule,
    build_cohort,
    clean_diagnosis_rows,
    participant_coverage,
    read_csv_member,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    parser.add_argument("--diagnosis-zip", type=Path, default=ROOT / "prism-uploads/Diagnosis.zip")
    parser.add_argument(
        "--characteristics-zip",
        type=Path,
        default=ROOT / "prism-uploads/Subject_Characteristics.zip",
    )
    parser.add_argument("--private-output", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--public-output", type=Path, default=ROOT / "results/cohort")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    diagnosis_raw = read_csv_member(args.diagnosis_zip, "DXSUM_PDXCONV_ADNIALL.csv")
    diagnosis, cleaning_audit = clean_diagnosis_rows(
        diagnosis_raw,
        minimum_valid_date=config["minimum_valid_date"],
        data_cutoff=config["data_cutoff"],
    )

    common = dict(
        baseline_visit_codes=tuple(config["baseline_visit_codes"]),
        baseline_diagnosis_code=int(config["baseline_diagnosis_code"]),
        ad_diagnosis_code=int(config["ad_diagnosis_code"]),
        conversion_horizon_days=int(config["conversion_horizon_days"]),
        stable_followup_min_days=int(config["stable_followup_min_days"]),
    )
    rules = [
        CohortRule(
            name="candidate_non_ad_followup",
            stable_diagnosis_codes=tuple(config["candidate_stable_diagnosis_codes"]),
            **common,
        ),
        CohortRule(
            name="strict_stable_mci",
            stable_diagnosis_codes=tuple(config["strict_stable_diagnosis_codes"]),
            **common,
        ),
    ]

    args.private_output.mkdir(parents=True, exist_ok=True)
    args.public_output.mkdir(parents=True, exist_ok=True)

    comparisons = []
    cohorts = {}
    audits = {}
    for rule in rules:
        cohort, audit = build_cohort(diagnosis, rule)
        cohorts[rule.name] = cohort
        audits[rule.name] = audit
        comparisons.append({"rule": rule.name, **audit})
        cohort.to_csv(args.private_output / f"{rule.name}.csv", index=False)

    demographics = read_csv_member(args.characteristics_zip, "PTDEMOG.csv")
    family_frames = [
        read_csv_member(args.characteristics_zip, member)
        for member in ["FHQ.csv", "RECFHQ.csv", "FAMXHPAR.csv", "FAMXHSIB.csv"]
    ]
    primary = cohorts["candidate_non_ad_followup"]
    family_union = pd.concat([frame[["RID"]] for frame in family_frames], ignore_index=True)
    coverage = {
        "demographics": participant_coverage(primary, [demographics]),
        "any_family_history_table": participant_coverage(primary, [family_union]),
    }

    comparison = pd.DataFrame(comparisons)
    comparison["delta_total_vs_reported"] = comparison["cohort_total"] - int(config["reported_total"])
    comparison["delta_converters_vs_reported"] = (
        comparison["converters"] - int(config["reported_converters"])
    )
    comparison["delta_stable_vs_reported"] = comparison["stable"] - int(config["reported_stable"])
    comparison.to_csv(args.public_output / "rule_comparison.csv", index=False)

    flow = pd.DataFrame(
        [
            {"stage": "raw diagnosis rows", "n": cleaning_audit["raw_rows"]},
            {"stage": "valid dated diagnosis rows", "n": cleaning_audit["retained_rows"]},
            {
                "stage": "participants with valid diagnoses",
                "n": cleaning_audit["retained_participants"],
            },
            {
                "stage": "baseline MCI candidates",
                "n": audits["candidate_non_ad_followup"]["baseline_mci_candidates"],
            },
            {
                "stage": "candidate labeled cohort",
                "n": audits["candidate_non_ad_followup"]["cohort_total"],
            },
        ]
    )
    flow.to_csv(args.public_output / "cohort_flow.csv", index=False)

    public_audit = {
        "config": config,
        "input_manifest": {
            "Diagnosis.zip": sha256(args.diagnosis_zip),
            "Subject_Characteristics.zip": sha256(args.characteristics_zip),
        },
        "cleaning": cleaning_audit,
        "rules": audits,
        "candidate_coverage": coverage,
        "status": (
            "reported_counts_reproduced"
            if audits["candidate_non_ad_followup"]["cohort_total"] == config["reported_total"]
            and audits["candidate_non_ad_followup"]["converters"] == config["reported_converters"]
            else "reported_counts_not_yet_reproduced"
        ),
    }
    (args.public_output / "cohort_audit.json").write_text(
        json.dumps(public_audit, indent=2), encoding="utf-8"
    )

    print(comparison.to_string(index=False))
    print(json.dumps({"cleaning": cleaning_audit, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
