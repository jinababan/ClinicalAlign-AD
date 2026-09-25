#!/usr/bin/env python3
"""Run leakage-safe classical baselines, ablations, and bootstrap uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicalalign_ad.modeling import (  # noqa: E402
    MedianStandardizer,
    binary_metrics,
    bootstrap_intervals,
    choose_youden_threshold,
    make_model,
    paired_bootstrap_auc_difference,
    stratified_holdout_indices,
    tune_parameter,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sex_female"] = out["sex"].map({"female": 1.0, "male": 0.0})
    nfl = pd.to_numeric(out["plasma_nfl"], errors="coerce")
    out["log_plasma_nfl"] = np.log1p(nfl.where(nfl.ge(0)))
    icv = pd.to_numeric(out["fs_icv"], errors="coerce")
    out["fs_lateral_ventricle_to_icv"] = (
        pd.to_numeric(out["fs_lateral_ventricle_volume"], errors="coerce") / icv
    )
    out["fs_total_gray_matter_to_icv"] = (
        pd.to_numeric(out["fs_total_gray_matter_volume"], errors="coerce") / icv
    )
    return out


def parameter_values(model_name: str, specification: dict[str, list[float]]) -> list[float]:
    return list(next(iter(specification[model_name].values())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/modeling.json")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/derived/multimodal_cohort.csv"
    )
    parser.add_argument("--private-output", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--public-output", type=Path, default=ROOT / "results/modeling")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = add_derived_features(pd.read_csv(args.dataset, low_memory=False))
    target = "mci_conversion_label_36m"
    y = pd.to_numeric(data[target], errors="raise").astype(int).to_numpy()
    train_indices, test_indices = stratified_holdout_indices(
        y, config["test_fraction"], config["seed"]
    )
    split = np.full(len(data), "train", dtype=object)
    split[test_indices] = "test"
    data["analysis_split"] = split

    args.private_output.mkdir(parents=True, exist_ok=True)
    args.public_output.mkdir(parents=True, exist_ok=True)
    data[["RID", target, "analysis_split"]].to_csv(
        args.private_output / "split_assignments.csv", index=False
    )

    feature_sets: dict[str, list[str]] = config["feature_sets"]
    missing_features = {
        name: sorted(set(features).difference(data.columns))
        for name, features in feature_sets.items()
        if set(features).difference(data.columns)
    }
    if missing_features:
        raise ValueError(f"Configured features are missing: {missing_features}")

    populations: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for feature_set in feature_sets:
        populations.append(("full_imputed", feature_set, train_indices, test_indices))
    for feature_set in config["complete_case_feature_sets"]:
        features = feature_sets[feature_set]
        complete = data[features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
        populations.append(
            (
                "complete_case",
                feature_set,
                train_indices[complete[train_indices]],
                test_indices[complete[test_indices]],
            )
        )

    performance_records = []
    interval_records = []
    tuning_records = []
    prediction_records = []
    prediction_lookup: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    feasibility_records = []
    run_number = 0

    for population, feature_set, population_train, population_test in populations:
        features = feature_sets[feature_set]
        y_train = y[population_train]
        y_test = y[population_test]
        train_counts = np.bincount(y_train, minlength=2)
        test_counts = np.bincount(y_test, minlength=2)
        eligible = bool(
            train_counts.min() >= int(config["minimum_train_class_count"])
            and test_counts.min() >= int(config["minimum_test_class_count"])
        )
        feasibility_records.append(
            {
                "analysis_population": population,
                "feature_set": feature_set,
                "train_n": len(population_train),
                "train_converters": int(train_counts[1]),
                "train_stable": int(train_counts[0]),
                "test_n": len(population_test),
                "test_converters": int(test_counts[1]),
                "test_stable": int(test_counts[0]),
                "eligible_for_modeling": eligible,
                "exclusion_reason": ""
                if eligible
                else "class count below configured minimum",
            }
        )
        if not eligible:
            continue
        n_folds = min(int(config["cv_folds"]), int(np.bincount(y_train).min()))
        if n_folds < 2:
            continue

        train_frame = data.iloc[population_train]
        test_frame = data.iloc[population_test]
        for model_name in config["models"]:
            run_number += 1
            chosen, tuning, training_oof_probabilities = tune_parameter(
                model_name,
                parameter_values(model_name, config["models"]),
                train_frame,
                y_train,
                features,
                n_folds,
                int(config["seed"]) + run_number,
            )
            if config["threshold_strategy"] == "training_youden":
                selected_threshold = choose_youden_threshold(
                    y_train, training_oof_probabilities
                )
            else:
                selected_threshold = float(config["threshold_strategy"])
            for record in tuning.to_dict("records"):
                tuning_records.append(
                    {
                        "analysis_population": population,
                        "feature_set": feature_set,
                        "model": model_name,
                        **record,
                        "selected": bool(record["parameter"] == chosen),
                    }
                )

            preprocessor = MedianStandardizer()
            x_train = preprocessor.fit_transform(train_frame, features)
            x_test = preprocessor.transform(test_frame)
            model = make_model(model_name, chosen).fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)
            metrics = binary_metrics(y_test, probabilities, selected_threshold)
            intervals = bootstrap_intervals(
                y_test,
                probabilities,
                selected_threshold,
                config["bootstrap_replicates"],
                int(config["seed"]) + 10000 + run_number,
            )
            performance_records.append(
                {
                    "analysis_population": population,
                    "feature_set": feature_set,
                    "model": model_name,
                    "selected_parameter": chosen,
                    "selected_threshold": selected_threshold,
                    "feature_count": len(features),
                    "train_n": len(population_train),
                    "train_converters": int(y_train.sum()),
                    "test_n": len(population_test),
                    "test_converters": int(y_test.sum()),
                    **metrics,
                    "auc_ci_lower": intervals["auc"][0],
                    "auc_ci_upper": intervals["auc"][1],
                }
            )
            for metric_name, (lower, upper) in intervals.items():
                interval_records.append(
                    {
                        "analysis_population": population,
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": metric_name,
                        "estimate": metrics[metric_name],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "bootstrap_replicates": config["bootstrap_replicates"],
                    }
                )
            for row_index, probability in zip(population_test, probabilities):
                prediction_records.append(
                    {
                        "RID": int(data.iloc[row_index]["RID"]),
                        "analysis_population": population,
                        "feature_set": feature_set,
                        "model": model_name,
                        "observed": int(y[row_index]),
                        "probability": float(probability),
                        "selected_threshold": selected_threshold,
                    }
                )
            prediction_lookup[(population, feature_set, model_name)] = (
                population_test,
                y_test,
                probabilities,
            )

    performance = pd.DataFrame(performance_records).sort_values(
        ["analysis_population", "feature_set", "auc"], ascending=[True, True, False]
    )
    performance.to_csv(args.public_output / "model_performance.csv", index=False)
    pd.DataFrame(interval_records).to_csv(
        args.public_output / "bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(tuning_records).to_csv(
        args.public_output / "training_cv_tuning.csv", index=False
    )
    pd.DataFrame(prediction_records).to_csv(
        args.private_output / "test_predictions.csv", index=False
    )
    pd.DataFrame(feasibility_records).to_csv(
        args.public_output / "analysis_population_feasibility.csv", index=False
    )

    comparison_records = []
    reference_key = ("full_imputed", "clinical_core", "logistic_l2")
    reference_indices, reference_y, reference_probabilities = prediction_lookup[reference_key]
    for feature_set in feature_sets:
        key = ("full_imputed", feature_set, "logistic_l2")
        candidate_indices, candidate_y, candidate_probabilities = prediction_lookup[key]
        if not np.array_equal(candidate_indices, reference_indices) or not np.array_equal(
            candidate_y, reference_y
        ):
            raise RuntimeError("Paired ablation predictions are not aligned")
        difference, lower, upper = paired_bootstrap_auc_difference(
            reference_y,
            candidate_probabilities,
            reference_probabilities,
            config["bootstrap_replicates"],
            int(config["seed"]) + 20000 + len(comparison_records),
        )
        comparison_records.append(
            {
                "model": "logistic_l2",
                "reference_feature_set": "clinical_core",
                "candidate_feature_set": feature_set,
                "auc_difference": difference,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    pd.DataFrame(comparison_records).to_csv(
        args.public_output / "ablation_auc_differences.csv", index=False
    )

    split_summary = (
        data.groupby("analysis_split")[target]
        .agg(n="size", converters="sum")
        .reset_index()
    )
    split_summary["stable"] = split_summary["n"] - split_summary["converters"]
    split_summary.to_csv(args.public_output / "split_summary.csv", index=False)

    dependency_names = ["numpy", "pandas", "scipy", "sklearn", "torch", "xgboost", "catboost"]
    audit = {
        "config": config,
        "dataset_sha256": sha256(args.dataset),
        "candidate_cohort_n": int(len(data)),
        "train_test_rid_overlap_n": int(
            len(set(data.iloc[train_indices]["RID"]) & set(data.iloc[test_indices]["RID"]))
        ),
        "duplicate_rid_n": int(data["RID"].duplicated().sum()),
        "split_summary": split_summary.to_dict("records"),
        "models_implemented_without_optional_ml_dependencies": [
            "logistic_l2",
            "gaussian_nb",
            "knn",
        ],
        "dependency_available": {
            name: bool(importlib.util.find_spec(name)) for name in dependency_names
        },
        "leakage_controls": [
            "participant-disjoint fixed holdout",
            "hyperparameter selection restricted to training folds",
            "imputation and scaling fit within each training fold",
            "classification threshold selected from training out-of-fold predictions",
            "test set evaluated once per predeclared model/feature set",
            "class-stratified paired bootstrap on fixed test predictions",
        ],
        "interpretation_status": "candidate_702_participant_cohort_not_legacy_695_count",
    }
    (args.public_output / "modeling_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    display = performance.loc[
        performance["analysis_population"].eq("full_imputed"),
        [
            "feature_set",
            "model",
            "test_n",
            "auc",
            "auc_ci_lower",
            "auc_ci_upper",
            "balanced_accuracy",
            "brier",
        ],
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()