#!/usr/bin/env python3
"""Generate manuscript tables and figures from aggregate reproducibility outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRIMARY_FEATURE_SETS = [
    "clinical_core",
    "clinical_extended",
    "clinical_core_plus_nfl",
    "clinical_extended_plus_nfl",
    "clinical_extended_plus_dti",
    "clinical_extended_plus_dti_standard",
]
EXPLORATORY_FEATURE_SETS = [
    "clinical_extended_plus_freesurfer",
    "multimodal_all",
]

FEATURE_LABELS = {
    "clinical_core": "Clinical core",
    "clinical_extended": "Clinical extended",
    "clinical_core_plus_nfl": "Clinical core + NfL",
    "clinical_extended_plus_nfl": "Clinical extended + NfL",
    "clinical_extended_plus_dti": "Clinical extended + DTI",
    "clinical_extended_plus_dti_standard": "Clinical extended + standard-mean DTI",
    "clinical_extended_plus_freesurfer": "Clinical extended + FreeSurfer*",
    "multimodal_all": "All modalities*",
}

MODALITY_LABELS = {
    "demographics": "Demographics",
    "family_history": "Family history",
    "vitals": "Vital signs",
    "neurological_exam": "Neurological exam",
    "medical_history": "Medical history",
    "plasma_nfl": "Plasma NfL",
    "dti": "DTI (primary)",
    "dti_standard_mean": "DTI (standard mean)",
    "freesurfer": "FreeSurfer*",
}


def mean_sd(series: pd.Series, decimals: int = 1) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "--"
    return f"{values.mean():.{decimals}f} ({values.std(ddof=1):.{decimals}f})"


def count_percent(series: pd.Series, positive: object = 1) -> str:
    known = series.dropna()
    if known.empty:
        return "--"
    count = int(known.eq(positive).sum())
    return rf"{count} ({100 * count / len(known):.1f}\%)"


def escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_cohort_table(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    groups = [
        ("Overall", data),
        ("Stable", data.loc[data["mci_conversion_label_36m"].eq(0)]),
        ("Converter", data.loc[data["mci_conversion_label_36m"].eq(1)]),
    ]
    rows = []
    for label, group in groups:
        rows.append(
            {
                "Group": label,
                "N": len(group),
                "Age": mean_sd(group["age_at_baseline"]),
                "Female": count_percent(group["sex"], "female"),
                "Education": mean_sd(group["education_years"]),
                "Family history": count_percent(group["family_history_any"]),
                "NfL available": count_percent(group["has_plasma_nfl"]),
                "DTI available": count_percent(group["has_dti"]),
                "FreeSurfer available": count_percent(group["has_freesurfer"]),
            }
        )
    summary = pd.DataFrame(rows)
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Candidate-cohort characteristics and modality availability. Continuous variables are mean (SD); categorical variables are n (\%).}",
        r"\label{tab:cohort_characteristics}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Group & N & Age, years & Female & Education, years & Family history & NfL available & DTI available & FreeSurfer available \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["Group"],
                row["N"],
                row["Age"],
                row["Female"],
                row["Education"],
                row["Family history"],
                row["NfL available"],
                row["DTI available"],
                row["FreeSurfer available"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_model_table(performance: pd.DataFrame, output: Path) -> pd.DataFrame:
    selected = performance.loc[
        performance["analysis_population"].eq("full_imputed")
        & performance["model"].eq("logistic_l2")
        & performance["feature_set"].isin(PRIMARY_FEATURE_SETS + EXPLORATORY_FEATURE_SETS)
    ].copy()
    order = PRIMARY_FEATURE_SETS + EXPLORATORY_FEATURE_SETS
    selected["_order"] = selected["feature_set"].map({name: i for i, name in enumerate(order)})
    selected = selected.sort_values("_order")

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Held-out performance of the prespecified L2-regularized logistic-regression ablations. AUC confidence intervals use 1,000 class-stratified bootstrap samples.}",
        r"\label{tab:model_performance}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Feature configuration & Test n & AUC (95\% CI) & Balanced accuracy & Sensitivity & Specificity & Brier score \\",
        r"\midrule",
    ]
    for row in selected.to_dict("records"):
        label = FEATURE_LABELS[row["feature_set"]]
        lines.append(
            f"{label} & {int(row['test_n'])} & "
            f"{row['auc']:.3f} ({row['auc_ci_lower']:.3f}--{row['auc_ci_upper']:.3f}) & "
            f"{row['balanced_accuracy']:.3f} & {row['sensitivity']:.3f} & "
            f"{row['specificity']:.3f} & {row['brier']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize",
            r"*Exploratory structural-MRI analyses were not part of the original clinical/NfL/DTI analysis scope. Classification thresholds were selected from training-only out-of-fold predictions.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected.drop(columns="_order")


def write_complete_case_table(
    performance: pd.DataFrame, feasibility: pd.DataFrame, output: Path
) -> pd.DataFrame:
    selected = performance.loc[
        performance["analysis_population"].eq("complete_case")
        & performance["model"].eq("logistic_l2")
    ].copy()
    selected = selected.loc[
        selected["feature_set"].isin(
            [
                "clinical_core_plus_nfl",
                "clinical_extended_plus_nfl",
                "clinical_extended_plus_freesurfer",
            ]
        )
    ]
    order = [
        "clinical_core_plus_nfl",
        "clinical_extended_plus_nfl",
        "clinical_extended_plus_freesurfer",
    ]
    selected["_order"] = selected["feature_set"].map({name: i for i, name in enumerate(order)})
    selected = selected.sort_values("_order")
    infeasible = feasibility.loc[
        feasibility["analysis_population"].eq("complete_case")
        & ~feasibility["eligible_for_modeling"]
    ]

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Complete-case sensitivity analyses using logistic regression.}",
        r"\label{tab:complete_case}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Feature configuration & Train n & Test n & Test converters & AUC (95\% CI) \\",
        r"\midrule",
    ]
    for row in selected.to_dict("records"):
        lines.append(
            f"{FEATURE_LABELS[row['feature_set']]} & {int(row['train_n'])} & {int(row['test_n'])} & "
            f"{int(row['test_converters'])} & {row['auc']:.3f} "
            f"({row['auc_ci_lower']:.3f}--{row['auc_ci_upper']:.3f}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{minipage}{0.95\linewidth}\footnotesize",
            "Complete-case DTI analyses were not estimated because the robust-mean DTI test subset contained only "
            f"{int(infeasible.loc[infeasible['feature_set'].eq('clinical_extended_plus_dti'), 'test_converters'].iloc[0])} converters; "
            "the all-modality complete-case subset contained one converter.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected.drop(columns="_order")


def plot_workflow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "ADNI diagnosis\nand modality files"),
        (0.22, "36-month cohort\nrule audit"),
        (0.42, "Temporal matching\nand modality QC"),
        (0.62, "Training-only\npreprocessing and CV"),
        (0.82, "Held-out test +\nbootstrap uncertainty"),
    ]
    colors = ["#dceaf7", "#e8e1f4", "#dff0e4", "#fff0cf", "#f8dddd"]
    for index, ((x, label), color) in enumerate(zip(boxes, colors)):
        patch = FancyBboxPatch(
            (x, 0.28),
            0.16,
            0.44,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + 0.08, 0.50, label, ha="center", va="center", fontsize=10)
        if index < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.165, 0.50),
                    (boxes[index + 1][0] - 0.005, 0.50),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.2,
                    color="#475569",
                )
            )
    fig.tight_layout()
    fig.savefig(
        output,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        dpi=300,
    )
    plt.close(fig)


def plot_coverage(coverage: pd.DataFrame, output: Path) -> None:
    plot = coverage.copy()
    plot["label"] = plot["modality"].map(MODALITY_LABELS)
    plot["converter_percent"] = 100 * plot["converter_n"] / 255
    plot["stable_percent"] = 100 * plot["stable_n"] / 447
    y = np.arange(len(plot))
    height = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(y - height, plot["available_percent"], height, label="Overall", color="#334155")
    ax.barh(y, plot["converter_percent"], height, label="Converters", color="#c2413b")
    ax.barh(y + height, plot["stable_percent"], height, label="Stable", color="#3b82a0")
    ax.set_yticks(y, plot["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Participants with available modality (%)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        output,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        dpi=300,
    )
    plt.close(fig)


def plot_auc(performance: pd.DataFrame, output: Path) -> None:
    order = PRIMARY_FEATURE_SETS + EXPLORATORY_FEATURE_SETS
    selected = performance.loc[
        performance["analysis_population"].eq("full_imputed")
        & performance["model"].eq("logistic_l2")
        & performance["feature_set"].isin(order)
    ].copy()
    selected["_order"] = selected["feature_set"].map({name: i for i, name in enumerate(order)})
    selected = selected.sort_values("_order", ascending=False)
    labels = selected["feature_set"].map(FEATURE_LABELS)
    y = np.arange(len(selected))
    colors = [
        "#9b4d96" if name in EXPLORATORY_FEATURE_SETS else "#256d85"
        for name in selected["feature_set"]
    ]
    lower = selected["auc"] - selected["auc_ci_lower"]
    upper = selected["auc_ci_upper"] - selected["auc"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for index, (_, row) in enumerate(selected.iterrows()):
        ax.errorbar(
            row["auc"],
            y[index],
            xerr=[[lower.iloc[index]], [upper.iloc[index]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=6,
        )
    ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("Held-out ROC AUC (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(0.405, len(y) - 0.45, "* exploratory structural MRI", color="#9b4d96", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        output,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        dpi=300,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/derived/multimodal_cohort.csv"
    )
    parser.add_argument(
        "--performance", type=Path, default=ROOT / "results/modeling/model_performance.csv"
    )
    parser.add_argument(
        "--feasibility",
        type=Path,
        default=ROOT / "results/modeling/analysis_population_feasibility.csv",
    )
    parser.add_argument(
        "--coverage", type=Path, default=ROOT / "results/modalities/modality_coverage.csv"
    )
    parser.add_argument("--tables", type=Path, default=ROOT / "tables")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--public-output", type=Path, default=ROOT / "results/manuscript"
    )
    args = parser.parse_args()

    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.public_output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.dataset, low_memory=False)
    performance = pd.read_csv(args.performance)
    feasibility = pd.read_csv(args.feasibility)
    coverage = pd.read_csv(args.coverage)

    cohort_summary = write_cohort_table(
        data, args.tables / "cohort_characteristics.tex"
    )
    model_summary = write_model_table(
        performance, args.tables / "model_performance.tex"
    )
    complete_case = write_complete_case_table(
        performance, feasibility, args.tables / "complete_case.tex"
    )
    cohort_summary.to_csv(args.public_output / "cohort_characteristics.csv", index=False)
    model_summary.to_csv(args.public_output / "manuscript_model_performance.csv", index=False)
    complete_case.to_csv(args.public_output / "complete_case_summary.csv", index=False)

    plot_workflow(args.figures / "reproducible_workflow.png")
    plot_coverage(coverage, args.figures / "modality_coverage.png")
    plot_auc(performance, args.figures / "heldout_auc_ablation.png")
    print("Generated manuscript tables and figures.")


if __name__ == "__main__":
    main()