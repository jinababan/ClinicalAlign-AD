# ClinicalAlign-AD

Python analysis code for an ADNI-based study of 36-month conversion from mild cognitive impairment to Alzheimer's disease. The pipeline builds an auditable diagnosis cohort, aligns clinical, plasma NfL, DTI, and FreeSurfer MRI records, and compares classical machine-learning baselines with a participant-disjoint test set.

**Status:** Research code for a manuscript in development. These models have not been externally validated and are not intended for clinical use.

## Contents

- `configs/`: cohort rules, temporal windows, feature groups and model settings.
- `src/clinicalalign_ad/`: cohort construction, modality selection and modeling.
- `scripts/`: four sequential analysis stages.
- `tests/`: cohort, temporal-alignment and modeling checks.

This repository contains **code and configuration only**. It excludes ADNI source data, participant-level derived data, predictions, aggregate results, generated tables and figures, and the unfinished manuscript. The scripts create analysis outputs locally in Git-ignored directories. Manuscript authorship and publication details are still being finalized.

## Setup and authorized data

Use Python 3.12 and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

Obtain your own authorized access to [ADNI data](https://adni.loni.usc.edu/data-samples/adni-data/). See [`data/README.md`](data/README.md) for expected filenames. By default, scripts read source archives and CSVs from `prism-uploads/` at the repository root. Each script's `--help` lists options to supply different authorized local paths. Never commit ADNI data, participant-level derived tables, or individual predictions.

## Run the pipeline

From the project root, after placing authorized data in `prism-uploads/`:

```bash
python scripts/01_build_cohort.py
python scripts/02_build_multimodal_dataset.py
python scripts/03_run_baselines.py
python scripts/04_make_manuscript_assets.py
```

The stages write participant tables, split assignments, and test predictions to ignored `data/derived/`; local audits and model metrics to ignored `results/`; and generated `tables/` and `figures/`. Review any output under applicable ADNI data-use terms before sharing it. Stage 4 needs the first three stages' outputs.

The modeling stage uses a fixed stratified 80/20 split, five training-only cross-validation folds, training-only imputation and scaling, logistic regression, Gaussian naive Bayes, and k-nearest neighbors. It evaluates declared feature sets and bootstrap intervals. See `configs/modeling.json` for exact settings. Deep-learning and gradient-boosting libraries are not used in this version.

## Interpretation and limitations

The explicit candidate cohort rule selects **702 participants** (255 converters, 447 stable), whereas an earlier manuscript draft reported 695 (255 converters, 440 stable). The seven-person difference remains unresolved; the code does not exclude people arbitrarily to reproduce the earlier count. DTI availability is limited, and several complete-case analyses have too few converters for reliable evaluation. FreeSurfer comparisons are exploratory. Declared temporal windows permit measurements up to 90 days after baseline if they precede the outcome date. Replication requires authorized ADNI files and compatible file versions.

No license has been selected yet. Ask the owner before reusing the code beyond GitHub's default viewing and forking permissions.
