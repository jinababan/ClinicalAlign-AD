# Data inputs

Participant-level ADNI data must remain local and must not be committed.

Stage 1 expects these archives in `prism-uploads/` at the repository root:

- `Diagnosis.zip`
  - `DXSUM_PDXCONV_ADNIALL.csv`
- `Subject_Characteristics.zip`
  - `PTDEMOG.csv`
  - `FHQ.csv`
  - `RECFHQ.csv`
  - `FAMXHPAR.csv`
  - `FAMXHSIB.csv`

Later stages additionally use the locally supplied NfL, DTI, FreeSurfer, vital
sign, medical history, and neurological examination files. Analysis outputs
remain local and are ignored by Git, including aggregate summaries.

Stage 2 expects the following additional local files:

- `NFL.zip`
- `VITALS_18Sep2026.csv`
- `NEUROEXM_18Sep2026.csv`
- `MEDHIST_18Sep2026.csv`
- `INITHEALTH_18Sep2026.csv`
- `ADNI_DTIROI_V1_18Sep2026.csv`
- `DTIROI_ROBUSTMEAN_18Sep2026.csv`
- `DTIROI_MEAN_18Sep2026.csv`
- `UCSFFSX7_18Sep2026.csv`

These dated filenames are the script defaults. Use the command-line input
options shown by `python scripts/02_build_multimodal_dataset.py --help` if your
authorized downloads have different names. File columns and formats must still
match what the pipeline expects.
