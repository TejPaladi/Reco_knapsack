# IITR-Teaming Evaluation

This folder contains the fairness evaluation pipeline and the saved fairness outputs for the canonical IITR `100 x 46` experiment.

## Main Files

- `generate_fairness_tables.py`
  Generates Table 2 style and Table 3 style fairness outputs from saved Teaming method CSV files.

## Output Folders

- `output_v0/`
  Fairness results for the `100 proposals / 46 researchers` `v0_teaming` experiment.

## Generated Artifacts

Each output folder contains:

- `fairness_table2_gender.csv`
  Population distribution versus method-level gender distribution, with SP.

- `fairness_table3_gender.csv`
  Decision-based fairness metrics: SP, CSP, EO, PRP, and TE.

- `fairness_support_gender.csv`
  Supporting counts and utility summaries used by the fairness analysis.

- `fairness_metadata_gender.json`
  Metadata describing the evaluation slice, gender coverage, and alignment details.

## Typical Use

From the repository root:

```bash
.venv/bin/python evaluation/generate_fairness_tables.py
```
