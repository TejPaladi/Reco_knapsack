# Teaming Project

This is the cleaned USC Teaming workspace prepared for GitHub release.

The single GitHub-facing published output root in this project is:

- `data/output/`

## What This Project Does

The Teaming project recommends research teams for proposal calls under the UC1
anchor-fixed setting. Given a researcher anchor and a proposal call, the system
returns ranked candidate teams built by several methods:

- `M0` Random baseline
- `M1` String / lexical matching
- `M2` Semantic mapper matching
- `M3` BoostSRL / relational learning
- `M6` Exact knapsack coverage
- `M7` Marginal-utility knapsack

The published USC result in this repo corresponds to the `442` call by `202`
researcher run used in the paper-facing comparison tables.

## Canonical Published Outputs

The cleaned release keeps one clear published root:

- `data/output/teaming_uc1_m0.csv`
- `data/output/teaming_uc1_m1.csv`
- `data/output/teaming_uc1_m2.csv`
- `data/output/teaming_uc1_m3.csv`
- `data/output/teaming_uc1_m6.csv`
- `data/output/teaming_uc1_m7.csv`
- `data/output/comparisons/comparison_summary.csv`
- `data/output/comparisons/fairness_table2_gender.csv`
- `data/output/comparisons/fairness_table3_gender.csv`
- `data/output/comparisons/fairness_support_gender.csv`
- `data/output/comparisons/fairness_metadata_gender.json`
- `data/output/comparisons/published_metadata.json`

## Important Clarification

This USC workspace is older and notebook-first, so the published root is a
clean consolidation of the paper-aligned outputs rather than a brand-new
pipeline directory.

The published files come from two maintained source roots:

1. quality and raw method exports come from
   `data/v1_output_teaming/teaming_1698proposals_316researchers/`
2. paper-facing fairness tables come from `evaluation/output_paper_subset/`

This means the GitHub-facing `data/output/comparisons/` folder intentionally
combines:

- the full USC quality summary that matches the paper's USC quality table
- the USC fairness subset outputs that match the paper-facing fairness tables

The older folders are still kept in the repo for traceability, but future
users should start from `data/output/`.

## Repository Layout

```text
Teaming/
├── README.md
├── code/
├── data/
│   ├── output/
│   ├── v0_output_teaming/
│   ├── v1_input_files/
│   └── v1_output_teaming/
└── evaluation/
```

## Input CSVs

The USC Teaming input files are stored in `data/v1_input_files/`.

- `v1_researchers.csv` - primary researcher/anchor input used by the UC1
  notebooks. Key columns are `names`, `descriptions`, `titles`, and `research`.
- `v1_proposal_links_title_synopsis.csv` - primary proposal-call input used by
  the UC1 notebooks. Key columns are `nsf_proposal_links_v1`, `title`, and
  `synopsis`.
- `v1_og_faculty.csv` - original faculty metadata used by fairness-table
  generation together with `v1_researchers.csv`. Columns include `name`,
  `designation`, `titles`, `research`, `background`, `keywords`, `links`, and
  `area`.
- `v1_og_rfps.csv` - original RFP metadata retained with the workspace. Columns
  include `agency`, `title`, `deadline`, `reference`, `budget`, `details`, and
  `keywords`.
- `usc_combined_interests.csv` - compact researcher-interest reference table
  with `names` and `research`.

## Where To Start

This workspace is notebook-first. There is not one single canonical
one-command runner like the cleaned Meal project. For future users, the best
entry points are:

- `code/teaming_uc1_m0.ipynb`
- `code/teaming_uc1_m1.ipynb`
- `code/teaming_uc1_m2.ipynb`
- `code/teaming_uc1_m3.ipynb`
- `code/teaming_uc1_m6.ipynb`
- `code/teaming_uc1_m7.ipynb`
- `code/M6.py`
- `code/M7.py`
- `code/Results.ipynb`

The main published results are already collected under `data/output/`, so many
GitHub users will not need to rerun the notebooks unless they want to extend
the study.

## Current Published Ranking

From `data/output/comparisons/comparison_summary.csv`, the USC ranking is:

1. `M7` Marginal-utility knapsack
2. `M3` BoostSRL / relational learning
3. `M6` Exact knapsack coverage
4. `M2` Semantic mapper matching
5. `M1` String / lexical matching
6. `M0` Random baseline

Summary:

| Method | Average Goodness | Average Volume |
|---|---:|---:|
| M0 | 0.0872 | 10.0000 |
| M1 | 0.3649 | 10.0000 |
| M2 | 0.4297 | 10.0000 |
| M3 | 0.5894 | 6.8360 |
| M6 | 0.4520 | 5.2891 |
| M7 | 0.6176 | 5.5594 |

## Current Published Fairness Snapshot

Published composition fairness in `data/output/comparisons/fairness_table2_gender.csv`:

| Method | SP Gap |
|---|---:|
| M0 | 0.0000 |
| M1 | 0.0729 |
| M2 | 0.0055 |
| M3 | 0.0090 |
| M6 | 0.0053 |
| M7 | 0.0002 |

Published decision fairness in `data/output/comparisons/fairness_table3_gender.csv`:

- USC `M3` is best on `SP`, `CSP`, and `EO`
- USC `M7` is best on `PRP`
- USC `M6` is best on `TE`

## Legacy Folders Kept For Reference

These are intentionally still present, but they are not the GitHub-facing
published root:

- `data/v0_output_teaming/`
- `data/v1_output_teaming/`
- `evaluation/output/`
- `evaluation/output_paper_subset/`

## Notes For GitHub Users

- Start with `data/output/` if you want the published USC outputs.
- Use the notebooks in `code/` if you want to inspect method generation logic.
- The published fairness tables are the paper-aligned USC subset outputs.
- The published raw method CSVs are the cleaned USC method exports used for the
  quality summary.
