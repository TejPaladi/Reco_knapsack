# IITR-Teaming Project

This is the cleaned IITR Teaming workspace prepared for GitHub release.

The single GitHub-facing published output root in this project is:

- `data/output/`

## What This Project Does

The IITR-Teaming project recommends anchor-fixed research teams for the audited
IITR slice used in the paper:

- first `100` proposal calls from `data/v0_data/archive_proposals.csv`
- all `46` researchers from `data/v0_data/researchers.csv`

The published methods are:

- `M0` Random baseline
- `M1` String / lexical matching
- `M2` Semantic mapper matching
- `M3` BoostSRL / relational learning
- `M6` Exact knapsack coverage
- `M7` Marginal-utility knapsack

## Canonical Runner

The authoritative IITR regeneration script is:

- `code/run_v0_teaming_pipeline.py`

Run from the project root:

```bash
.venv/bin/python code/run_v0_teaming_pipeline.py
```

To regenerate fairness only:

```bash
.venv/bin/python evaluation/generate_fairness_tables.py
```

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

The canonical published IITR outputs are copied from:

- `data/v0_teaming/`
- `code/compare_outputs/comparison_summary_v0.csv`
- `evaluation/output_v0/`

Older `paper_style` folders are kept only for historical reproduction and
comparison. They are not the GitHub-facing published outputs:

- `data/v0_teaming_paper_style/`
- `evaluation/output_v0_paper_style/`

Future users should start from `data/output/`.

## Repository Layout

```text
IITR-Teaming/
├── README.md
├── code/
├── data/
│   ├── output/
│   ├── v0_data/
│   ├── v0_teaming/
│   └── v0_teaming_paper_style/
└── evaluation/
```

## Core Files

Method implementations:

- `code/M0.py`
- `code/M1.py`
- `code/M2.py`
- `code/M3.py`
- `code/M6.py`
- `code/M7.py`

Shared utilities:

- `code/metrics_scorer.py`
- `code/nlp_techniques.py`
- `code/mapper4_main/`
- `evaluation/generate_fairness_tables.py`

BoostSRL assets used by `M3`:

- `code/boosted_results/`

## Current Published Ranking

From `data/output/comparisons/comparison_summary.csv`, the IITR ranking is:

1. `M7` Marginal-utility knapsack
2. `M3` BoostSRL / relational learning
3. `M1` String / lexical matching
4. `M2` Semantic mapper matching
5. `M6` Exact knapsack coverage
6. `M0` Random baseline

Summary:

| Method | Average Goodness | Average Volume |
|---|---:|---:|
| M0 | 0.1263 | 10.0000 |
| M1 | 0.4953 | 10.0000 |
| M2 | 0.4913 | 10.0000 |
| M3 | 0.5757 | 4.0996 |
| M6 | 0.4365 | 5.6557 |
| M7 | 0.6118 | 4.0476 |

## Current Published Fairness Snapshot

Published composition fairness in `data/output/comparisons/fairness_table2_gender.csv`:

| Method | SP Gap |
|---|---:|
| M0 | 0.0295 |
| M1 | 0.0405 |
| M2 | 0.0228 |
| M3 | 0.1143 |
| M6 | 0.0257 |
| M7 | 0.0070 |

Published decision fairness in `data/output/comparisons/fairness_table3_gender.csv`:

- IITR `M1` is best on `SP`
- IITR `M6` is best on `CSP` and `EO`
- IITR `M0` is best on `PRP` and `TE`

## Notes For GitHub Users

- Start with `data/output/` if you want the published IITR outputs.
- Use `code/run_v0_teaming_pipeline.py` if you want to regenerate the
  canonical IITR run.
- Treat the `paper_style` folders as legacy reproductions, not as the main
  published result.
