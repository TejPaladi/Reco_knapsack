# Meal Recommendation Pipeline

This is the cleaned Meal recommendation project prepared for GitHub release.

The canonical published result in this repo is the **closest BEACON-style
proxy** for the meal domain. Future users should treat the following folder as
the single source of truth:

- `data/output/`

The repository keeps one final published output root and one canonical entry
point:

- final pipeline command: `code/run_pipeline.py`
- final published outputs: `data/output/`

## What This Project Does

Given a user meal request and a target anchor meal, the system builds small
meal bundles and evaluates five methods:

- `M0` Random baseline
- `M1` Sequential greedy
- `M3` Boosted Bandit
- `M6` Exact knapsack
- `M7` Marginal utility

The final exported bundle quality is reported with the **closest BEACON-style
proxy** supported by the current category-only meal dataset:

- `dm_proxy = unique_items / total_items`
- `mc_proxy = covered_required_categories / required_categories`
- `uc_proxy = average per-item positive alignment`
- `G_BP = (dm_proxy + mc_proxy + uc_proxy) / 3`

## Important Clarification

The BEACON-style proxy output is produced in two stages:

1. the pipeline first builds internal source bundles using the source meal
   profile needed by the current generation code
2. those same returned bundles are then **rescored** with the BEACON-style
   proxy and published to `data/output/`

So the final published proxy results do **not** regenerate different bundles at
the proxy step. They rescore the same returned bundles and then compute the
final comparison and fairness tables from that rescored output root.

## Repository Layout

```text
Meal/
├── README.md
├── boosted_bandit/
├── code/
│   ├── run_pipeline.py
│   ├── rescore_outputs_beacon_proxy.py
│   ├── generate_fairness_tables.py
│   ├── build_final_outputs.py
│   ├── gen_boosted_data.py
│   ├── boosted_bandit_postprocess.py
│   ├── meal_methods.py
│   ├── meal_utils.py
│   └── metrics_scorer.py
└── data/
    ├── input_data/
    │   ├── meal_categories.csv
    │   └── user_meal_requirements.csv
    └── output/
        ├── meal_uc1_m0.csv
        ├── meal_uc1_m1.csv
        ├── meal_uc1_m2.csv
        ├── meal_uc1_m3.csv
        ├── meal_uc1_m4.csv
        └── comparisons/
```

## Canonical Dataset

The current cleaned dataset contains:

- `154` user requests
- `77` meals
- `5` meal occasions
- `16` unique requirement patterns

Each method export is a cross-product style table, so each method file contains:

- `154 x 77 = 11,858` rows

## How To Run The Canonical Pipeline

From the project root:

```bash
cd /Users/tej/Desktop/Recommendation_Projects/Meal
.venv/bin/python code/run_pipeline.py
```

This command performs the full canonical workflow:

1. generates BoostSRL facts from `data/input_data/`
2. runs BoostSRL training and testing in `boosted_bandit/`
3. builds internal source outputs in a temporary intermediate folder
4. rescales those same returned bundles with the BEACON-style proxy
5. publishes the final output CSVs to `data/output/`
6. generates fairness tables from the published proxy outputs

By default the intermediate source folder is deleted at the end of the run.

If you want to inspect the intermediate source outputs:

```bash
.venv/bin/python code/run_pipeline.py --keep-intermediate
```

## Main Published Files

After the canonical run, the most important files are:

- `data/output/meal_uc1_m0.csv`
- `data/output/meal_uc1_m1.csv`
- `data/output/meal_uc1_m2.csv`
- `data/output/meal_uc1_m3.csv`
- `data/output/meal_uc1_m4.csv`
- `data/output/comparisons/comparison_summary.csv`
- `data/output/comparisons/method_comparison_summary.csv`
- `data/output/comparisons/method_comparison_detailed.csv`
- `data/output/comparisons/beacon_proxy_metadata.json`
- `data/output/comparisons/fairness_table2_gender.csv`
- `data/output/comparisons/fairness_table3_gender.csv`
- `data/output/comparisons/fairness_support_gender.csv`
- `data/output/comparisons/fairness_metadata_gender.json`

## Published File Mapping

The exported CSV filenames keep the historical pipeline numbering, even though
the methods discussed in the paper are `M0`, `M1`, `M3`, `M6`, and `M7`.
For GitHub users, the mapping is:

| Paper method | Meaning | Published file |
|---|---|---|
| `M0` | Random baseline | `data/output/meal_uc1_m0.csv` |
| `M1` | Sequential greedy | `data/output/meal_uc1_m1.csv` |
| `M3` | Boosted Bandit | `data/output/meal_uc1_m2.csv` |
| `M6` | Exact Knapsack | `data/output/meal_uc1_m3.csv` |
| `M7` | Marginal Utility | `data/output/meal_uc1_m4.csv` |

This numbering is preserved only for file compatibility with the existing
pipeline and comparison scripts.

## Current Published Ranking

Latest canonical BEACON-style proxy ranking:

1. `M7` Marginal Utility
2. `M6` Exact Knapsack
3. `M1` Sequential greedy
4. `M3` Boosted Bandit
5. `M0` Random baseline

Summary:

| Method | Average Goodness | Average Volume |
|---|---:|---:|
| M0 | 0.6638 | 10.0000 |
| M1 | 0.8158 | 6.2302 |
| M3 | 0.7761 | 6.2693 |
| M6 | 0.8169 | 5.7268 |
| M7 | 0.8429 | 5.1203 |

## Current Published Fairness Snapshot

Composition-gap fairness under the canonical BEACON-style proxy:

| Method | Composition Gap |
|---|---:|
| M0 | 0.0061 |
| M1 | 0.0003 |
| M3 | 0.0020 |
| M6 | 0.0001 |
| M7 | 0.0006 |

Decision-level fairness under the canonical BEACON-style proxy:

| Method | SP | CSP | EO | PRP | TE |
|---|---:|---:|---:|---:|---:|
| M0 | 0.0721 | 0.0734 | 0.1015 | 0.1556 | 0.0000 |
| M1 | 0.0819 | 0.0591 | 0.0260 | 0.0365 | 0.0659 |
| M3 | 0.0721 | 0.0825 | 0.1015 | 0.1556 | 0.0000 |
| M6 | 0.0535 | 0.0559 | 0.0047 | 0.0085 | 0.0000 |
| M7 | 0.0164 | 0.0524 | 0.0385 | 0.1036 | 0.0000 |

## Notes For GitHub Users

- The published output root in this repo is `data/output/`.
- That output root is the BEACON-style proxy version.
- The intermediate source-generation folder is intentionally not part of the
  published release.
- The proxy output keeps the same returned bundles and rescales them under the
  BEACON-style proxy before generating summary and fairness tables.
