# Reco Knapsack

This repository collects three recommendation and team-formation workspaces
used for knapsack-style recommendation experiments:

- `IITR-Teaming/` - IITR research team recommendation pipeline and published outputs.
- `Meal/` - meal bundle recommendation pipeline with outputs.
- `Teaming/` - USC research team recommendation notebooks, scripts, and published outputs.

Each workspace has its own README with deeper setup, runner, data, and output
details:

- `IITR-Teaming/README.md`
- `Meal/README.md`
- `Teaming/README.md`

## Repository Layout

```text
Reco_knapsack/
├── IITR-Teaming/
│   ├── code/
│   ├── data/
│   └── evaluation/
├── Meal/
│   ├── boosted_bandit/
│   ├── code/
│   └── data/
└── Teaming/
    ├── code/
    ├── data/
    └── evaluation/
```

## Where To Start

The main published outputs for each project are under that workspace's
`data/output/` folder.

### IITR-Teaming

`IITR-Teaming/` is a cleaned IITR research team recommendation workspace for a
fixed slice of `100` proposal calls by `46` researchers.

- Inputs: `IITR-Teaming/data/v0_data/archive_proposals.csv` and
  `IITR-Teaming/data/v0_data/researchers.csv`.
- Published outputs: `IITR-Teaming/data/output/`.
- Canonical runner from `IITR-Teaming/`:

```bash
python code/run_v0_teaming_pipeline.py
```

- Fairness-only runner from `IITR-Teaming/`:

```bash
python evaluation/generate_fairness_tables.py
```

Published methods are `M0` random, `M1` lexical matching, `M2` semantic mapper,
`M3` BoostSRL, `M6` exact knapsack coverage, and `M7` marginal-utility
knapsack. The current published ranking is `M7`, `M3`, `M1`, `M2`, `M6`, `M0`.

Legacy paper-style outputs are retained under `IITR-Teaming/data/v0_teaming_paper_style/`
and `IITR-Teaming/evaluation/output_v0_paper_style/`; new users should start
with `IITR-Teaming/data/output/`.

### Meal

`Meal/` builds meal bundles for user meal requests and evaluates them with a
BEACON-style proxy score:

```text
G_BP = (dm_proxy + mc_proxy + uc_proxy) / 3
```

- Inputs: `Meal/data/input_data/meal_categories.csv` and
  `Meal/data/input_data/user_meal_requirements.csv`.
- Published outputs: `Meal/data/output/`.
- Canonical runner from `Meal/`:

```bash
python code/run_pipeline.py
```

The pipeline generates source bundles, rescales the returned bundles with the
BEACON-style proxy, publishes results to `Meal/data/output/`, and creates
comparison and fairness tables. The BoostSRL assets used by the boosted bandit
method are bundled under `Meal/boosted_bandit/`, including the required jar
files. Java is required for those BoostSRL steps.

Published method files map to `M0` random, `M1` sequential greedy, `M3` boosted
bandit, `M6` exact knapsack, and `M7` marginal utility. The current published
ranking is `M7`, `M6`, `M1`, `M3`, `M0`.

### Teaming

`Teaming/` is the USC research team recommendation workspace. It is
notebook-first rather than a single one-command pipeline.

- Inputs: `Teaming/data/v1_input_files/`.
- Published outputs: `Teaming/data/output/`.
- Maintained source output root:
  `Teaming/data/v1_output_teaming/teaming_1698proposals_316researchers/`.
- Paper-facing fairness source outputs:
  `Teaming/evaluation/output_paper_subset/`.
- Notebook entry points: `Teaming/code/teaming_uc1_m0.ipynb` through
  `Teaming/code/teaming_uc1_m3.ipynb`, plus `Teaming/code/teaming_uc1_m6.ipynb`,
  `Teaming/code/teaming_uc1_m7.ipynb`, and `Teaming/code/Results.ipynb`.
- Script entry points: `Teaming/code/M6.py`, `Teaming/code/M7.py`,
  `Teaming/code/tune_m7.py`, and `Teaming/evaluation/generate_fairness_tables.py`.

Published methods are `M0` random, `M1` lexical matching, `M2` semantic mapper,
`M3` BoostSRL, `M6` exact knapsack coverage, and `M7` marginal-utility
knapsack. The current published ranking is `M7`, `M3`, `M6`, `M2`, `M1`, `M0`.

## Environment Notes

Virtual environments are intentionally not committed. Create a local Python
environment before running scripts or notebooks. Some workspaces also use
BoostSRL jar files and require Java for those stages.

## Version-Control Notes

Local virtual environments, Python caches, notebook checkpoints, and macOS
metadata are ignored.

Several historical/generated Teaming CSV files are larger than GitHub's 100 MB
regular Git object limit. They are intentionally ignored unless Git LFS or an
external artifact store is configured. The copied files can remain available
locally, but the normal Git push excludes them. The pushable published outputs
remain in the project folders, including each workspace's `data/output/`
contents.
