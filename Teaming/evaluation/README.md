### Metrics:

1. Redundancy of skill set - 0: no overlap, 1: total redundancy 
2. Set size - revenue per person - what is the ideal team size? (Usually, it's [total funding]/[$50K])
3. Coverage - skill set covered
4. k-robustness - how many team members can we remove before the team starts to fall apart? Can only happen if redundancy=1
5. Teaming potential - bonus points if team members already know each other very well (e.g., past paper/project collaborations)
6. Diversity - inverse to redundancy

### Fairness

- `generate_fairness_tables.py`: builds Table 2 / Table 3 style fairness summaries for the available UC1 methods, now including `M6 = teaming_uc1_m6.csv` and `M7 = teaming_uc1_m7.csv`. The current `M6` and `M7` outputs both use the `M1` lexical pseudo-skill stage before their optimization logic. The script supports a paper-style first-name ensemble (`gender-guesser` + `Genderize` + `SexMachine`) with biography-pronoun fallback, and defaults to a `100 proposal / 46 anchor researcher` paper-style subset saved under `evaluation/output_paper_subset/`. Exported fairness tables include readable method names in their headers. Pass `--full-dataset --results-dir evaluation/output/` to reproduce the larger full-data run.
- `M6_M7_publishability_notes.md`: literature-backed notes on how to improve the experimental knapsack-style Teaming methods so they can be positioned more credibly in a paper.
