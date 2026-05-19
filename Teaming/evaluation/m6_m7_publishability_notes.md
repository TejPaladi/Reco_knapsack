# M6 / M7 Publishability Notes

## Current method map

- `M6`: [M6.py](/Users/tej/Desktop/Recommendation_Projects/Teaming/code/M6.py)
- `M6` legacy notebook: [teaming_uc1_m6_legacy_greedy_coverage.ipynb](/Users/tej/Desktop/Recommendation_Projects/Teaming/code/teaming_uc1_m6_legacy_greedy_coverage.ipynb)
- `M7`: [M7.py](/Users/tej/Desktop/Recommendation_Projects/Teaming/code/M7.py)
- `M7` legacy notebook: [teaming_uc1_m7_legacy_dynamic_knapsack_MUBO.ipynb](/Users/tej/Desktop/Recommendation_Projects/Teaming/code/teaming_uc1_m7_legacy_dynamic_knapsack_MUBO.ipynb)

## Short answer

The current `M6` and `M7` notebooks are useful experimental baselines, but they are not yet framed or implemented in the way most publishable team-formation papers are. The literature around expert or research team formation usually uses one of these formulations instead:

- graph optimization over collaboration networks
- integer programming / branch-and-bound / branch-and-price
- generalized assignment / set covering
- submodular maximization under cardinality or knapsack constraints
- multi-objective evolutionary optimization

I did not find a strong canonical line of expert-team-formation papers that simply present the method as a plain "knapsack" algorithm. The more publishable route is to either:

1. make `M6` a true exact optimization baseline with a transparent mathematical formulation, or
2. make `M7` a principled submodular/knapsack method with explicit approximation-aware greedy selection.

## Relevant literature

- Lappas et al. is the classic starting point for communication-aware expert team formation using social-network structure and communication-cost objectives.
- Gajewar and Das Sarma, *Multi-skill Collaborative Teams based on Densest Subgraphs* (arXiv:1102.3340): treats team formation as graph optimization with compatibility structure rather than plain coverage scoring.
- Rangapuram et al., *Towards Realistic Team Formation in Social Networks based on Densest Subgraphs* (arXiv:1505.06661): explicitly adds practical constraints such as designated leaders, team-size or cost restrictions, and locality.
- Berktaş and Yaman, *A Branch-and-Bound Algorithm for Team Formation on Social Networks* (INFORMS Journal on Computing, 2021): formulates team formation as a constrained quadratic set covering problem with communication-cost objectives.
- Muniz and Flamand, *A column generation approach for the team formation problem* (Computers & Operations Research, 2024): shows that solver-based decomposition methods can give near-optimal solutions for richer team-formation variants.
- Amanatidis et al., *Fast Adaptive Non-Monotone Submodular Maximization Subject to a Knapsack Constraint* (JAIR 2022, arXiv:2007.05014): not a team-formation paper by itself, but directly relevant because it treats team formation as a natural application area for knapsack-constrained submodular optimization.

## Current local weaknesses

### M6

- `M6` is now a true exact anchor-fixed knapsack baseline implemented in `code/M6.py`.
- It uses dynamic programming to maximize distinct proposal-skill coverage under a 5-seat budget, with the anchor researcher forced into the team.
- The older greedy notebook has been retained only as a legacy reference and should not be presented as the canonical `M6` method anymore.
- The method still depends on the shared Teaming skill files, so a future paper version would be stronger if the skill extraction pipeline were packaged more explicitly as part of the method description.

### M7

- `M7` is now a cleaner marginal-utility extension of `M6`, implemented in `code/M7.py`.
- It uses weighted diminishing-return utility, a per-seat cost, and shortlist-based exact enumeration instead of the older stochastic greedy notebook flow.
- This is a much stronger methodological story than the old notebook, but it is still shortlist-based rather than globally exact across the full researcher population.
- It still has no explicit collaboration-graph or communication-cost term, so it remains a skills-first optimizer rather than a social-network team-formation model.
- For publishability, the biggest next step would be adding an explicit compatibility term and benchmarking shortlist sensitivity.

## What to implement next

### Upgrade path A: make M6 a true exact baseline

Use a mixed-integer formulation on a shortlist of candidate researchers:

- Decision variable `x_i`: choose researcher `i`
- Decision variable `y_s`: required skill `s` is covered
- Optional decision variable `z_ij`: both `i` and `j` are selected, for collaboration terms

Possible objective:

`maximize  sum_s w_s y_s + lambda * compatibility(T) - mu * team_cost(T) - rho * redundancy(T)`

Subject to:

- anchor researcher is selected
- team size budget
- skill coverage constraints
- optional communication-diameter or pairwise-distance constraints

Why this helps:

- the method becomes honestly "exact" on the shortlisted problem
- it aligns naturally with branch-and-bound / integer-programming team-formation literature
- you can report optimality gaps on small and medium instances

Good practical implementation path:

- shortlist top `K` candidates from `M1`, `M2`, or `M3`
- solve the exact model with Pyomo + Gurobi or OR-Tools CP-SAT
- compare exact shortlist optimum against the current greedy `M6`

### Upgrade path B: make M7 a publishable submodular-knapsack method

Recast the method as budgeted submodular maximization:

`F(T) = coverage(T) + lambda * compatibility(T) - rho * redundancy(T)`

where:

- `coverage(T)` uses weighted skill coverage
- compatibility comes from a coauthor/social graph
- redundancy is penalized with diminishing returns

Then optimize under:

- cardinality constraint or explicit per-person cost budget
- anchor inclusion

Good practical implementation path:

- use lazy greedy or stochastic greedy as the main scalable solver
- add a true per-person cost instead of only a global seat-cost constant
- document whether the objective is monotone or non-monotone
- compare against an exact solver on small shortlisted instances

Why this helps:

- the method becomes much easier to justify theoretically
- the "knapsack" label becomes accurate
- it connects directly to constrained submodular optimization literature

## Strong improvements for both methods

- Add a collaboration graph term:
  Use coauthorship, shared department, or publication overlap so team quality is not based only on skill coverage.

- Separate optimization from evaluation:
  Optimize one explicit objective, then evaluate with ULTRA and fairness afterward. Right now the story is blurry because the pipelines partly reuse other methods' artifacts.

- Remove duplicated output padding:
  Repeated best-team padding is not publication-grade. Keep only unique alternatives and report when fewer than `N` good variants exist.

- Add deterministic reproducibility:
  Fix seeds, store configs, and export run metadata for every notebook run.

- Add ablations:
  Report the effect of turning off compatibility, diminishing returns, seat cost, and shortlist size.

- Add exact-vs-heuristic comparisons:
  Even a small exact benchmark subset would make the paper much stronger.

- Be careful with method naming:
  If `M6` stays greedy, rename it as greedy coverage or greedy coverage-with-anchor instead of knapsack.
  If `M7` stays greedy, rename it as marginal-utility greedy or budgeted marginal utility.

## Recommended repo plan

1. Keep `M6.py` as the exact baseline and the renamed legacy notebook only as historical context.
2. Add a solver-backed `M6_exact_shortlist.py` or notebook that extends the exact model with explicit shortlist controls or collaboration terms.
3. Add a graph-aware `M7_submodular.py` or notebook with:
   - weighted coverage
   - compatibility graph term
   - explicit budget
   - lazy-greedy optimization
4. Benchmark all of `M1`-`M3`, `M6`, `M7`, plus the improved exact/submodular variants on:
   - ULTRA goodness
   - team size
   - fairness
   - runtime
   - solution diversity
5. Write the paper around:
   - one exact optimizer baseline
   - one scalable graph-aware heuristic
   - one fairness analysis section

## Bottom line

If the goal is publishability, the strongest move is not to defend the current notebooks as-is. The strongest move is to use them as stepping stones:

- `M6` becomes the exact optimization baseline
- `M7` becomes the scalable submodular/graph-aware heuristic

That gives you a clean and publishable methodological story.
