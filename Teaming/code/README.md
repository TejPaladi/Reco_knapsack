This folder contains the code and methods used for the ULTRA-Matching system.

### Methods:
- [M0_Matching_Notebook.ipynb](https://github.com/ai4society/Ultra-Matching/blob/likitha/code/code/M0_Matching_Notebook.ipynb "M0_Matching_Notebook.ipynb") - original code containing the first matching algorithm (M0 - string matching), where: if any one of the researcher's interests match with any if the terms in proposal description, then recommend that proposal to the researcher.
- [M1.ipynb](https://github.com/ai4society/Ultra-Matching/blob/likitha/code/code/M1.ipynb "M1.ipynb") - semantic query matching. 
- `teaming_uc1_m6.ipynb` - canonical notebook entry point for `M6`; running it regenerates `teaming_uc1_m6.csv` with the latest exact knapsack logic from `M6.py`.
- `M6.py` - canonical exact knapsack baseline for Teaming UC1. It first uses the `M1` string-matching pseudo-skill pipeline to build a proposal-specific lexical shortlist, keeps the anchor researcher fixed, and then solves the remaining seats with dynamic programming.
- `teaming_uc1_m6_legacy_greedy_coverage.ipynb` - legacy exploratory notebook from the earlier greedy coverage experiment that previously wore the `M6` label.
- `teaming_uc1_m7.ipynb` - canonical notebook entry point for `M7`; running it regenerates `teaming_uc1_m7.csv` with the latest marginal-utility knapsack logic from `M7.py`.
- `M7.py` - canonical marginal-utility extension of `M6`. It uses the same `M1` lexical prefilter first, then scores shortlist teams with weighted diminishing-return utility and a seat-cost penalty, and exports a diverse representative set whose width is derived from row-level branching structure and proposal complexity.
- `teaming_uc1_m7_legacy_dynamic_knapsack_MUBO.ipynb` - earlier exploratory notebook that informed the current `M7` implementation.

### APIs:
- [mapper4-main](https://github.com/ai4society/Ultra-Matching/tree/likitha/likitha/code/mapper4-main "mapper4-main") - code for [Text to Classification Mapper](http://casy.cse.sc.edu/mapper/). Given a subject/topic entry and matching threshold, the mapper returns the corresponding ACM/JEL classification codes/text. 
