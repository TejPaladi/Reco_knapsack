"""
meal_utils.py — Shared utilities for all meal recommendation notebooks.

Mirrors the role of M1.py / M3.py in the teaming notebooks:
  - Data loading
  - IDF skill weights
  - Target item selection
  - MealMetricScorer (patched MetricScorer for meal domain)
"""

import os, ast, math, sys

# Allow importing metrics_scorer from the same code/ directory
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import pandas as pd
import numpy as np
from metrics_scorer import MetricScorer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MEAL_SIZE      = 4    # hard cap on items per recommendation  (≈ TEAM_SIZE=5)
NUM_VARIATIONS = 10   # recommendation variants per user      (≈ NUM_TEAMS=10)
QUALITY_THRESH = 0.90 # M3 only: keep variant if ≥ anchor × this

GOODNESS_PROFILES = {
    # Historical meal pipeline behaviour used by the existing exported outputs.
    "legacy_meal": {
        "coefficients": (2 / 14, 2 / 14, 5 / 14, 5 / 14),
        "description": "Legacy meal scorer used in the existing output exports.",
    },
    # Paper formula previously used for the meal domain.
    "meal": {
        "coefficients": (-0.125, -0.125, 0.375, 0.375),
        "description": "Meal-domain goodness with heavier emphasis on coverage and robustness.",
    },
    # Teaming-domain formula: equal weight on penalty and reward terms.
    "teaming": {
        "coefficients": (-0.25, -0.25, 0.25, 0.25),
        "description": "Teaming-domain goodness reused for meal evaluation.",
    },
}

PROJECT_ROOT = os.path.dirname(CODE_DIR)


def get_input_dir():
    return os.environ.get("MEAL_INPUT_DIR", os.path.join(PROJECT_ROOT, "data", "input_data"))


def get_output_dir():
    return os.environ.get("MEAL_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "data", "_intermediate_legacy_source"))


def get_meal_categories_file():
    return os.environ.get("MEAL_CATEGORIES_FILE", "meal_categories.csv")


def get_user_requirements_file():
    return os.environ.get("MEAL_USERS_FILE", "user_meal_requirements.csv")

OUTPUT_COLS = [
    "meal_request_id", "user_id", "name",
    "meal_occasion", "required_categories",
    "target_item", "recommended_meals", "goodness",
    "avg_redundancy", "avg_setsize", "avg_coverage", "avg_krobust"
]

OUTPUT_FILE_ALIASES = {
    "m0_random": "m0",
    "m1_sequential": "m1",
    "m4_boosted_bandit": "m2",
    "m2_boosted_bandit": "m2",
    "m2_knapsack_v1": "m3",
    "m2_knapsack_exact": "m3",
    "m3_knapsack": "m3",
    "m3_dynamic_knapsack": "m4",
    "m3_marginal_utility": "m4",
    "m4_marginal_utility": "m4",
}

# ─────────────────────────────────────────────────────────────────────────────
# PATCHED METRIC SCORER
# ─────────────────────────────────────────────────────────────────────────────
class MealMetricScorer(MetricScorer):
    """
    Patched MetricScorer for the meal domain.

    Relative to the original Teaming scorer, this subclass keeps the meal-size
    normalization and the fixed k-robustness condition, and it exposes named
    goodness profiles so we can rerun the same meal outputs under different
    evaluation formulas without rewriting the pipeline.
    """

    def __init__(self, meal_size=MEAL_SIZE, goodness_profile="legacy_meal"):
        super().__init__()
        self.meal_size = meal_size
        if goodness_profile not in GOODNESS_PROFILES:
            valid = ", ".join(sorted(GOODNESS_PROFILES))
            raise ValueError(f"Unknown goodness profile '{goodness_profile}'. Expected one of: {valid}")
        self.goodness_profile = goodness_profile
        self.goodness_coefficients = GOODNESS_PROFILES[goodness_profile]["coefficients"]
        self.w_r, self.w_s, self.w_c, self.w_k = self.goodness_coefficients

    def calc_setsize(self, size=None):
        size = size or self.meal_size
        self.setsize = len(self.team) / size

    def calc_krobust(self):
        import itertools
        og_team        = self.team.copy()
        og_researchers = self.researchers.copy()
        self.calc_coverage()
        og_coverage    = self.coverage

        for i in range(1, len(og_team)):
            for subset in itertools.combinations(og_team, i):
                self.team        = og_team.copy()
                self.researchers = og_researchers.copy()
                for j in subset:
                    self.team.remove(j)
                    self.researchers.pop(j)
                self.calc_coverage()
                # Fixed condition: ≥ half demand covered AND enough members to drop one
                if og_coverage >= 0.5 and og_coverage <= self.coverage and len(og_team) >= 2:
                    self.krobust += 1
                    break

        self.team        = og_team.copy()
        self.researchers = og_researchers.copy()
        self.coverage    = og_coverage
        if self.krobust > 0:
            self.krobust = 1

    def score_team(self, demand, team, item_categories):
        """
        Score a single team recommendation.

        Args:
            demand         : set/list of required tags
            team           : list of meal item names
            item_categories: dict {item_name: set(tags)}

        Returns:
            dict with redundancy, setsize, coverage, krobust, goodness
        """
        self.demand      = list(demand)
        self.team        = list(team)
        self.researchers = {item: item_categories.get(item, set()) for item in team}
        # reset scores only (not weights)
        self.redundancy = self.setsize = self.coverage = self.krobust = self.goodness = 0
        self.calc_redundancy()
        self.calc_setsize()
        self.calc_coverage()
        self.calc_krobust()
        self.goodness_measure()
        return {
            "redundancy": round(self.redundancy, 4),
            "setsize"   : round(self.setsize,    4),
            "coverage"  : round(self.coverage,   4),
            "krobust"   : self.krobust,
            "goodness"  : round(self.goodness,   4),
        }

    def goodness_measure(self):
        w_r, w_s, w_c, w_k = self.goodness_coefficients
        self.goodness = (
            w_r * self.redundancy
            + w_s * self.setsize
            + w_c * self.coverage
            + w_k * self.krobust
        )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(user_limit=None):
    """
    Load meal_categories.csv and user_meal_requirements.csv.
    Returns:
        item_categories : dict {meal_name: set(tags)}
        user_required   : dict {user_id:   set(required_tags)}
        df_users        : DataFrame of user rows
        all_items       : list of all meal item names
    """
    input_dir = get_input_dir()
    meal_categories_file = get_meal_categories_file()
    user_requirements_file = get_user_requirements_file()

    df_meals = pd.read_csv(os.path.join(input_dir, meal_categories_file))
    df_users = pd.read_csv(os.path.join(input_dir, user_requirements_file))

    if user_limit:
        df_users = df_users.head(user_limit).reset_index(drop=True)

    item_categories = {}
    for _, row in df_meals.iterrows():
        cats = ast.literal_eval(row["categories"])
        item_categories[row["meal_name"]] = set(cats)

    user_required = {}
    for _, row in df_users.iterrows():
        cats = ast.literal_eval(row["required_categories"])
        user_required[row["user_id"]] = set(cats)

    all_items = list(item_categories.keys())
    return item_categories, user_required, df_users, all_items


# ─────────────────────────────────────────────────────────────────────────────
# IDF WEIGHTS  (mirrors skill_weights in MUBO notebook — used by M3 only)
# ─────────────────────────────────────────────────────────────────────────────
def build_idf_weights(item_categories):
    """
    Compute IDF weight per tag:
        weight(t) = log(N / (count(t) + 1)) + 1
    Rare tags → high weight.  Common tags → low weight.
    """
    total = len(item_categories)
    tag_counts = {}
    for cats in item_categories.values():
        for c in cats:
            tag_counts[c] = tag_counts.get(c, 0) + 1
    return {
        c: math.log(total / (cnt + 1)) + 1
        for c, cnt in tag_counts.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ITEM SELECTION  (= pick target researcher in teaming)
# ─────────────────────────────────────────────────────────────────────────────
def pick_target_item(required, item_categories, all_items):
    """Best single item covering the most required tags (anchor / target)."""
    return max(all_items, key=lambda i: len(item_categories[i] & required))


# ─────────────────────────────────────────────────────────────────────────────
# BUILD OUTPUT ROW  (same schema as teaming output CSVs)
# ─────────────────────────────────────────────────────────────────────────────
def build_output_row(user_row, target_item, teams, scored_teams, user_required, row_id=None):
    """
    Sorts teams by goodness descending and returns a dict matching OUTPUT_COLS.
    scored_teams: list of dicts from MealMetricScorer.score_team()

    row_id: optional explicit ID — use when one user produces multiple rows
            (e.g. cross-product mode where each user × target_item is a row).
            Defaults to the user's original CSV index.
    """
    paired  = sorted(zip(scored_teams, teams), key=lambda x: x[0]["goodness"], reverse=True)
    s_teams  = [p[1]                  for p in paired]
    s_scores = [p[0]["goodness"]      for p in paired]

    uid = user_row["user_id"]
    return {
        "meal_request_id"    : row_id if row_id is not None else int(user_row.get("Unnamed: 0", user_row.name)),
        "user_id"            : uid,
        "name"               : user_row.get("name", ""),
        "meal_occasion"      : user_row["meal_occasion"],
        "required_categories": str(user_required[uid]),
        "target_item"        : target_item,
        "recommended_meals"  : s_teams,
        "goodness"           : s_scores,
        "avg_redundancy"     : round(np.mean([s["redundancy"] for s in scored_teams]), 4),
        "avg_setsize"        : round(np.mean([s["setsize"]    for s in scored_teams]), 4),
        "avg_coverage"       : round(np.mean([s["coverage"]   for s in scored_teams]), 4),
        "avg_krobust"        : round(np.mean([s["krobust"]    for s in scored_teams]), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS  (Table 3.6 style — mirrors teaming notebooks)
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(method_name, rows):
    """
    Print Table 3.6-style stats matching the Teaming paper.

    Goodness (G) and Volume (#T) are computed *per user* first (averaging
    across all target items for that user), then mean ± STD is reported
    across users — exactly as the paper reports "per researcher (r_j)".
    """
    # Group rows by user
    from collections import defaultdict
    user_rows = defaultdict(list)
    for row in rows:
        user_rows[row["user_id"]].append(row)

    per_user_g, per_user_vol, all_sz, all_cov, all_red = [], [], [], [], []
    for _, u_rows in user_rows.items():
        # Mean goodness across all (user, target_item) pairs for this user
        user_g = [float(np.mean(row["goodness"])) for row in u_rows if row["goodness"]]
        per_user_g.append(float(np.mean(user_g)) if user_g else 0.0)

        # Mean volume across all (user, target_item) pairs for this user
        user_vol = [len(row["goodness"]) for row in u_rows]
        per_user_vol.append(float(np.mean(user_vol)))

        for row in u_rows:
            all_sz.extend([len(team) for team in row["recommended_meals"]])
            all_cov.append(row["avg_coverage"])
            all_red.append(row["avg_redundancy"])

    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    print(f"  Avg Goodness (G)  : {np.mean(per_user_g):.4f} ± {np.std(per_user_g):.4f}")
    print(f"  Avg Volume  (#T)  : {np.mean(per_user_vol):.2f}   (lower = more selective)")
    print(f"  Avg Team Size     : {np.mean(all_sz):.2f}")
    print(f"  Avg Coverage      : {np.mean(all_cov):.4f}")
    print(f"  Avg Redundancy    : {np.mean(all_red):.4f}")
    print(f"  Quality signal    : G↑ + #T↓ = better method")
    return {
        "method"        : method_name,
        "avg_goodness"  : round(float(np.mean(per_user_g)),  4),
        "std_goodness"  : round(float(np.std(per_user_g)),   4),
        "avg_volume"    : round(float(np.mean(per_user_vol)), 2),
        "avg_team_size" : round(float(np.mean(all_sz)),  2),
        "avg_coverage"  : round(float(np.mean(all_cov)), 4),
        "avg_redundancy": round(float(np.mean(all_red)), 4),
    }


def save_output(rows, method_name):
    """Save a CSV into the active pipeline output folder."""
    canonical_name = OUTPUT_FILE_ALIASES.get(method_name, method_name)
    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    output_dir = get_output_dir()
    path = os.path.join(output_dir, f"meal_uc1_{canonical_name}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved → {path}  ({len(df)} rows)")
    return df
