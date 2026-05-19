"""
Build the internal source outputs used by the canonical BEACON-style pipeline.

This script generates one CSV per method plus comparison summaries for the
source-generation stage. In the GitHub-facing pipeline, these files are written
to a temporary intermediate directory and then rescored into the published
BEACON-style proxy output root.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from boosted_bandit_postprocess import build_teaming_style_bandit_teams, lexical_item_ranking, create_fallback_teams_for_target, parse_results_db
from meal_methods import (
    build_scorer,
    load_boosted_test_split,
    load_core_data,
    rows_to_summary,
    run_crossproduct_method,
    run_standard_method,
)
from meal_utils import MEAL_SIZE, QUALITY_THRESH, build_output_row, pick_target_item


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOSTED_MATCHING_THRESHOLD = float(os.environ.get("BOOSTED_MATCHING_THRESHOLD", "0.3"))
BOOSTED_NUM_TEAMS = int(os.environ.get("BOOSTED_NUM_TEAMS", "10"))

METHOD_EXPORTS = [
    ("m0_random", "meal_uc1_m0.csv", "M0", "Random baseline"),
    ("m1_sequential", "meal_uc1_m1.csv", "M1", "Sequential greedy"),
    ("m4_boosted_bandit", "meal_uc1_m2.csv", "M3", "Boosted Bandit"),
    ("m2_knapsack_exact", "meal_uc1_m3.csv", "M6", "Knapsack"),
    ("m3_marginal_utility", "meal_uc1_m4.csv", "M7", "Marginal Utility"),
]


def get_output_root():
    return Path(os.environ.get("MEAL_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "_intermediate_legacy_source")))


def get_comparisons_root():
    return get_output_root() / "comparisons"


def get_bandit_dir():
    return Path(os.environ.get("MEAL_BANDIT_DIR", str(PROJECT_ROOT / "boosted_bandit")))


def get_boosted_results_db():
    return get_bandit_dir() / "test" / "results_team.db"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["boosted_test", "all"],
        default="all",
        help="User cohort for the final export.",
    )
    parser.add_argument(
        "--user-limit",
        type=int,
        default=None,
        help="Optional cap after split filtering.",
    )
    parser.add_argument(
        "--goodness-profile",
        choices=["legacy_meal", "meal", "teaming"],
        default=os.environ.get("MEAL_GOODNESS_PROFILE", "legacy_meal"),
        help="Goodness formula to use when scoring meal outputs.",
    )
    return parser.parse_args()


def build_user_stats(rows):
    grouped = {}
    for row in rows:
        uid = row["user_id"]
        scores = [float(score) for score in row["goodness"]]
        volume = len(scores)

        entry = grouped.setdefault(
            uid,
            {
                "user_id": uid,
                "name": row.get("name", ""),
                "meal_occasion": row.get("meal_occasion", ""),
                "sum_avg_goodness": 0.0,
                "sum_volume": 0.0,
                "row_count": 0,
            },
        )
        entry["sum_avg_goodness"] += float(np.mean(scores)) if scores else 0.0
        entry["sum_volume"] += volume
        entry["row_count"] += 1

    user_rows = []
    for uid in sorted(grouped):
        entry = grouped[uid]
        row_count = max(1, entry["row_count"])
        user_rows.append(
            {
                "user_id": uid,
                "name": entry["name"],
                "meal_occasion": entry["meal_occasion"],
                "avg_goodness_per_row": round(entry["sum_avg_goodness"] / row_count, 6),
                "volume": round(entry["sum_volume"] / row_count, 6),
                "row_count": entry["row_count"],
            }
        )

    return pd.DataFrame(user_rows)


def build_comparison_row(method_label, method_name, dataset_name, rows, user_stats):
    goodness = user_stats["avg_goodness_per_row"]
    volume = user_stats["volume"]
    return {
        "Method": method_label,
        "Method_name": method_name,
        "dataset": dataset_name,
        "G_mean": round(float(goodness.mean()), 6),
        "G_std": round(float(goodness.std(ddof=0)) if len(goodness) > 1 else 0.0, 6),
        "Volume_mean": round(float(volume.mean()), 6),
        "n_users": int(user_stats["user_id"].nunique()),
        "n_rows": int(len(rows)),
    }


def build_m2_boosted_rows(df_users, user_required, item_categories, scorer):
    bandit_dir = get_bandit_dir()
    results_db = get_boosted_results_db()
    if not results_db.exists():
        legacy_results_db = bandit_dir / "test" / "results_recommendation.db"
        if legacy_results_db.exists():
            results_db = legacy_results_db
        else:
            raise FileNotFoundError(f"Missing BoostSRL result database: {results_db}")

    user_probs, pos_items, neg_items = parse_results_db(
        results_db,
        item_categories,
        user_required,
    )

    rows = []
    skipped = []

    for idx, user_row in df_users.iterrows():
        uid = user_row["user_id"]

        required = user_required[uid]
        target_item = pick_target_item(required, item_categories, list(item_categories.keys()))
        teams = build_teaming_style_bandit_teams(
            uid,
            target_item=target_item,
            required=required,
            item_categories=item_categories,
            user_probs=user_probs,
            pos_items=pos_items,
            neg_items=neg_items,
            meal_size=MEAL_SIZE,
            n=BOOSTED_NUM_TEAMS,
            matching_threshold=BOOSTED_MATCHING_THRESHOLD,
            rng=random.Random(42 + idx),
        )
        if not teams:
            # build_teaming_style_bandit_teams should already return fallback teams,
            # but if it returns empty (edge case), generate lexical fallback here so
            # every user always appears in the output — matching Teaming behaviour.
            ranking = lexical_item_ranking(item_categories, required)
            teams = create_fallback_teams_for_target(
                ranking, target_item, BOOSTED_NUM_TEAMS, rng=random.Random(42 + idx)
            )
        if not teams:
            skipped.append(uid)
            continue

        scored = [scorer.score_team(required, team, item_categories) for team in teams]
        if scored:
            best = max(s["goodness"] for s in scored)
            threshold = best * QUALITY_THRESH
            filtered = [(t, s) for t, s in zip(teams, scored) if s["goodness"] >= threshold]
            if filtered:
                teams, scored = zip(*filtered)
                teams, scored = list(teams), list(scored)
        rows.append(build_output_row(user_row, target_item, teams, scored, user_required))

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(df_users)}]  {uid}")

    print(
        "Built Teaming-style boosted rows directly from BoostSRL results "
        f"for {len(rows)} users; skipped {len(skipped)} without usable bandit output."
    )
    return rows


# Backward-compatible alias for older notebook cells.
build_m4_rows = build_m2_boosted_rows


def main():
    args = parse_args()
    os.environ["MEAL_GOODNESS_PROFILE"] = args.goodness_profile
    output_root = get_output_root()
    comparisons_root = get_comparisons_root()
    output_root.mkdir(parents=True, exist_ok=True)
    comparisons_root.mkdir(parents=True, exist_ok=True)

    selected_user_ids = None
    if args.split == "boosted_test":
        selected_user_ids = load_boosted_test_split(PROJECT_ROOT)

    item_categories, user_required, df_users, all_items = load_core_data(
        selected_user_ids=selected_user_ids,
        user_limit=args.user_limit,
    )
    scorer = build_scorer()

    print(f"Users in export cohort: {len(df_users)}")
    print(f"Goodness profile: {args.goodness_profile}")

    exported = []
    detailed_summary_rows = []

    for method_name, output_name, method_label, method_display in METHOD_EXPORTS:
        print(f"\n=== Preparing {method_label} ({method_display}) ===")

        if method_name == "m4_boosted_bandit":
            rows = run_crossproduct_method(
                "m2_boosted_bandit",
                df_users,
                user_required,
                item_categories,
                all_items,
                scorer,
            )
        else:
            rows = run_crossproduct_method(
                method_name,
                df_users,
                user_required,
                item_categories,
                all_items,
                scorer,
            )

        output_path = output_root / output_name
        pd.DataFrame(rows).to_csv(output_path, index=False)

        user_stats = build_user_stats(rows)
        user_stats_path = comparisons_root / f"{output_name}_user_stats.csv"
        user_stats.to_csv(user_stats_path, index=False)

        exported.append(build_comparison_row(method_label, method_display, output_name, rows, user_stats))

        method_summary = rows_to_summary(method_name, rows)
        detailed_summary_rows.append(
            {
                "Method": method_label,
                "Method_name": method_display,
                "dataset": output_name,
                "Average goodness": round(float(user_stats["avg_goodness_per_row"].mean()), 6),
                "Average volume": round(float(user_stats["volume"].mean()), 6),
                **method_summary,
            }
        )

        print(f"Saved {output_path}")
        print(f"Saved {user_stats_path}")

    comparison_df = pd.DataFrame(exported).sort_values(["G_mean", "Volume_mean"], ascending=[False, True])
    comparison_path = comparisons_root / "comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    method_summary_df = pd.DataFrame(detailed_summary_rows).sort_values(
        ["Average goodness", "Average volume"], ascending=[False, True]
    )
    method_summary_path = comparisons_root / "method_comparison_summary.csv"
    method_summary_df[["Method", "Method_name", "dataset", "Average goodness", "Average volume"]].to_csv(
        method_summary_path,
        index=False,
    )

    detailed_path = comparisons_root / "method_comparison_detailed.csv"
    method_summary_df.to_csv(detailed_path, index=False)

    print(f"\nSaved comparison summary -> {comparison_path}")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved method comparison summary -> {method_summary_path}")
    print(f"Saved detailed method comparison -> {detailed_path}")


if __name__ == "__main__":
    main()
