"""
Rescore canonical meal outputs with the closest BEACON-style proxy supported by
the current category-only meal dataset.

Why a proxy?
- Exact BEACON needs duplicate score (dm), meal coverage score (mc), and
  user-constraint score (uc) built from role labels plus ingredient preference
  annotations such as dairy/meat/nuts.
- Our current meal dataset only provides category-level requirements and meal
  category tags.

Closest proxy used here:
- dm_proxy(bundle) = unique_items / total_items
- mc_proxy(bundle) = covered_required_categories / required_categories
- uc_proxy(bundle) = average per-item positive alignment, i.e.
  mean_i( matched_required_categories(item_i) / required_categories )
- goodness_proxy(bundle) = (dm_proxy + mc_proxy + uc_proxy) / 3

This mirrors the BEACON table-style combined metric `(uc + dm + mc) / 3`
without claiming to reproduce the full role/ingredient formulation.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "_intermediate_legacy_source"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"
DEFAULT_MEALS_FILE = PROJECT_ROOT / "data" / "input_data" / "meal_categories.csv"
METHOD_SPECS = [
    ("M0", "Random baseline", "meal_uc1_m0.csv"),
    ("M1", "Sequential greedy", "meal_uc1_m1.csv"),
    ("M3", "Boosted Bandit", "meal_uc1_m2.csv"),
    ("M6", "Knapsack", "meal_uc1_m3.csv"),
    ("M7", "Marginal Utility", "meal_uc1_m4.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--meals-file", default=str(DEFAULT_MEALS_FILE))
    return parser.parse_args()


def load_item_categories(meals_file: Path) -> dict[str, set[str]]:
    df = pd.read_csv(meals_file)
    return {
        row["meal_name"]: set(ast.literal_eval(row["categories"]))
        for _, row in df.iterrows()
    }


def score_bundle(bundle: list[str], required: set[str], item_categories: dict[str, set[str]]) -> tuple[float, float, float, float]:
    if not bundle:
        return 0.0, 0.0, 0.0, 0.0

    denom = max(1, len(required))
    dm = len(set(bundle)) / len(bundle)

    covered = set()
    uc_parts = []
    for item in bundle:
        cats = item_categories.get(item, set())
        covered |= cats
        uc_parts.append(len(cats & required) / denom)

    mc = len(covered & required) / denom
    uc = sum(uc_parts) / len(uc_parts)
    goodness = (dm + mc + uc) / 3
    return goodness, dm, mc, uc


def build_user_stats(rows: list[dict]) -> pd.DataFrame:
    grouped: dict[str, dict[str, float | int | str]] = {}
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
        entry["sum_avg_goodness"] += sum(scores) / len(scores) if scores else 0.0
        entry["sum_volume"] += volume
        entry["row_count"] += 1

    user_rows = []
    for uid in sorted(grouped):
        entry = grouped[uid]
        row_count = max(1, int(entry["row_count"]))
        user_rows.append(
            {
                "user_id": uid,
                "name": entry["name"],
                "meal_occasion": entry["meal_occasion"],
                "avg_goodness_per_row": round(float(entry["sum_avg_goodness"]) / row_count, 6),
                "volume": round(float(entry["sum_volume"]) / row_count, 6),
                "row_count": row_count,
            }
        )
    return pd.DataFrame(user_rows)


def build_comparison_row(method_label: str, method_name: str, dataset_name: str, rows: list[dict], user_stats: pd.DataFrame) -> dict:
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


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    comparisons_root = output_root / "comparisons"
    output_root.mkdir(parents=True, exist_ok=True)
    comparisons_root.mkdir(parents=True, exist_ok=True)

    item_categories = load_item_categories(Path(args.meals_file).resolve())

    comparison_rows = []
    detailed_rows = []

    for method_label, method_name, filename in METHOD_SPECS:
        source_path = source_root / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source output: {source_path}")

        df = pd.read_csv(source_path)
        rescored_rows = []
        avg_dm_values = []
        avg_mc_values = []
        avg_uc_values = []

        for _, row in df.iterrows():
            required = set(ast.literal_eval(row["required_categories"]))
            bundles = ast.literal_eval(row["recommended_meals"])

            goodness_values = []
            dm_values = []
            mc_values = []
            uc_values = []
            for bundle in bundles:
                goodness, dm, mc, uc = score_bundle(bundle, required, item_categories)
                goodness_values.append(round(goodness, 6))
                dm_values.append(round(dm, 6))
                mc_values.append(round(mc, 6))
                uc_values.append(round(uc, 6))

            row_dict = row.to_dict()
            row_dict["goodness"] = goodness_values
            row_dict["avg_dm_proxy"] = round(sum(dm_values) / len(dm_values), 6) if dm_values else 0.0
            row_dict["avg_mc_proxy"] = round(sum(mc_values) / len(mc_values), 6) if mc_values else 0.0
            row_dict["avg_uc_proxy"] = round(sum(uc_values) / len(uc_values), 6) if uc_values else 0.0
            rescored_rows.append(row_dict)

            avg_dm_values.append(row_dict["avg_dm_proxy"])
            avg_mc_values.append(row_dict["avg_mc_proxy"])
            avg_uc_values.append(row_dict["avg_uc_proxy"])

        output_path = output_root / filename
        pd.DataFrame(rescored_rows).to_csv(output_path, index=False)

        user_stats = build_user_stats(rescored_rows)
        user_stats_path = comparisons_root / f"{filename}_user_stats.csv"
        user_stats.to_csv(user_stats_path, index=False)

        comparison_rows.append(build_comparison_row(method_label, method_name, filename, rescored_rows, user_stats))
        detailed_rows.append(
            {
                "Method": method_label,
                "Method_name": method_name,
                "dataset": filename,
                "Average goodness": round(float(user_stats["avg_goodness_per_row"].mean()), 6),
                "Average volume": round(float(user_stats["volume"].mean()), 6),
                "avg_dm_proxy": round(sum(avg_dm_values) / len(avg_dm_values), 6),
                "avg_mc_proxy": round(sum(avg_mc_values) / len(avg_mc_values), 6),
                "avg_uc_proxy": round(sum(avg_uc_values) / len(avg_uc_values), 6),
            }
        )

        print(f"Saved rescored output -> {output_path}")
        print(f"Saved user stats      -> {user_stats_path}")

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["G_mean", "Volume_mean"], ascending=[False, True])
    comparison_path = comparisons_root / "comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    method_summary_df = pd.DataFrame(detailed_rows).sort_values(["Average goodness", "Average volume"], ascending=[False, True])
    method_summary_path = comparisons_root / "method_comparison_summary.csv"
    method_summary_df[["Method", "Method_name", "dataset", "Average goodness", "Average volume"]].to_csv(
        method_summary_path,
        index=False,
    )
    detailed_path = comparisons_root / "method_comparison_detailed.csv"
    method_summary_df.to_csv(detailed_path, index=False)

    metadata = {
        "profile": "beacon_proxy",
        "description": (
            "Closest BEACON-style proxy supported by the current category-only meal dataset: "
            "dm_proxy = unique_items/total_items, "
            "mc_proxy = covered_required_categories/required_categories, "
            "uc_proxy = average per-item positive alignment, "
            "goodness = (dm_proxy + mc_proxy + uc_proxy) / 3."
        ),
        "source_stage_note": (
            "These published proxy outputs are computed by rescoring internally generated "
            "source bundles. The source-generation directory may be temporary and may not "
            "be retained in the GitHub-facing release."
        ),
        "source_root": str(source_root),
        "source_root_retained": source_root.exists(),
        "output_root": str(output_root),
        "meals_file": str(Path(args.meals_file).resolve()),
        "methods": {label: {"name": name, "filename": filename} for label, name, filename in METHOD_SPECS},
    }
    metadata_path = comparisons_root / "beacon_proxy_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved comparison summary -> {comparison_path}")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved method comparison summary -> {method_summary_path}")
    print(f"Saved detailed method comparison -> {detailed_path}")
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
