from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import M7
import generate_fairness_tables as fairness


OUTPUT_BASE = PROJECT_ROOT / "data" / "v1_output_teaming" / "teaming_1698proposals_316researchers"
TUNING_DIR = CODE_DIR / "tuning_outputs"
BASELINE_METHOD_FILES = [
    "teaming_uc1_m0.csv",
    "teaming_uc1_m1.csv",
    "teaming_uc1_m2.csv",
    "teaming_uc1_m3.csv",
    "teaming_uc1_m6.csv",
]


def parse_float_list(raw_value: str) -> list[float]:
    return [float(piece.strip()) for piece in raw_value.split(",") if piece.strip()]


def parse_int_list(raw_value: str) -> list[int]:
    return [int(piece.strip()) for piece in raw_value.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune Teaming M7 across alpha, seat-cost multiplier, and shortlist size. "
            "The script runs a coarse sweep first, then a paper-style finalist round."
        )
    )
    parser.add_argument("--alphas", default="0.35,0.5,0.65")
    parser.add_argument("--cost-multipliers", default="0.14,0.2,0.26")
    parser.add_argument("--max-candidates", default="14,18,22")
    parser.add_argument("--team-size", type=int, default=M7.DEFAULT_TEAM_SIZE)
    parser.add_argument("--num-teams", type=int, default=M7.DEFAULT_NUM_TEAMS)
    parser.add_argument("--coarse-proposals", type=int, default=40)
    parser.add_argument("--coarse-anchors", type=int, default=25)
    parser.add_argument("--final-proposals", type=int, default=100)
    parser.add_argument("--final-anchors", type=int, default=46)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--lexical-shortlist", type=int, default=max(M7.DEFAULT_LEXICAL_SHORTLIST, 30))
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument(
        "--gender-inference-cache",
        default=str(PROJECT_ROOT / "evaluation" / "output_paper_subset" / "fairness_gender_inference.csv"),
    )
    parser.add_argument("--results-dir", default=str(TUNING_DIR))
    return parser.parse_args()


def subset_skeleton_rows(
    skeleton_rows: list[dict[str, str]],
    proposal_count: int,
    anchor_count: int,
) -> list[dict[str, str]]:
    proposal_ids = []
    seen_proposals = set()
    for row in skeleton_rows:
        proposal_id = row["proposal_id"]
        if proposal_id not in seen_proposals:
            seen_proposals.add(proposal_id)
            proposal_ids.append(proposal_id)
        if len(proposal_ids) >= proposal_count:
            break

    proposal_filtered = [row for row in skeleton_rows if row["proposal_id"] in set(proposal_ids)]

    anchor_names = []
    seen_anchors = set()
    for row in proposal_filtered:
        anchor = row["researcher_name"]
        if anchor not in seen_anchors:
            seen_anchors.add(anchor)
            anchor_names.append(anchor)
        if len(anchor_names) >= anchor_count:
            break

    anchor_set = set(anchor_names)
    return [row for row in proposal_filtered if row["researcher_name"] in anchor_set]


def compute_candidate_rows(
    subset_rows: list[dict[str, str]],
    proposal_payloads: dict[str, dict[str, object]],
    proposal_pseudo_skills: dict[str, dict[str, set[str]]],
    alpha: float,
    cost_multiplier: float,
    max_candidates: int,
    team_size: int,
    num_teams: int,
    round_digits: int,
) -> list[dict[str, object]]:
    rows = []
    for row in subset_rows:
        proposal_link = row["proposal_link"]
        payload = proposal_payloads[proposal_link]
        required_skills = payload["required_skills"]
        anchor = row["researcher_name"]
        pseudo_skill_map = proposal_pseudo_skills.get(proposal_link, {})

        teams = M7.generate_marginal_utility_teams(
            anchor=anchor,
            proposal_payload=payload,
            alpha=alpha,
            cost_multiplier=cost_multiplier,
            team_size=team_size,
            num_teams=num_teams,
            max_candidates=max_candidates,
        )
        scored_variants = []
        for team in teams:
            profile = M7.evaluate_team_profile(required_skills, team, pseudo_skill_map)
            scored_variants.append((profile, team))

        scored_variants.sort(
            key=lambda item: (round(float(item[0]["goodness"]), round_digits), tuple(item[1])),
            reverse=True,
        )
        scored_variants = M7.select_representative_variants(
            anchor=anchor,
            variants=scored_variants,
            round_digits=round_digits,
            required_skills=required_skills,
            team_size=team_size,
            num_teams=num_teams,
        )
        rows.append(
            {
                "proposal_id": row["proposal_id"],
                "year": row["year"],
                "proposal_link": proposal_link,
                "title": row["title"],
                "skills": required_skills,
                "researcher_name": anchor,
                "team": [team for _, team in scored_variants],
                "goodness": [round(float(profile["goodness"]), round_digits) for profile, _ in scored_variants],
            }
        )
    return rows


def compute_utility_summary(candidate_rows: list[dict[str, object]]) -> dict[str, float]:
    df = pd.DataFrame(candidate_rows)
    df["volume"] = df["team"].apply(len)
    df["avg_goodness_per_row"] = df["goodness"].apply(lambda scores: float(np.mean(scores)) if scores else 0.0)

    researcher_stats = df.groupby("researcher_name").agg(
        avg_goodness_per_row=("avg_goodness_per_row", "mean"),
        volume=("volume", "mean"),
    ).reset_index()

    return {
        "G_mean": float(researcher_stats["avg_goodness_per_row"].mean()),
        "G_std": float(researcher_stats["avg_goodness_per_row"].std(ddof=0)),
        "Volume_mean": float(researcher_stats["volume"].mean()),
        "top_goodness_mean": float(df["goodness"].apply(lambda scores: scores[0] if scores else 0.0).mean()),
        "n_rows": int(len(df)),
        "n_researchers": int(df["researcher_name"].nunique()),
        "n_proposals": int(df["proposal_id"].nunique()),
    }


def write_candidate_file(path: Path, candidate_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["proposal_id", "year", "proposal_link", "title", "skills", "researcher_name", "team", "goodness"],
        )
        writer.writeheader()
        writer.writerows(candidate_rows)


def evaluate_fairness(
    candidate_rows: list[dict[str, object]],
    gender_map: dict[str, str],
    groups: list[str],
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="m7_tune_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for filename in BASELINE_METHOD_FILES:
            os.symlink(OUTPUT_BASE / filename, temp_dir / filename)
        write_candidate_file(temp_dir / "teaming_uc1_m7.csv", candidate_rows)

        aligned, method_meta = fairness.build_aligned_rows(temp_dir)
        table2 = fairness.build_table2(aligned, gender_map, groups, method_meta)
        decisions = fairness.build_decision_frame(aligned, gender_map, groups, method_meta)
        table3, support = fairness.build_table3_and_support(decisions, groups, "year", method_meta)

        support_row = support[support["Method"] == "M7"].iloc[0]
        return {
            "table2_sp": float(table2.loc[table2["Group"] == "SP", "M7"].iloc[0]),
            "table3_sp": float(table3.loc[table3["Metric"] == "SP", "M7"].iloc[0]),
            "table3_csp": float(table3.loc[table3["Metric"] == "CSP", "M7"].iloc[0]),
            "table3_eo": float(table3.loc[table3["Metric"] == "EO", "M7"].iloc[0]),
            "table3_prp": float(table3.loc[table3["Metric"] == "PRP", "M7"].iloc[0]),
            "table3_te": float(table3.loc[table3["Metric"] == "TE", "M7"].iloc[0]),
            "fairness_top_goodness_mean": float(support_row["top_goodness_mean"]),
        }


def evaluate_config(
    subset_rows: list[dict[str, str]],
    proposal_payloads: dict[str, dict[str, object]],
    proposal_pseudo_skills: dict[str, dict[str, set[str]]],
    gender_map: dict[str, str],
    groups: list[str],
    alpha: float,
    cost_multiplier: float,
    max_candidates: int,
    team_size: int,
    num_teams: int,
    round_digits: int,
) -> dict[str, object]:
    candidate_rows = compute_candidate_rows(
        subset_rows=subset_rows,
        proposal_payloads=proposal_payloads,
        proposal_pseudo_skills=proposal_pseudo_skills,
        alpha=alpha,
        cost_multiplier=cost_multiplier,
        max_candidates=max_candidates,
        team_size=team_size,
        num_teams=num_teams,
        round_digits=round_digits,
    )
    utility = compute_utility_summary(candidate_rows)
    fairness_metrics = evaluate_fairness(candidate_rows, gender_map, groups)

    return {
        "alpha": alpha,
        "cost_multiplier": cost_multiplier,
        "max_candidates": max_candidates,
        **utility,
        **fairness_metrics,
    }


def select_finalists(results_df: pd.DataFrame, finalists: int) -> pd.DataFrame:
    baseline = results_df[
        (results_df["alpha"] == M7.DEFAULT_ALPHA)
        & (results_df["cost_multiplier"] == M7.DEFAULT_COST_MULTIPLIER)
        & (results_df["max_candidates"] == M7.DEFAULT_MAX_CANDIDATES)
    ]
    if baseline.empty:
        baseline = results_df.iloc[[0]]
    baseline_row = baseline.iloc[0]

    t2_limit = baseline_row["table2_sp"] + 0.03
    t3_limit = baseline_row["table3_sp"] + 0.03
    volume_limit = max(6.5, baseline_row["Volume_mean"] - 1.0)

    eligible = results_df[
        (results_df["table2_sp"] <= t2_limit)
        & (results_df["table3_sp"] <= t3_limit)
        & (results_df["Volume_mean"] >= volume_limit)
    ].copy()

    if eligible.empty:
        eligible = results_df.copy()
        eligible["selection_score"] = (
            eligible["G_mean"]
            - 0.35 * eligible["table2_sp"]
            - 0.35 * eligible["table3_sp"]
            - 0.15 * eligible["table3_prp"]
        )
        return eligible.sort_values("selection_score", ascending=False).head(finalists)

    return eligible.sort_values(["G_mean", "table3_sp", "table2_sp"], ascending=[False, True, True]).head(finalists)


def choose_winner(results_df: pd.DataFrame) -> pd.Series:
    baseline = results_df[
        (results_df["alpha"] == M7.DEFAULT_ALPHA)
        & (results_df["cost_multiplier"] == M7.DEFAULT_COST_MULTIPLIER)
        & (results_df["max_candidates"] == M7.DEFAULT_MAX_CANDIDATES)
    ]
    if baseline.empty:
        baseline = results_df.iloc[[0]]
    baseline_row = baseline.iloc[0]

    t2_limit = baseline_row["table2_sp"] + 0.02
    t3_limit = baseline_row["table3_sp"] + 0.02
    volume_limit = max(6.5, baseline_row["Volume_mean"] - 0.75)

    eligible = results_df[
        (results_df["table2_sp"] <= t2_limit)
        & (results_df["table3_sp"] <= t3_limit)
        & (results_df["Volume_mean"] >= volume_limit)
    ].copy()
    if eligible.empty:
        eligible = results_df.copy()
        eligible["selection_score"] = (
            eligible["G_mean"]
            - 0.40 * eligible["table2_sp"]
            - 0.40 * eligible["table3_sp"]
            - 0.15 * eligible["table3_prp"]
        )
        return eligible.sort_values("selection_score", ascending=False).iloc[0]

    return eligible.sort_values(["G_mean", "table3_sp", "table2_sp"], ascending=[False, True, True]).iloc[0]


def load_or_build_gender_map(
    cache_path: Path,
    researcher_skills: dict[str, set[str]],
) -> dict[str, str]:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"researcher_name_norm", "gender"}.issubset(cached.columns):
            gender_map = dict(zip(cached["researcher_name_norm"], cached["gender"]))
            missing = [name for name in researcher_skills if fairness.normalize_name(name) not in gender_map]
            if not missing:
                return gender_map

    profiles = fairness.load_profiles(
        fairness.Path(fairness.PROJECT_ROOT / "data" / "v1_input_files" / "v1_researchers.csv"),
        fairness.Path(fairness.PROJECT_ROOT / "data" / "v1_input_files" / "v1_og_faculty.csv"),
    )
    inference_df = fairness.infer_gender_map(set(researcher_skills.keys()), profiles, "paper_ensemble")
    return dict(zip(inference_df["researcher_name_norm"], inference_df["gender"]))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    alphas = parse_float_list(args.alphas)
    cost_multipliers = parse_float_list(args.cost_multipliers)
    candidate_caps = parse_int_list(args.max_candidates)

    researcher_skills = M7.load_researcher_skills(Path(M7.DEFAULT_RESEARCHER_SKILLS_FILE))
    proposal_skills = M7.load_proposal_skills(Path(M7.DEFAULT_PROPOSAL_SKILLS_FILE))
    proposal_pseudo_skills = M7.load_proposal_pseudo_skills(Path(M7.DEFAULT_PSEUDO_SKILLS_FILE))
    skeleton_rows = M7.load_skeleton_rows(Path(M7.DEFAULT_SKELETON_FILE))
    skill_weights = M7.build_idf_weights(researcher_skills)
    proposal_payloads = {
        proposal_link: M7.prepare_proposal_payload(
            required,
            proposal_pseudo_skills.get(proposal_link, {}),
            skill_weights,
            args.lexical_shortlist,
        )
        for proposal_link, required in proposal_skills.items()
    }
    gender_map = load_or_build_gender_map(Path(args.gender_inference_cache), researcher_skills)
    groups = ["male", "female"]

    coarse_rows = subset_skeleton_rows(skeleton_rows, args.coarse_proposals, args.coarse_anchors)
    final_rows = subset_skeleton_rows(skeleton_rows, args.final_proposals, args.final_anchors)

    grid = list(itertools.product(alphas, cost_multipliers, candidate_caps))
    coarse_results = []
    print(f"[tune_m7] coarse sweep on {len(coarse_rows)} rows across {len(grid)} configs", flush=True)
    for idx, (alpha, cost_multiplier, max_candidates) in enumerate(grid, start=1):
        print(
            f"[tune_m7] coarse {idx}/{len(grid)} "
            f"(alpha={alpha}, cost={cost_multiplier}, shortlist={max_candidates})"
        , flush=True)
        coarse_results.append(
            evaluate_config(
                subset_rows=coarse_rows,
                proposal_payloads=proposal_payloads,
                proposal_pseudo_skills=proposal_pseudo_skills,
                gender_map=gender_map,
                groups=groups,
                alpha=alpha,
                cost_multiplier=cost_multiplier,
                max_candidates=max_candidates,
                team_size=args.team_size,
                num_teams=args.num_teams,
                round_digits=args.round_digits,
            )
        )

    coarse_df = pd.DataFrame(coarse_results).sort_values(["G_mean", "table3_sp", "table2_sp"], ascending=[False, True, True])
    coarse_path = results_dir / "m7_sweep_coarse.csv"
    coarse_df.to_csv(coarse_path, index=False)

    finalists_df = select_finalists(coarse_df, args.finalists)
    finalist_configs = finalists_df[["alpha", "cost_multiplier", "max_candidates"]].drop_duplicates()

    final_results = []
    print(f"[tune_m7] finalist round on {len(final_rows)} rows across {len(finalist_configs)} configs", flush=True)
    for idx, finalist in enumerate(finalist_configs.itertuples(index=False), start=1):
        print(
            f"[tune_m7] finalist {idx}/{len(finalist_configs)} "
            f"(alpha={finalist.alpha}, cost={finalist.cost_multiplier}, shortlist={finalist.max_candidates})"
        , flush=True)
        final_results.append(
            evaluate_config(
                subset_rows=final_rows,
                proposal_payloads=proposal_payloads,
                proposal_pseudo_skills=proposal_pseudo_skills,
                gender_map=gender_map,
                groups=groups,
                alpha=float(finalist.alpha),
                cost_multiplier=float(finalist.cost_multiplier),
                max_candidates=int(finalist.max_candidates),
                team_size=args.team_size,
                num_teams=args.num_teams,
                round_digits=args.round_digits,
            )
        )

    final_df = pd.DataFrame(final_results).sort_values(["G_mean", "table3_sp", "table2_sp"], ascending=[False, True, True])
    final_path = results_dir / "m7_sweep_final.csv"
    final_df.to_csv(final_path, index=False)

    winner = choose_winner(final_df)
    summary = {
        "coarse_results": str(coarse_path),
        "final_results": str(final_path),
        "winner": {
            "alpha": float(winner["alpha"]),
            "cost_multiplier": float(winner["cost_multiplier"]),
            "max_candidates": int(winner["max_candidates"]),
            "G_mean": float(winner["G_mean"]),
            "Volume_mean": float(winner["Volume_mean"]),
            "table2_sp": float(winner["table2_sp"]),
            "table3_sp": float(winner["table3_sp"]),
            "table3_csp": float(winner["table3_csp"]),
            "table3_prp": float(winner["table3_prp"]),
        },
        "defaults": {
            "alpha": M7.DEFAULT_ALPHA,
            "cost_multiplier": M7.DEFAULT_COST_MULTIPLIER,
            "max_candidates": M7.DEFAULT_MAX_CANDIDATES,
            "lexical_shortlist": args.lexical_shortlist,
        },
    }
    summary_path = results_dir / "m7_tuning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[tune_m7] coarse results -> {coarse_path}", flush=True)
    print(f"[tune_m7] final results  -> {final_path}", flush=True)
    print(f"[tune_m7] summary       -> {summary_path}", flush=True)
    print(
        "[tune_m7] winner: "
        f"alpha={winner['alpha']}, cost={winner['cost_multiplier']}, shortlist={winner['max_candidates']}, "
        f"G_mean={winner['G_mean']:.4f}, Volume_mean={winner['Volume_mean']:.4f}, "
        f"Table2_SP={winner['table2_sp']:.4f}, Table3_SP={winner['table3_sp']:.4f}"
    , flush=True)


if __name__ == "__main__":
    main()
