"""
Shared method implementations for meal_project-3.

This module lifts the notebook logic into reusable Python so the project can be
run and compared from one place.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from collections import deque

import numpy as np
import pandas as pd

from meal_utils import (
    MEAL_SIZE,
    NUM_VARIATIONS,
    QUALITY_THRESH,
    MealMetricScorer,
    build_idf_weights,
    build_output_row,
    load_data,
    pick_target_item,
)


DEFAULT_WEIGHT = 1.0
MARGINAL_UTILITY_ALPHA = 0.50
MARGINAL_UTILITY_SEAT_MULTIPLIER = 0.20
MARGINAL_UTILITY_TOPK = 2


def filter_users(df_users, user_required, selected_user_ids=None, user_limit=None):
    """Return a filtered user dataframe plus aligned requirement mapping."""
    if selected_user_ids is not None:
        selected_user_ids = set(selected_user_ids)
        df_users = df_users[df_users["user_id"].isin(selected_user_ids)].reset_index(drop=True)
    elif user_limit is not None:
        df_users = df_users.head(user_limit).reset_index(drop=True)

    filtered_required = {uid: user_required[uid] for uid in df_users["user_id"]}
    return df_users, filtered_required


def load_boosted_test_split(project_root):
    split_path = os.environ.get("MEAL_USER_SPLIT_FILE")
    if not split_path:
        bandit_dir = os.environ.get("MEAL_BANDIT_DIR", f"{project_root}/boosted_bandit")
        split_path = f"{bandit_dir}/user_split.json"
    with open(split_path, encoding="utf-8") as handle:
        split = json.load(handle)
    return split["test_users"]


def generate_random_teams(target_item, required, all_items, n=NUM_VARIATIONS, meal_size=MEAL_SIZE, rng=None):
    """M0 baseline: anchor on the best single item, then fill randomly."""
    rng = rng or random
    teams = []
    attempts = 0
    while len(teams) < n and attempts < n * 10:
        attempts += 1
        pool = [item for item in all_items if item != target_item]
        rng.shuffle(pool)
        team = [target_item] + pool[: meal_size - 1]
        if sorted(team) not in [sorted(existing) for existing in teams]:
            teams.append(team)
    return teams


def generate_sequential_teams(
    target_item,
    required,
    item_categories,
    all_items,
    n=NUM_VARIATIONS,
    meal_size=MEAL_SIZE,
    rng=None,
):
    """M1 baseline: sequential greedy coverage over uncovered tags."""
    rng = rng or random

    def build(shuffle=False, noise=False):
        team = [target_item]
        covered = item_categories.get(target_item, set()) & required
        todo = sorted(required - covered)
        if shuffle:
            rng.shuffle(todo)

        for tag in todo:
            if len(team) >= meal_size:
                break
            if tag in covered:
                continue

            remaining = required - covered
            candidates = [
                (item, len(item_categories[item] & remaining))
                for item in all_items
                if item not in team and tag in item_categories[item]
            ]
            if not candidates:
                candidates = [
                    (item, len(item_categories[item] & remaining))
                    for item in all_items
                    if item not in team
                ]
            if not candidates:
                break

            candidates.sort(key=lambda row: row[1], reverse=True)
            chosen = rng.choice(candidates[:3])[0] if noise else candidates[0][0]
            team.append(chosen)
            covered |= item_categories[chosen] & required
        return team

    base = build(shuffle=False, noise=False)
    teams = [base]
    attempts = 0
    while len(teams) < n and attempts < n * 8:
        attempts += 1
        variant = build(shuffle=True, noise=True)
        if sorted(variant) not in [sorted(existing) for existing in teams]:
            teams.append(variant)
    return teams


def _required_mask_setup(required, cat_weights):
    required_tags = sorted(required)
    tag_index = {tag: idx for idx, tag in enumerate(required_tags)}
    mask_scores = [0.0] * (1 << len(required_tags))

    for mask in range(1, len(mask_scores)):
        low_bit = mask & -mask
        bit_index = low_bit.bit_length() - 1
        tag = required_tags[bit_index]
        mask_scores[mask] = mask_scores[mask ^ low_bit] + cat_weights.get(tag, DEFAULT_WEIGHT)

    return required_tags, tag_index, mask_scores


def _item_mask(item_name, required, tag_index, item_categories):
    mask = 0
    for tag in item_categories[item_name] & required:
        mask |= 1 << tag_index[tag]
    return mask


def exact_knapsack_bundle(
    required,
    item_categories,
    all_items,
    cat_weights,
    meal_size=MEAL_SIZE,
    banned_items=None,
):
    """
    Exact dynamic-programming method for weighted maximum coverage with a seat
    budget. This is a real DP, unlike the legacy greedy M2 notebook.
    """
    banned_items = banned_items or set()
    if not required:
        return []

    _, tag_index, mask_scores = _required_mask_setup(required, cat_weights)

    items = []
    for item_name in all_items:
        if item_name in banned_items:
            continue
        mask = _item_mask(item_name, required, tag_index, item_categories)
        if mask:
            items.append((item_name, mask))

    if not items:
        return []

    # dp[k][mask] = (redundancy_hits, team_tuple)
    dp = [{0: (0, tuple())}] + [{} for _ in range(meal_size)]

    for item_name, item_mask in items:
        for used in range(meal_size - 1, -1, -1):
            snapshot = list(dp[used].items())
            for current_mask, (redundancy_hits, team) in snapshot:
                new_mask = current_mask | item_mask
                if new_mask == current_mask:
                    continue

                new_redundancy = redundancy_hits + (current_mask & item_mask).bit_count()
                new_team = team + (item_name,)
                existing = dp[used + 1].get(new_mask)

                if existing is None or (new_redundancy, new_team) < (existing[0], existing[1]):
                    dp[used + 1][new_mask] = (new_redundancy, new_team)

    best_team = []
    best_key = None
    for used in range(1, meal_size + 1):
        for mask, (redundancy_hits, team) in dp[used].items():
            candidate_key = (
                mask_scores[mask],
                -redundancy_hits,
                -used,
                tuple(sorted(team)),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_team = list(team)

    return best_team


def generate_exact_knapsack_teams(
    required,
    item_categories,
    all_items,
    cat_weights,
    n=NUM_VARIATIONS,
    meal_size=MEAL_SIZE,
):
    """
    Exact knapsack method: anchor is the exact DP optimum; variants come from reruns
    with small exclusion sets so we keep near-optimal but distinct bundles.
    """
    teams = []
    seen = set()
    queue = deque([frozenset()])
    visited_bans = {frozenset()}

    while queue and len(teams) < n:
        banned_items = set(queue.popleft())
        team = exact_knapsack_bundle(
            required,
            item_categories,
            all_items,
            cat_weights,
            meal_size=meal_size,
            banned_items=banned_items,
        )
        if not team:
            continue

        signature = tuple(sorted(team))
        if signature in seen:
            continue

        seen.add(signature)
        teams.append(team)

        if len(banned_items) < 2:
            for item in team:
                new_ban = frozenset(set(banned_items) | {item})
                if new_ban not in visited_bans:
                    visited_bans.add(new_ban)
                    queue.append(new_ban)

    return teams


def generate_exact_knapsack_teams_with_anchor(
    target_item,
    required,
    item_categories,
    all_items,
    cat_weights,
    n=NUM_VARIATIONS,
    meal_size=MEAL_SIZE,
):
    """
    Cross-product exact-knapsack variant: target_item is always the anchor (slot 1).
    Fills the remaining meal_size-1 slots via exact DP on all other items.
    Mirrors generate_exact_knapsack_teams but with a forced first element.
    """
    teams = []
    seen = set()
    queue = deque([frozenset()])
    visited_bans = {frozenset()}

    while queue and len(teams) < n:
        extra_bans = set(queue.popleft())
        banned_items = extra_bans | {target_item}  # always exclude anchor from DP

        filler = exact_knapsack_bundle(
            required, item_categories, all_items, cat_weights,
            meal_size=meal_size - 1,
            banned_items=banned_items,
        )
        team = [target_item] + filler
        sig = tuple(sorted(team))
        if sig in seen:
            continue

        seen.add(sig)
        teams.append(team)

        if len(extra_bans) < 2:
            for item in filler:
                new_ban = frozenset(extra_bans | {item})
                if new_ban not in visited_bans:
                    visited_bans.add(new_ban)
                    queue.append(new_ban)

    return teams if teams else [[target_item]]


def generate_marginal_utility_teams(
    target_item,
    required,
    item_categories,
    all_items,
    scorer,
    cat_weights,
    n=NUM_VARIATIONS,
    meal_size=MEAL_SIZE,
    quality_thresh=QUALITY_THRESH,
    rng=None,
):
    """
    Marginal-utility method: weighted diminishing-return selection.
    """
    rng = rng or random

    def avg_weight(tags):
        if not tags:
            return DEFAULT_WEIGHT
        return sum(cat_weights.get(tag, DEFAULT_WEIGHT) for tag in tags) / len(tags)

    seat_cost = avg_weight(required) * MARGINAL_UTILITY_SEAT_MULTIPLIER

    def build(stochastic=False):
        team = [target_item]
        coverage_counts = {tag: 0 for tag in required}
        for tag in item_categories.get(target_item, set()) & required:
            coverage_counts[tag] += 1

        while len(team) < meal_size:
            candidates = []
            for item_name in all_items:
                if item_name in team:
                    continue
                item_tags = item_categories[item_name] & required
                if not item_tags:
                    continue
                utility = sum(
                    cat_weights.get(tag, DEFAULT_WEIGHT) * (MARGINAL_UTILITY_ALPHA ** coverage_counts[tag])
                    for tag in item_tags
                )
                net_utility = utility - seat_cost
                if net_utility <= 0:
                    continue

                candidates.append((item_name, net_utility))

            if not candidates:
                break

            candidates.sort(key=lambda row: (-row[1], row[0]))
            chosen = rng.choice(candidates[:MARGINAL_UTILITY_TOPK])[0] if stochastic else candidates[0][0]
            team.append(chosen)
            for tag in item_categories[chosen] & required:
                coverage_counts[tag] += 1

        return team

    anchor = build(stochastic=False)
    anchor_score = scorer.score_team(required, anchor, item_categories)["goodness"]
    teams = [anchor]

    for _ in range(60):
        if len(teams) >= n:
            break
        variant = build(stochastic=True)
        variant_score = scorer.score_team(required, variant, item_categories)["goodness"]
        if variant_score >= anchor_score * quality_thresh:
            if sorted(variant) not in [sorted(existing) for existing in teams]:
                teams.append(variant)

    return teams


def select_target_from_team(team, required, item_categories):
    if not team:
        return ""
    return max(team, key=lambda item: len(item_categories.get(item, set()) & required))


def run_standard_method(
    method_name,
    df_users,
    user_required,
    item_categories,
    all_items,
    scorer,
    rng_seed=42,
):
    rng = random.Random(rng_seed)
    cat_weights = build_idf_weights(item_categories)
    rows = []

    for idx, user_row in df_users.iterrows():
        uid = user_row["user_id"]
        required = user_required[uid]

        if method_name == "m0_random":
            target_item = pick_target_item(required, item_categories, all_items)
            teams = generate_random_teams(target_item, required, all_items, rng=rng)
        elif method_name == "m1_sequential":
            target_item = pick_target_item(required, item_categories, all_items)
            teams = generate_sequential_teams(target_item, required, item_categories, all_items, rng=rng)
        elif method_name == "m2_knapsack_exact":
            teams = generate_exact_knapsack_teams(required, item_categories, all_items, cat_weights)
            target_item = select_target_from_team(teams[0], required, item_categories) if teams else ""
        elif method_name == "m3_marginal_utility":
            target_item = pick_target_item(required, item_categories, all_items)
            teams = generate_marginal_utility_teams(
                target_item,
                required,
                item_categories,
                all_items,
                scorer,
                cat_weights,
                rng=rng,
            )
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        scored = [scorer.score_team(required, team, item_categories) for team in teams]
        rows.append(build_output_row(user_row, target_item, teams, scored, user_required))

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(df_users)}]  {uid}")

    return rows


def run_crossproduct_method(
    method_name,
    df_users,
    user_required,
    item_categories,
    all_items,
    scorer,
    rng_seed=42,
):
    """
    Teaming-style cross product: for every (user × item) pair, build bundles
    anchored on that item — mirroring Teaming's (researcher × proposal) loop.

    ALL items are used as targets, including those with zero category overlap,
    so reviewers can observe how bundle quality degrades when the anchor item
    is irrelevant to the user's requirements.

    Output: one row per (user, target_item) pair.
    """
    rng = random.Random(rng_seed)
    cat_weights = build_idf_weights(item_categories)
    rows = []
    row_id = 0
    boosted_payload = None
    cached_teams = {}

    if method_name == "m2_boosted_bandit":
        from boosted_bandit_postprocess import build_teaming_style_bandit_teams, parse_results_db

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bandit_dir = os.environ.get("MEAL_BANDIT_DIR", f"{project_root}/boosted_bandit")
        boosted_num_teams = int(os.environ.get("BOOSTED_NUM_TEAMS", "10"))
        results_db = os.path.join(bandit_dir, "test", "results_team.db")
        if not os.path.exists(results_db):
            legacy_results_db = os.path.join(bandit_dir, "test", "results_recommendation.db")
            if os.path.exists(legacy_results_db):
                results_db = legacy_results_db
            else:
                raise FileNotFoundError(
                    "Missing BoostSRL result database: "
                    f"{results_db}. Run the boosted notebook's data-generation "
                    "and train/test cells first for the active MEAL_BANDIT_DIR."
                )

        user_probs, pos_items, neg_items = parse_results_db(
            results_db,
            item_categories,
            user_required,
        )
        boosted_payload = {
            "user_probs": user_probs,
            "pos_items": pos_items,
            "neg_items": neg_items,
            "num_teams": boosted_num_teams,
            "matching_threshold": float(os.environ.get("BOOSTED_MATCHING_THRESHOLD", "0.3")),
            "build_teaming_style_bandit_teams": build_teaming_style_bandit_teams,
        }

    for idx, user_row in df_users.iterrows():
        uid = user_row["user_id"]
        required = user_required[uid]

        for target_item in all_items:
            cache_key = None
            if method_name == "m0_random":
                teams = generate_random_teams(target_item, required, all_items, rng=rng)
            elif method_name == "m1_sequential":
                teams = generate_sequential_teams(target_item, required, item_categories, all_items, rng=rng)
            elif method_name == "m2_boosted_bandit":
                teams = boosted_payload["build_teaming_style_bandit_teams"](
                    uid,
                    target_item,
                    required,
                    item_categories,
                    boosted_payload["user_probs"],
                    boosted_payload["pos_items"],
                    boosted_payload["neg_items"],
                    meal_size=MEAL_SIZE,
                    n=boosted_payload["num_teams"],
                    matching_threshold=boosted_payload["matching_threshold"],
                    rng=rng,
                )
            elif method_name == "m2_knapsack_exact":
                cache_key = (method_name, tuple(sorted(required)), target_item)
                teams = cached_teams.get(cache_key)
                if teams is None:
                    teams = generate_exact_knapsack_teams_with_anchor(
                        target_item, required, item_categories, all_items, cat_weights
                    )
                    cached_teams[cache_key] = [list(team) for team in teams]
                else:
                    teams = [list(team) for team in teams]
            elif method_name == "m3_marginal_utility":
                cache_key = (method_name, tuple(sorted(required)), target_item)
                teams = cached_teams.get(cache_key)
                if teams is None:
                    teams = generate_marginal_utility_teams(
                        target_item, required, item_categories, all_items, scorer, cat_weights, rng=rng
                    )
                    cached_teams[cache_key] = [list(team) for team in teams]
                else:
                    teams = [list(team) for team in teams]
            else:
                raise ValueError(f"Cross-product mode not supported for method: {method_name}")

            scored = [scorer.score_team(required, team, item_categories) for team in teams]
            if method_name == "m2_boosted_bandit" and scored:
                best_g = max(s["goodness"] for s in scored)
                threshold = best_g * QUALITY_THRESH
                filtered = [(t, s) for t, s in zip(teams, scored) if s["goodness"] >= threshold]
                if filtered:
                    teams, scored = zip(*filtered)
                    teams, scored = list(teams), list(scored)
            rows.append(build_output_row(user_row, target_item, teams, scored, user_required, row_id=row_id))
            row_id += 1

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(df_users)}]  {uid}  ({row_id} rows so far)")

    return rows


def rows_to_summary(method_name, rows):
    """User-level and variant-level summary metrics."""
    variant_scores = []
    volumes = []
    avg_coverages = []
    avg_redundancies = []
    avg_sizes = []
    user_mean_scores = []
    best_scores = []

    for row in rows:
        scores = list(row["goodness"])
        teams = list(row["recommended_meals"])
        variant_scores.extend(scores)
        volumes.append(len(scores))
        avg_coverages.append(row["avg_coverage"])
        avg_redundancies.append(row["avg_redundancy"])
        avg_sizes.extend(len(team) for team in teams)
        user_mean_scores.append(float(np.mean(scores)))
        best_scores.append(float(np.max(scores)))

    return {
        "method": method_name,
        "users": len(rows),
        "variant_weighted_g": round(float(np.mean(variant_scores)), 4),
        "user_mean_g": round(float(np.mean(user_mean_scores)), 4),
        "user_best_g": round(float(np.mean(best_scores)), 4),
        "avg_volume": round(float(np.mean(volumes)), 2),
        "avg_team_size": round(float(np.mean(avg_sizes)), 2),
        "avg_coverage": round(float(np.mean(avg_coverages)), 4),
        "avg_redundancy": round(float(np.mean(avg_redundancies)), 4),
    }


def save_rows(rows, output_dir, method_name):
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"meal_uc1_{method_name}.csv"
    df.to_csv(path, index=False)
    return path


def load_core_data(selected_user_ids=None, user_limit=None):
    item_categories, user_required, df_users, all_items = load_data(user_limit=None)
    df_users, user_required = filter_users(
        df_users,
        user_required,
        selected_user_ids=selected_user_ids,
        user_limit=user_limit,
    )
    return item_categories, user_required, df_users, all_items


def compare_method_best_scores(rows_a, rows_b, method_a, method_b):
    """
    Compare best-per-user goodness across two methods.
    Returns a dataframe suitable for diagnostics.
    """
    by_user_a = {row["user_id"]: row for row in rows_a}
    by_user_b = {row["user_id"]: row for row in rows_b}

    rows = []
    for uid in sorted(set(by_user_a) & set(by_user_b)):
        row_a = by_user_a[uid]
        row_b = by_user_b[uid]
        best_a = max(row_a["goodness"])
        best_b = max(row_b["goodness"])
        rows.append(
            {
                "user_id": uid,
                f"{method_a}_best_g": round(best_a, 4),
                f"{method_b}_best_g": round(best_b, 4),
                "gap_b_minus_a": round(best_b - best_a, 4),
                f"{method_a}_best_team": row_a["recommended_meals"][row_a["goodness"].index(best_a)],
                f"{method_b}_best_team": row_b["recommended_meals"][row_b["goodness"].index(best_b)],
                "better_method": method_b if best_b > best_a else method_a if best_a > best_b else "tie",
            }
        )

    return pd.DataFrame(rows)


def build_scorer():
    goodness_profile = os.environ.get("MEAL_GOODNESS_PROFILE", "legacy_meal")
    return MealMetricScorer(meal_size=MEAL_SIZE, goodness_profile=goodness_profile)
