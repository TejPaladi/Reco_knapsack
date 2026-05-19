"""
Helpers for post-processing BoostSRL meal recommendation output.

This keeps the meal M4 notebook closer to the Teaming M3 flow:
  1. Parse BoostSRL results into positive/negative candidate pools.
  2. Use the positive pool as the main candidate set.
  3. Build a small set of high-coverage meal bundles instead of always
     forcing 10 random variants.
"""

from __future__ import annotations

import itertools
import random
import re
from difflib import SequenceMatcher


def to_atom(name):
    token = re.sub(r"[^a-z0-9_]", "_", str(name).lower().strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return ("x_" + token) if (token and token[0].isdigit()) else token


def parse_results_db(db_path, item_categories, user_required):
    """
    Parse BoostSRL results into:
      - user_probs: ranked per-user item scores
      - pos_items: items that came from positive literals
      - neg_items: items that came from negated literals

    Notes:
      - A line like `team(U,I) 0.14` contributes a positive-class
        probability of 0.14.
      - A line like `!team(U,I) 0.86` contributes a positive-class
        probability of `1 - 0.86 = 0.14`.
      - The positive/negative split is still useful as a Teaming-style
        candidate filter even when the probabilities are coarse.
    """

    item_atom_to_name = {}
    user_atom_to_id = {}

    for name in item_categories:
        atom = to_atom(name)
        item_atom_to_name[atom] = name
        item_atom_to_name[f"i_{atom}"] = name

    for uid in user_required:
        atom = to_atom(uid)
        user_atom_to_id[atom] = uid
        user_atom_to_id[f"u_{atom}"] = uid

    user_probs = {}
    pos_items = {}
    neg_items = {}

    with open(db_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            is_pos_literal = not line.startswith("!")
            clean_line = line.lstrip("!")
            match = re.match(r"(?:team|recommendation)\(([^,]+),\s*([^)]+)\)\s+([0-9.]+)", clean_line)
            if not match:
                continue

            user_atom, item_atom, prob_str = match.groups()
            uid = user_atom_to_id.get(user_atom)
            item_name = item_atom_to_name.get(item_atom)
            if uid is None or item_name is None:
                continue

            raw_prob = float(prob_str)
            positive_prob = raw_prob if is_pos_literal else (1.0 - raw_prob)

            record = {
                "item": item_name,
                "positive_prob": positive_prob,
                "raw_prob": raw_prob,
                "is_pos_literal": is_pos_literal,
            }
            user_probs.setdefault(uid, []).append(record)

            target = pos_items if is_pos_literal else neg_items
            target.setdefault(uid, set()).add(item_name)

    for uid, records in user_probs.items():
        records.sort(
            key=lambda record: (
                -(1 if record["is_pos_literal"] else 0),
                -record["positive_prob"],
                record["item"],
            )
        )

    return user_probs, pos_items, neg_items


def _bundle_proxy_score(bundle, required, item_categories, pos_pool):
    covered = set()
    duplicate_hits = 0
    positive_hits = 0

    for item in bundle:
        tags = item_categories.get(item, set()) & required
        duplicate_hits += len(covered & tags)
        covered.update(tags)
        if item in pos_pool:
            positive_hits += 1

    return (
        2.5 * len(covered)
        + 0.5 * positive_hits
        - 0.75 * duplicate_hits
        - 0.25 * abs(len(bundle) - min(4, max(1, len(bundle))))
    )


def _rank_items(required, items, item_categories, user_probs_by_item, pos_pool):
    ranked = []
    for item in items:
        tags = item_categories.get(item, set())
        overlap = len(tags & required)
        if overlap == 0 and item not in pos_pool:
            continue

        ranked.append(
            (
                item,
                1 if item in pos_pool else 0,
                overlap,
                -len(tags - required),
                user_probs_by_item.get(item, 0.0),
            )
        )

    ranked.sort(key=lambda row: (-row[1], -row[2], -row[3], -row[4], row[0]))
    return [item for item, *_ in ranked]


def build_bandit_teams(uid, user_probs, pos_items, required, item_categories, meal_size=4, n=10):
    """
    Build a selective set of candidate bundles for a user.

    The Teaming-style adaptation here is:
      - use BoostSRL-positive literals as the main candidate pool
      - supplement only when the positive pool is too small
      - enumerate compact bundles from that pool instead of sampling random
        top-k sets, which keeps method volume lower
    """

    records = user_probs.get(uid, [])
    if not records:
        return []

    pos_pool = set(pos_items.get(uid, set()))
    probs_by_item = {record["item"]: record["positive_prob"] for record in records}

    positive_candidates = _rank_items(required, pos_pool, item_categories, probs_by_item, pos_pool)
    all_ranked = _rank_items(
        required,
        [record["item"] for record in records],
        item_categories,
        probs_by_item,
        pos_pool,
    )

    pool = positive_candidates[:]
    min_bundle_size = min(meal_size, max(1, len(pool)))

    if len(pool) < meal_size:
        for item in all_ranked:
            if item not in pool:
                pool.append(item)
            if len(pool) >= min(meal_size + 2, len(all_ranked)):
                break
        min_bundle_size = min(meal_size, max(1, len(pool)))

    if not pool:
        return []

    pool = pool[: min(len(pool), max(meal_size + 4, 8))]

    candidates = []
    seen = set()
    for size in range(min_bundle_size, meal_size + 1):
        if size > len(pool):
            break
        for bundle in itertools.combinations(pool, size):
            signature = tuple(sorted(bundle))
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                (
                    _bundle_proxy_score(bundle, required, item_categories, pos_pool),
                    bundle,
                )
            )

    if not candidates:
        return []

    candidates.sort(key=lambda row: (-row[0], row[1]))
    return [list(bundle) for _, bundle in candidates[:n]]


def lexical_item_ranking(item_categories, required_tags, matching_threshold=0.3):
    """
    Teaming-style lexical ranking adapted to meal items.

    For each required tag, collect meal items whose categories fuzzily match it.
    Items are ranked by the fraction of required tags they matched.
    """
    required_tags = list(required_tags)
    matched_required_by_item = {item_name: [] for item_name in item_categories}

    for required_tag in required_tags:
        for item_name, item_tags in item_categories.items():
            for item_tag in item_tags:
                if SequenceMatcher(None, str(required_tag), str(item_tag)).ratio() >= matching_threshold:
                    matched_required_by_item[item_name].append(required_tag)
                    break

    denom = max(len(required_tags), 1)
    ranking = {
        item_name: len(set(matched_required_by_item[item_name])) / denom
        for item_name in item_categories
    }
    return dict(sorted(ranking.items(), key=lambda item: item[1]))


def create_fallback_teams_for_target(ranking, target_item, num_teams, meal_size=4, rng=None):
    """
    Teaming-style fallback team creation:
    - sample from the top lexical candidates
    - keep the target item as anchor
    - allow repeated attempts but not guaranteed full volume
    """
    rng = rng or random.Random(42)
    teams = []
    pseudo_count = 0

    candidate_pool = list(ranking.keys())[-20:] if ranking else [target_item]
    if target_item not in candidate_pool:
        candidate_pool.append(target_item)

    while len(teams) < num_teams:
        pseudo_count += 1
        if pseudo_count > num_teams + 10:
            break

        if meal_size <= 1:
            team = [target_item]
        else:
            extra_size = rng.randint(1, max(1, meal_size - 1))
            sample_size = min(len(candidate_pool), extra_size)
            sampled = rng.sample(candidate_pool, sample_size)
            if target_item in sampled:
                sampled.remove(target_item)
            team = [target_item] + sampled

        if team not in teams:
            teams.append(team)

    return teams


def build_teaming_style_bandit_teams(
    uid,
    target_item,
    required,
    item_categories,
    user_probs,
    pos_items,
    neg_items,
    meal_size=4,
    n=10,
    matching_threshold=0.3,
    rng=None,
):
    """
    Reproduce the Teaming boosted-bandit post-processing pattern for meals.

    The exact structure mirrors Teaming's notebook flow:
    1. Build lexical fallback teams first.
    2. If BoostSRL predicted positive user-item pairs, replace the fallback
       teams with random teams sampled from that positive pool.
    3. Keep only the unique random bandit teams that were actually generated.

    Notes:
    - This intentionally does not add a score-based quality filter.
    - The attempted negative-pool fallback in the Teaming notebook did not
      affect the final exported teams, so this function preserves that overall
      behavior by returning either fallback teams or positive-pool teams.
    """
    rng = rng or random.Random(42)

    ranking = lexical_item_ranking(
        item_categories=item_categories,
        required_tags=required,
        matching_threshold=matching_threshold,
    )
    fallback_teams = create_fallback_teams_for_target(
        ranking=ranking,
        target_item=target_item,
        num_teams=n,
        meal_size=meal_size,
        rng=rng,
    )

    if uid not in pos_items and uid not in neg_items:
        return fallback_teams

    pos_members = list(pos_items.get(uid, set()))
    if not pos_members:
        return fallback_teams

    bandit_teams = []
    seen_signatures = set()
    count = 0
    sample_size = min(len(pos_members), max(1, meal_size - 1))

    while count < n:
        count += 1
        sampled = rng.sample(pos_members, sample_size)
        if target_item in sampled:
            sampled.remove(target_item)
        team = [target_item] + sampled

        signature = tuple(sorted(team))
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            bandit_teams.append(team)

    return bandit_teams if bandit_teams else fallback_teams
