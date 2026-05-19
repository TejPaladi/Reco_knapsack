"""
Generate BoostSRL files for the meal recommendation task.

This now mirrors the Teaming `prepare_data.py` pipeline much more literally:
- facts use the same `skill(...)` / `interest(...)` predicate pattern
- positives are `team(user,item)` pairs from fuzzy tag overlap
- negatives are random non-positive `team(user,item)` pairs
- positives and negatives are shuffled globally, then split 80/20

The existing `user_split.json` is still written for the non-BoostSRL notebooks,
but the BoostSRL train/test split itself now follows Teaming's pair-level split.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import random
import re
from difflib import SequenceMatcher


random.seed(12345)

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Notebook `exec(open(...).read())` does not define __file__.
    PROJECT_ROOT = os.path.abspath("..")
INPUT_DIR = os.environ.get("MEAL_INPUT_DIR", os.path.join(PROJECT_ROOT, "data", "input_data"))
BANDIT_DIR = os.environ.get("MEAL_BANDIT_DIR", os.path.join(PROJECT_ROOT, "boosted_bandit"))
MEAL_CATEGORIES_FILE = os.environ.get("MEAL_CATEGORIES_FILE", "meal_categories.csv")
USER_REQUIREMENTS_FILE = os.environ.get("MEAL_USERS_FILE", "user_meal_requirements.csv")
TRAIN_DIR = os.path.join(BANDIT_DIR, "train")
TEST_DIR = os.path.join(BANDIT_DIR, "test")
USER_SPLIT_PATH = os.path.join(BANDIT_DIR, "user_split.json")
TRAIN_BK_PATH = os.path.join(TRAIN_DIR, "train_bk.txt")
TEST_BK_PATH = os.path.join(TEST_DIR, "test_bk.txt")

TRAIN_SPLIT = float(os.environ.get("BOOSTED_TRAIN_SPLIT", "0.8"))
MATCH_THRESHOLD = float(os.environ.get("BOOSTED_MATCH_THRESHOLD", "0.8"))
NEG_RATIO = int(os.environ.get("BOOSTED_NEG_RATIO", "1"))


def sanitize_token(value):
    if value is None:
        return "unknown"
    token = str(value).strip().strip(" {}'\"").lower()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def user_atom(user_id):
    return sanitize_token(user_id)


def item_atom(item_name):
    return sanitize_token(item_name)


def parse_tag_set(raw_value):
    if raw_value is None:
        return set()
    parsed = ast.literal_eval(str(raw_value))
    return {sanitize_token(tag) for tag in parsed if sanitize_token(tag)}


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def load_items():
    path = os.path.join(INPUT_DIR, MEAL_CATEGORIES_FILE)
    items = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["meal_name"]
            items.append(
                {
                    "name": name,
                    "atom": item_atom(name),
                    "tags": parse_tag_set(row["categories"]),
                }
            )
    return items


def load_users():
    path = os.path.join(INPUT_DIR, USER_REQUIREMENTS_FILE)
    users = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uid = row["user_id"]
            users.append(
                {
                    "user_id": uid,
                    "atom": user_atom(uid),
                    "required_tags": parse_tag_set(row["required_categories"]),
                }
            )
    return users


def load_or_create_user_split(users):
    current_user_ids = sorted(user["user_id"] for user in users)
    if os.path.exists(USER_SPLIT_PATH):
        with open(USER_SPLIT_PATH, encoding="utf-8") as handle:
            split = json.load(handle)
        if "train_users" in split and "test_users" in split:
            split_user_ids = sorted(set(split["train_users"]) | set(split["test_users"]))
            if split_user_ids == current_user_ids:
                return split

    user_ids = list(current_user_ids)
    random.shuffle(user_ids)
    cutoff = int(len(user_ids) * TRAIN_SPLIT)
    split = {
        "train_users": sorted(user_ids[:cutoff]),
        "test_users": sorted(user_ids[cutoff:]),
    }
    with open(USER_SPLIT_PATH, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    return split


def write_bk_imports():
    write_lines(TRAIN_BK_PATH, ['import: "../teams_bk.txt"'])
    write_lines(TEST_BK_PATH, ['import: "../teams_bk.txt"'])


def build_facts(users, items):
    facts = set()
    for user in users:
        for tag in user["required_tags"]:
            facts.add(f"skill({user['atom']},{tag}).")
    for item in items:
        for tag in item["tags"]:
            facts.add(f"interest({item['atom']},{tag}).")
    return sorted(facts)


def is_positive_match(required_tags, item_tags):
    for required_tag in required_tags:
        for item_tag in item_tags:
            if SequenceMatcher(None, required_tag, item_tag).ratio() >= MATCH_THRESHOLD:
                return True
    return False


def build_positive_pairs(users, items):
    positives = set()
    for user in users:
        for item in items:
            if is_positive_match(user["required_tags"], item["tags"]):
                positives.add((user["user_id"], item["name"]))
    return positives


def sample_negative_pairs(user_ids, item_names, positives, target_count):
    candidates = [
        (user_id, item_name)
        for user_id in user_ids
        for item_name in item_names
        if (user_id, item_name) not in positives
    ]

    if target_count > len(candidates):
        print(
            f"Requested {target_count} negatives, but only {len(candidates)} "
            "non-positive pairs are available. Using all available negatives."
        )
        target_count = len(candidates)

    random.shuffle(candidates)
    return set(candidates[:target_count])


def format_pairs(pairs):
    return [
        f"team({user_atom(uid)},{item_atom(item_name)})."
        for uid, item_name in pairs
    ]


def main():
    users = load_users()
    items = load_items()
    split = load_or_create_user_split(users)

    user_ids = [user["user_id"] for user in users]
    item_names = [item["name"] for item in items]

    positives = build_positive_pairs(users, items)
    num_pos = len(positives)
    num_neg = NEG_RATIO * num_pos

    negatives = sample_negative_pairs(
        user_ids=user_ids,
        item_names=item_names,
        positives=positives,
        target_count=num_neg,
    )

    pos_list = list(positives)
    neg_list = list(negatives)
    random.shuffle(pos_list)
    random.shuffle(neg_list)

    num_pos_train = int(TRAIN_SPLIT * len(pos_list))
    num_neg_train = int(TRAIN_SPLIT * len(neg_list))

    train_pos_pairs = pos_list[:num_pos_train]
    test_pos_pairs = pos_list[num_pos_train:]
    train_neg_pairs = neg_list[:num_neg_train]
    test_neg_pairs = neg_list[num_neg_train:]

    facts = build_facts(users, items)
    write_bk_imports()

    write_lines(os.path.join(TRAIN_DIR, "train_pos.txt"), format_pairs(train_pos_pairs))
    write_lines(os.path.join(TRAIN_DIR, "train_neg.txt"), format_pairs(train_neg_pairs))
    write_lines(os.path.join(TRAIN_DIR, "train_facts.txt"), facts)
    write_lines(os.path.join(TEST_DIR, "test_pos.txt"), format_pairs(test_pos_pairs))
    write_lines(os.path.join(TEST_DIR, "test_neg.txt"), format_pairs(test_neg_pairs))
    write_lines(os.path.join(TEST_DIR, "test_facts.txt"), facts)

    print("Generated Teaming-style BoostSRL files for meal_project-3")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Bandit dir: {BANDIT_DIR}")
    print(f"Users: {len(users)} | Items: {len(items)}")
    print(f"comparison split users={len(split['train_users'])}/{len(split['test_users'])} (kept for notebook comparisons)")
    print(f"match_threshold={MATCH_THRESHOLD:.2f} | neg_ratio={NEG_RATIO}")
    print(
        f"train_pos={len(train_pos_pairs)} train_neg={len(train_neg_pairs)} "
        f"test_pos={len(test_pos_pairs)} test_neg={len(test_neg_pairs)}"
    )


if __name__ == "__main__":
    main()
