from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

import M0
import M1
import M2
import M3
import nlp_techniques


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESEARCHERS_FILE = PROJECT_ROOT / "data" / "v0_data" / "researchers.csv"
DEFAULT_PROPOSALS_FILE = PROJECT_ROOT / "data" / "v0_data" / "archive_proposals.csv"
DEFAULT_BOOSTED_WORKSPACE = PROJECT_ROOT / "code" / "boosted_results" / "v0_slice_100x46"
DEFAULT_RESULTS_FILE = DEFAULT_BOOSTED_WORKSPACE / "test" / "results_team.db"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "v0_teaming"
DEFAULT_FAIRNESS_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "output_v0"
DEFAULT_COMPARE_RESULTS_FILE = PROJECT_ROOT / "code" / "compare_outputs" / "comparison_summary_v0.csv"
DEFAULT_NUM_PROPOSALS = 100
DEFAULT_NUM_TEAMS = 10
DEFAULT_TEAM_SIZE = 5
DEFAULT_RANDOM_SEED = 12345
DEFAULT_MATCH_THRESHOLD = 0.5
DEFAULT_M3_MATCH_THRESHOLD = 0.30
DEFAULT_MAPPER_THRESHOLD = 0.3
DEFAULT_BANDIT_TREES = 20
DEFAULT_BANDIT_NEGATIVE_MULTIPLIER = 1.0
DEFAULT_BANDIT_TRAIN_RATIO = 0.8
DEFAULT_BANDIT_SKILL_MATCH_THRESHOLD = 0.9

NOISY_SKILL_TOKENS = {
    "acknowledgement",
    "acknowledgments",
    "apply",
    "area",
    "application",
    "applications",
    "article",
    "assistant",
    "associate",
    "authority",
    "author",
    "background",
    "benefit",
    "benefits",
    "book",
    "call",
    "center",
    "centre",
    "challenge",
    "challenges",
    "college",
    "concept",
    "concepts",
    "conference",
    "copyright",
    "country",
    "credit",
    "current",
    "dean",
    "degree",
    "department",
    "develop",
    "development",
    "developmental",
    "different",
    "director",
    "doctoral",
    "doctorate",
    "education",
    "encouragement",
    "encouraging",
    "expert",
    "field",
    "faculty",
    "foremost",
    "form",
    "foundation",
    "foundations",
    "future",
    "general",
    "government",
    "grant",
    "grants",
    "high",
    "huge",
    "impact",
    "importance",
    "india",
    "indian",
    "information",
    "initiation",
    "institute",
    "international",
    "interest",
    "interests",
    "interaction",
    "interactions",
    "knowledge",
    "letter",
    "level",
    "link",
    "links",
    "local",
    "main",
    "major",
    "manuscript",
    "member",
    "mean",
    "ministry",
    "mom",
    "motivation",
    "national",
    "need",
    "needs",
    "number",
    "objective",
    "objectives",
    "ongoing",
    "opportunity",
    "outcome",
    "outcomes",
    "overall",
    "paper",
    "persistent",
    "phd",
    "place",
    "plan",
    "potential",
    "procedure",
    "procedures",
    "professional",
    "prof",
    "profound",
    "present",
    "priority",
    "proceeding",
    "proceedings",
    "process",
    "processes",
    "professor",
    "program",
    "programme",
    "project",
    "proposal",
    "proposals",
    "purpose",
    "range",
    "reason",
    "relevant",
    "researcher",
    "researchers",
    "resource",
    "resources",
    "research",
    "result",
    "results",
    "review",
    "role",
    "science",
    "second",
    "sincere",
    "source",
    "sources",
    "spark",
    "step",
    "steps",
    "strength",
    "student",
    "study",
    "submission",
    "support",
    "system",
    "technology",
    "thank",
    "thanks",
    "theses",
    "thesis",
    "trend",
    "type",
    "types",
    "university",
    "understanding",
    "work",
    "world",
    "year",
    "young",
}

PROPOSAL_GENERIC_TOKENS = {
    "analysis",
    "applied",
    "component",
    "condition",
    "financial",
    "framework",
    "increase",
    "institution",
    "key",
    "order",
    "part",
    "product",
    "quality",
    "solution",
    "technical",
}

PROPOSAL_GENERIC_PHRASES = {
    "key area",
    "key areas",
}

NOISY_SKILL_PHRASES = {
    "assistant professor",
    "brazil prof",
    "developmental professional",
    "developmental professional education",
    "faculty initiation",
    "faculty initiation grant",
    "associate professor",
    "indian institute",
    "mom kusum",
    "ministry science",
    "professional education",
    "prof nagaria",
    "prof shailly",
    "prof shailly tomar",
    "profound understanding",
    "science technology",
    "research interest",
    "sincere appreciation",
    "sincere thank",
    "thank mhrd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the IITR Teaming methods on the paper-style v0 slice: "
            "the first 100 archive proposals and the 46 researchers from "
            "v0_data/researchers.csv. The script generates v0_teaming method "
            "outputs, regenerates a slice-specific BoostSRL workspace for M3, "
            "runs M6/M7 on top of the v0 M1 artifacts, and exports fairness "
            "tables for that slice."
        )
    )
    parser.add_argument("--researchers-file", default=str(DEFAULT_RESEARCHERS_FILE))
    parser.add_argument("--proposals-file", default=str(DEFAULT_PROPOSALS_FILE))
    parser.add_argument("--results-file", default=str(DEFAULT_RESULTS_FILE))
    parser.add_argument("--boosted-workspace", default=str(DEFAULT_BOOSTED_WORKSPACE))
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--fairness-results-dir", default=str(DEFAULT_FAIRNESS_RESULTS_DIR))
    parser.add_argument("--comparison-summary-file", default=str(DEFAULT_COMPARE_RESULTS_FILE))
    parser.add_argument("--num-proposals", type=int, default=DEFAULT_NUM_PROPOSALS)
    parser.add_argument("--num-teams", type=int, default=DEFAULT_NUM_TEAMS)
    parser.add_argument("--team-size", type=int, default=DEFAULT_TEAM_SIZE)
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--m3-match-threshold", type=float, default=DEFAULT_M3_MATCH_THRESHOLD)
    parser.add_argument("--mapper-threshold", type=float, default=DEFAULT_MAPPER_THRESHOLD)
    parser.add_argument("--bandit-trees", type=int, default=DEFAULT_BANDIT_TREES)
    parser.add_argument("--bandit-negative-multiplier", type=float, default=DEFAULT_BANDIT_NEGATIVE_MULTIPLIER)
    parser.add_argument("--bandit-train-ratio", type=float, default=DEFAULT_BANDIT_TRAIN_RATIO)
    parser.add_argument("--bandit-skill-match-threshold", type=float, default=DEFAULT_BANDIT_SKILL_MATCH_THRESHOLD)
    parser.add_argument("--skip-bandit-refresh", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--skip-fairness", action="store_true")
    return parser.parse_args()


def sanitize_token(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().strip(" '{}\"").lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"^_+", "", text)
    return text if text else "unknown"


def normalize_name(name: object) -> str:
    return " ".join(str(name or "").replace("\xa0", " ").split()).strip()


def normalize_skill_phrase(
    phrase: object,
    faculty_name: str | None = None,
    global_name_tokens: set[str] | None = None,
) -> str:
    cleaned = nlp_techniques.preprocess(str(phrase or ""))
    if not cleaned:
        return ""

    tokens = [token for token in cleaned.split() if token]
    if faculty_name:
        faculty_tokens = set(nlp_techniques.preprocess(faculty_name).split())
        tokens = [token for token in tokens if token not in faculty_tokens]
    if global_name_tokens:
        tokens = [token for token in tokens if token not in global_name_tokens]

    tokens = [token for token in tokens if len(token) >= 3 and token not in NOISY_SKILL_TOKENS]
    if not tokens:
        return ""

    normalized = " ".join(tokens)
    if normalized in NOISY_SKILL_PHRASES:
        return ""
    return normalized


def collect_phrase_skills(
    phrase: object,
    faculty_name: str | None = None,
    global_name_tokens: set[str] | None = None,
) -> set[str]:
    normalized = normalize_skill_phrase(
        phrase,
        faculty_name=faculty_name,
        global_name_tokens=global_name_tokens,
    )
    if not normalized:
        return set()

    tokens = normalized.split()
    skills = {normalized}
    if len(tokens) >= 2:
        skills.update(ngram for ngram in nlp_techniques.generate_N_grams(normalized, ngram=2) if ngram and ngram not in NOISY_SKILL_PHRASES)
    return skills


def build_skill_document_frequency(researcher_skills: dict[str, set[str]]) -> dict[str, int]:
    df: dict[str, int] = defaultdict(int)
    for skills in researcher_skills.values():
        for skill in set(skills):
            df[skill] += 1
    return dict(df)


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_researchers(path: Path) -> pd.DataFrame:
    researchers = pd.read_csv(path).copy()
    researchers["FacultyName"] = researchers["FacultyName"].map(normalize_name)
    researchers["Research Interests"] = researchers["Research Interests"].fillna("")
    return researchers


def load_v0_proposals(path: Path, limit: int) -> pd.DataFrame:
    proposals = pd.read_csv(path).copy()
    proposals["Title"] = proposals["Title"].fillna("general").map(normalize_name)
    proposals["Attachment Text"] = proposals["Attachment Text"].fillna("")
    proposals = proposals.head(limit).reset_index(drop=True)
    proposals["title_token"] = proposals["Title"].map(sanitize_token)
    proposals["proposal_link"] = [
        f"v0_{index:03d}_{title_token}"
        for index, title_token in enumerate(proposals["title_token"], start=1)
    ]
    # Use the same unique, slice-stable identifier everywhere so duplicate
    # proposal titles do not collapse distinct rows inside BoostSRL.
    proposals["proposal_id"] = proposals["proposal_link"]
    return proposals


def build_full_bandit_examples(
    proposals: pd.DataFrame,
    researchers: pd.DataFrame,
    proposal_skills: dict[str, set[str]],
    researcher_skills: dict[str, set[str]],
    bandit_skill_match_threshold: float,
) -> tuple[list[str], list[str]]:
    sanitized_researcher_skills = {
        researcher_name: sorted({sanitize_token(skill) for skill in skills if sanitize_token(skill) != "unknown"})
        for researcher_name, skills in researcher_skills.items()
    }

    positive_examples: set[str] = set()
    all_examples: set[str] = set()

    for proposal_row in proposals.itertuples(index=False):
        proposal_id = proposal_row.proposal_id
        required_skills = {
            sanitize_token(skill)
            for skill in proposal_skills[proposal_row.proposal_link]
            if sanitize_token(skill) != "unknown"
        }
        for researcher_name in researchers["FacultyName"]:
            researcher_id = sanitize_token(researcher_name)
            pair = f"team({proposal_id},{researcher_id})."
            all_examples.add(pair)

            researcher_interest_tokens = sanitized_researcher_skills.get(researcher_name, [])
            if any(
                SequenceMatcher(None, proposal_skill, researcher_skill).ratio() >= bandit_skill_match_threshold
                for proposal_skill in required_skills
                for researcher_skill in researcher_interest_tokens
            ):
                positive_examples.add(pair)

    negative_examples = all_examples - positive_examples
    return sorted(positive_examples), sorted(negative_examples)


def run_boostsrl_inference(
    workspace: Path,
    boostsrl_jar: Path,
    auc_dir: Path,
    bandit_trees: int,
    log_filename: str,
) -> Path:
    log_path = workspace / log_filename
    with log_path.open("w", encoding="utf-8") as test_log:
        subprocess.run(
            [
                "java",
                "-jar",
                str(boostsrl_jar),
                "-i",
                "-model",
                "train/models/",
                "-test",
                "test/",
                "-target",
                "team",
                "-aucJarPath",
                str(auc_dir),
                "-trees",
                str(bandit_trees),
            ],
            check=True,
            cwd=workspace,
            stdout=test_log,
            stderr=subprocess.STDOUT,
        )

    results_path = workspace / "test" / "results_team.db"
    if not results_path.exists():
        raise FileNotFoundError(f"BoostSRL inference did not produce {results_path}")
    return results_path


def prepare_boosted_workspace(
    workspace: Path,
    proposals: pd.DataFrame,
    researchers: pd.DataFrame,
    proposal_skills: dict[str, set[str]],
    researcher_skills: dict[str, set[str]],
    bandit_trees: int,
    bandit_negative_multiplier: float,
    bandit_train_ratio: float,
    bandit_skill_match_threshold: float,
    seed: int,
) -> Path:
    boosted_root = PROJECT_ROOT / "code" / "boosted_results"
    prepare_script = boosted_root / "prepare_data.py"
    teams_bk_source = boosted_root / "teams_bk.txt"
    boostsrl_jar = boosted_root / "boostsrl_v1.1.1.jar"
    auc_dir = boosted_root

    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "train").mkdir(parents=True, exist_ok=True)
    (workspace / "test").mkdir(parents=True, exist_ok=True)

    shutil.copy2(teams_bk_source, workspace / "teams_bk.txt")
    write_text_lines(workspace / "train" / "train_bk.txt", ['import: "../teams_bk.txt".'])
    write_text_lines(workspace / "test" / "test_bk.txt", ['import: "../teams_bk.txt".'])

    proposal_rows = [
        {
            "hyperlink": row.proposal_id,
            "skills": ", ".join(sorted(proposal_skills[row.proposal_link])),
        }
        for row in proposals.itertuples(index=False)
    ]
    researcher_rows = [
        {
            "researcher_name": row.FacultyName,
            "skills": ", ".join(sorted(researcher_skills[row.FacultyName])),
        }
        for row in researchers.itertuples(index=False)
    ]
    write_rows(workspace / "proposal_skills.csv", proposal_rows, ["hyperlink", "skills"])
    write_rows(workspace / "researchers_skills.csv", researcher_rows, ["researcher_name", "skills"])

    unique_proposal_ids = {row["hyperlink"] for row in proposal_rows}
    if len(unique_proposal_ids) != len(proposal_rows):
        raise ValueError(
            "BoostSRL workspace preparation found duplicate proposal identifiers. "
            "Each proposal in the 100x46 slice must map to a unique logic ID."
        )

    subprocess.run(
        [
            str(Path(sys.executable)),
            str(prepare_script),
            "--negative-multiplier",
            str(bandit_negative_multiplier),
            "--train-ratio",
            str(bandit_train_ratio),
            "--skill-match-threshold",
            str(bandit_skill_match_threshold),
            "--seed",
            str(seed),
        ],
        check=True,
        cwd=workspace,
    )

    holdout_test_pos = read_text_lines(workspace / "test" / "test_pos.txt")
    holdout_test_neg = read_text_lines(workspace / "test" / "test_neg.txt")

    query_lines = [
        f"team({proposal_row.proposal_id},{sanitize_token(researcher_name)})"
        for proposal_row in proposals.itertuples(index=False)
        for researcher_name in researchers["FacultyName"]
    ]
    write_text_lines(workspace / "test" / "query_team.db", query_lines)
    expected_query_count = len(proposals) * len(researchers)
    if len(query_lines) != expected_query_count:
        raise ValueError(
            f"Expected {expected_query_count} BoostSRL queries for the 100x46 slice, "
            f"but generated {len(query_lines)}."
        )

    full_inference_pos, full_inference_neg = build_full_bandit_examples(
        proposals,
        researchers,
        proposal_skills,
        researcher_skills,
        bandit_skill_match_threshold,
    )

    with (workspace / "output_train.txt").open("w", encoding="utf-8") as train_log:
        subprocess.run(
            [
                "java",
                "-jar",
                str(boostsrl_jar),
                "-l",
                "-combine",
                "-train",
                "train/",
                "-target",
                "team",
                "-trees",
                str(bandit_trees),
            ],
            check=True,
            cwd=workspace,
            stdout=train_log,
            stderr=subprocess.STDOUT,
        )

    heldout_results_path = run_boostsrl_inference(
        workspace,
        boostsrl_jar,
        auc_dir,
        bandit_trees,
        "output_test_holdout.txt",
    )

    heldout_eval_dir = workspace / "heldout_eval"
    heldout_eval_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in [
        ("test/test_pos.txt", "test_pos.txt"),
        ("test/test_neg.txt", "test_neg.txt"),
        ("test/test_facts.txt", "test_facts.txt"),
        ("test/test_infer_dribble.txt", "test_infer_dribble.txt"),
        ("test/results_team.db", "results_team.db"),
        ("test/query_team.db", "query_team.db"),
        ("output_test_holdout.txt", "output_test_holdout.txt"),
    ]:
        shutil.copy2(workspace / source_name, heldout_eval_dir / target_name)

    heldout_results_lines = read_text_lines(heldout_eval_dir / "results_team.db")

    write_text_lines(workspace / "test" / "test_pos.txt", full_inference_pos)
    write_text_lines(workspace / "test" / "test_neg.txt", full_inference_neg)
    write_text_lines(workspace / "test" / "query_team.db", query_lines)

    results_path = run_boostsrl_inference(
        workspace,
        boostsrl_jar,
        auc_dir,
        bandit_trees,
        "output_test.txt",
    )

    full_inference_results = read_text_lines(results_path)
    workspace_metadata = {
        "dataset": "IITR-Teaming v0 paper slice",
        "proposal_count": int(len(proposals)),
        "anchor_researcher_count": int(len(researchers)),
        "unique_proposal_ids": int(len(unique_proposal_ids)),
        "query_count": int(len(query_lines)),
        "expected_query_count": int(expected_query_count),
        "negative_multiplier": float(bandit_negative_multiplier),
        "train_ratio": float(bandit_train_ratio),
        "skill_match_threshold": float(bandit_skill_match_threshold),
        "trees": int(bandit_trees),
        "seed": int(seed),
        "heldout_evaluation": {
            "test_pos_count": int(len(holdout_test_pos)),
            "test_neg_count": int(len(holdout_test_neg)),
            "predicted_example_count": int(len(heldout_results_lines)),
            "results_file": str(heldout_eval_dir / "results_team.db"),
        },
        "full_inference": {
            "pos_count": int(len(full_inference_pos)),
            "neg_count": int(len(full_inference_neg)),
            "predicted_example_count": int(len(full_inference_results)),
            "results_file": str(results_path),
        },
    }
    (workspace / "workspace_metadata.json").write_text(
        json.dumps(workspace_metadata, indent=2),
        encoding="utf-8",
    )

    return results_path


def build_global_name_tokens(researchers: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for name in researchers["FacultyName"]:
        tokens.update(token for token in nlp_techniques.preprocess(str(name or "")).split() if len(token) >= 3)
    return tokens


def build_researcher_skills(researchers: pd.DataFrame) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    global_name_tokens = build_global_name_tokens(researchers)
    for _, row in researchers.iterrows():
        researcher = row["FacultyName"]
        interests = str(row["Research Interests"] or "")

        cleaned_skills: set[str] = set()
        for fragment in interests.split("|"):
            fragment = fragment.strip()
            if not fragment:
                continue
            for comma_piece in fragment.split(","):
                cleaned_skills.update(
                    collect_phrase_skills(
                        comma_piece,
                        faculty_name=researcher,
                        global_name_tokens=global_name_tokens,
                    )
                )

        mapping[researcher] = cleaned_skills or {"general"}

    return mapping


def build_all_researcher_skills(researcher_skills: dict[str, set[str]]) -> list[str]:
    all_skills: list[str] = []
    for skills in researcher_skills.values():
        for skill in skills:
            if skill not in all_skills:
                all_skills.append(skill)
    return all_skills


def build_proposal_skills(
    proposals: pd.DataFrame,
    all_researcher_skills: list[str],
    skill_document_frequency: dict[str, int],
    researcher_count: int,
) -> dict[str, set[str]]:
    proposal_skills: dict[str, set[str]] = {}
    allowed_skills = set(all_researcher_skills)
    max_skill_document_frequency = max(6, math.ceil(researcher_count * 0.2))

    for _, row in proposals.iterrows():
        title_text = nlp_techniques.preprocess(row["Title"])
        attachment_text = nlp_techniques.preprocess(row["Attachment Text"])

        title_tokens = [
            token
            for token in title_text.split()
            if token and token not in NOISY_SKILL_TOKENS and token not in PROPOSAL_GENERIC_TOKENS
        ]
        attachment_tokens = [
            token
            for token in attachment_text.split()
            if token and token not in NOISY_SKILL_TOKENS and token not in PROPOSAL_GENERIC_TOKENS
        ]
        filtered_title_text = " ".join(title_tokens)
        filtered_attachment_text = " ".join(attachment_tokens)

        keywords = title_tokens + attachment_tokens
        title_ngrams = nlp_techniques.generate_N_grams(filtered_title_text, ngram=2)
        attachment_ngrams = nlp_techniques.generate_N_grams(filtered_attachment_text, ngram=2)
        all_keywords = {
            keyword
            for keyword in (keywords + title_ngrams + attachment_ngrams)
            if keyword and keyword not in PROPOSAL_GENERIC_PHRASES
        }
        filtered = {
            keyword
            for keyword in all_keywords
            if keyword in allowed_skills and skill_document_frequency.get(keyword, 0) <= max_skill_document_frequency
        }
        proposal_skills[row["proposal_link"]] = filtered or {"general"}

    return proposal_skills


def compute_string_matches(
    researcher_skills: dict[str, set[str]],
    proposal_skills: set[str],
    matching_threshold: float,
    skill_document_frequency: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    pseudo_researcher_skills: dict[str, list[str]] = {}
    ranking_rows = []
    total_weight = 0.0
    if skill_document_frequency:
        total_weight = sum(1.0 / max(1, skill_document_frequency.get(skill, 1)) for skill in proposal_skills)

    for researcher, skills in researcher_skills.items():
        matched_skills: list[str] = []
        for proposal_skill in proposal_skills:
            for skill in skills:
                if SequenceMatcher(None, proposal_skill, skill).ratio() >= matching_threshold:
                    matched_skills.append(proposal_skill)
                    break
        deduped = sorted(set(matched_skills))
        pseudo_researcher_skills[researcher] = deduped
        if skill_document_frequency and total_weight > 0:
            weighted_overlap = sum(1.0 / max(1, skill_document_frequency.get(skill, 1)) for skill in deduped)
            ranking_score = weighted_overlap / total_weight
        else:
            ranking_score = len(deduped) / max(1, len(proposal_skills))
        ranking_rows.append((researcher, ranking_score))

    ranking_rows.sort(key=lambda item: (item[1], item[0]))
    return dict(ranking_rows), pseudo_researcher_skills


def create_ranked_teams(
    ranking: dict[str, float],
    target_researcher: str,
    num_teams: int,
    shortlist_size: int = 20,
) -> list[list[str]]:
    ranked_names = [name for name in ranking if name != target_researcher]
    candidate_pool = ranked_names[-shortlist_size:] if shortlist_size > 0 else ranked_names
    candidate_pool = list(dict.fromkeys(candidate_pool))

    if not candidate_pool:
        return [[target_researcher]]

    teams: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0

    while len(teams) < num_teams and attempts < num_teams + 50:
        attempts += 1
        size = min(random.randint(1, 4), len(candidate_pool))
        chosen = random.sample(candidate_pool, size) if size > 0 else []
        team = [target_researcher, *chosen]
        signature = (target_researcher, *tuple(sorted(chosen)))
        if signature in seen:
            continue
        seen.add(signature)
        teams.append(team)

    return teams or [[target_researcher]]


def create_candidate_teams(
    candidate_researchers: list[str],
    target_researcher: str,
    num_teams: int,
) -> list[list[str]]:
    candidate_pool = [name for name in dict.fromkeys(candidate_researchers) if name != target_researcher]
    if not candidate_pool:
        return [[target_researcher]]

    teams: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0

    while len(teams) < num_teams and attempts < num_teams + 50:
        attempts += 1
        size = min(random.randint(1, 4), len(candidate_pool))
        chosen = random.sample(candidate_pool, size) if size > 0 else []
        team = [target_researcher, *chosen]
        signature = (target_researcher, *tuple(sorted(chosen)))
        if signature in seen:
            continue
        seen.add(signature)
        teams.append(team)

    return teams or [[target_researcher]]


def score_and_sort_variants(
    teams: list[list[str]],
    proposal_skills: list[str] | set[str],
    skill_map: dict[str, list[str] | set[str]],
    scorer,
    round_digits: int = 4,
) -> tuple[list[list[str]], list[float]]:
    scored = []
    for team in teams:
        deduped_team = list(dict.fromkeys(team))
        score = float(scorer(proposal_skills, deduped_team, skill_map))
        scored.append((round(score, round_digits), deduped_team))

    scored.sort(key=lambda item: (item[0], tuple(item[1])), reverse=True)
    return [team for _, team in scored], [score for score, _ in scored]


def flatten_team_rows(
    proposal_link: str,
    researcher_name: str,
    sorted_teams: list[list[str]],
    sorted_scores: list[float],
    researcher_field: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    teaming_rows = [
        {
            "nsf_proposal_links_v0": proposal_link,
            researcher_field: researcher_name,
            "team": team,
        }
        for team in sorted_teams
    ]
    goodness_rows = [
        {
            "nsf_proposal_links_v0": proposal_link,
            "researcher_name": researcher_name,
            "goodness": score,
        }
        for score in sorted_scores
    ]
    return teaming_rows, goodness_rows


def parse_mapper_payload(results: tuple[object, object]) -> list[object]:
    codes = list(results[0]) if isinstance(results[0], (list, tuple)) else [results[0]]
    raw_terms = list(results[1]) if isinstance(results[1], (list, tuple)) else [[results[1]]]
    terms = [item for sublist in raw_terms for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
    terms = [str(term).strip() for term in terms if str(term).strip()]
    preprocessed_terms = [nlp_techniques.preprocess(term) for term in terms]
    return [codes, terms, preprocessed_terms]


def mapper_lookup(skill: str, threshold: float, cache: dict[str, list[object]]) -> list[object]:
    if skill not in cache:
        with io.StringIO() as buffer, redirect_stdout(buffer), redirect_stderr(buffer):
            with M2.app.app_context():
                try:
                    raw_payload = M2.getValue(skill, threshold, "acm")
                except Exception:
                    raw_payload = (["A.0"], [["GENERAL", ""]])
                cache[skill] = parse_mapper_payload(raw_payload)
    return cache[skill]


def build_bandit_name_variants(researcher_name: str) -> list[str]:
    base = sanitize_token(researcher_name)
    return [
        base,
        f"dr_{base}",
        f"dr__{base}",
    ]


def parse_bandit_results(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    pos_teams: dict[str, dict[str, float]] = defaultdict(dict)
    neg_teams: dict[str, dict[str, float]] = defaultdict(dict)
    pattern = re.compile(r"^(!?team)\(([^,]+),([^)]+)\)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            predicate, proposal_id, researcher_id, score_text = match.groups()
            proposal_id = proposal_id.strip()
            researcher_id = researcher_id.strip()
            bucket = neg_teams if predicate.startswith("!team") else pos_teams
            bucket[proposal_id][researcher_id] = float(score_text)

    return dict(pos_teams), dict(neg_teams)


def build_combined_ranking(
    lexical_ranking: dict[str, float],
    positive_scores: dict[str, float],
    negative_scores: dict[str, float],
) -> dict[str, float]:
    combined_rows = []
    for researcher_name, lexical_score in lexical_ranking.items():
        score = lexical_score
        variants = build_bandit_name_variants(researcher_name)
        matching_pos = [positive_scores[variant] for variant in variants if variant in positive_scores]
        matching_neg = [negative_scores[variant] for variant in variants if variant in negative_scores]

        if matching_pos:
            score += 1.0 + max(matching_pos)
        elif matching_neg:
            score -= 1.0 + max(abs(value) for value in matching_neg)

        combined_rows.append((researcher_name, score))

    combined_rows.sort(key=lambda item: (item[1], item[0]))
    return dict(combined_rows)


def compute_m3_target_width(positive_count: int, num_teams: int) -> int:
    if num_teams < 1:
        return 0
    if positive_count <= 0:
        return min(num_teams, 8)
    structural_width = round(2.5 * math.sqrt(positive_count))
    return max(1, min(num_teams, structural_width))


def run_m0(
    output_base: Path,
    proposals: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    num_teams: int,
) -> None:
    method_dir = output_base / "data_uc1_m0"
    all_researchers = list(researcher_skills.keys())

    proposal_skill_rows = []
    researcher_skill_rows = []
    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for proposal_link, skills in proposal_skills.items():
        proposal_skill_rows.append({"nsf_proposal_links_v0": proposal_link, "skills": sorted(skills)})

    for researcher_name, skills in researcher_skills.items():
        researcher_skill_rows.append({"researcher_name": researcher_name, "skills": sorted(skills)})

    total_proposals = len(proposals)
    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        required_skills = proposal_skills[proposal_row.proposal_link]
        for target_researcher in all_researchers:
            teams = create_candidate_teams(all_researchers, target_researcher, num_teams)
            sorted_teams, sorted_scores = score_and_sort_variants(
                teams,
                required_skills,
                researcher_skills,
                M0.apply_ultra_metric,
            )
            method_teaming_rows, method_goodness_rows = flatten_team_rows(
                proposal_row.proposal_link,
                target_researcher,
                sorted_teams,
                sorted_scores,
                "researcher_name",
            )
            teaming_rows.extend(method_teaming_rows)
            goodness_rows.extend(method_goodness_rows)
            final_rows.append(
                {
                    "proposal_link": proposal_row.proposal_link,
                    "title": proposal_row.Title,
                    "skills": sorted(required_skills),
                    "researcher_name": target_researcher,
                    "team": sorted_teams,
                    "goodness": sorted_scores,
                }
            )
        if proposal_index % 25 == 0 or proposal_index == total_proposals:
            print(f"[v0:M0] processed {proposal_index}/{total_proposals} proposals")

    write_rows(method_dir / "m0_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    write_rows(method_dir / "m0_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    write_rows(method_dir / "m0_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher_name", "team"])
    write_rows(method_dir / "m0_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    write_rows(output_base / "teaming_uc1_m0.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])


def run_m1(
    output_base: Path,
    proposals: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    num_teams: int,
    matching_threshold: float,
    skill_document_frequency: dict[str, int],
) -> None:
    method_dir = output_base / "data_uc1_m1"
    all_skill_rows = [{"all_skills": skill} for skill in build_all_researcher_skills(researcher_skills)]
    researcher_skill_rows = [{"researcher_name": name, "skills": sorted(skills)} for name, skills in researcher_skills.items()]
    proposal_skill_rows = []
    pseudo_skill_rows: list[dict[str, object]] = []
    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    total_proposals = len(proposals)
    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        required_skills = proposal_skills[proposal_row.proposal_link]
        ranking, pseudo_skill_map = compute_string_matches(
            researcher_skills,
            required_skills,
            matching_threshold,
            skill_document_frequency,
        )
        proposal_skill_rows.append({"nsf_proposal_links_v0": proposal_row.proposal_link, "skills": sorted(required_skills)})
        for researcher_name, pseudo_skills in pseudo_skill_map.items():
            pseudo_skill_rows.append(
                {
                    "nsf_proposal_links_v0": proposal_row.proposal_link,
                    "researcher": researcher_name,
                    "pseudo_skills": pseudo_skills,
                }
            )

        pseudo_skill_map_for_metric = {name: set(skills) for name, skills in pseudo_skill_map.items()}
        for target_researcher in researcher_skills:
            teams = create_ranked_teams(ranking, target_researcher, num_teams)
            sorted_teams, sorted_scores = score_and_sort_variants(
                teams,
                required_skills,
                pseudo_skill_map_for_metric,
                M1.apply_ultra_metric,
            )
            method_teaming_rows, method_goodness_rows = flatten_team_rows(
                proposal_row.proposal_link,
                target_researcher,
                sorted_teams,
                sorted_scores,
                "researcher_name",
            )
            teaming_rows.extend(method_teaming_rows)
            goodness_rows.extend(method_goodness_rows)
            final_rows.append(
                {
                    "proposal_link": proposal_row.proposal_link,
                    "title": proposal_row.Title,
                    "skills": sorted(required_skills),
                    "researcher_name": target_researcher,
                    "team": sorted_teams,
                    "goodness": sorted_scores,
                }
            )

        if proposal_index % 25 == 0 or proposal_index == total_proposals:
            print(f"[v0:M1] processed {proposal_index}/{total_proposals} proposals")

    write_rows(method_dir / "m1_all_researcher_skills.csv", all_skill_rows, ["all_skills"])
    write_rows(method_dir / "m1_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    write_rows(method_dir / "m1_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    write_rows(method_dir / "m1_pseudo_researcher_skills.csv", pseudo_skill_rows, ["nsf_proposal_links_v0", "researcher", "pseudo_skills"])
    write_rows(method_dir / "m1_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher_name", "team"])
    write_rows(method_dir / "m1_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    write_rows(output_base / "teaming_uc1_m1.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])


def run_m2(
    output_base: Path,
    proposals: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    num_teams: int,
    mapper_threshold: float,
) -> None:
    method_dir = output_base / "data_uc1_m2"
    mapper_cache: dict[str, list[object]] = {}

    all_skill_rows = [{"all_skills": skill} for skill in build_all_researcher_skills(researcher_skills)]
    researcher_skill_rows = [{"researcher_name": name, "skills": sorted(skills)} for name, skills in researcher_skills.items()]
    proposal_skill_rows = [{"nsf_proposal_links_v0": row.proposal_link, "skills": sorted(proposal_skills[row.proposal_link])} for row in proposals.itertuples(index=False)]

    m2_proposal_skills_mapper: dict[str, dict[str, list[object]]] = {}
    m2_researcher_skills_mapper: dict[str, dict[str, list[object]]] = {}
    m2_proposal_mapper_category_sets: dict[str, set[str]] = {}
    m2_researcher_mapper_category_sets: dict[str, set[str]] = {}

    for proposal_row in proposals.itertuples(index=False):
        proposal_link = proposal_row.proposal_link
        m2_proposal_skills_mapper[proposal_link] = {}
        m2_proposal_mapper_category_sets[proposal_link] = set()
        for skill in sorted(proposal_skills[proposal_link]):
            payload = mapper_lookup(skill, mapper_threshold, mapper_cache)
            m2_proposal_skills_mapper[proposal_link][skill] = payload
            for code in payload[0]:
                code_text = str(code)
                m2_proposal_mapper_category_sets[proposal_link].add(code_text[:3] if len(code_text) > 3 else code_text)

    for researcher_name, skills in researcher_skills.items():
        m2_researcher_skills_mapper[researcher_name] = {}
        m2_researcher_mapper_category_sets[researcher_name] = set()
        for skill in sorted(skills):
            payload = mapper_lookup(skill, mapper_threshold, mapper_cache)
            m2_researcher_skills_mapper[researcher_name][skill] = payload
            for code in payload[0]:
                code_text = str(code)
                m2_researcher_mapper_category_sets[researcher_name].add(code_text[:3] if len(code_text) > 3 else code_text)

    m2_proposal_mapper_categories = {
        proposal: sorted(values)
        for proposal, values in m2_proposal_mapper_category_sets.items()
    }
    m2_researcher_mapper_categories = {
        researcher: sorted(values)
        for researcher, values in m2_researcher_mapper_category_sets.items()
    }

    proposal_mapper_rows = [{"proposal": proposal, "skill_categories": values} for proposal, values in m2_proposal_mapper_categories.items()]
    researcher_mapper_rows = [{"researcher": researcher, "skill_categories": values} for researcher, values in m2_researcher_mapper_categories.items()]
    proposal_skills_mapper_rows = [
        {"proposal": proposal, "skill": skill, "mapper_payload": payload}
        for proposal, skill_map in m2_proposal_skills_mapper.items()
        for skill, payload in skill_map.items()
    ]
    researcher_skills_mapper_rows = [
        {"researcher": researcher, "skill": skill, "mapper_payload": payload}
        for researcher, skill_map in m2_researcher_skills_mapper.items()
        for skill, payload in skill_map.items()
    ]

    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    total_proposals = len(proposals)

    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        proposal_link = proposal_row.proposal_link
        candidate_researchers = [
            researcher
            for researcher, categories in m2_researcher_mapper_category_sets.items()
            if categories & m2_proposal_mapper_category_sets[proposal_link]
        ]
        for target_researcher in researcher_skills:
            teams = create_candidate_teams(candidate_researchers, target_researcher, num_teams)
            sorted_teams, sorted_scores = score_and_sort_variants(
                teams,
                m2_proposal_mapper_categories[proposal_link],
                m2_researcher_mapper_categories,
                M2.apply_ultra_metric,
            )
            method_teaming_rows, method_goodness_rows = flatten_team_rows(
                proposal_link,
                target_researcher,
                sorted_teams,
                sorted_scores,
                "researcher_name",
            )
            teaming_rows.extend(method_teaming_rows)
            goodness_rows.extend(method_goodness_rows)
            final_rows.append(
                {
                    "proposal_link": proposal_link,
                    "title": proposal_row.Title,
                    "skills": sorted(proposal_skills[proposal_link]),
                    "researcher_name": target_researcher,
                    "team": sorted_teams,
                    "goodness": sorted_scores,
                }
            )

        if proposal_index % 25 == 0 or proposal_index == total_proposals:
            print(f"[v0:M2] processed {proposal_index}/{total_proposals} proposals")

    write_rows(method_dir / "m2_all_researcher_skills.csv", all_skill_rows, ["all_skills"])
    write_rows(method_dir / "m2_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    write_rows(method_dir / "m2_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    write_rows(method_dir / "m2_proposal_mapper_categories.csv", proposal_mapper_rows, ["proposal", "skill_categories"])
    write_rows(method_dir / "m2_researcher_mapper_categories.csv", researcher_mapper_rows, ["researcher", "skill_categories"])
    write_rows(method_dir / "m2_proposal_skills_mapper.csv", proposal_skills_mapper_rows, ["proposal", "skill", "mapper_payload"])
    write_rows(method_dir / "m2_researcher_skills_mapper.csv", researcher_skills_mapper_rows, ["researcher", "skill", "mapper_payload"])
    write_rows(method_dir / "m2_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher_name", "team"])
    write_rows(method_dir / "m2_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    write_rows(output_base / "teaming_uc1_m2.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])


def run_m3(
    output_base: Path,
    proposals: pd.DataFrame,
    researchers: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    results_file: Path,
    num_teams: int,
    matching_threshold: float,
    skill_document_frequency: dict[str, int],
    team_size: int,
) -> None:
    method_dir = output_base / "data_uc1_m3"
    pos_teams, neg_teams = parse_bandit_results(results_file)

    researcher_skill_rows = [{"researcher_name": name, "skills": sorted(skills)} for name, skills in researcher_skills.items()]
    all_skill_rows = [{"all_skills": skill} for skill in build_all_researcher_skills(researcher_skills)]
    proposal_skill_rows = [{"nsf_proposal_links_v0": proposal_link, "skills": sorted(skills)} for proposal_link, skills in proposal_skills.items()]

    m3_teaming: dict[str, list[list[object]]] = {}
    m3_pseudo_researcher_skills: dict[str, dict[str, list[str]]] = {}
    alignment_rows: list[dict[str, object]] = []

    total_proposals = len(proposals)
    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        proposal_link = proposal_row.proposal_link
        proposal_id = proposal_row.proposal_id
        required_skills = proposal_skills[proposal_link]
        lexical_ranking, pseudo_researcher_skills = compute_string_matches(
            researcher_skills,
            required_skills,
            matching_threshold,
            skill_document_frequency,
        )

        positive_scores = pos_teams.get(proposal_id, {})
        negative_scores = neg_teams.get(proposal_id, {})
        combined_ranking = build_combined_ranking(
            lexical_ranking,
            positive_scores,
            negative_scores,
        )

        alignment_rows.append(
            {
                "proposal_link": proposal_link,
                "proposal_id": proposal_id,
                "title": proposal_row.Title,
                "bandit_positive_count": len(positive_scores),
                "bandit_negative_count": len(negative_scores),
                "bandit_matched": bool(positive_scores or negative_scores),
            }
        )

        # When BoostSRL has positive matches, restrict the candidate pool to
        # those researchers — this mirrors the paper pseudocode ("build teams
        # mainly from the positive pool") and produces volume < num_teams when
        # the positive pool is small, just like USC M3.  Fall back to the full
        # combined ranking only when BoostSRL has no positive predictions.
        if positive_scores:
            pos_researcher_ranking = {
                name: combined_ranking[name]
                for name in combined_ranking
                if any(v in positive_scores for v in build_bandit_name_variants(name))
            }
            team_ranking = pos_researcher_ranking if pos_researcher_ranking else combined_ranking
        else:
            team_ranking = combined_ranking

        m3_pseudo_researcher_skills[proposal_link] = pseudo_researcher_skills
        m3_teaming[proposal_link] = []
        for target_researcher in researchers["FacultyName"]:
            teams = create_ranked_teams(team_ranking, target_researcher, num_teams)
            m3_teaming[proposal_link].append([target_researcher, teams])

        if proposal_index % 25 == 0 or proposal_index == total_proposals:
            print(f"[v0:M3] processed {proposal_index}/{total_proposals} proposals")

    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    pseudo_skill_rows: list[dict[str, object]] = []

    for proposal_link, pseudo_skill_map in m3_pseudo_researcher_skills.items():
        for researcher_name, pseudo_skills in pseudo_skill_map.items():
            pseudo_skill_rows.append(
                {
                    "nsf_proposal_links_v0": proposal_link,
                    "researcher": researcher_name,
                    "pseudo_skills": pseudo_skills,
                }
            )

    positive_counts_by_proposal = {
        str(row["proposal_link"]): int(row["bandit_positive_count"])
        for row in alignment_rows
    }

    for proposal_row in proposals.itertuples(index=False):
        proposal_link = proposal_row.proposal_link
        pseudo_skill_map_for_metric = {name: set(skills) for name, skills in m3_pseudo_researcher_skills[proposal_link].items()}
        researcher_team_rows = m3_teaming[proposal_link]
        for researcher_name, teams in researcher_team_rows:
            sorted_teams, sorted_scores = score_and_sort_variants(
                teams,
                proposal_skills[proposal_link],
                pseudo_skill_map_for_metric,
                M3.apply_ultra_metric,
            )
            target_width = compute_m3_target_width(
                positive_counts_by_proposal.get(str(proposal_link), 0),
                num_teams,
            )
            sorted_teams = sorted_teams[:target_width]
            sorted_scores = sorted_scores[:target_width]
            method_teaming_rows, method_goodness_rows = flatten_team_rows(
                proposal_link,
                researcher_name,
                sorted_teams,
                sorted_scores,
                "researcher",
            )
            teaming_rows.extend(method_teaming_rows)
            goodness_rows.extend(method_goodness_rows)
            final_rows.append(
                {
                    "proposal_link": proposal_link,
                    "title": proposal_row.Title,
                    "skills": sorted(proposal_skills[proposal_link]),
                    "researcher_name": researcher_name,
                    "team": sorted_teams,
                    "goodness": sorted_scores,
                }
            )

    write_rows(method_dir / "m3_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    write_rows(method_dir / "m3_all_researcher_skills.csv", all_skill_rows, ["all_skills"])
    write_rows(method_dir / "m3_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    write_rows(method_dir / "m3_pseudo_researcher_skills.csv", pseudo_skill_rows, ["nsf_proposal_links_v0", "researcher", "pseudo_skills"])
    write_rows(method_dir / "m3_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher", "team"])
    write_rows(method_dir / "m3_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    write_rows(method_dir / "m3_bandit_alignment.csv", alignment_rows, ["proposal_link", "proposal_id", "title", "bandit_positive_count", "bandit_negative_count", "bandit_matched"])
    write_rows(output_base / "teaming_uc1_m3.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])

    metadata = {
        "method": "M3",
        "description": (
            "v0 IITR M3 run on the fairness-paper slice using the first 100 archive proposals "
            "and the 46 researchers in v0_data/researchers.csv. A slice-specific BoostSRL workspace "
            "is regenerated before inference. The workspace preserves a held-out BoostSRL evaluation "
            "artifact, then reruns inference on the full 100x46 slice so the final proposal-member "
            "scores used by M3 cover every proposal-anchor pair. Those scores are then used to build "
            "teams with lexical fallback where the learned positive pool is empty."
        ),
        "researchers_file": str(DEFAULT_RESEARCHERS_FILE),
        "proposals_file": str(DEFAULT_PROPOSALS_FILE),
        "results_file": str(results_file),
        "proposal_count": int(len(proposals)),
        "anchor_researcher_count": int(researchers["FacultyName"].nunique()),
        "unique_proposal_ids": int(proposals["proposal_id"].nunique()),
        "expected_query_count": int(len(proposals) * len(researchers)),
        "bandit_scored_pairs": int(sum(len(values) for values in pos_teams.values()) + sum(len(values) for values in neg_teams.values())),
        "bandit_matched_proposals": int(sum(1 for row in alignment_rows if row["bandit_matched"])),
        "bandit_unmatched_proposals": int(sum(1 for row in alignment_rows if not row["bandit_matched"])),
        "uses_representative_variant_selection": False,
        "uses_pool_based_width_cap": True,
        "m3_target_width_rule": "round(2.5 * sqrt(positive_count)), capped to [1, num_teams], with zero-positive fallback to 8",
    }
    (method_dir / "m3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_knapsack_methods(output_base: Path, team_size: int, num_teams: int) -> None:
    python_bin = Path(sys.executable)
    base_args = [
        str(python_bin),
    ]

    m6_command = base_args + [
        str(PROJECT_ROOT / "code" / "M6.py"),
        "--skeleton-file",
        str(output_base / "teaming_uc1_m3.csv"),
        "--proposal-skills-file",
        str(output_base / "data_uc1_m1" / "m1_proposal_skills.csv"),
        "--pseudo-skills-file",
        str(output_base / "data_uc1_m1" / "m1_pseudo_researcher_skills.csv"),
        "--output-dir",
        str(output_base / "data_uc1_m6"),
        "--output-file",
        str(output_base / "teaming_uc1_m6.csv"),
        "--team-size",
        str(team_size),
        "--num-teams",
        str(num_teams),
    ]
    subprocess.run(m6_command, check=True, cwd=PROJECT_ROOT / "code")

    m7_command = base_args + [
        str(PROJECT_ROOT / "code" / "M7.py"),
        "--skeleton-file",
        str(output_base / "teaming_uc1_m3.csv"),
        "--proposal-skills-file",
        str(output_base / "data_uc1_m1" / "m1_proposal_skills.csv"),
        "--researcher-skills-file",
        str(output_base / "data_uc1_m1" / "m1_researcher_skills.csv"),
        "--pseudo-skills-file",
        str(output_base / "data_uc1_m1" / "m1_pseudo_researcher_skills.csv"),
        "--output-dir",
        str(output_base / "data_uc1_m7"),
        "--output-file",
        str(output_base / "teaming_uc1_m7.csv"),
        "--team-size",
        str(team_size),
        "--num-teams",
        str(num_teams),
    ]
    subprocess.run(m7_command, check=True, cwd=PROJECT_ROOT / "code")


def summarize_outputs(output_base: Path, summary_file: Path) -> None:
    rows = []
    for filename in [
        "teaming_uc1_m0.csv",
        "teaming_uc1_m1.csv",
        "teaming_uc1_m2.csv",
        "teaming_uc1_m3.csv",
        "teaming_uc1_m6.csv",
        "teaming_uc1_m7.csv",
    ]:
        path = output_base / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        goodness_means = []
        volumes = []
        for _, row in df.iterrows():
            goodness = [float(value) for value in ast.literal_eval(row["goodness"])]
            teams = ast.literal_eval(row["team"])
            goodness_means.append(sum(goodness) / len(goodness) if goodness else 0.0)
            volumes.append(len(teams))
        rows.append(
            {
                "dataset": filename,
                "G_mean": float(pd.Series(goodness_means).mean()),
                "G_std": float(pd.Series(goodness_means).std(ddof=0)),
                "Volume_mean": float(pd.Series(volumes).mean()),
                "n_researchers": int(df["researcher_name"].nunique()),
                "n_rows": int(len(df)),
            }
        )

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_file, index=False)


def run_fairness(output_base: Path, researchers_file: Path, results_dir: Path) -> None:
    command = [
        str(Path(sys.executable)),
        str(PROJECT_ROOT / "evaluation" / "generate_fairness_tables.py"),
        "--output-dir",
        str(output_base),
        "--researchers-file",
        str(researchers_file),
        "--results-dir",
        str(results_dir),
        "--conditioning-attribute",
        "proposal_link",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT / "evaluation")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    researchers_file = Path(args.researchers_file)
    proposals_file = Path(args.proposals_file)
    results_file = Path(args.results_file)
    boosted_workspace = Path(args.boosted_workspace)
    output_base = Path(args.output_base)
    fairness_results_dir = Path(args.fairness_results_dir)
    comparison_summary_file = Path(args.comparison_summary_file)

    output_base.mkdir(parents=True, exist_ok=True)

    researchers = load_researchers(researchers_file)
    proposals = load_v0_proposals(proposals_file, args.num_proposals)
    researcher_skills = build_researcher_skills(researchers)
    skill_document_frequency = build_skill_document_frequency(researcher_skills)
    all_researcher_skills = build_all_researcher_skills(researcher_skills)
    proposal_skills = build_proposal_skills(
        proposals,
        all_researcher_skills,
        skill_document_frequency,
        len(researchers),
    )

    if len(researchers) != 46:
        print(f"[v0] warning: expected 46 researchers for the paper slice, found {len(researchers)}")
    if len(proposals) != args.num_proposals:
        print(f"[v0] warning: requested {args.num_proposals} proposals, loaded {len(proposals)}")

    if args.skip_bandit_refresh:
        print(f"[v0:M3] reusing existing BoostSRL results from {results_file}")
        existing_results_count = len(read_text_lines(results_file))
        expected_results_count = len(proposals) * len(researchers)
        if existing_results_count != expected_results_count:
            print(
                f"[v0:M3] warning: expected {expected_results_count} full-slice BoostSRL scores "
                f"but found {existing_results_count} in {results_file}"
            )
    else:
        print(f"[v0:M3] regenerating BoostSRL workspace at {boosted_workspace}")
        results_file = prepare_boosted_workspace(
            boosted_workspace,
            proposals,
            researchers,
            proposal_skills,
            researcher_skills,
            args.bandit_trees,
            args.bandit_negative_multiplier,
            args.bandit_train_ratio,
            args.bandit_skill_match_threshold,
            args.seed,
        )
        print(f"[v0:M3] fresh BoostSRL results ready at {results_file}")

    random.seed(args.seed)
    run_m0(output_base, proposals, researcher_skills, proposal_skills, args.num_teams)
    random.seed(args.seed)
    run_m1(
        output_base,
        proposals,
        researcher_skills,
        proposal_skills,
        args.num_teams,
        args.match_threshold,
        skill_document_frequency,
    )
    random.seed(args.seed)
    run_m2(output_base, proposals, researcher_skills, proposal_skills, args.num_teams, args.mapper_threshold)
    random.seed(args.seed)
    run_m3(
        output_base,
        proposals,
        researchers,
        researcher_skills,
        proposal_skills,
        results_file,
        args.num_teams,
        args.m3_match_threshold,
        skill_document_frequency,
        args.team_size,
    )
    run_knapsack_methods(output_base, args.team_size, args.num_teams)
    summarize_outputs(output_base, comparison_summary_file)

    slice_metadata = {
        "dataset": "IITR-Teaming v0 fairness-paper slice",
        "researchers_file": str(researchers_file),
        "proposals_file": str(proposals_file),
        "results_file": str(results_file),
        "boosted_workspace": str(boosted_workspace),
        "proposal_rows_used": int(len(proposals)),
        "unique_title_count": int(proposals["Title"].nunique()),
        "unique_proposal_id_count": int(proposals["proposal_id"].nunique()),
        "anchor_researcher_count": int(researchers["FacultyName"].nunique()),
        "proposal_link_examples": proposals["proposal_link"].head(5).tolist(),
    }
    (output_base / "slice_metadata.json").write_text(json.dumps(slice_metadata, indent=2), encoding="utf-8")

    if not args.skip_fairness:
        run_fairness(output_base, researchers_file, fairness_results_dir)

    print(f"[v0] wrote method outputs to {output_base}")
    if not args.skip_fairness:
        print(f"[v0] wrote fairness outputs to {fairness_results_dir}")


if __name__ == "__main__":
    main()
