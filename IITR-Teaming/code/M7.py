from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import metrics_scorer as metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "data" / "v1_teaming"
DEFAULT_SKELETON_FILE = OUTPUT_BASE / "teaming_uc1_m3.csv"
DEFAULT_PROPOSAL_SKILLS_FILE = OUTPUT_BASE / "data_uc1_m1" / "m1_proposal_skills.csv"
DEFAULT_RESEARCHER_SKILLS_FILE = OUTPUT_BASE / "data_uc1_m1" / "m1_researcher_skills.csv"
DEFAULT_PSEUDO_SKILLS_FILE = OUTPUT_BASE / "data_uc1_m1" / "m1_pseudo_researcher_skills.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_BASE / "data_uc1_m7"
DEFAULT_OUTPUT_FILE = OUTPUT_BASE / "teaming_uc1_m7.csv"
DEFAULT_TEAM_SIZE = 5
DEFAULT_NUM_TEAMS = 10
DEFAULT_ALPHA = 0.5
DEFAULT_COST_MULTIPLIER = 0.14
DEFAULT_MAX_CANDIDATES = 18
DEFAULT_LEXICAL_SHORTLIST = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IITR Teaming M7 as a marginal-utility extension of M6. "
            "It keeps the anchor researcher fixed, scores candidate teams with "
            "weighted diminishing-return utility, and exports the top exact "
            "teams found on a proposal-specific shortlist."
        )
    )
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--proposal-skills-file", default=str(DEFAULT_PROPOSAL_SKILLS_FILE))
    parser.add_argument("--researcher-skills-file", default=str(DEFAULT_RESEARCHER_SKILLS_FILE))
    parser.add_argument("--pseudo-skills-file", default=str(DEFAULT_PSEUDO_SKILLS_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--team-size", type=int, default=DEFAULT_TEAM_SIZE)
    parser.add_argument("--num-teams", type=int, default=DEFAULT_NUM_TEAMS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--cost-multiplier", type=float, default=DEFAULT_COST_MULTIPLIER)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--lexical-shortlist", type=int, default=DEFAULT_LEXICAL_SHORTLIST)
    parser.add_argument("--round-digits", type=int, default=4)
    return parser.parse_args()


def parse_literal_skills(raw_value: str) -> list[str]:
    value = ast.literal_eval(raw_value)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    raise ValueError(f"Unsupported skills payload: {type(value)!r}")


def load_researcher_skills(path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[row["researcher_name"]] = set(parse_literal_skills(row["skills"]))
    return mapping


def load_proposal_skills(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            proposal_link = row.get("nsf_proposal_links_v0") or row.get("nsf_proposal_links_v1")
            if not proposal_link:
                continue
            mapping[proposal_link] = parse_literal_skills(row["skills"])
    return mapping


def load_skeleton_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "proposal_link": row["proposal_link"],
                "title": row["title"],
                "researcher_name": row["researcher_name"],
            }
            for row in csv.DictReader(handle)
        ]


def load_proposal_pseudo_skills(path: Path) -> dict[str, dict[str, set[str]]]:
    mapping: dict[str, dict[str, set[str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            proposal_link = row.get("nsf_proposal_links_v0") or row.get("nsf_proposal_links_v1")
            if not proposal_link:
                continue
            mapping[proposal_link][row["researcher"]] = set(parse_literal_skills(row["pseudo_skills"]))
    return mapping


def build_lexical_shortlist(
    pseudo_skill_map: dict[str, set[str]],
    shortlist_size: int,
) -> list[str]:
    ranked = [name for name, skills in pseudo_skill_map.items() if skills]
    ranked.sort(key=lambda name: (-len(pseudo_skill_map[name]), name))
    if shortlist_size > 0:
        ranked = ranked[:shortlist_size]
    return ranked


def build_idf_weights(researcher_skills: dict[str, set[str]]) -> dict[str, float]:
    total = len(researcher_skills) or 1
    skill_counts: dict[str, int] = {}
    for skills in researcher_skills.values():
        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    return {
        skill: math.log(total / (count + 1)) + 1
        for skill, count in skill_counts.items()
    }


def apply_ultra_metric(
    proposal_skills: list[str],
    team: list[str],
    all_researchers_skills: dict[str, set[str]],
) -> float:
    return evaluate_team_profile(proposal_skills, team, all_researchers_skills)["goodness"]


def evaluate_team_profile(
    proposal_skills: list[str],
    team: list[str],
    all_researchers_skills: dict[str, set[str]],
) -> dict[str, object]:
    if not proposal_skills or not team:
        return {
            "goodness": 0.0,
            "redundancy": 0.0,
            "setsize": 0.0,
            "coverage": 0.0,
            "krobust": 0,
            "covered_skills": tuple(),
        }

    scorer = metrics.MetricScorer()
    scorer.demand = list(proposal_skills)
    scorer.team = list(team)
    for researcher_name in scorer.team:
        scorer.researchers[researcher_name] = all_researchers_skills.get(researcher_name, set())
    scorer.set_new_weights([-1, -1, 1, 1])
    scorer.run_metrics()
    covered_skills = sorted(
        {
            skill
            for researcher_name in scorer.team
            for skill in all_researchers_skills.get(researcher_name, set())
            if skill in proposal_skills
        }
    )
    return {
        "goodness": float(scorer.goodness),
        "redundancy": float(scorer.redundancy),
        "setsize": float(scorer.setsize),
        "coverage": float(scorer.coverage),
        "krobust": int(scorer.krobust),
        "covered_skills": tuple(covered_skills),
    }


def variant_outcome_signature(profile: dict[str, object], round_digits: int) -> tuple[object, ...]:
    return (
        profile["covered_skills"],
        round(float(profile["redundancy"]), round_digits),
        round(float(profile["setsize"]), round_digits),
        round(float(profile["coverage"]), round_digits),
        int(profile["krobust"]),
        round(float(profile["goodness"]), round_digits),
    )


def collapse_equivalent_variants(
    variants: list[tuple[dict[str, object], list[str]]],
    round_digits: int,
) -> list[tuple[dict[str, object], list[str]]]:
    collapsed: list[tuple[dict[str, object], list[str]]] = []
    seen_signatures: set[tuple[object, ...]] = set()

    for profile, team in variants:
        signature = variant_outcome_signature(profile, round_digits)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        collapsed.append((profile, team))

    return collapsed


def member_set_distance(anchor: str, team_a: list[str], team_b: list[str]) -> float:
    members_a = {member for member in team_a if member != anchor}
    members_b = {member for member in team_b if member != anchor}
    if not members_a and not members_b:
        return 0.0
    union = members_a | members_b
    if not union:
        return 0.0
    return 1.0 - (len(members_a & members_b) / len(union))


def select_representative_variants(
    anchor: str,
    variants: list[tuple[dict[str, object], list[str]]],
    round_digits: int,
    required_skills: list[str],
    team_size: int,
    num_teams: int,
) -> list[tuple[dict[str, object], list[str]]]:
    if not variants:
        return []

    extras_by_variant = [{member for member in team if member != anchor} for _, team in variants]
    union_members = set().union(*extras_by_variant) if extras_by_variant else set()
    core_members = set.intersection(*extras_by_variant) if extras_by_variant else set()
    branching_options = max(0, len(union_members - core_members))
    if branching_options == 0:
        return [variants[0]]

    complexity = max(1, len(required_skills))
    structural_width = math.sqrt(max(1, len(union_members)) * complexity) / 1.3
    target_count = max(1, min(num_teams, len(variants), round(structural_width)))

    ranked_variants = sorted(
        variants,
        key=lambda item: (
            round(float(item[0]["goodness"]), round_digits),
            round(float(item[0]["coverage"]), round_digits),
            -round(float(item[0]["redundancy"]), round_digits),
            tuple(item[1]),
        ),
        reverse=True,
    )

    selected: list[tuple[dict[str, object], list[str]]] = [ranked_variants[0]]
    remaining = ranked_variants[1:]

    while len(selected) < target_count and remaining:
        selected_teams = [team for _, team in selected]
        best_index = 0
        best_key: tuple[float, float, float, tuple[str, ...]] | None = None
        for idx, (profile, team) in enumerate(remaining):
            novelty = min(member_set_distance(anchor, team, existing_team) for existing_team in selected_teams)
            candidate_key = (
                novelty,
                round(float(profile["goodness"]), round_digits),
                round(float(profile["coverage"]), round_digits),
                tuple(team),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_index = idx

        selected.append(remaining.pop(best_index))

    return selected


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_proposal_payload(
    required_skills: list[str],
    pseudo_skill_map: dict[str, set[str]],
    skill_weights: dict[str, float],
    lexical_shortlist: int,
) -> dict[str, object]:
    required = [skill for skill in required_skills if skill]
    skill_index = {skill: idx for idx, skill in enumerate(required)}
    weights_by_idx = [skill_weights.get(skill, 1.0) for skill in required]

    researcher_skill_indices: dict[str, tuple[int, ...]] = {}
    for researcher_name, skills in pseudo_skill_map.items():
        overlap = sorted(skill_index[skill] for skill in skills if skill in skill_index)
        researcher_skill_indices[researcher_name] = tuple(overlap)

    lexical_candidates = build_lexical_shortlist(pseudo_skill_map, lexical_shortlist)

    avg_weight = sum(weights_by_idx) / len(weights_by_idx) if weights_by_idx else 1.0
    return {
        "required_skills": required,
        "weights_by_idx": weights_by_idx,
        "researcher_skill_indices": researcher_skill_indices,
        "lexical_candidates": lexical_candidates,
        "seat_cost": avg_weight,
    }


def score_marginal_utility_team(
    anchor_indices: tuple[int, ...],
    combo: tuple[tuple[str, tuple[int, ...], float], ...],
    weights_by_idx: list[float],
    alpha: float,
    seat_cost: float,
) -> tuple[float, float]:
    counts = {idx: 1 for idx in anchor_indices}
    objective = 0.0

    for _, skill_indices, _ in combo:
        marginal_gain = 0.0
        for idx in skill_indices:
            covered_count = counts.get(idx, 0)
            marginal_gain += weights_by_idx[idx] * (alpha ** covered_count)
            counts[idx] = covered_count + 1
        objective += marginal_gain - seat_cost

    weighted_unique_coverage = sum(weights_by_idx[idx] for idx in counts)
    return objective, weighted_unique_coverage


def generate_marginal_utility_teams(
    anchor: str,
    proposal_payload: dict[str, object],
    alpha: float,
    cost_multiplier: float,
    team_size: int,
    num_teams: int,
    max_candidates: int,
) -> list[list[str]]:
    researcher_skill_indices = proposal_payload["researcher_skill_indices"]
    lexical_candidates = proposal_payload["lexical_candidates"]
    weights_by_idx = proposal_payload["weights_by_idx"]
    anchor_indices = researcher_skill_indices.get(anchor, tuple())
    seat_cost = proposal_payload["seat_cost"] * cost_multiplier
    extra_capacity = max(0, team_size - 1)

    candidates: list[tuple[str, tuple[int, ...], float]] = []
    for candidate_name in lexical_candidates:
        if candidate_name == anchor:
            continue
        skill_indices = researcher_skill_indices.get(candidate_name, tuple())
        if not skill_indices:
            continue
        initial_gain = sum(weights_by_idx[idx] * (alpha ** (1 if idx in anchor_indices else 0)) for idx in skill_indices)
        initial_marginal_utility = initial_gain - seat_cost
        if initial_marginal_utility > 0:
            candidates.append((candidate_name, skill_indices, initial_marginal_utility))

    candidates.sort(key=lambda row: (-row[2], row[0]))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    candidate_rows: list[tuple[float, float, int, tuple[str, ...], list[str]]] = []
    anchor_weighted_coverage = sum(weights_by_idx[idx] for idx in anchor_indices)
    candidate_rows.append((0.0, anchor_weighted_coverage, 1, (anchor,), [anchor]))

    max_size = min(extra_capacity, len(candidates))
    for subset_size in range(1, max_size + 1):
        for combo in combinations(candidates, subset_size):
            objective, weighted_unique_coverage = score_marginal_utility_team(
                anchor_indices=anchor_indices,
                combo=combo,
                weights_by_idx=weights_by_idx,
                alpha=alpha,
                seat_cost=seat_cost,
            )
            team = [anchor, *[candidate_name for candidate_name, _, _ in combo]]
            candidate_rows.append(
                (
                    objective,
                    weighted_unique_coverage,
                    len(team),
                    tuple(sorted(team)),
                    team,
                )
            )

    candidate_rows.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))

    teams: list[list[str]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for _, _, _, signature, team in candidate_rows:
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        teams.append(team)
        if len(teams) >= num_teams:
            break

    return teams


def main() -> None:
    args = parse_args()

    skeleton_file = Path(args.skeleton_file)
    proposal_skills_file = Path(args.proposal_skills_file)
    researcher_skills_file = Path(args.researcher_skills_file)
    pseudo_skills_file = Path(args.pseudo_skills_file)
    output_dir = Path(args.output_dir)
    output_file = Path(args.output_file)

    researcher_skills = load_researcher_skills(researcher_skills_file)
    proposal_skills = load_proposal_skills(proposal_skills_file)
    proposal_pseudo_skills = load_proposal_pseudo_skills(pseudo_skills_file)
    skeleton_rows = load_skeleton_rows(skeleton_file)
    skill_weights = build_idf_weights(researcher_skills)

    rows_by_proposal: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in skeleton_rows:
        rows_by_proposal[row["proposal_link"]].append(row)

    proposal_payloads = {
        proposal_link: prepare_proposal_payload(
            required,
            proposal_pseudo_skills.get(proposal_link, {}),
            skill_weights,
            args.lexical_shortlist,
        )
        for proposal_link, required in proposal_skills.items()
    }

    final_rows: list[dict[str, object]] = []
    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []

    total_proposals = len(rows_by_proposal)
    for proposal_index, (proposal_link, proposal_rows) in enumerate(rows_by_proposal.items(), start=1):
        payload = proposal_payloads.get(
            proposal_link,
            prepare_proposal_payload([], {}, skill_weights, args.lexical_shortlist),
        )
        required_skills = payload["required_skills"]
        pseudo_skill_map = proposal_pseudo_skills.get(proposal_link, {})

        for row in proposal_rows:
            anchor = row["researcher_name"]
            teams = generate_marginal_utility_teams(
                anchor=anchor,
                proposal_payload=payload,
                alpha=args.alpha,
                cost_multiplier=args.cost_multiplier,
                team_size=args.team_size,
                num_teams=args.num_teams,
                max_candidates=args.max_candidates,
            )
            scored_variants = []
            for team in teams:
                profile = evaluate_team_profile(required_skills, team, pseudo_skill_map)
                scored_variants.append((profile, team))

            scored_variants.sort(
                key=lambda item: (round(float(item[0]["goodness"]), args.round_digits), tuple(item[1])),
                reverse=True,
            )
            scored_variants = select_representative_variants(
                anchor=anchor,
                variants=scored_variants,
                round_digits=args.round_digits,
                required_skills=required_skills,
                team_size=args.team_size,
                num_teams=args.num_teams,
            )
            sorted_goodness = [round(float(profile["goodness"]), args.round_digits) for profile, _ in scored_variants]
            sorted_teams = [team for _, team in scored_variants]

            final_rows.append(
                {
                    "proposal_link": proposal_link,
                    "title": row["title"],
                    "skills": required_skills,
                    "researcher_name": anchor,
                    "team": sorted_teams,
                    "goodness": sorted_goodness,
                }
            )
            for profile, team in scored_variants:
                goodness = round(float(profile["goodness"]), args.round_digits)
                teaming_rows.append(
                    {
                        "nsf_proposal_links_v0": proposal_link,
                        "researcher_name": anchor,
                        "team": team,
                    }
                )
                goodness_rows.append(
                    {
                        "nsf_proposal_links_v0": proposal_link,
                        "researcher_name": anchor,
                        "goodness": goodness,
                    }
                )

        if proposal_index % 25 == 0 or proposal_index == total_proposals:
            print(f"[M7] solved {proposal_index}/{total_proposals} proposals")

    write_rows(
        output_dir / "m7_teaming.csv",
        ["nsf_proposal_links_v0", "researcher_name", "team"],
        teaming_rows,
    )
    write_rows(
        output_dir / "m7_goodness_scores.csv",
        ["nsf_proposal_links_v0", "researcher_name", "goodness"],
        goodness_rows,
    )
    write_rows(
        output_file,
        ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"],
        final_rows,
    )

    metadata = {
        "method": "M7",
        "description": (
            "Marginal-utility extension of M6 for IITR Teaming on top of the M1 lexical matching pipeline. "
            "The top string-matched researchers are shortlisted first using proposal-specific pseudo skills, "
            "the anchor researcher is fixed, additional teammates are scored by weighted diminishing-return "
            "utility minus a per-seat cost, and the exported variants are selected automatically as a diverse "
            "representative set whose width is derived from row-level branching structure and proposal complexity."
        ),
        "skeleton_file": str(skeleton_file),
        "proposal_skills_file": str(proposal_skills_file),
        "researcher_skills_file": str(researcher_skills_file),
        "pseudo_skills_file": str(pseudo_skills_file),
        "team_size": int(args.team_size),
        "num_teams_requested": int(args.num_teams),
        "alpha": float(args.alpha),
        "cost_multiplier": float(args.cost_multiplier),
        "max_candidates": int(args.max_candidates),
        "lexical_shortlist": int(args.lexical_shortlist),
        "rows_written": len(final_rows),
        "proposal_count": len(rows_by_proposal),
        "anchor_researcher_count": len({row["researcher_name"] for row in skeleton_rows}),
        "round_digits": int(args.round_digits),
        "auto_selects_diverse_representatives": True,
        "representative_budget_rule": "round(sqrt(unique_optional_members * max(1, required_skill_count)) / 1.3)",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[M7] wrote final output to {output_file}")


if __name__ == "__main__":
    main()
