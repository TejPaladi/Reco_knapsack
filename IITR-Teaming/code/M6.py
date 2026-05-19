from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import metrics_scorer as metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "data" / "v1_teaming"
DEFAULT_SKELETON_FILE = OUTPUT_BASE / "teaming_uc1_m3.csv"
DEFAULT_PROPOSAL_SKILLS_FILE = OUTPUT_BASE / "data_uc1_m1" / "m1_proposal_skills.csv"
DEFAULT_PSEUDO_SKILLS_FILE = OUTPUT_BASE / "data_uc1_m1" / "m1_pseudo_researcher_skills.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_BASE / "data_uc1_m6"
DEFAULT_OUTPUT_FILE = OUTPUT_BASE / "teaming_uc1_m6.csv"
DEFAULT_TEAM_SIZE = 5
DEFAULT_NUM_TEAMS = 10
DEFAULT_MAX_EXCLUSION_ORDER = DEFAULT_TEAM_SIZE - 1
DEFAULT_LEXICAL_SHORTLIST = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IITR Teaming M6 as a pure exact knapsack-style baseline: "
            "anchor the query researcher, then solve weighted maximum coverage "
            "under a fixed team-size budget with dynamic programming."
        )
    )
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--proposal-skills-file", default=str(DEFAULT_PROPOSAL_SKILLS_FILE))
    parser.add_argument("--pseudo-skills-file", default=str(DEFAULT_PSEUDO_SKILLS_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--team-size", type=int, default=DEFAULT_TEAM_SIZE)
    parser.add_argument("--num-teams", type=int, default=DEFAULT_NUM_TEAMS)
    parser.add_argument("--max-exclusion-order", type=int, default=DEFAULT_MAX_EXCLUSION_ORDER)
    parser.add_argument("--lexical-shortlist", type=int, default=DEFAULT_LEXICAL_SHORTLIST)
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--enable-fallback-padding", action="store_true")
    return parser.parse_args()


def parse_literal_skills(raw_value: str) -> list[str]:
    value = ast.literal_eval(raw_value)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    raise ValueError(f"Unsupported skills payload: {type(value)!r}")


def load_proposal_pseudo_skills(path: Path) -> dict[str, dict[str, set[str]]]:
    mapping: dict[str, dict[str, set[str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            proposal_link = row.get("nsf_proposal_links_v0") or row.get("nsf_proposal_links_v1")
            if not proposal_link:
                continue
            mapping[proposal_link][row["researcher"]] = set(parse_literal_skills(row["pseudo_skills"]))
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


def build_lexical_shortlist(
    pseudo_skill_map: dict[str, set[str]],
    shortlist_size: int,
) -> list[str]:
    ranked = [name for name, skills in pseudo_skill_map.items() if skills]
    ranked.sort(key=lambda name: (-len(pseudo_skill_map[name]), name))
    if shortlist_size > 0:
        ranked = ranked[:shortlist_size]
    return ranked


def mask_for_skills(skills: set[str], skill_index: dict[str, int]) -> int:
    mask = 0
    for skill in skills:
        bit_index = skill_index.get(skill)
        if bit_index is not None:
            mask |= 1 << bit_index
    return mask


def build_candidate_masks(
    required_skills: list[str],
    pseudo_skill_map: dict[str, set[str]],
    lexical_candidates: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    skill_index = {skill: idx for idx, skill in enumerate(required_skills)}
    candidate_masks: dict[str, int] = {}
    for researcher_name in lexical_candidates:
        skills = pseudo_skill_map.get(researcher_name, set())
        mask = mask_for_skills(skills, skill_index)
        if mask:
            candidate_masks[researcher_name] = mask
    return skill_index, candidate_masks


def exact_knapsack_team_with_anchor(
    anchor: str,
    skill_index: dict[str, int],
    candidate_masks: dict[str, int],
    researcher_skills: dict[str, set[str]],
    team_size: int,
    banned_members: set[str] | None = None,
) -> list[str]:
    if team_size < 1:
        raise ValueError("team_size must be at least 1")

    banned_members = banned_members or set()
    anchor_mask = mask_for_skills(researcher_skills.get(anchor, set()), skill_index)
    extra_capacity = max(0, team_size - 1)
    items = sorted(
        (name, mask)
        for name, mask in candidate_masks.items()
        if name != anchor and name not in banned_members
    )

    # dp[k][mask] = (redundancy_hits, extra_members_tuple)
    dp: list[dict[int, tuple[int, tuple[str, ...]]]] = [{anchor_mask: (0, tuple())}]
    dp.extend({} for _ in range(extra_capacity))

    for researcher_name, researcher_mask in items:
        for used in range(extra_capacity - 1, -1, -1):
            snapshot = list(dp[used].items())
            for current_mask, (redundancy_hits, chosen_members) in snapshot:
                new_mask = current_mask | researcher_mask
                if new_mask == current_mask:
                    continue

                new_redundancy = redundancy_hits + (current_mask & researcher_mask).bit_count()
                new_members = chosen_members + (researcher_name,)
                existing = dp[used + 1].get(new_mask)

                if existing is None or (new_redundancy, len(new_members), tuple(sorted(new_members))) < (
                    existing[0],
                    len(existing[1]),
                    tuple(sorted(existing[1])),
                ):
                    dp[used + 1][new_mask] = (new_redundancy, new_members)

    best_members: tuple[str, ...] = tuple()
    best_key: tuple[int, int, int, tuple[str, ...]] | None = None

    for used in range(extra_capacity + 1):
        for covered_mask, (redundancy_hits, chosen_members) in dp[used].items():
            candidate_key = (
                covered_mask.bit_count(),
                -redundancy_hits,
                -used,
                tuple(sorted(chosen_members)),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_members = chosen_members

    return [anchor, *best_members]


def team_signature(anchor: str, team: list[str]) -> tuple[str, ...]:
    return (anchor, *tuple(sorted(member for member in team if member != anchor)))


def generate_exact_knapsack_variants(
    anchor: str,
    skill_index: dict[str, int],
    candidate_masks: dict[str, int],
    researcher_skills: dict[str, set[str]],
    team_size: int,
    num_teams: int,
    max_exclusion_order: int,
    fallback_pool: list[str] | None = None,
    rng: random.Random | None = None,
) -> list[list[str]]:
    if num_teams < 1:
        return []

    frontier: deque[frozenset[str]] = deque([frozenset()])
    queued_subproblems = {frozenset()}
    explored_subproblems: set[frozenset[str]] = set()
    seen_teams: set[tuple[str, ...]] = set()
    variants: list[list[str]] = []

    # Search a broader exact frontier by recursively banning members from the
    # currently optimal team of each solved subproblem.
    max_exclusion_order = max(0, min(max_exclusion_order, team_size - 1))
    max_subproblems = max(num_teams * 8, 32)

    while frontier and len(variants) < num_teams and len(explored_subproblems) < max_subproblems:
        banned_members = frontier.popleft()
        explored_subproblems.add(banned_members)

        team = exact_knapsack_team_with_anchor(
            anchor=anchor,
            skill_index=skill_index,
            candidate_masks=candidate_masks,
            researcher_skills=researcher_skills,
            team_size=team_size,
            banned_members=set(banned_members),
        )
        signature = team_signature(anchor, team)
        if signature not in seen_teams:
            seen_teams.add(signature)
            variants.append(team)

        if len(banned_members) >= max_exclusion_order:
            continue

        for member in sorted(member for member in team if member != anchor):
            next_banned = frozenset((*banned_members, member))
            if len(next_banned) > max_exclusion_order:
                continue
            if next_banned in explored_subproblems or next_banned in queued_subproblems:
                continue
            frontier.append(next_banned)
            queued_subproblems.add(next_banned)

    # Fallback: when all researchers share identical skill coverage with the anchor
    # (common for generic IITR proposal vocabulary), the exact frontier produces
    # only the solo-anchor team.  Supplement with random samples from the lexical
    # pool so that M6 degrades gracefully to lexical-random rather than returning
    # a single-member team for every proposal-anchor pair.
    if len(variants) < num_teams and fallback_pool:
        _rng = rng or random.Random()
        others = [c for c in fallback_pool if c != anchor]
        seen_sigs = {team_signature(anchor, t) for t in variants}
        extra_size = min(team_size - 1, len(others))
        attempts = 0
        max_attempts = num_teams * 20
        while len(variants) < num_teams and extra_size > 0 and attempts < max_attempts:
            attempts += 1
            sample = _rng.sample(others, extra_size)
            candidate = [anchor] + sample
            sig = team_signature(anchor, candidate)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                variants.append(candidate)

    return variants


def apply_ultra_metric(
    proposal_skills: list[str],
    team: list[str],
    all_researchers_skills: dict[str, set[str]],
) -> float:
    if not proposal_skills or not team:
        return 0.0

    scorer = metrics.MetricScorer()
    scorer.demand = list(proposal_skills)
    scorer.team = list(team)
    for researcher_name in scorer.team:
        scorer.researchers[researcher_name] = all_researchers_skills.get(researcher_name, set())
    scorer.set_new_weights([-1, -1, 1, 1])
    scorer.run_metrics()
    return float(scorer.goodness)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    skeleton_file = Path(args.skeleton_file)
    proposal_skills_file = Path(args.proposal_skills_file)
    pseudo_skills_file = Path(args.pseudo_skills_file)
    output_dir = Path(args.output_dir)
    output_file = Path(args.output_file)
    rng = random.Random(args.seed)

    proposal_skills = load_proposal_skills(proposal_skills_file)
    proposal_pseudo_skills = load_proposal_pseudo_skills(pseudo_skills_file)
    skeleton_rows = load_skeleton_rows(skeleton_file)

    rows_by_proposal: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in skeleton_rows:
        rows_by_proposal[row["proposal_link"]].append(row)

    final_rows: list[dict[str, object]] = []
    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []

    total_proposals = len(rows_by_proposal)
    for proposal_index, (proposal_link, proposal_rows) in enumerate(rows_by_proposal.items(), start=1):
        required_skills = proposal_skills.get(proposal_link, [])
        pseudo_skill_map = proposal_pseudo_skills.get(proposal_link, {})
        lexical_candidates = build_lexical_shortlist(pseudo_skill_map, args.lexical_shortlist)
        skill_index, candidate_masks = build_candidate_masks(required_skills, pseudo_skill_map, lexical_candidates)

        for row in proposal_rows:
            anchor = row["researcher_name"]
            teams = generate_exact_knapsack_variants(
                anchor=anchor,
                skill_index=skill_index,
                candidate_masks=candidate_masks,
                researcher_skills=pseudo_skill_map,
                team_size=args.team_size,
                num_teams=args.num_teams,
                max_exclusion_order=args.max_exclusion_order,
                fallback_pool=lexical_candidates if args.enable_fallback_padding else None,
                rng=rng,
            )
            scored_variants = []
            for team in teams:
                goodness = round(apply_ultra_metric(required_skills, team, pseudo_skill_map), args.round_digits)
                scored_variants.append((goodness, team))

            scored_variants.sort(key=lambda item: (item[0], tuple(item[1])), reverse=True)
            sorted_goodness = [score for score, _ in scored_variants]
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
            for goodness, team in scored_variants:
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
            print(f"[M6] solved {proposal_index}/{total_proposals} proposals")

    write_rows(
        output_dir / "m6_teaming.csv",
        ["nsf_proposal_links_v0", "researcher_name", "team"],
        teaming_rows,
    )
    write_rows(
        output_dir / "m6_goodness_scores.csv",
        ["nsf_proposal_links_v0", "researcher_name", "goodness"],
        goodness_rows,
    )
    write_rows(
        output_file,
        ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"],
        final_rows,
    )

    metadata = {
        "method": "M6",
        "description": (
            "Exact anchor-fixed knapsack baseline for IITR Teaming on top of the M1 lexical matching pipeline. "
            "Proposal-specific pseudo skills from string matching define the candidate coverage map, "
            "the top lexical shortlist is filtered first, the anchor researcher is mandatory, and the "
            "solver maximizes distinct matched proposal-skill coverage under the team-size budget while "
            "expanding an exact frontier of recursively excluded subproblems to surface more diverse exact variants. "
            "Optional lexical fallback padding can be enabled explicitly for stress-testing noisy IITR inputs, "
            "but it is disabled by default so the exported volume reflects the exact frontier rather than padded teams."
        ),
        "skeleton_file": str(skeleton_file),
        "proposal_skills_file": str(proposal_skills_file),
        "pseudo_skills_file": str(pseudo_skills_file),
        "team_size": int(args.team_size),
        "num_teams_requested": int(args.num_teams),
        "max_exclusion_order": int(args.max_exclusion_order),
        "lexical_shortlist": int(args.lexical_shortlist),
        "seed": int(args.seed),
        "fallback_padding_enabled": bool(args.enable_fallback_padding),
        "rows_written": len(final_rows),
        "proposal_count": len(rows_by_proposal),
        "anchor_researcher_count": len({row["researcher_name"] for row in skeleton_rows}),
        "round_digits": int(args.round_digits),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m6_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[M6] wrote final output to {output_file}")


if __name__ == "__main__":
    main()
