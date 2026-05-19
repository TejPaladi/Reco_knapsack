from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import pandas as pd

import M1
import M2
import M3
import nlp_techniques
import run_v0_teaming_pipeline as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESEARCHERS_FILE = PROJECT_ROOT / "data" / "v0_data" / "researchers.csv"
DEFAULT_PROPOSALS_FILE = PROJECT_ROOT / "data" / "v0_data" / "archive_proposals.csv"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "v0_teaming_paper_style"
DEFAULT_COMPARE_RESULTS_FILE = PROJECT_ROOT / "code" / "compare_outputs" / "comparison_summary_v0_paper_style.csv"
DEFAULT_BOOSTED_WORKSPACE = PROJECT_ROOT / "code" / "boosted_results" / "v0_paper_style_100x46"
DEFAULT_NUM_PROPOSALS = 100
DEFAULT_NUM_TEAMS = 10
DEFAULT_TEAM_SIZE = 5
DEFAULT_RANDOM_SEED = 12345
DEFAULT_M1_MATCH_THRESHOLD = 0.5
DEFAULT_M2_MAPPER_THRESHOLD = 0.3
DEFAULT_M3_MATCH_THRESHOLD = 0.3
DEFAULT_BANDIT_TREES = 20
DEFAULT_BANDIT_NEGATIVE_MULTIPLIER = 1.0
DEFAULT_BANDIT_TRAIN_RATIO = 0.8
DEFAULT_BANDIT_SKILL_MATCH_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the older AI Magazine-style IITR IN-1 workflow on the "
            "100 proposal x 46 researcher slice. This keeps the current cleaned "
            "canonical pipeline separate and writes paper-style outputs to a "
            "different folder."
        )
    )
    parser.add_argument("--researchers-file", default=str(DEFAULT_RESEARCHERS_FILE))
    parser.add_argument("--proposals-file", default=str(DEFAULT_PROPOSALS_FILE))
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--comparison-summary-file", default=str(DEFAULT_COMPARE_RESULTS_FILE))
    parser.add_argument("--boosted-workspace", default=str(DEFAULT_BOOSTED_WORKSPACE))
    parser.add_argument("--num-proposals", type=int, default=DEFAULT_NUM_PROPOSALS)
    parser.add_argument("--num-teams", type=int, default=DEFAULT_NUM_TEAMS)
    parser.add_argument("--team-size", type=int, default=DEFAULT_TEAM_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--m1-threshold", type=float, default=DEFAULT_M1_MATCH_THRESHOLD)
    parser.add_argument("--m2-mapper-threshold", type=float, default=DEFAULT_M2_MAPPER_THRESHOLD)
    parser.add_argument("--m3-threshold", type=float, default=DEFAULT_M3_MATCH_THRESHOLD)
    parser.add_argument("--bandit-trees", type=int, default=DEFAULT_BANDIT_TREES)
    parser.add_argument("--bandit-negative-multiplier", type=float, default=DEFAULT_BANDIT_NEGATIVE_MULTIPLIER)
    parser.add_argument("--bandit-train-ratio", type=float, default=DEFAULT_BANDIT_TRAIN_RATIO)
    parser.add_argument("--bandit-skill-match-threshold", type=float, default=DEFAULT_BANDIT_SKILL_MATCH_THRESHOLD)
    parser.add_argument("--skip-bandit-refresh", action="store_true")
    return parser.parse_args()


def build_legacy_researcher_skills(researchers: pd.DataFrame) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for _, row in researchers.iterrows():
        researcher = row["FacultyName"]
        interests_text = str(row.get("Research Interests", "") or "")

        processed_interests: list[str] = []
        for fragment in interests_text.split("|"):
            fragment = fragment.strip()
            if not fragment:
                continue
            for comma_piece in fragment.split(","):
                cleaned = nlp_techniques.preprocess(comma_piece)
                if cleaned:
                    processed_interests.append(cleaned)

        ngrams = []
        for interest in processed_interests:
            ngrams.extend(ngram for ngram in nlp_techniques.generate_N_grams(interest, ngram=2) if ngram)

        merged = {value for value in processed_interests + ngrams if value}
        mapping[researcher] = merged or {"general"}

    return mapping


def build_legacy_proposal_skills(
    proposals: pd.DataFrame,
    all_researcher_skills: list[str],
) -> dict[str, set[str]]:
    allowed_skills = set(all_researcher_skills)
    proposal_skills: dict[str, set[str]] = {}

    for _, row in proposals.iterrows():
        title = row["Title"] if isinstance(row["Title"], str) else "general"
        synopsis = row.get("Attachment Text", "")
        synopsis = synopsis if isinstance(synopsis, str) else ""

        title = nlp_techniques.preprocess(title)
        synopsis = nlp_techniques.preprocess(synopsis)

        keywords = title.split(" ") + synopsis.split(" ")
        title_ngrams = nlp_techniques.generate_N_grams(title, ngram=2)
        synopsis_ngrams = nlp_techniques.generate_N_grams(synopsis, ngram=2)
        all_keywords = {keyword for keyword in (keywords + title_ngrams + synopsis_ngrams) if keyword}
        filtered = {keyword for keyword in all_keywords if keyword in allowed_skills}

        proposal_skills[row["proposal_link"]] = filtered or {"general"}

    return proposal_skills


def run_m1_paper_style(
    output_base: Path,
    proposals: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    num_teams: int,
    matching_threshold: float,
) -> None:
    method_dir = output_base / "data_uc1_m1"
    all_skill_rows = [{"all_skills": skill} for skill in base.build_all_researcher_skills(researcher_skills)]
    researcher_skill_rows = [{"researcher_name": name, "skills": sorted(skills)} for name, skills in researcher_skills.items()]
    proposal_skill_rows = []
    pseudo_skill_rows: list[dict[str, object]] = []
    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        proposal_link = proposal_row.proposal_link
        required_skills = proposal_skills[proposal_link]
        ranking, pseudo_skill_map = M1.string_matching_ranking(
            researcher_skills,
            required_skills,
            {},
            matching_threshold=matching_threshold,
        )
        proposal_skill_rows.append({"nsf_proposal_links_v0": proposal_link, "skills": sorted(required_skills)})
        for researcher_name, pseudo_skills in pseudo_skill_map.items():
            pseudo_skill_rows.append(
                {
                    "nsf_proposal_links_v0": proposal_link,
                    "researcher": researcher_name,
                    "pseudo_skills": sorted(set(pseudo_skills)),
                }
            )

        pseudo_skill_map_for_metric = {name: set(skills) for name, skills in pseudo_skill_map.items()}
        for target_researcher in researcher_skills:
            teams = M1.create_teams_for_each_person(ranking, target_researcher, num_teams)
            sorted_teams, sorted_scores = base.score_and_sort_variants(
                teams,
                required_skills,
                pseudo_skill_map_for_metric,
                M1.apply_ultra_metric,
            )
            method_teaming_rows, method_goodness_rows = base.flatten_team_rows(
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
                    "skills": sorted(required_skills),
                    "researcher_name": target_researcher,
                    "team": sorted_teams,
                    "goodness": sorted_scores,
                }
            )

        if proposal_index % 25 == 0 or proposal_index == len(proposals):
            print(f"[paper:M1] processed {proposal_index}/{len(proposals)} proposals")

    base.write_rows(method_dir / "m1_all_researcher_skills.csv", all_skill_rows, ["all_skills"])
    base.write_rows(method_dir / "m1_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    base.write_rows(method_dir / "m1_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    base.write_rows(method_dir / "m1_pseudo_researcher_skills.csv", pseudo_skill_rows, ["nsf_proposal_links_v0", "researcher", "pseudo_skills"])
    base.write_rows(method_dir / "m1_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher_name", "team"])
    base.write_rows(method_dir / "m1_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    base.write_rows(output_base / "teaming_uc1_m1.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])


def run_m2_paper_style(
    output_base: Path,
    proposals: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    num_teams: int,
    mapper_threshold: float,
) -> None:
    method_dir = output_base / "data_uc1_m2"
    all_skill_rows = [{"all_skills": skill} for skill in base.build_all_researcher_skills(researcher_skills)]
    researcher_skill_rows = [{"researcher_name": name, "skills": sorted(skills)} for name, skills in researcher_skills.items()]
    proposal_skill_rows = [{"nsf_proposal_links_v0": row.proposal_link, "skills": sorted(proposal_skills[row.proposal_link])} for row in proposals.itertuples(index=False)]
    mapper_cache: dict[str, list[object]] = {}

    proposal_mapper_categories: dict[str, set[str]] = {}
    researcher_mapper_categories: dict[str, set[str]] = {}
    proposal_mapper_rows: list[dict[str, object]] = []
    researcher_mapper_rows: list[dict[str, object]] = []

    for proposal_row in proposals.itertuples(index=False):
        proposal_link = proposal_row.proposal_link
        codes: set[str] = set()
        for skill in sorted(proposal_skills[proposal_link]):
            payload = base.mapper_lookup(skill, mapper_threshold, mapper_cache)
            codes.update(str(code) for code in payload[0])
        proposal_mapper_categories[proposal_link] = codes
        proposal_mapper_rows.append({"proposal": proposal_link, "skill_categories": sorted(codes)})

    for researcher_name, skills in researcher_skills.items():
        codes: set[str] = set()
        for skill in sorted(skills):
            payload = base.mapper_lookup(skill, mapper_threshold, mapper_cache)
            codes.update(str(code) for code in payload[0])
        researcher_mapper_categories[researcher_name] = codes
        researcher_mapper_rows.append({"researcher": researcher_name, "skill_categories": sorted(codes)})

    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        proposal_link = proposal_row.proposal_link
        candidate_researchers = [
            researcher
            for researcher, categories in researcher_mapper_categories.items()
            if categories & proposal_mapper_categories[proposal_link]
        ]
        metric_skill_map = {name: sorted(values) for name, values in researcher_mapper_categories.items()}
        for target_researcher in researcher_skills:
            teams = M2.create_teams_for_each_person(candidate_researchers, target_researcher, num_teams)
            sorted_teams, sorted_scores = base.score_and_sort_variants(
                teams,
                sorted(proposal_mapper_categories[proposal_link]),
                metric_skill_map,
                M2.apply_ultra_metric,
            )
            method_teaming_rows, method_goodness_rows = base.flatten_team_rows(
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

        if proposal_index % 25 == 0 or proposal_index == len(proposals):
            print(f"[paper:M2] processed {proposal_index}/{len(proposals)} proposals")

    base.write_rows(method_dir / "m2_all_researcher_skills.csv", all_skill_rows, ["all_skills"])
    base.write_rows(method_dir / "m2_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    base.write_rows(method_dir / "m2_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    base.write_rows(method_dir / "m2_proposal_mapper_categories.csv", proposal_mapper_rows, ["proposal", "skill_categories"])
    base.write_rows(method_dir / "m2_researcher_mapper_categories.csv", researcher_mapper_rows, ["researcher", "skill_categories"])
    base.write_rows(method_dir / "m2_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher_name", "team"])
    base.write_rows(method_dir / "m2_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    base.write_rows(output_base / "teaming_uc1_m2.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])


def run_m3_paper_style(
    output_base: Path,
    proposals: pd.DataFrame,
    researchers: pd.DataFrame,
    researcher_skills: dict[str, set[str]],
    proposal_skills: dict[str, set[str]],
    heldout_results_file: Path,
    num_teams: int,
    matching_threshold: float,
) -> None:
    method_dir = output_base / "data_uc1_m3"
    pos_teams, neg_teams = base.parse_bandit_results(heldout_results_file)

    researcher_skill_rows = [{"researcher_name": name, "skills": sorted(skills)} for name, skills in researcher_skills.items()]
    all_skill_rows = [{"all_skills": skill} for skill in base.build_all_researcher_skills(researcher_skills)]
    proposal_skill_rows = [{"nsf_proposal_links_v0": proposal_link, "skills": sorted(skills)} for proposal_link, skills in proposal_skills.items()]
    pseudo_skill_rows: list[dict[str, object]] = []
    teaming_rows: list[dict[str, object]] = []
    goodness_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []

    id_to_name: dict[str, str] = {}
    for researcher_name in researchers["FacultyName"]:
        for variant in base.build_bandit_name_variants(researcher_name):
            id_to_name[variant] = researcher_name

    for proposal_index, proposal_row in enumerate(proposals.itertuples(index=False), start=1):
        proposal_link = proposal_row.proposal_link
        proposal_id = proposal_row.proposal_id
        required_skills = proposal_skills[proposal_link]
        ranking, pseudo_skill_map = M3.string_matching_ranking(
            researcher_skills,
            required_skills,
            {},
            matching_threshold=matching_threshold,
        )
        positive_scores = pos_teams.get(proposal_id, {})
        negative_scores = neg_teams.get(proposal_id, {})
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

        pseudo_skill_map_for_metric = {name: set(skills) for name, skills in pseudo_skill_map.items()}
        for researcher_name, pseudo_skills in pseudo_skill_map.items():
            pseudo_skill_rows.append(
                {
                    "nsf_proposal_links_v0": proposal_link,
                    "researcher": researcher_name,
                    "pseudo_skills": sorted(set(pseudo_skills)),
                }
            )

        scored_pos_members: dict[str, float] = {}
        if positive_scores:
            for researcher_id, score in positive_scores.items():
                if researcher_id in id_to_name:
                    researcher_name = id_to_name[researcher_id]
                    scored_pos_members[researcher_name] = max(score, scored_pos_members.get(researcher_name, float("-inf")))

        for target_researcher in researchers["FacultyName"]:
            lexical_teams = M3.create_teams_for_each_person(ranking, target_researcher, num_teams)
            chosen_teams = lexical_teams[:1] if lexical_teams else [[target_researcher]]

            if scored_pos_members:
                ranked_members = [
                    name
                    for name, _ in sorted(
                        scored_pos_members.items(),
                        key=lambda item: (item[1], item[0]),
                        reverse=True,
                    )
                    if name != target_researcher
                ]
                greedy_team = [target_researcher, *ranked_members[:4]]
                chosen_teams = [greedy_team]

            sorted_teams, sorted_scores = base.score_and_sort_variants(
                chosen_teams,
                required_skills,
                pseudo_skill_map_for_metric,
                M3.apply_ultra_metric,
            )
            method_teaming_rows, method_goodness_rows = base.flatten_team_rows(
                proposal_link,
                target_researcher,
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
                    "skills": sorted(required_skills),
                    "researcher_name": target_researcher,
                    "team": sorted_teams,
                    "goodness": sorted_scores,
                }
            )

        if proposal_index % 25 == 0 or proposal_index == len(proposals):
            print(f"[paper:M3] processed {proposal_index}/{len(proposals)} proposals")

    base.write_rows(method_dir / "m3_researcher_skills.csv", researcher_skill_rows, ["researcher_name", "skills"])
    base.write_rows(method_dir / "m3_all_researcher_skills.csv", all_skill_rows, ["all_skills"])
    base.write_rows(method_dir / "m3_proposal_skills.csv", proposal_skill_rows, ["nsf_proposal_links_v0", "skills"])
    base.write_rows(method_dir / "m3_pseudo_researcher_skills.csv", pseudo_skill_rows, ["nsf_proposal_links_v0", "researcher", "pseudo_skills"])
    base.write_rows(method_dir / "m3_teaming.csv", teaming_rows, ["nsf_proposal_links_v0", "researcher", "team"])
    base.write_rows(method_dir / "m3_goodness_scores.csv", goodness_rows, ["nsf_proposal_links_v0", "researcher_name", "goodness"])
    base.write_rows(method_dir / "m3_bandit_alignment.csv", alignment_rows, ["proposal_link", "proposal_id", "title", "bandit_positive_count", "bandit_negative_count", "bandit_matched"])
    base.write_rows(output_base / "teaming_uc1_m3.csv", final_rows, ["proposal_link", "title", "skills", "researcher_name", "team", "goodness"])

    metadata = {
        "method": "M3-paper-style",
        "description": (
            "Closest local reproduction of the older AI Magazine IITR IN-1 notebook flow. "
            "It uses legacy lexical extraction, M3 lexical threshold 0.3, and builds teams "
            "from held-out BoostSRL proposal-member predictions rather than full-slice inference."
        ),
        "heldout_results_file": str(heldout_results_file),
        "proposal_count": int(len(proposals)),
        "anchor_researcher_count": int(researchers["FacultyName"].nunique()),
        "bandit_scored_pairs": int(sum(len(values) for values in pos_teams.values()) + sum(len(values) for values in neg_teams.values())),
    }
    (method_dir / "m3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    researchers_file = Path(args.researchers_file)
    proposals_file = Path(args.proposals_file)
    output_base = Path(args.output_base)
    summary_file = Path(args.comparison_summary_file)
    boosted_workspace = Path(args.boosted_workspace)

    output_base.mkdir(parents=True, exist_ok=True)

    researchers = base.load_researchers(researchers_file)
    proposals = base.load_v0_proposals(proposals_file, args.num_proposals)
    researcher_skills = build_legacy_researcher_skills(researchers)
    all_researcher_skills = base.build_all_researcher_skills(researcher_skills)
    proposal_skills = build_legacy_proposal_skills(proposals, all_researcher_skills)

    if args.skip_bandit_refresh:
        heldout_results_file = boosted_workspace / "heldout_eval" / "results_team.db"
        print(f"[paper:M3] reusing held-out BoostSRL results from {heldout_results_file}")
    else:
        print(f"[paper:M3] regenerating BoostSRL workspace at {boosted_workspace}")
        base.prepare_boosted_workspace(
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
        heldout_results_file = boosted_workspace / "heldout_eval" / "results_team.db"
        print(f"[paper:M3] held-out BoostSRL results ready at {heldout_results_file}")

    random.seed(args.seed)
    base.run_m0(output_base, proposals, researcher_skills, proposal_skills, args.num_teams)
    random.seed(args.seed)
    run_m1_paper_style(
        output_base,
        proposals,
        researcher_skills,
        proposal_skills,
        args.num_teams,
        args.m1_threshold,
    )
    random.seed(args.seed)
    run_m2_paper_style(
        output_base,
        proposals,
        researcher_skills,
        proposal_skills,
        args.num_teams,
        args.m2_mapper_threshold,
    )
    random.seed(args.seed)
    run_m3_paper_style(
        output_base,
        proposals,
        researchers,
        researcher_skills,
        proposal_skills,
        heldout_results_file,
        args.num_teams,
        args.m3_threshold,
    )
    base.run_knapsack_methods(output_base, args.team_size, args.num_teams)
    base.summarize_outputs(output_base, summary_file)

    metadata = {
        "dataset": "IITR-Teaming v0 paper-style reproduction",
        "researchers_file": str(researchers_file),
        "proposals_file": str(proposals_file),
        "output_base": str(output_base),
        "comparison_summary_file": str(summary_file),
        "boosted_workspace": str(boosted_workspace),
        "heldout_results_file": str(heldout_results_file),
        "proposal_rows_used": int(len(proposals)),
        "anchor_researcher_count": int(researchers["FacultyName"].nunique()),
        "team_size": int(args.team_size),
        "m1_threshold": float(args.m1_threshold),
        "m2_mapper_threshold": float(args.m2_mapper_threshold),
        "m3_threshold": float(args.m3_threshold),
    }
    (output_base / "slice_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[paper] wrote reproduction outputs to {output_base}")
    print(f"[paper] wrote comparison summary to {summary_file}")


if __name__ == "__main__":
    main()
