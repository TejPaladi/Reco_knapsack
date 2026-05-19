#!/usr/bin/env python3
"""Generate M5 LLM Teaming outputs.

This script mirrors the existing Teaming CSV contract, but uses an LLM to
generate candidate teams instead of M0/M1/M2/M3/M6/M7 logic.

Supported domains:
  - teaming:      USC Teaming, using Teaming/data/v1_input_files/
  - iitr-teaming: IITR Teaming, using IITR-Teaming/data/v0_data/

Default output files:
  - Teaming/data/output/teaming_uc1_m5_<model>.csv
  - IITR-Teaming/data/output/teaming_uc1_m5_<model>.csv

The script prints the input rows used and the output rows written after it
finishes. It does not hardcode API keys; set GEMINI_API_KEY or GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

USC_BAD_PROPOSAL_LINKS = {
    "https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=505073",
}

MODEL_ALIASES = {
    "gemini-3-flash-live": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemma-4-31b": "models/gemma-4-31b-it",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "with",
    "research",
    "proposal",
    "program",
    "project",
    "science",
    "scientific",
    "foundation",
    "national",
    "university",
}


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    year: str
    proposal_link: str
    title: str
    prompt_title: str
    summary: str
    skills: list[str]


@dataclass(frozen=True)
class Researcher:
    name: str
    title: str
    description: str
    skills: list[str]


@dataclass(frozen=True)
class DomainBundle:
    domain: str
    researchers_path: Path
    proposals_path: Path
    raw_researcher_rows: int
    raw_proposal_rows: int
    researchers: list[Researcher]
    proposals: list[Proposal]
    output_columns: list[str]
    output_dir: Path


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def clean_text(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = repo_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first_value(row: dict[str, Any], *columns: str) -> str:
    for column in columns:
        value = clean_text(row.get(column))
        if value:
            return value
    return ""


def sanitize_token(value: object) -> str:
    text = clean_text(value).lower().strip(" '{}\"")
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"^_+", "", text)
    return text or "unknown"


def normalize_name(value: object) -> str:
    return " ".join(clean_text(value).replace("\xa0", " ").split()).strip()


def parse_literal_or_split(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parsed: Any | None = None
    if text[0] in "[{(":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, dict):
        values = parsed.values()
    elif isinstance(parsed, (list, tuple, set)):
        values = parsed
    else:
        values = re.split(r"\s*\|\s*|\s*,\s*", text)
    return [clean_text(item) for item in values if clean_text(item)]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+-]*", clean_text(text).lower())
    return [word for word in words if len(word) >= 3 and word not in STOPWORDS]


def extract_skills(*values: Any, max_skills: int = 40) -> list[str]:
    tokens: list[str] = []
    for value in values:
        parts = parse_literal_or_split(value)
        if not parts:
            parts = [clean_text(value)]
        for part in parts:
            tokens.extend(tokenize(part))
            part_tokens = tokenize(part)
            tokens.extend(
                f"{part_tokens[index]} {part_tokens[index + 1]}"
                for index in range(max(0, len(part_tokens) - 1))
            )
    seen: dict[str, None] = {}
    for token in tokens:
        seen.setdefault(token, None)
    return list(seen)[:max_skills] or ["general"]


def model_slug(model_alias: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_alias.lower()).strip("_")


def model_id_for(args: argparse.Namespace) -> str:
    return args.model_id or MODEL_ALIASES[args.model]


def output_path_for(bundle: DomainBundle, args: argparse.Namespace) -> Path:
    if args.output_csv:
        return repo_path(args.output_csv)
    return bundle.output_dir / f"teaming_uc1_m5_{model_slug(args.model)}.csv"


def load_usc_teaming(args: argparse.Namespace) -> DomainBundle:
    researchers_path = repo_path(args.researchers_csv)
    proposals_path = repo_path(args.proposals_csv)
    researcher_rows = read_csv_rows(researchers_path)
    proposal_rows = read_csv_rows(proposals_path)

    researchers: list[Researcher] = []
    for row in researcher_rows:
        name = normalize_name(first_value(row, "names", "name", "FacultyName"))
        if not name:
            continue
        research = first_value(row, "research", "Research Interests")
        researchers.append(
            Researcher(
                name=name,
                title=clean_text(first_value(row, "titles", "designation")),
                description=clean_text(first_value(row, "descriptions", "background"), args.max_field_chars),
                skills=extract_skills(research, max_skills=args.max_skills_per_record),
            )
        )

    proposals_sorted = sorted(
        proposal_rows,
        key=lambda row: first_value(row, "nsf_proposal_links_v1"),
        reverse=True,
    )
    proposals_sorted = [
        row
        for row in proposals_sorted
        if first_value(row, "nsf_proposal_links_v1") not in USC_BAD_PROPOSAL_LINKS
    ]
    if args.limit_proposals > 0:
        proposals_sorted = proposals_sorted[: args.limit_proposals]

    proposals: list[Proposal] = []
    for row in proposals_sorted:
        link = first_value(row, "nsf_proposal_links_v1")
        title = first_value(row, "title", "Title") or "general"
        year = ""
        proposal_id = ""
        link_parts = link.split("/")
        if len(link_parts) > 5:
            year = link_parts[4]
            proposal_id = link_parts[5]
        summary = clean_text(first_value(row, "synopsis", "details"), args.max_field_chars)
        skills = extract_skills(title, summary, max_skills=args.max_skills_per_record)
        display_title = f"{title} ({year})" if year else title
        proposals.append(
            Proposal(
                proposal_id=proposal_id,
                year=year,
                proposal_link=link,
                title=display_title,
                prompt_title=title,
                summary=summary,
                skills=skills,
            )
        )

    if args.limit_researchers > 0:
        researchers = researchers[: args.limit_researchers]

    return DomainBundle(
        domain="teaming",
        researchers_path=researchers_path,
        proposals_path=proposals_path,
        raw_researcher_rows=len(researcher_rows),
        raw_proposal_rows=len(proposal_rows),
        researchers=researchers,
        proposals=proposals,
        output_columns=[
            "proposal_id",
            "year",
            "proposal_link",
            "title",
            "skills",
            "researcher_name",
            "team",
            "goodness",
        ],
        output_dir=REPO_ROOT / "Teaming" / "data" / "output",
    )


def load_iitr_teaming(args: argparse.Namespace) -> DomainBundle:
    researchers_path = repo_path(args.iitr_researchers_csv)
    proposals_path = repo_path(args.iitr_proposals_csv)
    researcher_rows = read_csv_rows(researchers_path)
    proposal_rows = read_csv_rows(proposals_path)

    researchers: list[Researcher] = []
    for row in researcher_rows:
        name = normalize_name(first_value(row, "FacultyName", "names", "name"))
        if not name:
            continue
        interests = first_value(row, "Research Interests", "research")
        researchers.append(
            Researcher(
                name=name,
                title="",
                description="",
                skills=extract_skills(interests, max_skills=args.max_skills_per_record),
            )
        )

    selected_proposals = proposal_rows[: args.iitr_num_proposals]
    if args.limit_proposals > 0:
        selected_proposals = selected_proposals[: args.limit_proposals]

    proposals: list[Proposal] = []
    for index, row in enumerate(selected_proposals, start=1):
        title = first_value(row, "Title", "title") or "general"
        summary = clean_text(first_value(row, "Attachment Text", "synopsis", "details"), args.max_field_chars)
        proposal_link = f"v0_{index:03d}_{sanitize_token(title)}"
        proposals.append(
            Proposal(
                proposal_id=proposal_link,
                year="",
                proposal_link=proposal_link,
                title=title,
                prompt_title=title,
                summary=summary,
                skills=extract_skills(title, summary, max_skills=args.max_skills_per_record),
            )
        )

    if args.limit_researchers > 0:
        researchers = researchers[: args.limit_researchers]

    return DomainBundle(
        domain="iitr-teaming",
        researchers_path=researchers_path,
        proposals_path=proposals_path,
        raw_researcher_rows=len(researcher_rows),
        raw_proposal_rows=len(proposal_rows),
        researchers=researchers,
        proposals=proposals,
        output_columns=[
            "proposal_link",
            "title",
            "skills",
            "researcher_name",
            "team",
            "goodness",
        ],
        output_dir=REPO_ROOT / "IITR-Teaming" / "data" / "output",
    )


def load_domain(args: argparse.Namespace) -> DomainBundle:
    if args.domain == "teaming":
        return load_usc_teaming(args)
    if args.domain == "iitr-teaming":
        return load_iitr_teaming(args)
    raise ValueError(f"Unsupported domain: {args.domain}")


def format_researcher(person: Researcher) -> str:
    parts = [f"name={person.name}", f"skills={', '.join(person.skills)}"]
    if person.title:
        parts.append(f"title={person.title}")
    if person.description:
        parts.append(f"description={person.description}")
    return " | ".join(parts)


def build_prompt(
    proposal: Proposal,
    anchor: Researcher,
    candidates: list[Researcher],
    args: argparse.Namespace,
) -> str:
    candidate_lines = "\n".join(f"- {format_researcher(person)}" for person in candidates)
    return f"""You are generating the M5 LLM baseline for research team recommendation.

Return CSV only. Do not use markdown fences or explanations.
Required header:
team

Task:
Recommend exactly {args.num_teams} candidate teams for the anchor researcher and proposal below.

Rules:
- Use only names from Available Researchers.
- Every team must include the anchor researcher exactly as written.
- Each team should contain 1 to {args.team_size} researchers total, including the anchor.
- Use pipe-separated exact names inside the team field, for example: Name One | Name Two.
- Do not use semicolons inside the team field.
- Do not use commas to separate names; some researcher names already contain commas.
- Prefer teams whose combined skills match the proposal skills and summary.

Anchor Researcher:
{format_researcher(anchor)}

Proposal:
title={proposal.prompt_title}
skills={', '.join(proposal.skills)}
summary={proposal.summary}

Available Researchers:
{candidate_lines}
"""


def api_key_for(args: argparse.Namespace) -> str:
    for env_name in args.api_key_env:
        value = os.environ.get(env_name)
        if value:
            return value
    raise RuntimeError(f"Missing API key. Set one of: {', '.join(args.api_key_env)}")


def call_model(prompt: str, args: argparse.Namespace) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai before running with --run.") from exc

    client = genai.Client(api_key=api_key_for(args))
    config = types.GenerateContentConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    response = client.models.generate_content(
        model=model_id_for(args),
        contents=prompt,
        config=config,
    )
    return clean_text(getattr(response, "text", ""))


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:csv|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def split_team_field(value: str) -> list[str]:
    text = clean_text(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if not text:
        return []
    if text[:1] in "[(":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [clean_text(item) for item in parsed if clean_text(item)]
        except (SyntaxError, ValueError):
            pass
    if "|" in text:
        return [
            clean_text(part).strip(" '\"")
            for part in text.split("|")
            if clean_text(part).strip(" '\"")
        ]
    return [
        clean_text(part).strip(" '\"")
        for part in text.split(",")
        if clean_text(part).strip(" '\"")
    ]


def exact_names_in_text(value: str, candidates: list[Researcher]) -> list[str]:
    text = clean_text(value).lower()
    matches: list[tuple[int, str]] = []
    for person in candidates:
        name = person.name.lower()
        index = text.find(name)
        if index >= 0:
            matches.append((index, person.name))
    ordered: list[str] = []
    for _, name in sorted(matches, key=lambda item: item[0]):
        if name not in ordered:
            ordered.append(name)
    return ordered


def parse_model_teams(
    raw_text: str,
    anchor: Researcher,
    candidates: list[Researcher],
    candidates_by_key: dict[str, Researcher],
    args: argparse.Namespace,
) -> list[list[str]]:
    cleaned = strip_fences(raw_text)
    teams: list[list[str]] = []

    reader = csv.reader(io.StringIO(cleaned), delimiter=";")
    for row in reader:
        if not row:
            continue
        parts = [clean_text(part) for part in row if clean_text(part)]
        if not parts:
            continue
        if parts[0].lower() in {"team", "team recommended"}:
            continue
        team_field = parts[-1]
        names = exact_names_in_text(team_field, candidates) or split_team_field(team_field)
        canonical: list[str] = []
        for name in names:
            candidate = candidates_by_key.get(normalize_name(name).lower())
            if candidate and candidate.name not in canonical:
                canonical.append(candidate.name)

        if anchor.name not in canonical:
            canonical.insert(0, anchor.name)
        canonical = canonical[: args.team_size]
        if canonical and canonical not in teams:
            teams.append(canonical)
        if len(teams) >= args.num_teams:
            break

    if not teams:
        teams = [[anchor.name]]
    return teams[: args.num_teams]


def score_team(
    proposal_skills: list[str],
    team: list[str],
    researcher_skills: dict[str, list[str]],
    max_team_size: int,
) -> float:
    demand = proposal_skills or ["general"]
    demand_set = set(demand)
    team = list(dict.fromkeys(team))

    covered: set[str] = set()
    redundant: set[str] = set()
    for member in team:
        for skill in researcher_skills.get(member, []):
            if skill not in demand_set:
                continue
            if skill in covered:
                redundant.add(skill)
            else:
                covered.add(skill)

    redundancy = len(redundant) / max(1, len(demand))
    set_size = len(team) / max(1, max_team_size)
    coverage = len(covered) / max(1, len(demand))
    k_robust = 0.0

    if coverage >= 1.0 and len(team) >= max_team_size:
        for remove_count in range(1, len(team)):
            for removed_index in range(len(team)):
                reduced = team[:removed_index] + team[removed_index + remove_count :]
                reduced_covered = {
                    skill
                    for member in reduced
                    for skill in researcher_skills.get(member, [])
                    if skill in demand_set
                }
                if len(reduced_covered) / max(1, len(demand)) >= coverage:
                    k_robust = 1.0
                    break
            if k_robust:
                break

    weights = [0.125, 0.125, 0.375, 0.375]
    goodness = (
        weights[0] * redundancy
        + weights[1] * set_size
        + weights[2] * coverage
        + weights[3] * k_robust
    )
    return round(float(goodness), 4)


def row_for_output(
    bundle: DomainBundle,
    proposal: Proposal,
    anchor: Researcher,
    teams: list[list[str]],
    goodness: list[float],
) -> dict[str, object]:
    if bundle.domain == "teaming":
        return {
            "proposal_id": proposal.proposal_id,
            "year": proposal.year,
            "proposal_link": proposal.proposal_link,
            "title": proposal.title,
            "skills": repr(proposal.skills),
            "researcher_name": anchor.name,
            "team": repr(teams),
            "goodness": repr(goodness),
        }
    return {
        "proposal_link": proposal.proposal_link,
        "title": proposal.title,
        "skills": repr(proposal.skills),
        "researcher_name": anchor.name,
        "team": repr(teams),
        "goodness": repr(goodness),
    }


def candidate_pool_for(anchor: Researcher, researchers: list[Researcher], args: argparse.Namespace) -> list[Researcher]:
    candidates = researchers
    if args.limit_candidates > 0:
        others = [person for person in researchers if person.name != anchor.name]
        candidates = [anchor, *others[: max(0, args.limit_candidates - 1)]]
    return candidates


def write_output_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def print_counts(bundle: DomainBundle, output_path: Path | None, output_rows: int | None) -> None:
    print("Dataset usage")
    print(f"  domain: {bundle.domain}")
    print(f"  researchers csv: {bundle.researchers_path}")
    print(f"  researchers rows: raw={bundle.raw_researcher_rows}, used={len(bundle.researchers)}")
    print(f"  proposals csv: {bundle.proposals_path}")
    print(f"  proposals rows: raw={bundle.raw_proposal_rows}, used={len(bundle.proposals)}")
    print(f"  planned output rows: {len(bundle.researchers) * len(bundle.proposals)}")
    if output_path is not None and output_rows is not None:
        print("Output")
        print(f"  csv: {output_path}")
        print(f"  rows written: {output_rows}")


def write_prompt_manifest(path: Path, prompts: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    bundle = load_domain(args)
    output_path = output_path_for(bundle, args)
    call_count = len(bundle.researchers) * len(bundle.proposals)

    print_counts(bundle, None, None)
    print(f"Model alias: {args.model}")
    print(f"Model id: {model_id_for(args)}")

    if args.run and args.max_calls > 0 and call_count > args.max_calls:
        raise RuntimeError(
            f"Refusing to make {call_count} model calls because --max-calls is {args.max_calls}. "
            "Use limits for a test run or pass --max-calls 0 for a full run."
        )

    researcher_skills = {person.name: person.skills for person in bundle.researchers}
    all_rows: list[dict[str, object]] = []
    prompt_manifest: list[dict[str, str]] = []
    raw_dir = output_path.parent / f"{output_path.stem}_raw"

    for proposal_index, proposal in enumerate(bundle.proposals, start=1):
        for anchor_index, anchor in enumerate(bundle.researchers, start=1):
            candidates = candidate_pool_for(anchor, bundle.researchers, args)
            prompt = build_prompt(proposal, anchor, candidates, args)
            candidates_by_key = {normalize_name(person.name).lower(): person for person in candidates}

            if not args.run:
                if len(prompt_manifest) < args.max_prompt_manifest:
                    prompt_manifest.append(
                        {
                            "proposal_link": proposal.proposal_link,
                            "researcher_name": anchor.name,
                            "prompt": prompt,
                        }
                    )
                continue

            raw_text = call_model(prompt, args)
            if args.save_raw:
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_name = f"{proposal_index:04d}_{anchor_index:04d}_{sanitize_token(anchor.name)}.txt"
                (raw_dir / raw_name).write_text(raw_text, encoding="utf-8")

            teams = parse_model_teams(raw_text, anchor, candidates, candidates_by_key, args)
            goodness = [
                score_team(proposal.skills, team, researcher_skills, args.team_size)
                for team in teams
            ]
            scored = sorted(zip(goodness, teams, strict=True), key=lambda item: item[0], reverse=True)
            sorted_goodness = [score for score, _ in scored]
            sorted_teams = [team for _, team in scored]
            all_rows.append(row_for_output(bundle, proposal, anchor, sorted_teams, sorted_goodness))
            time.sleep(args.sleep_seconds)

        print(f"processed proposals: {proposal_index}/{len(bundle.proposals)}")

    if not args.run:
        manifest_path = (
            REPO_ROOT
            / "llm_outputs"
            / "prompt_manifests"
            / bundle.domain
            / f"{output_path.stem}_prompt_manifest.jsonl"
        )
        write_prompt_manifest(manifest_path, prompt_manifest)
        print("Dry run only; no output CSV generated.")
        print(f"Wrote prompt examples: {manifest_path}")
        return 0

    output_rows = write_output_csv(output_path, bundle.output_columns, all_rows)
    print_counts(bundle, output_path, output_rows)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M5 LLM output CSVs matching Teaming/IITR Teaming formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", choices=["teaming", "iitr-teaming"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), default="gemini-3-flash-live")
    parser.add_argument("--model-id", default="", help="Override exact API model id.")
    parser.add_argument("--run", action="store_true", help="Call the model and write the M5 CSV.")
    parser.add_argument("--output-csv", default="", help="Override output CSV path.")
    parser.add_argument("--api-key-env", nargs="+", default=["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    parser.add_argument("--num-teams", type=int, default=10)
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--sleep-seconds", type=float, default=4.1)
    parser.add_argument("--max-calls", type=int, default=100, help="Safety limit for API calls; use 0 for no limit.")
    parser.add_argument("--save-raw", action="store_true", help="Save raw model responses beside the output CSV.")
    parser.add_argument("--limit-proposals", type=int, default=0, help="Use first N effective proposals; 0 means all.")
    parser.add_argument("--limit-researchers", type=int, default=0, help="Use first N researchers; 0 means all.")
    parser.add_argument("--limit-candidates", type=int, default=0, help="Limit candidate researchers in prompt; 0 means all.")
    parser.add_argument("--max-field-chars", type=int, default=900)
    parser.add_argument("--max-skills-per-record", type=int, default=40)
    parser.add_argument("--max-prompt-manifest", type=int, default=5)

    parser.add_argument(
        "--researchers-csv",
        default="Teaming/data/v1_input_files/v1_researchers.csv",
        help="USC Teaming researchers input.",
    )
    parser.add_argument(
        "--proposals-csv",
        default="Teaming/data/v1_input_files/v1_proposal_links_title_synopsis.csv",
        help="USC Teaming proposal input.",
    )
    parser.add_argument(
        "--iitr-researchers-csv",
        default="IITR-Teaming/data/v0_data/researchers.csv",
        help="IITR Teaming researchers input.",
    )
    parser.add_argument(
        "--iitr-proposals-csv",
        default="IITR-Teaming/data/v0_data/archive_proposals.csv",
        help="IITR Teaming proposal input.",
    )
    parser.add_argument(
        "--iitr-num-proposals",
        type=int,
        default=100,
        help="Number of IITR proposals to use before applying --limit-proposals.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
