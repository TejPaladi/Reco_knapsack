#!/usr/bin/env python3
"""Generate M5 LLM recommendation outputs.

This script mirrors the existing CSV contracts, but uses an LLM to generate
candidate recommendations instead of M0/M1/M2/M3/M6/M7 logic.

Supported domains:
  - teaming:      USC Teaming, using Teaming/data/v1_input_files/
  - iitr-teaming: IITR Teaming, using IITR-Teaming/data/v0_data/
  - meal:         Meal recommendation, using Meal/data/input_data/

Default output files:
  - Teaming/data/output/teaming_uc1_m5_<model>.csv
  - IITR-Teaming/data/output/teaming_uc1_m5_<model>.csv
  - Meal/data/output/meal_uc1_m5_<model>.csv

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
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

USC_BAD_PROPOSAL_LINKS = {
    "https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=505073",
}

MODEL_ALIASES = {
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3-flash-live": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-2.5-flash": "models/gemini-2.5-flash",
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
    meta: dict[str, str] | None = None


@dataclass(frozen=True)
class Researcher:
    name: str
    title: str
    description: str
    skills: list[str]
    meta: dict[str, str] | None = None


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
    candidates: list[Researcher] | None = None


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
    if bundle.domain == "meal":
        return bundle.output_dir / f"meal_uc1_m5_{model_slug(args.model)}.csv"
    return bundle.output_dir / f"teaming_uc1_m5_{model_slug(args.model)}.csv"


def load_metric_module(domain: str):
    if domain == "iitr-teaming":
        metric_path = repo_path("IITR-Teaming/code/metrics_scorer.py")
    elif domain == "teaming":
        metric_path = repo_path("Teaming/code/metrics_scorer.py")
    elif domain == "meal":
        return None
    else:
        raise ValueError(f"Unsupported domain for metrics: {domain}")

    spec = importlib.util.spec_from_file_location(f"{domain.replace('-', '_')}_metrics_scorer", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load metric scorer from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def stable_set_repr(values: list[str] | set[str]) -> str:
    return "{" + ", ".join(repr(value) for value in sorted(values)) + "}"


def load_meal(args: argparse.Namespace) -> DomainBundle:
    users_path = repo_path(args.meal_users_csv)
    items_path = repo_path(args.meal_items_csv)
    user_rows = read_csv_rows(users_path)
    item_rows = read_csv_rows(items_path)

    users: list[Researcher] = []
    for row in user_rows:
        user_id = first_value(row, "user_id")
        if not user_id:
            continue
        required_categories = parse_literal_or_split(first_value(row, "required_categories"))
        users.append(
            Researcher(
                name=first_value(row, "name") or user_id,
                title=first_value(row, "meal_occasion"),
                description=f"user_id={user_id}",
                skills=required_categories or ["general"],
                meta={
                    "user_id": user_id,
                    "name": first_value(row, "name") or user_id,
                    "meal_occasion": first_value(row, "meal_occasion"),
                    "gender": first_value(row, "gender"),
                    "ethnicity": first_value(row, "ethnicity"),
                },
            )
        )

    meal_items: list[Researcher] = []
    for row in item_rows:
        meal_name = first_value(row, "meal_name")
        if not meal_name:
            continue
        categories = parse_literal_or_split(first_value(row, "categories"))
        meal_items.append(
            Researcher(
                name=meal_name,
                title="",
                description="",
                skills=categories or ["general"],
                meta={"categories": stable_set_repr(categories or ["general"])},
            )
        )

    selected_items = meal_items
    if args.limit_proposals > 0:
        selected_items = selected_items[: args.limit_proposals]

    proposals = [
        Proposal(
            proposal_id=sanitize_token(item.name),
            year="",
            proposal_link=sanitize_token(item.name),
            title=item.name,
            prompt_title=item.name,
            summary=f"Meal item categories: {', '.join(item.skills)}",
            skills=item.skills,
            meta={"target_item": item.name},
        )
        for item in selected_items
    ]

    return DomainBundle(
        domain="meal",
        researchers_path=users_path,
        proposals_path=items_path,
        raw_researcher_rows=len(user_rows),
        raw_proposal_rows=len(item_rows),
        researchers=users,
        proposals=proposals,
        output_columns=[
            "meal_request_id",
            "user_id",
            "name",
            "meal_occasion",
            "required_categories",
            "target_item",
            "recommended_meals",
            "goodness",
            "avg_redundancy",
            "avg_setsize",
            "avg_coverage",
            "avg_krobust",
            "avg_dm_proxy",
            "avg_mc_proxy",
            "avg_uc_proxy",
        ],
        output_dir=REPO_ROOT / "Meal" / "data" / "output",
        candidates=meal_items,
    )


def load_domain(args: argparse.Namespace) -> DomainBundle:
    if args.domain == "teaming":
        return load_usc_teaming(args)
    if args.domain == "iitr-teaming":
        return load_iitr_teaming(args)
    if args.domain == "meal":
        return load_meal(args)
    raise ValueError(f"Unsupported domain: {args.domain}")


def apply_index_slice(items: list[Any], start_index: int, end_index: int) -> list[Any]:
    if start_index <= 0 and end_index <= 0:
        return items
    start = max(0, start_index - 1) if start_index > 0 else 0
    stop = end_index if end_index > 0 else len(items)
    return items[start:stop]


def format_researcher(person: Researcher) -> str:
    parts = [f"name={person.name}", f"skills={', '.join(person.skills)}"]
    if person.title:
        parts.append(f"title={person.title}")
    if person.description:
        parts.append(f"description={person.description}")
    return " | ".join(parts)


def format_meal_user(user: Researcher) -> str:
    meta = user.meta or {}
    parts = [
        f"user_id={meta.get('user_id', user.description.replace('user_id=', ''))}",
        f"name={meta.get('name', user.name)}",
        f"meal_occasion={meta.get('meal_occasion', user.title)}",
        f"required_categories={', '.join(user.skills)}",
    ]
    if meta.get("gender"):
        parts.append(f"gender={meta['gender']}")
    if meta.get("ethnicity"):
        parts.append(f"ethnicity={meta['ethnicity']}")
    return " | ".join(parts)


def format_meal_item(item: Researcher) -> str:
    return f"meal_name={item.name} | categories={', '.join(item.skills)}"


def build_prompt(
    bundle: DomainBundle,
    proposal: Proposal,
    anchor: Researcher,
    candidates: list[Researcher],
    args: argparse.Namespace,
) -> str:
    if bundle.domain == "meal":
        candidate_lines = "\n".join(f"- {format_meal_item(item)}" for item in candidates)
        example_items = [item.name for item in candidates if item.name != proposal.prompt_title][: max(args.num_teams, 1)]
        if not example_items:
            example_items = [proposal.prompt_title]
        example_rows = "\n".join(
            f"{proposal.prompt_title} | {example_items[index % len(example_items)]}"
            for index in range(args.num_teams)
        )
        return f"""You are generating the M5 LLM baseline for meal recommendation.

Return plain text CSV only.
Do not use markdown fences.
Do not add explanations.
Do not put multiple meal bundles on one line.

Required CSV header:
team

Task:
Recommend exactly {args.num_teams} distinct meal bundles for the user request and target item below.
The CSV must contain exactly {args.num_teams + 1} lines total:
- line 1 is exactly: team
- lines 2 through {args.num_teams + 1} are exactly one meal bundle per line

Rules:
- Use only meal names from Available Meal Items.
- Every bundle must include the target item exactly as written.
- Each bundle should contain 2 to {args.team_size} meal items total, including the target item.
- Use pipe-separated exact meal names inside each bundle row.
- Do not use semicolons inside the bundle field.
- Do not use commas to separate meal names; some names may contain punctuation.
- Do not quote the bundle rows.
- Do not include blank rows or duplicate bundles.
- Prefer compact bundles whose combined categories satisfy the user's required categories.
- Prefer non-redundant, high-confidence bundles appropriate for the meal occasion.
- Do not return a single-item bundle.
- If uncertain, still return exactly {args.num_teams} valid distinct multi-item bundles.

Required output format example:
team
{example_rows}

Invalid formats:
- team Coffee | Tea Coffee | Milk
- "Coffee | Tea" "Coffee | Milk"
- team;Coffee | Tea

User Request:
{format_meal_user(anchor)}

Target Item:
meal_name={proposal.prompt_title}
categories={', '.join(proposal.skills)}

Available Meal Items:
{candidate_lines}
"""

    candidate_lines = "\n".join(f"- {format_researcher(person)}" for person in candidates)
    example_partners = [person.name for person in candidates if person.name != anchor.name][: max(args.num_teams, 1)]
    if not example_partners:
        example_partners = [anchor.name]
    example_rows = "\n".join(
        f"{anchor.name} | {example_partners[index % len(example_partners)]}"
        for index in range(args.num_teams)
    )
    return f"""You are generating the M5 LLM baseline for research team recommendation.

Return plain text CSV only.
Do not use markdown fences.
Do not add explanations.
Do not put multiple teams on one line.

Required CSV header:
team

Task:
Recommend exactly {args.num_teams} distinct candidate teams for the anchor researcher and proposal below.
The CSV must contain exactly {args.num_teams + 1} lines total:
- line 1 is exactly: team
- lines 2 through {args.num_teams + 1} are exactly one team per line

Rules:
- Use only names from Available Researchers.
- Every team must include the anchor researcher exactly as written.
- Each team should contain 2 to {args.team_size} researchers total, including the anchor.
- Use pipe-separated exact names inside each team row.
- Do not use semicolons inside the team field.
- Do not use commas to separate names; some researcher names already contain commas.
- Do not quote the team rows.
- Do not include blank rows or duplicate teams.
- Prefer teams whose combined skills match the proposal skills and summary.
- Prefer teams that are compact, non-redundant, and high-confidence.
- Do not return a single-person team.
- If uncertain, still return exactly {args.num_teams} valid distinct multi-person teams.

Required output format example:
team
{example_rows}

Invalid formats:
- team Name A | Name B Name A | Name C
- "Name A | Name B" "Name A | Name C"
- team;Name A | Name B

Anchor Researcher:
{format_researcher(anchor)}

Proposal:
title={proposal.prompt_title}
skills={', '.join(proposal.skills)}
summary={proposal.summary}

Available Researchers:
{candidate_lines}
"""


def build_repair_prompt(
    original_prompt: str,
    existing_teams: list[list[str]],
    missing_count: int,
) -> str:
    existing = "\n".join(
        f"- {' | '.join(team)}"
        for team in existing_teams
    )
    existing_block = existing or "- none"
    return f"""{original_prompt}

Correction:
The previous response did not contain enough valid distinct multi-person teams.
Return CSV only with the same required header:
team

Return exactly {missing_count} additional valid rows, one team per row.
Do not repeat any of these already valid teams:
{existing_block}
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
    last_error: Exception | None = None
    for attempt in range(1, args.api_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id_for(args),
                contents=prompt,
                config=config,
            )
            return clean_text(getattr(response, "text", ""))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            message = str(exc)
            transient = any(token in message for token in ("503", "500", "429", "UNAVAILABLE", "INTERNAL"))
            if not transient or attempt >= args.api_retries:
                raise
            time.sleep(min(args.retry_sleep_seconds * attempt, args.max_retry_sleep_seconds))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Model call failed without raising an error.")


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
    matches: list[tuple[int, int, str]] = []
    for person in sorted(candidates, key=lambda candidate: len(candidate.name), reverse=True):
        name = person.name.lower()
        pattern = rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            span = (match.start(), match.end())
            if any(not (span[1] <= start or span[0] >= end) for start, end, _ in matches):
                continue
            matches.append((span[0], span[1], person.name))
    return [name for _, _, name in sorted(matches, key=lambda item: item[0])]


def team_signature(anchor: Researcher, team: list[str]) -> tuple[str, ...]:
    return (anchor.name, *sorted(member for member in team if member != anchor.name))


def quoted_team_fields(text: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return [clean_text(field) for field in quoted if clean_text(field)]
    return []


def repeated_anchor_team_fields(text: str, anchor: Researcher) -> list[str]:
    pattern = rf"(?<![a-z0-9]){re.escape(anchor.name)}(?![a-z0-9])"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if len(matches) < 2:
        return []

    fields: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        field = clean_text(text[start:stop]).strip(" ;,")
        if field:
            fields.append(field)
    return fields


def parse_model_teams(
    raw_text: str,
    anchor: Researcher,
    candidates: list[Researcher],
    candidates_by_key: dict[str, Researcher],
    args: argparse.Namespace,
) -> list[list[str]]:
    cleaned = strip_fences(raw_text)
    teams: list[list[str]] = []
    seen_signatures: set[tuple[str, ...]] = set()

    def add_team_from_text(team_text: str) -> None:
        names = exact_names_in_text(team_text, candidates) or split_team_field(team_text)
        canonical: list[str] = []
        for name in names:
            candidate = candidates_by_key.get(normalize_name(name).lower())
            if candidate and candidate.name not in canonical:
                canonical.append(candidate.name)

        if anchor.name not in canonical:
            canonical.insert(0, anchor.name)
        canonical = canonical[: args.team_size]
        signature = team_signature(anchor, canonical)
        if len(canonical) >= 2 and signature not in seen_signatures:
            seen_signatures.add(signature)
            teams.append(canonical)

    quoted_fields = quoted_team_fields(cleaned)
    if quoted_fields and len(quoted_fields) >= min(args.num_teams, 2):
        for field in quoted_fields:
            add_team_from_text(field)
            if len(teams) >= args.num_teams:
                break
        return teams[: args.num_teams]

    repeated_anchor_fields = repeated_anchor_team_fields(cleaned, anchor)
    if repeated_anchor_fields:
        for field in repeated_anchor_fields:
            add_team_from_text(field)
            if len(teams) >= args.num_teams:
                break
        return teams[: args.num_teams]

    reader = csv.reader(io.StringIO(cleaned), delimiter=";")
    for row in reader:
        if not row:
            continue
        parts = [clean_text(part) for part in row if clean_text(part)]
        if not parts:
            continue
        if parts[0].lower() in {"team", "team recommended", "professor name", "proposal name"}:
            continue
        row_text = "; ".join(parts)
        add_team_from_text(row_text)
        if len(teams) >= args.num_teams:
            break

    return teams[: args.num_teams]


def collect_exact_model_teams(
    raw_text: str,
    prompt: str,
    anchor: Researcher,
    candidates: list[Researcher],
    candidates_by_key: dict[str, Researcher],
    args: argparse.Namespace,
) -> list[list[str]]:
    teams = parse_model_teams(raw_text, anchor, candidates, candidates_by_key, args)
    if args.allow_short_output:
        return teams

    for _ in range(args.repair_retries):
        missing_count = args.num_teams - len(teams)
        if missing_count <= 0:
            break

        repair_prompt = build_repair_prompt(prompt, teams, missing_count)
        repair_text = call_model(repair_prompt, args)
        repair_teams = parse_model_teams(repair_text, anchor, candidates, candidates_by_key, args)
        seen = {team_signature(anchor, team) for team in teams}
        for team in repair_teams:
            signature = team_signature(anchor, team)
            if signature not in seen:
                teams.append(team)
                seen.add(signature)
            if len(teams) >= args.num_teams:
                break

    if len(teams) != args.num_teams:
        raise RuntimeError(
            f"Model returned {len(teams)} valid teams for anchor={anchor.name!r} after "
            f"{args.repair_retries} repair retries; expected exactly {args.num_teams}. "
            "Use --allow-short-output to keep partial rows."
        )
    return teams[: args.num_teams]


def score_team(
    proposal_skills: list[str],
    team: list[str],
    researcher_skills: dict[str, list[str]],
    metrics_module: Any,
) -> float:
    if not proposal_skills or not team:
        return 0.0

    scorer = metrics_module.MetricScorer()
    scorer.demand = list(proposal_skills)
    scorer.team = list(team)
    for researcher_name in scorer.team:
        scorer.researchers[researcher_name] = set(researcher_skills.get(researcher_name, []))
    scorer.set_new_weights([-1, -1, 1, 1])
    scorer.run_metrics()
    return round(float(scorer.goodness), 4)


def score_meal_bundle(
    required_categories: list[str],
    bundle: list[str],
    item_categories: dict[str, list[str]],
    meal_size: int,
) -> dict[str, float]:
    if not required_categories or not bundle:
        return {
            "redundancy": 0.0,
            "setsize": 0.0,
            "coverage": 0.0,
            "krobust": 0.0,
            "goodness": 0.0,
            "dm_proxy": 0.0,
            "mc_proxy": 0.0,
            "uc_proxy": 0.0,
        }

    required = set(required_categories)
    denom = max(1, len(required))
    covered: set[str] = set()
    seen_required: set[str] = set()
    redundant_required: set[str] = set()
    uc_parts: list[float] = []

    for item in bundle:
        categories = set(item_categories.get(item, []))
        item_required = categories & required
        covered |= item_required
        for category in item_required:
            if category in seen_required:
                redundant_required.add(category)
            seen_required.add(category)
        uc_parts.append(len(item_required) / denom)

    coverage = len(covered) / denom
    redundancy = len(redundant_required) / denom
    setsize = len(bundle) / max(1, meal_size)
    krobust = 0.0
    if coverage >= 0.5 and len(bundle) >= 2:
        for index in range(len(bundle)):
            remaining = bundle[:index] + bundle[index + 1 :]
            remaining_covered: set[str] = set()
            for item in remaining:
                remaining_covered |= set(item_categories.get(item, [])) & required
            if len(remaining_covered) / denom >= coverage:
                krobust = 1.0
                break

    dm_proxy = len(set(bundle)) / len(bundle)
    mc_proxy = coverage
    uc_proxy = sum(uc_parts) / len(uc_parts) if uc_parts else 0.0
    goodness = (dm_proxy + mc_proxy + uc_proxy) / 3
    return {
        "redundancy": round(redundancy, 6),
        "setsize": round(setsize, 6),
        "coverage": round(coverage, 6),
        "krobust": round(krobust, 6),
        "goodness": round(goodness, 6),
        "dm_proxy": round(dm_proxy, 6),
        "mc_proxy": round(mc_proxy, 6),
        "uc_proxy": round(uc_proxy, 6),
    }


def mean_metric(metrics: list[dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    return round(sum(metric[key] for metric in metrics) / len(metrics), 6)


def row_for_output(
    bundle: DomainBundle,
    proposal: Proposal,
    anchor: Researcher,
    teams: list[list[str]],
    goodness: list[float],
    metrics: list[dict[str, float]] | None = None,
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
    if bundle.domain == "meal":
        meta = anchor.meta or {}
        metrics = metrics or []
        return {
            "meal_request_id": 0,
            "user_id": meta.get("user_id", anchor.description.replace("user_id=", "")),
            "name": meta.get("name", anchor.name),
            "meal_occasion": meta.get("meal_occasion", anchor.title),
            "required_categories": stable_set_repr(anchor.skills),
            "target_item": proposal.prompt_title,
            "recommended_meals": repr(teams),
            "goodness": repr(goodness),
            "avg_redundancy": mean_metric(metrics, "redundancy"),
            "avg_setsize": mean_metric(metrics, "setsize"),
            "avg_coverage": mean_metric(metrics, "coverage"),
            "avg_krobust": mean_metric(metrics, "krobust"),
            "avg_dm_proxy": mean_metric(metrics, "dm_proxy"),
            "avg_mc_proxy": mean_metric(metrics, "mc_proxy"),
            "avg_uc_proxy": mean_metric(metrics, "uc_proxy"),
        }
    return {
        "proposal_link": proposal.proposal_link,
        "title": proposal.title,
        "skills": repr(proposal.skills),
        "researcher_name": anchor.name,
        "team": repr(teams),
        "goodness": repr(goodness),
    }


def candidate_pool_for(
    anchor: Researcher,
    researchers: list[Researcher],
    args: argparse.Namespace,
    candidates: list[Researcher] | None = None,
) -> list[Researcher]:
    if candidates is not None:
        return candidates[: args.limit_candidates] if args.limit_candidates > 0 else candidates

    candidates = researchers
    if args.limit_candidates > 0:
        others = [person for person in researchers if person.name != anchor.name]
        candidates = [anchor, *others[: max(0, args.limit_candidates - 1)]]
    return candidates


def write_output_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    append: bool = False,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    write_header = mode == "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
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
    full_bundle = load_domain(args)
    anchor_researchers = apply_index_slice(full_bundle.researchers, args.researcher_start, args.researcher_end)
    if args.limit_researchers > 0:
        anchor_researchers = anchor_researchers[: args.limit_researchers]
    bundle = DomainBundle(
        domain=full_bundle.domain,
        researchers_path=full_bundle.researchers_path,
        proposals_path=full_bundle.proposals_path,
        raw_researcher_rows=full_bundle.raw_researcher_rows,
        raw_proposal_rows=full_bundle.raw_proposal_rows,
        researchers=anchor_researchers,
        proposals=apply_index_slice(full_bundle.proposals, args.proposal_start, args.proposal_end),
        output_columns=full_bundle.output_columns,
        output_dir=full_bundle.output_dir,
        candidates=full_bundle.candidates,
    )
    output_path = output_path_for(bundle, args)
    metrics_module = load_metric_module(bundle.domain)
    call_count = len(bundle.researchers) * len(bundle.proposals)
    if args.run and not args.append_output and output_path.exists():
        output_path.unlink()

    print_counts(bundle, None, None)
    print(f"Model alias: {args.model}")
    print(f"Model id: {model_id_for(args)}")

    max_possible_calls = call_count
    if not args.allow_short_output:
        max_possible_calls *= 1 + max(0, args.repair_retries)
    if args.run and args.max_calls > 0 and max_possible_calls > args.max_calls:
        raise RuntimeError(
            f"Refusing to make up to {max_possible_calls} model calls because --max-calls is {args.max_calls}. "
            "Use limits for a test run or pass --max-calls 0 for a full run."
        )

    skill_records = full_bundle.candidates or full_bundle.researchers
    researcher_skills = {person.name: person.skills for person in skill_records}
    all_rows: list[dict[str, object]] = []
    prompt_manifest: list[dict[str, str]] = []
    raw_dir = output_path.parent / f"{output_path.stem}_raw"
    initial_output_rows = 0
    if args.run and args.append_output and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            initial_output_rows = sum(1 for _ in csv.DictReader(handle))
    output_rows = initial_output_rows
    planned_total_rows = initial_output_rows + len(bundle.researchers) * len(bundle.proposals)

    proposal_index_offset = max(0, args.proposal_start - 1) if args.proposal_start > 0 else 0
    researcher_index_offset = max(0, args.researcher_start - 1) if args.researcher_start > 0 else 0
    proposal_display_total = proposal_index_offset + len(bundle.proposals)
    researcher_display_total = researcher_index_offset + len(bundle.researchers)

    for proposal_offset, proposal in enumerate(bundle.proposals, start=1):
        proposal_index = proposal_index_offset + proposal_offset
        proposal_rows: list[dict[str, object]] = []
        for anchor_offset, anchor in enumerate(bundle.researchers, start=1):
            anchor_index = researcher_index_offset + anchor_offset
            candidates = candidate_pool_for(anchor, full_bundle.researchers, args, full_bundle.candidates)
            prompt = build_prompt(bundle, proposal, anchor, candidates, args)
            candidates_by_key = {normalize_name(person.name).lower(): person for person in candidates}
            required_member = anchor
            if bundle.domain == "meal":
                required_member = next(
                    (person for person in candidates if person.name == proposal.prompt_title),
                    Researcher(name=proposal.prompt_title, title="", description="", skills=proposal.skills),
                )

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

            teams = collect_exact_model_teams(raw_text, prompt, required_member, candidates, candidates_by_key, args)
            if bundle.domain == "meal":
                scored_metrics = [
                    score_meal_bundle(proposal.skills, team, researcher_skills, args.team_size)
                    for team in teams
                ]
                scored = sorted(
                    zip([metric["goodness"] for metric in scored_metrics], teams, scored_metrics),
                    key=lambda item: item[0],
                    reverse=True,
                )
                sorted_goodness = [score for score, _, _ in scored]
                sorted_teams = [team for _, team, _ in scored]
                sorted_metrics = [metric for _, _, metric in scored]
            else:
                goodness = [
                    score_team(proposal.skills, team, researcher_skills, metrics_module)
                    for team in teams
                ]
                scored = sorted(zip(goodness, teams), key=lambda item: item[0], reverse=True)
                sorted_goodness = [score for score, _ in scored]
                sorted_teams = [team for _, team in scored]
                sorted_metrics = None
            output_row = row_for_output(bundle, proposal, anchor, sorted_teams, sorted_goodness, sorted_metrics)
            if bundle.domain == "meal":
                output_row["meal_request_id"] = initial_output_rows + len(all_rows) + len(proposal_rows)
            proposal_rows.append(output_row)
            if args.run:
                output_rows = write_output_csv(output_path, bundle.output_columns, [output_row], append=True)
                print(
                    f"processed rows: {output_rows}/{planned_total_rows} "
                    f"(proposal {proposal_index}/{proposal_display_total}, "
                    f"anchor {anchor_index}/{researcher_display_total})"
                )
            time.sleep(args.sleep_seconds)

        all_rows.extend(proposal_rows)
        print(f"processed proposals: {proposal_index}/{proposal_display_total}")

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

    if args.run:
        # The proposal-level writes already persisted the rows; this final pass is only used
        # to report a stable row count if the run completed normally.
        output_rows = write_output_csv(output_path, bundle.output_columns, [], append=True)
    print_counts(bundle, output_path, output_rows)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M5 LLM output CSVs matching Teaming/IITR/Meal formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", choices=["teaming", "iitr-teaming", "meal"], required=True)
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
    parser.add_argument("--api-retries", type=int, default=4, help="Retry transient model API failures.")
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0, help="Base backoff sleep between API retries.")
    parser.add_argument("--max-retry-sleep-seconds", type=float, default=12.0, help="Maximum sleep between API retries.")
    parser.add_argument(
        "--repair-retries",
        type=int,
        default=2,
        help="Additional model calls used to repair a response that returns fewer than --num-teams valid teams.",
    )
    parser.add_argument(
        "--allow-short-output",
        action="store_true",
        help="Keep rows even when the model returns fewer than --num-teams valid multi-person teams.",
    )
    parser.add_argument("--max-calls", type=int, default=100, help="Safety limit for API calls; use 0 for no limit.")
    parser.add_argument("--save-raw", action="store_true", help="Save raw model responses beside the output CSV.")
    parser.add_argument("--limit-proposals", type=int, default=0, help="Use first N effective proposals; 0 means all.")
    parser.add_argument("--limit-researchers", type=int, default=0, help="Use first N anchor researchers; 0 means all.")
    parser.add_argument("--limit-candidates", type=int, default=0, help="Limit available candidate researchers in prompt; 0 means all.")
    parser.add_argument("--proposal-start", type=int, default=0, help="1-based proposal slice start; 0 means all.")
    parser.add_argument("--proposal-end", type=int, default=0, help="1-based proposal slice end; 0 means all.")
    parser.add_argument("--researcher-start", type=int, default=0, help="1-based anchor researcher slice start; 0 means all.")
    parser.add_argument("--researcher-end", type=int, default=0, help="1-based anchor researcher slice end; 0 means all.")
    parser.add_argument("--append-output", action="store_true", help="Append to an existing output CSV instead of overwriting it.")
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
    parser.add_argument(
        "--meal-users-csv",
        default="Meal/data/input_data/user_meal_requirements.csv",
        help="Meal user requirements input.",
    )
    parser.add_argument(
        "--meal-items-csv",
        default="Meal/data/input_data/meal_categories.csv",
        help="Meal item/category input.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
