#!/usr/bin/env python3
"""Parameterized LLM recommendation runner for Reco_knapsack.

This script builds prompts for the repository's supported domains and can call
Google GenAI models when `--run` is supplied. Without `--run`, it writes prompt
files only, which is useful for inspection before spending API quota.

Examples:
    python scripts/llm_recommender.py --domain teaming --model gemini-3-flash-live
    python scripts/llm_recommender.py --domain meal --model gemma-4-31b --run
    python scripts/llm_recommender.py --domain iitr-teaming --target-name "Arindam" --run

API keys are read from GEMINI_API_KEY or GOOGLE_API_KEY by default. Do not put
API keys in notebooks or source files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ALIASES = {
    # User-facing alias requested for this project. The Google model code for
    # text generation is currently gemini-3-flash-preview.
    "gemini-3-flash-live": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    # Keep the requested Gemma label as a configurable alias. If the serving
    # endpoint uses a different model ID, pass --model-id in the next phase.
    "gemma-4-31b": "gemma-4-31b",
}


@dataclass(frozen=True)
class DomainData:
    domain: str
    kind: str
    targets: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    items: list[dict[str, Any]]
    output_columns: list[str]


@dataclass(frozen=True)
class PromptTask:
    task_id: str
    target_name: str
    prompt: str
    metadata: dict[str, Any]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_csv(path: str | Path) -> pd.DataFrame:
    import pandas as pd

    csv_path = repo_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    return pd.read_csv(csv_path)


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


def parse_collection(value: Any, max_chars: int | None = None) -> list[str]:
    text = clean_text(value)
    if not text:
        return []

    parsed: Any | None = None
    if text[:1] in {"[", "{", "("}:
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

    cleaned = []
    for item in values:
        item_text = clean_text(item, max_chars=max_chars)
        if item_text:
            cleaned.append(item_text)
    return sorted(dict.fromkeys(cleaned))


def first_value(row: Any, *columns: str) -> Any:
    for column in columns:
        if column in row and clean_text(row[column]):
            return row[column]
    return ""


def normalize_name(name: Any) -> str:
    return clean_text(name).lower()


def format_list(values: list[str], max_items: int = 18) -> str:
    if not values:
        return "none listed"
    clipped = values[:max_items]
    suffix = "" if len(values) <= max_items else f", ... ({len(values)} total)"
    return ", ".join(clipped) + suffix


def load_meal(args: argparse.Namespace) -> DomainData:
    meals_df = read_csv(args.meal_categories_csv)
    users_df = read_csv(args.meal_requirements_csv)

    meals = []
    for _, row in meals_df.iterrows():
        meals.append(
            {
                "name": clean_text(first_value(row, "meal_name")),
                "categories": parse_collection(first_value(row, "categories")),
            }
        )

    users = []
    for _, row in users_df.iterrows():
        users.append(
            {
                "id": clean_text(first_value(row, "user_id")),
                "name": clean_text(first_value(row, "name")),
                "meal_occasion": clean_text(first_value(row, "meal_occasion")),
                "required_categories": parse_collection(
                    first_value(row, "required_categories")
                ),
            }
        )

    return DomainData(
        domain="meal",
        kind="meal",
        targets=users,
        candidates=meals,
        items=meals,
        output_columns=[
            "User ID",
            "User Name",
            "Meal Occasion",
            "Meal Bundle Recommended",
        ],
    )


def load_usc_reference_maps(args: argparse.Namespace) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    faculty_context: dict[str, list[str]] = {}
    if args.faculty_csv and repo_path(args.faculty_csv).exists():
        faculty_df = read_csv(args.faculty_csv)
        for _, row in faculty_df.iterrows():
            name = normalize_name(first_value(row, "name", "names", "FacultyName"))
            if not name:
                continue
            parts: list[str] = []
            for column in ("research", "keywords", "area", "background", "titles"):
                if column in row:
                    parts.extend(parse_collection(row[column], max_chars=args.max_field_chars))
            faculty_context[name] = sorted(dict.fromkeys(parts))

    if args.interests_csv and repo_path(args.interests_csv).exists():
        interests_df = read_csv(args.interests_csv)
        for _, row in interests_df.iterrows():
            name = normalize_name(first_value(row, "names", "name", "FacultyName"))
            if not name:
                continue
            parts = faculty_context.setdefault(name, [])
            parts.extend(parse_collection(first_value(row, "research"), max_chars=args.max_field_chars))
            faculty_context[name] = sorted(dict.fromkeys(parts))

    rfp_items: list[dict[str, Any]] = []
    if args.rfps_csv and repo_path(args.rfps_csv).exists():
        rfps_df = read_csv(args.rfps_csv)
        for _, row in rfps_df.iterrows():
            title = clean_text(first_value(row, "title", "Title"), args.max_field_chars)
            if not title:
                continue
            details = clean_text(first_value(row, "details", "synopsis"), args.max_field_chars)
            keywords = parse_collection(first_value(row, "keywords"), max_chars=args.max_field_chars)
            rfp_items.append(
                {
                    "id": clean_text(first_value(row, "reference", "agencyId")),
                    "title": title,
                    "summary": details,
                    "skills": keywords,
                    "source": "v1_og_rfps.csv",
                }
            )

    return faculty_context, rfp_items


def load_teaming(args: argparse.Namespace) -> DomainData:
    researchers_df = read_csv(args.researchers_csv)
    proposals_df = read_csv(args.proposals_csv)
    faculty_context, rfp_items = load_usc_reference_maps(args)

    researchers = []
    for _, row in researchers_df.iterrows():
        name = clean_text(first_value(row, "names", "name", "FacultyName"))
        if not name:
            continue
        skills = parse_collection(first_value(row, "research", "Research Interests"), max_chars=args.max_field_chars)
        skills.extend(faculty_context.get(normalize_name(name), []))
        researchers.append(
            {
                "name": name,
                "title": clean_text(first_value(row, "titles", "designation"), args.max_field_chars),
                "description": clean_text(first_value(row, "descriptions", "background"), args.max_field_chars),
                "skills": sorted(dict.fromkeys(skills)),
            }
        )

    primary_items = []
    for _, row in proposals_df.iterrows():
        title = clean_text(first_value(row, "title", "Title"), args.max_field_chars)
        if not title:
            continue
        primary_items.append(
            {
                "id": clean_text(first_value(row, "nsf_proposal_links_v1", "reference")),
                "title": title,
                "summary": clean_text(first_value(row, "synopsis", "details"), args.max_field_chars),
                "skills": parse_collection(first_value(row, "keywords"), max_chars=args.max_field_chars),
                "source": Path(args.proposals_csv).name,
            }
        )

    if args.proposal_source == "primary":
        proposals = primary_items
    elif args.proposal_source == "rfps":
        proposals = rfp_items
    else:
        proposals = primary_items + rfp_items

    return DomainData(
        domain="teaming",
        kind="teaming",
        targets=researchers,
        candidates=researchers,
        items=proposals,
        output_columns=["Anchor Name", "Proposal Name", "Team Recommended"],
    )


def load_iitr_teaming(args: argparse.Namespace) -> DomainData:
    researchers_df = read_csv(args.iitr_researchers_csv)
    proposals_df = read_csv(args.iitr_proposals_csv)

    researchers = []
    for _, row in researchers_df.iterrows():
        name = clean_text(first_value(row, "FacultyName", "names", "name"))
        if not name:
            continue
        researchers.append(
            {
                "name": name,
                "title": "",
                "description": "",
                "skills": parse_collection(
                    first_value(row, "Research Interests", "research"),
                    max_chars=args.max_field_chars,
                ),
            }
        )

    proposals = []
    for _, row in proposals_df.iterrows():
        title = clean_text(first_value(row, "Title", "title"), args.max_field_chars)
        if not title:
            continue
        proposals.append(
            {
                "id": title,
                "title": title,
                "summary": clean_text(
                    first_value(row, "Attachment Text", "synopsis", "details"),
                    args.max_field_chars,
                ),
                "skills": [],
                "source": Path(args.iitr_proposals_csv).name,
            }
        )

    return DomainData(
        domain="iitr-teaming",
        kind="teaming",
        targets=researchers,
        candidates=researchers,
        items=proposals,
        output_columns=["Anchor Name", "Proposal Name", "Team Recommended"],
    )


def choose_targets(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = records
    if args.target_name:
        needle = args.target_name.lower()
        selected = [
            record
            for record in records
            if needle in clean_text(record.get("name")).lower()
            or needle in clean_text(record.get("id")).lower()
        ]
        if not selected:
            raise ValueError(f"No target matched --target-name {args.target_name!r}")

    if args.sample_targets:
        rng = random.Random(args.seed)
        selected = rng.sample(selected, min(len(selected), args.limit_targets))
    else:
        selected = selected[: args.limit_targets]
    return selected


def limit_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return records
    return records[:limit]


def ensure_targets_in_candidates(
    targets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected = limit_records(candidates, limit)
    seen = {normalize_name(item.get("name")) for item in selected}
    for target in targets:
        key = normalize_name(target.get("name"))
        if key and key not in seen:
            selected.insert(0, target)
            seen.add(key)
    return selected


def build_teaming_prompt(
    domain: DomainData,
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    professor_lines = []
    for person in candidates:
        line = (
            f"- {person['name']}"
            f" | title: {person.get('title') or 'none listed'}"
            f" | skills: {format_list(person.get('skills', []))}"
        )
        if person.get("description"):
            line += f" | description: {person['description']}"
        professor_lines.append(line)

    proposal_lines = []
    for item in proposals:
        line = f"- {item['title']}"
        if item.get("id"):
            line += f" | id: {item['id']}"
        if item.get("skills"):
            line += f" | skills: {format_list(item['skills'])}"
        if item.get("summary"):
            line += f" | summary: {item['summary']}"
        proposal_lines.append(line)

    return f"""You are an LLM recommendation baseline for research team formation.

Domain: {domain.domain}
Task: Recommend proposal-team matches for the target anchor professor.

Return semicolon-delimited CSV only. Do not use markdown fences or commentary.
Required header:
Anchor Name;Proposal Name;Team Recommended

Rules:
- Generate {args.recommendations_per_target} rows for the target anchor.
- Use only proposal names from Available Proposals.
- Use only professor names from Available Professors.
- Team Recommended must be a comma-separated list of 2 to {args.max_team_size} professors.
- Include the target anchor in every recommended team.
- Do not put semicolons inside Team Recommended.
- Prefer teams whose combined skills cover the proposal needs.

Target Anchor:
Name: {target['name']}
Title: {target.get('title') or 'none listed'}
Skills: {format_list(target.get('skills', []))}
Description: {target.get('description') or 'none listed'}

Available Professors:
{chr(10).join(professor_lines)}

Available Proposals:
{chr(10).join(proposal_lines)}
"""


def build_meal_prompt(
    domain: DomainData,
    target: dict[str, Any],
    meals: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    meal_lines = [
        f"- {meal['name']} | categories: {format_list(meal.get('categories', []), max_items=8)}"
        for meal in meals
    ]

    return f"""You are an LLM recommendation baseline for meal bundle recommendation.

Domain: {domain.domain}
Task: Recommend meal bundles for the target user request.

Return semicolon-delimited CSV only. Do not use markdown fences or commentary.
Required header:
User ID;User Name;Meal Occasion;Meal Bundle Recommended

Rules:
- Generate {args.recommendations_per_target} rows for the target user.
- Use only meal names from Available Meals.
- Meal Bundle Recommended must be a comma-separated list of 2 to {args.max_bundle_size} meals.
- Do not put semicolons inside Meal Bundle Recommended.
- Match the required categories as well as possible.
- Do not use demographic fields as recommendation criteria.

Target User Request:
User ID: {target['id']}
User Name: {target['name']}
Meal Occasion: {target['meal_occasion']}
Required Categories: {format_list(target.get('required_categories', []), max_items=8)}

Available Meals:
{chr(10).join(meal_lines)}
"""


def build_tasks(domain: DomainData, args: argparse.Namespace) -> list[PromptTask]:
    targets = choose_targets(domain.targets, args)
    tasks = []

    if domain.kind == "teaming":
        candidates = ensure_targets_in_candidates(targets, domain.candidates, args.limit_candidates)
        proposals = limit_records(domain.items, args.limit_items)
        for index, target in enumerate(targets, start=1):
            prompt = build_teaming_prompt(domain, target, candidates, proposals, args)
            tasks.append(
                PromptTask(
                    task_id=f"{index:04d}",
                    target_name=target["name"],
                    prompt=prompt,
                    metadata={
                        "domain": domain.domain,
                        "target": target,
                        "candidate_count": len(candidates),
                        "proposal_count": len(proposals),
                    },
                )
            )
    else:
        meals = limit_records(domain.items, args.limit_items)
        for index, target in enumerate(targets, start=1):
            prompt = build_meal_prompt(domain, target, meals, args)
            tasks.append(
                PromptTask(
                    task_id=f"{index:04d}",
                    target_name=target["name"],
                    prompt=prompt,
                    metadata={
                        "domain": domain.domain,
                        "target": target,
                        "meal_count": len(meals),
                    },
                )
            )

    return tasks


def model_id_for(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return MODEL_ALIASES[args.model]


def api_key_for(args: argparse.Namespace) -> str:
    for env_name in args.api_key_env:
        value = os.environ.get(env_name)
        if value:
            return value
    names = ", ".join(args.api_key_env)
    raise RuntimeError(f"Missing API key. Set one of: {names}")


def call_google_genai(prompt: str, args: argparse.Namespace) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-genai package. Install it before running with --run."
        ) from exc

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


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:csv|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_semicolon_csv(text: str, expected_columns: list[str]) -> list[dict[str, str]]:
    cleaned = strip_markdown_fences(text)
    reader = csv.reader(io.StringIO(cleaned), delimiter=";")
    rows: list[dict[str, str]] = []
    expected_lower = [column.lower() for column in expected_columns]

    for raw_row in reader:
        parts = [part.strip() for part in raw_row if part.strip()]
        if not parts:
            continue
        if [part.lower() for part in parts[: len(expected_columns)]] == expected_lower:
            continue
        if len(parts) < len(expected_columns):
            continue
        if len(parts) > len(expected_columns):
            parts = parts[: len(expected_columns) - 1] + [
                ", ".join(parts[len(expected_columns) - 1 :])
            ]
        rows.append(dict(zip(expected_columns, parts, strict=True)))
    return rows


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "target"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = repo_path(args.output_dir)
    output_dir = base / args.domain / args.model / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_domain(args: argparse.Namespace) -> DomainData:
    if args.domain == "meal":
        return load_meal(args)
    if args.domain == "teaming":
        return load_teaming(args)
    if args.domain == "iitr-teaming":
        return load_iitr_teaming(args)
    raise ValueError(f"Unsupported domain: {args.domain}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally run LLM recommendation prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domain",
        choices=["teaming", "iitr-teaming", "meal"],
        required=True,
        help="Dataset/domain to load.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_ALIASES),
        default="gemini-3-flash-live",
        help="Model alias.",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help="Override the model ID sent to the API.",
    )
    parser.add_argument(
        "--backend",
        choices=["google-genai"],
        default="google-genai",
        help="Serving backend.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Call the model. Without this flag, only prompt files are written.",
    )
    parser.add_argument(
        "--api-key-env",
        nargs="+",
        default=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        help="Environment variable names checked for the API key.",
    )
    parser.add_argument("--output-dir", default="llm_outputs")
    parser.add_argument("--target-name", default="", help="Substring filter for a target.")
    parser.add_argument("--limit-targets", type=int, default=1)
    parser.add_argument("--limit-candidates", type=int, default=60)
    parser.add_argument("--limit-items", type=int, default=40)
    parser.add_argument("--sample-targets", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recommendations-per-target", type=int, default=10)
    parser.add_argument("--max-team-size", type=int, default=5)
    parser.add_argument("--max-bundle-size", type=int, default=5)
    parser.add_argument("--max-field-chars", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--sleep-seconds", type=float, default=4.1)

    parser.add_argument(
        "--researchers-csv",
        default="Teaming/data/v1_input_files/v1_researchers.csv",
        help="USC Teaming researcher CSV.",
    )
    parser.add_argument(
        "--proposals-csv",
        default="Teaming/data/v1_input_files/v1_proposal_links_title_synopsis.csv",
        help="USC Teaming primary proposal CSV.",
    )
    parser.add_argument(
        "--faculty-csv",
        default="Teaming/data/v1_input_files/v1_og_faculty.csv",
        help="USC Teaming optional faculty metadata CSV.",
    )
    parser.add_argument(
        "--rfps-csv",
        default="Teaming/data/v1_input_files/v1_og_rfps.csv",
        help="USC Teaming optional original RFP CSV.",
    )
    parser.add_argument(
        "--interests-csv",
        default="Teaming/data/v1_input_files/usc_combined_interests.csv",
        help="USC Teaming optional combined interests CSV.",
    )
    parser.add_argument(
        "--proposal-source",
        choices=["primary", "rfps", "combined"],
        default="primary",
        help="Which USC Teaming proposal input format to prompt with.",
    )

    parser.add_argument(
        "--iitr-researchers-csv",
        default="IITR-Teaming/data/v0_data/researchers.csv",
        help="IITR Teaming researcher CSV.",
    )
    parser.add_argument(
        "--iitr-proposals-csv",
        default="IITR-Teaming/data/v0_data/archive_proposals.csv",
        help="IITR Teaming proposal CSV.",
    )

    parser.add_argument(
        "--meal-categories-csv",
        default="Meal/data/input_data/meal_categories.csv",
        help="Meal catalog CSV.",
    )
    parser.add_argument(
        "--meal-requirements-csv",
        default="Meal/data/input_data/user_meal_requirements.csv",
        help="Meal user requirements CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    random.seed(args.seed)

    domain = load_domain(args)
    tasks = build_tasks(domain, args)
    if not tasks:
        raise RuntimeError("No prompt tasks were built.")

    output_dir = make_output_dir(args)
    manifest_rows = []
    parsed_rows: list[dict[str, str]] = []

    for task in tasks:
        prompt_path = output_dir / f"prompt_{task.task_id}_{safe_name(task.target_name)}.txt"
        prompt_path.write_text(task.prompt, encoding="utf-8")

        manifest = {
            "task_id": task.task_id,
            "domain": domain.domain,
            "model_alias": args.model,
            "model_id": model_id_for(args),
            "target_name": task.target_name,
            "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
            "metadata": task.metadata,
        }

        if args.run:
            raw_text = call_google_genai(task.prompt, args)
            raw_path = output_dir / f"raw_{task.task_id}_{safe_name(task.target_name)}.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            rows = parse_semicolon_csv(raw_text, domain.output_columns)
            for row in rows:
                row["Domain"] = domain.domain
                row["Model"] = args.model
                row["Model ID"] = model_id_for(args)
                row["Task ID"] = task.task_id
            parsed_rows.extend(rows)
            manifest["raw_path"] = str(raw_path.relative_to(REPO_ROOT))
            manifest["parsed_rows"] = len(rows)
            time.sleep(args.sleep_seconds)

        manifest_rows.append(manifest)

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)

    if args.run:
        output_columns = domain.output_columns + ["Domain", "Model", "Model ID", "Task ID"]
        write_csv(output_dir / "recommendations.csv", parsed_rows, output_columns)
        print(f"Wrote {len(parsed_rows)} parsed recommendations to {output_dir / 'recommendations.csv'}")
    else:
        print(f"Dry run complete. Wrote {len(tasks)} prompt file(s) to {output_dir}")
        print("Use --run to call the configured model in the next phase.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
