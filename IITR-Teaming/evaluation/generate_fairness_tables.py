"""
Generate fairness tables for IITR-Teaming in the style of the fairness paper.

This script mirrors the Teaming fairness flow, but aligns rows on the IITR
schema: (proposal_link, researcher_name). It evaluates gender fairness of the
top recommended team for each method and exports paper-style Table 2 / Table 3
CSV files plus support metadata.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import gender_guesser.detector as gender

try:
    from genderize import Genderize
except Exception:  # pragma: no cover - optional dependency
    Genderize = None

try:
    import sexmachine.detector as sexmachine_detector
    import sexmachine.mapping as sexmachine_mapping
except Exception:  # pragma: no cover - optional dependency
    sexmachine_detector = None
    sexmachine_mapping = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_SPECS = [
    ("M0", "Random baseline", "teaming_uc1_m0.csv"),
    ("M1", "String / lexical matching", "teaming_uc1_m1.csv"),
    ("M2", "Semantic mapper matching", "teaming_uc1_m2.csv"),
    ("M3", "BoostSRL / relational learning", "teaming_uc1_m3.csv"),
    ("M6", "Exact knapsack coverage", "teaming_uc1_m6.csv"),
    ("M7", "Marginal-utility knapsack", "teaming_uc1_m7.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "v0_teaming"),
        help="Directory containing the canonical IITR Teaming UC1 CSVs.",
    )
    parser.add_argument(
        "--researchers-file",
        default=str(PROJECT_ROOT / "data" / "v0_data" / "researchers.csv"),
        help="IITR researcher roster used for name-based gender inference context.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "evaluation" / "output_v0"),
        help="Directory where fairness outputs will be written.",
    )
    parser.add_argument(
        "--name-inference-mode",
        choices=["basic", "paper_ensemble"],
        default="paper_ensemble",
        help="How to infer gender from faculty first names.",
    )
    parser.add_argument(
        "--conditioning-attribute",
        default="proposal_link",
        help="Decision-frame attribute used for CSP. Use proposal_link or title for IITR.",
    )
    parser.add_argument(
        "--subset-proposals",
        type=int,
        default=0,
        help="Optional: keep only the first N proposals by appearance before computing fairness.",
    )
    parser.add_argument(
        "--subset-anchor-researchers",
        type=int,
        default=0,
        help="Optional: keep only the first K anchor researchers by appearance after proposal filtering.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=4,
        help="Number of decimal places for saved fairness tables.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return " ".join((name or "").replace("\xa0", " ").strip().split()).lower()


def extract_first_name(full_name: str) -> str:
    cleaned = re.sub(r"\b(dr|prof|mr|mrs|ms)\.?\b", " ", full_name or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = cleaned.replace(".", " ").replace("-", " ").replace("/", " ")
    tokens = [token for token in re.split(r"[^A-Za-z]+", cleaned) if token]
    skip = {"jr", "sr", "phd", "md"}
    tokens = [token for token in tokens if token.lower() not in skip and len(token) > 1]
    return tokens[0].lower() if tokens else ""


def pronoun_gender(text: str) -> str | None:
    text = (text or "").lower()
    has_she = bool(re.search(r"\b(she|her)\b", text))
    has_he = bool(re.search(r"\b(he|his)\b", text))
    if has_she and not has_he:
        return "female"
    if has_he and not has_she:
        return "male"
    return None


def normalize_library_gender(raw_value: object) -> str | None:
    raw = str(raw_value or "").strip().lower()
    if raw in {"male", "mostly_male"}:
        return "male"
    if raw in {"female", "mostly_female"}:
        return "female"
    return None


def build_sexmachine_detector():
    if sexmachine_detector is None or sexmachine_mapping is None:
        return None

    def safe_map_name(value: str) -> str:
        text = value or ""
        for code, patterns in sexmachine_mapping.mappings:
            for pattern in patterns:
                text = text.replace(pattern, chr(code))
        return text

    def safe_most_popular_gender(self, name, counter):
        if name not in self.names:
            return self.unknown_value

        max_count = 0
        max_tie = 0
        best = next(iter(self.names[name]))
        for gender_name, country_values in self.names[name].items():
            count, tie = counter(country_values)
            if count > max_count or (count == max_count and tie > max_tie):
                max_count, max_tie, best = count, tie, gender_name
        return best if max_count > 0 else self.unknown_value

    def safe_get_gender(self, name, country=None):
        candidate = (name or "").lower() if not self.case_sensitive else (name or "")
        if candidate not in self.names:
            return self.unknown_value
        if not country:
            def counter(country_values):
                weights = [ord(char) for char in country_values.replace(" ", "")]
                return (
                    len(weights),
                    sum((value - 55) if value > 64 else (value - 48) for value in weights),
                )

            return safe_most_popular_gender(self, candidate, counter)
        if country in self.__class__.COUNTRIES:
            index = self.__class__.COUNTRIES.index(country)
            return safe_most_popular_gender(self, candidate, lambda values: (ord(values[index]) - 32, 0))
        raise sexmachine_detector.NoCountryError(f"No such country: {country}")

    sexmachine_mapping.map_name = safe_map_name
    sexmachine_detector.map_name = safe_map_name
    sexmachine_detector.Detector._most_popular_gender = safe_most_popular_gender
    sexmachine_detector.Detector.get_gender = safe_get_gender

    try:
        return sexmachine_detector.Detector(case_sensitive=False)
    except Exception:
        return None


def lookup_genderize(first_names: list[str]) -> dict[str, dict[str, object]]:
    if Genderize is None:
        return {}

    responses: dict[str, dict[str, object]] = {}
    client = Genderize()
    normalized_names = [name for name in sorted(set(first_names)) if name]

    for start in range(0, len(normalized_names), 10):
        chunk = normalized_names[start : start + 10]
        try:
            payload = client.get(chunk)
        except Exception:
            for name in chunk:
                responses[name] = {"gender": None, "probability": None, "count": None}
            continue

        for entry in payload:
            key = str(entry.get("name", "")).strip().lower()
            probability = entry.get("probability")
            responses[key] = {
                "gender": normalize_library_gender(entry.get("gender")),
                "probability": float(probability) if probability is not None else None,
                "count": entry.get("count"),
            }

    return responses


def resolve_ensemble_gender(
    gender_guesser_vote: str | None,
    genderize_vote: str | None,
    genderize_probability: float | None,
    sexmachine_vote: str | None,
    pronoun_vote: str | None,
) -> tuple[str, str]:
    votes = [vote for vote in [gender_guesser_vote, genderize_vote, sexmachine_vote] if vote in {"male", "female"}]
    counts = Counter(votes)
    if counts:
        top_vote, top_count = counts.most_common(1)[0]
        tied = sum(1 for count in counts.values() if count == top_count) > 1
        if not tied:
            return top_vote, "ensemble_majority"

    if genderize_vote in {"male", "female"} and (genderize_probability or 0.0) >= 0.9:
        return genderize_vote, "genderize_strong"
    if gender_guesser_vote in {"male", "female"}:
        return gender_guesser_vote, "gender_guesser_fallback"
    if sexmachine_vote in {"male", "female"}:
        return sexmachine_vote, "sexmachine_fallback"
    if pronoun_vote in {"male", "female"}:
        return pronoun_vote, "pronoun_fallback"
    if genderize_vote in {"male", "female"}:
        return genderize_vote, "genderize_fallback"
    return "unknown", "unknown"


def round_frame(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def apply_method_display_headers(df: pd.DataFrame, method_meta: dict[str, dict[str, str]]) -> pd.DataFrame:
    renamed = df.copy()
    display_map = {
        method_label: f"{method_label}: {method_meta[method_label]['name']}"
        for method_label in method_meta
        if method_label in renamed.columns
    }
    return renamed.rename(columns=display_map)


def smooth_rate(positive: int, total: int) -> float:
    return (positive + 0.5) / (total + 1.0)


def dp_gap_smoothed(df: pd.DataFrame, pred_col: str, group_col: str, groups: list[str]) -> float:
    rates = []
    for group in groups:
        mask = df[group_col] == group
        rates.append(smooth_rate(int(df.loc[mask, pred_col].sum()), int(mask.sum())))
    return abs(rates[0] - rates[1])


def csp_gap_smoothed(
    df: pd.DataFrame,
    pred_col: str,
    group_col: str,
    groups: list[str],
    conditioning_col: str,
) -> float:
    weighted_gap = 0.0
    total_weight = 0
    for _, subset in df.groupby(conditioning_col):
        if any((subset[group_col] == group).sum() == 0 for group in groups):
            continue
        gap = dp_gap_smoothed(subset, pred_col, group_col, groups)
        weighted_gap += len(subset) * gap
        total_weight += len(subset)
    return weighted_gap / total_weight if total_weight else float("nan")


def confusion_counts(df: pd.DataFrame, truth_col: str, pred_col: str, group_col: str, group: str) -> dict[str, int]:
    mask = df[group_col] == group
    return {
        "tp": int(((df[pred_col] == 1) & (df[truth_col] == 1) & mask).sum()),
        "fp": int(((df[pred_col] == 1) & (df[truth_col] == 0) & mask).sum()),
        "fn": int(((df[pred_col] == 0) & (df[truth_col] == 1) & mask).sum()),
        "tn": int(((df[pred_col] == 0) & (df[truth_col] == 0) & mask).sum()),
    }


def eo_gap_smoothed(df: pd.DataFrame, truth_col: str, pred_col: str, group_col: str, groups: list[str]) -> float:
    stats = {}
    for group in groups:
        counts = confusion_counts(df, truth_col, pred_col, group_col, group)
        stats[group] = {
            "tpr": smooth_rate(counts["tp"], counts["tp"] + counts["fn"]),
            "fpr": smooth_rate(counts["fp"], counts["fp"] + counts["tn"]),
        }
    return 0.5 * (
        abs(stats[groups[0]]["tpr"] - stats[groups[1]]["tpr"])
        + abs(stats[groups[0]]["fpr"] - stats[groups[1]]["fpr"])
    )


def prp_gap_smoothed(df: pd.DataFrame, truth_col: str, pred_col: str, group_col: str, groups: list[str]) -> float:
    ppv = {}
    for group in groups:
        counts = confusion_counts(df, truth_col, pred_col, group_col, group)
        ppv[group] = smooth_rate(counts["tp"], counts["tp"] + counts["fp"])
    return abs(ppv[groups[0]] - ppv[groups[1]])


def te_gap_smoothed(df: pd.DataFrame, truth_col: str, pred_col: str, group_col: str, groups: list[str]) -> float:
    ratios = {}
    for group in groups:
        counts = confusion_counts(df, truth_col, pred_col, group_col, group)
        ratios[group] = (counts["fn"] + 0.5) / (counts["fp"] + 0.5)
    return abs(ratios[groups[0]] - ratios[groups[1]])


def parse_top_team(team_text: str, goodness_text: str) -> tuple[list[str], float]:
    teams = ast.literal_eval(team_text)
    goodness = [float(value) for value in ast.literal_eval(goodness_text)]
    if not teams or not goodness:
        return [], float("nan")
    best_idx = max(range(len(goodness)), key=lambda idx: goodness[idx])
    return list(teams[best_idx]), float(goodness[best_idx])


def load_profiles(researchers_file: Path) -> dict[str, list[str]]:
    profiles: dict[str, list[str]] = defaultdict(list)
    researchers = pd.read_csv(researchers_file)

    name_col = next((col for col in ["FacultyName", "names", "name"] if col in researchers.columns), None)
    if name_col is None:
        raise ValueError(f"Could not find a faculty-name column in {researchers_file}")

    text_cols = [
        col
        for col in [
            "Research Interests",
            "research",
            "descriptions",
            "titles",
            "background",
            "area",
        ]
        if col in researchers.columns
    ]

    for _, row in researchers.iterrows():
        text = " ".join(str(row.get(col, "")) for col in text_cols if str(row.get(col, "")) != "nan")
        profiles[normalize_name(str(row[name_col]))].append(text)

    return profiles


def infer_gender_map(names: set[str], profiles: dict[str, list[str]], mode: str) -> pd.DataFrame:
    detector = gender.Detector(case_sensitive=False)
    sexmachine = build_sexmachine_detector() if mode == "paper_ensemble" else None
    first_names = {extract_first_name(raw_name) for raw_name in names}
    genderize_lookup = lookup_genderize(list(first_names)) if mode == "paper_ensemble" else {}
    rows = []

    for raw_name in sorted(names):
        key = normalize_name(raw_name)
        texts = profiles.get(key, [])
        profile_text = " ".join(texts)
        pronoun_guess = pronoun_gender(profile_text)
        first_name = extract_first_name(raw_name)
        gender_guesser_raw = detector.get_gender(first_name) if first_name else "unknown"
        gender_guesser_vote = normalize_library_gender(gender_guesser_raw)

        if mode == "paper_ensemble":
            genderize_payload = genderize_lookup.get(first_name, {})
            genderize_vote = genderize_payload.get("gender")
            genderize_probability = genderize_payload.get("probability")
            sexmachine_raw = sexmachine.get_gender(first_name) if sexmachine is not None and first_name else "unknown"
            sexmachine_vote = normalize_library_gender(sexmachine_raw)
            final_gender, source = resolve_ensemble_gender(
                gender_guesser_vote,
                genderize_vote,
                genderize_probability,
                sexmachine_vote,
                pronoun_guess,
            )
        else:
            genderize_vote = None
            genderize_probability = None
            sexmachine_raw = "not_used"
            sexmachine_vote = None
            final_gender = pronoun_guess or gender_guesser_vote or "unknown"
            source = "pronoun_fallback" if pronoun_guess and not gender_guesser_vote else ("gender_guesser_only" if gender_guesser_vote else "unknown")

        rows.append(
            {
                "researcher_name": raw_name,
                "researcher_name_norm": key,
                "first_name": first_name,
                "gender_guesser_raw": gender_guesser_raw,
                "gender_guesser_vote": gender_guesser_vote or "unknown",
                "genderize_vote": genderize_vote or "unknown",
                "genderize_probability": genderize_probability,
                "sexmachine_raw": sexmachine_raw,
                "sexmachine_vote": sexmachine_vote or "unknown",
                "pronoun_vote": pronoun_guess or "unknown",
                "gender": final_gender,
                "source": source,
            }
        )

    return pd.DataFrame(rows)


def load_method_frame(output_dir: Path, filename: str, method_label: str, method_name: str) -> pd.DataFrame:
    path = output_dir / filename
    df = pd.read_csv(path)
    top_payload = [parse_top_team(team_text, goodness_text) for team_text, goodness_text in zip(df["team"], df["goodness"])]
    df[f"{method_label}_top_team"] = [payload[0] for payload in top_payload]
    df[f"{method_label}_top_goodness"] = [payload[1] for payload in top_payload]

    keep = df[["proposal_link", "title", "researcher_name", f"{method_label}_top_team", f"{method_label}_top_goodness"]].copy()
    keep["researcher_name_norm"] = keep["researcher_name"].map(normalize_name)
    keep = keep.rename(columns={"title": f"{method_label}_title"})
    keep.attrs["method_name"] = method_name
    return keep


def build_aligned_rows(output_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    frames = []
    meta = {}
    for method_label, method_name, filename in METHOD_SPECS:
        path = output_dir / filename
        if not path.exists():
            continue
        frame = load_method_frame(output_dir, filename, method_label, method_name)
        frames.append(frame)
        meta[method_label] = {"name": method_name, "filename": filename}

    if len(frames) < 2:
        raise ValueError(f"Expected at least two method outputs in {output_dir}, found {len(frames)}")

    aligned = frames[0]
    for frame in frames[1:]:
        aligned = aligned.merge(
            frame,
            on=["proposal_link", "researcher_name", "researcher_name_norm"],
            how="inner",
        )

    title_cols = [column for column in aligned.columns if column.endswith("_title")]
    aligned["title"] = aligned[title_cols].bfill(axis=1).iloc[:, 0]
    aligned = aligned.drop(columns=title_cols)
    return aligned, meta


def apply_subset(aligned: pd.DataFrame, proposal_count: int, researcher_count: int) -> tuple[pd.DataFrame, dict[str, int] | None]:
    subset = aligned.copy()
    metadata = None

    if proposal_count > 0:
        proposal_links = pd.Index(subset["proposal_link"]).drop_duplicates().tolist()[:proposal_count]
        subset = subset[subset["proposal_link"].isin(proposal_links)].copy()

    if researcher_count > 0:
        anchor_researchers = pd.Index(subset["researcher_name_norm"]).drop_duplicates().tolist()[:researcher_count]
        subset = subset[subset["researcher_name_norm"].isin(anchor_researchers)].copy()

    if subset.empty:
        raise ValueError("Subset filtering removed every aligned IITR row.")

    if proposal_count > 0 or researcher_count > 0:
        metadata = {
            "requested_proposals": proposal_count,
            "actual_proposals": int(subset["proposal_link"].nunique()),
            "requested_anchor_researchers": researcher_count,
            "actual_anchor_researchers": int(subset["researcher_name_norm"].nunique()),
            "description": "Optional IITR subset by proposal appearance and anchor-researcher appearance.",
        }

    return subset, metadata


def build_population_names(aligned: pd.DataFrame, method_meta: dict[str, dict[str, str]]) -> set[str]:
    names: set[str] = set(aligned["researcher_name"].tolist())
    for method_label in method_meta:
        for teams in aligned[f"{method_label}_top_team"]:
            names.update(teams)
    return names


def build_table2(
    aligned: pd.DataFrame,
    gender_map: dict[str, str],
    groups: list[str],
    method_meta: dict[str, dict[str, str]],
) -> pd.DataFrame:
    population_names = {name for name in gender_map if gender_map[name] in groups}
    population_counts = Counter(gender_map[name] for name in population_names)
    population_total = sum(population_counts.values())
    population_distribution = {
        group: population_counts[group] / population_total if population_total else float("nan")
        for group in groups
    }

    rows = []
    for group in groups:
        row = {
            "Group": group.capitalize(),
            "Population Distribution": population_distribution[group],
        }
        for method_label in method_meta:
            counts = Counter()
            for team in aligned[f"{method_label}_top_team"]:
                for member in team:
                    inferred = gender_map.get(normalize_name(member), "unknown")
                    if inferred in groups:
                        counts[inferred] += 1
            total = sum(counts.values())
            row[method_label] = counts[group] / total if total else float("nan")
        rows.append(row)

    sp_row = {"Group": "SP", "Population Distribution": float("nan")}
    for method_label in method_meta:
        shares = {group: rows[idx][method_label] for idx, group in enumerate(groups)}
        sp_row[method_label] = 0.5 * sum(abs(shares[group] - population_distribution[group]) for group in groups)
    rows.append(sp_row)

    return pd.DataFrame(rows)


def build_decision_frame(
    aligned: pd.DataFrame,
    gender_map: dict[str, str],
    groups: list[str],
    method_meta: dict[str, dict[str, str]],
) -> pd.DataFrame:
    decision_rows = []
    method_order = list(method_meta.keys())

    for _, row in aligned.iterrows():
        top_teams = {method: list(row[f"{method}_top_team"]) for method in method_order}
        top_scores = {method: float(row[f"{method}_top_goodness"]) for method in method_order}
        best_score = max(top_scores.values())
        oracle_methods = [
            method for method in method_order if math.isclose(top_scores[method], best_score, rel_tol=1e-9, abs_tol=1e-9)
        ]

        oracle_members = set()
        candidate_members = set()
        for method in method_order:
            candidate_members.update(top_teams[method])
            if method in oracle_methods:
                oracle_members.update(top_teams[method])

        for member in candidate_members:
            inferred_gender = gender_map.get(normalize_name(member), "unknown")
            if inferred_gender not in groups:
                continue

            payload = {
                "proposal_link": row["proposal_link"],
                "title": row["title"],
                "anchor_researcher": row["researcher_name"],
                "candidate_member": member,
                "gender": inferred_gender,
                "oracle_positive": int(member in oracle_members),
            }
            for method in method_order:
                payload[f"{method}_pred_positive"] = int(member in set(top_teams[method]))
                payload[f"{method}_top_goodness"] = top_scores[method]
            decision_rows.append(payload)

    return pd.DataFrame(decision_rows)


def build_table3_and_support(
    decisions: pd.DataFrame,
    groups: list[str],
    conditioning_col: str,
    method_meta: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table3_rows = {
        "SP": {"Metric": "SP"},
        "CSP": {"Metric": "CSP"},
        "EO": {"Metric": "EO"},
        "PRP": {"Metric": "PRP"},
        "TE": {"Metric": "TE"},
    }
    support_rows = []

    for method_label in method_meta:
        pred_col = f"{method_label}_pred_positive"
        table3_rows["SP"][method_label] = dp_gap_smoothed(decisions, pred_col, "gender", groups)
        table3_rows["CSP"][method_label] = csp_gap_smoothed(decisions, pred_col, "gender", groups, conditioning_col)
        table3_rows["EO"][method_label] = eo_gap_smoothed(decisions, "oracle_positive", pred_col, "gender", groups)
        table3_rows["PRP"][method_label] = prp_gap_smoothed(decisions, "oracle_positive", pred_col, "gender", groups)
        table3_rows["TE"][method_label] = te_gap_smoothed(decisions, "oracle_positive", pred_col, "gender", groups)

        support_row = {
            "Method": method_label,
            "Method_name": method_meta[method_label]["name"],
            "decision_rows": int(len(decisions)),
            "predicted_positive_rate": float(decisions[pred_col].mean()),
            "oracle_positive_rate": float(decisions["oracle_positive"].mean()),
            "predicted_positive_count": int(decisions[pred_col].sum()),
            "oracle_positive_count": int(decisions["oracle_positive"].sum()),
            "top_goodness_mean": float(decisions[f"{method_label}_top_goodness"].mean()),
        }
        for group in groups:
            mask = decisions["gender"] == group
            support_row[f"{group}_predicted_positive_rate"] = float(decisions.loc[mask, pred_col].mean())
        support_rows.append(support_row)

    table3 = pd.DataFrame([table3_rows[key] for key in ["SP", "CSP", "EO", "PRP", "TE"]])
    support = pd.DataFrame(support_rows)
    return table3, support


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    aligned, method_meta = build_aligned_rows(output_dir)
    aligned, subset_meta = apply_subset(aligned, args.subset_proposals, args.subset_anchor_researchers)

    profiles = load_profiles(Path(args.researchers_file))
    population_names = build_population_names(aligned, method_meta)
    inference_df = infer_gender_map(population_names, profiles, args.name_inference_mode)
    gender_map = dict(zip(inference_df["researcher_name_norm"], inference_df["gender"]))

    resolved_df = inference_df[inference_df["gender"].isin(["male", "female"])].copy()
    groups = resolved_df["gender"].value_counts().index.tolist()
    if set(groups) != {"male", "female"}:
        raise ValueError(f"Expected binary gender groups after inference; found {groups}")
    groups = ["male", "female"]

    table2 = build_table2(aligned, gender_map, groups, method_meta)
    decisions = build_decision_frame(aligned, gender_map, groups, method_meta)

    if args.conditioning_attribute not in decisions.columns:
        raise ValueError(
            f"Conditioning attribute '{args.conditioning_attribute}' is not present. "
            f"Available columns: {sorted(decisions.columns.tolist())}"
        )

    table3, support = build_table3_and_support(decisions, groups, args.conditioning_attribute, method_meta)
    table2_display = apply_method_display_headers(table2, method_meta)
    table3_display = apply_method_display_headers(table3, method_meta)
    support_display = support.copy()
    support_display["Method_display"] = support_display.apply(
        lambda row: f"{row['Method']}: {row['Method_name']}",
        axis=1,
    )

    table2_path = results_dir / "fairness_table2_gender.csv"
    table3_path = results_dir / "fairness_table3_gender.csv"
    support_path = results_dir / "fairness_support_gender.csv"
    inference_path = results_dir / "fairness_gender_inference.csv"
    metadata_path = results_dir / "fairness_metadata_gender.json"

    round_frame(table2_display, args.round_digits).to_csv(table2_path, index=False)
    round_frame(table3_display, args.round_digits).to_csv(table3_path, index=False)
    round_frame(support_display, args.round_digits).to_csv(support_path, index=False)
    inference_df.to_csv(inference_path, index=False)

    metadata = {
        "project": "IITR-Teaming",
        "dataset": str(output_dir),
        "aligned_rows": int(len(aligned)),
        "decision_rows": int(len(decisions)),
        "protected_attribute": "gender",
        "protected_groups": groups,
        "conditioning_attribute": args.conditioning_attribute,
        "name_inference_mode": args.name_inference_mode,
        "gender_inference": {
            "resolved_count": int(inference_df["gender"].isin(groups).sum()),
            "unresolved_count": int((~inference_df["gender"].isin(groups)).sum()),
            "source_counts": inference_df["source"].value_counts().to_dict(),
            "description": (
                "Gender is inferred from first names using a paper-style ensemble of "
                "gender-guesser, Genderize, and SexMachine, with local research-interest "
                "text used only for pronoun fallback when available."
                if args.name_inference_mode == "paper_ensemble"
                else "Gender is inferred from gender-guesser, with local research-interest fallback when available."
            ),
        },
        "table2_interpretation": (
            "Table 2 compares the inferred gender distribution of faculty appearing in each "
            "method's top recommended team against the inferred IITR faculty baseline."
        ),
        "table3_interpretation": (
            "Table 3 operates on member-selection decisions. For each aligned "
            "(proposal_link, researcher_name) row, the oracle label is the union of members in "
            "the highest-goodness top team(s) across methods; each method predicts the members "
            "in its own top team. Add-0.5 smoothing is applied to all rate-based metrics."
        ),
        "table3_metrics": {
            "SP": "Smoothed demographic-parity gap in member selection.",
            "CSP": f"Weighted smoothed demographic-parity gap conditioned on {args.conditioning_attribute}.",
            "EO": "Average of the smoothed TPR gap and smoothed FPR gap.",
            "PRP": "Smoothed predictive-rate-parity gap using PPV.",
            "TE": "Smoothed treatment-equality gap using FN/FP ratios.",
        },
        "methods": method_meta,
        "files": {
            "table2": str(table2_path),
            "table3": str(table3_path),
            "support": str(support_path),
            "inference": str(inference_path),
        },
    }
    if subset_meta is not None:
        metadata["subset"] = subset_meta
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved Table 2 -> {table2_path}")
    print(table2_display.to_string(index=False))
    print(f"\nSaved Table 3 -> {table3_path}")
    print(table3_display.to_string(index=False))
    print(f"\nSaved fairness support -> {support_path}")
    print(support_display.to_string(index=False))
    print(f"\nSaved gender inference -> {inference_path}")
    print(
        inference_df["gender"].value_counts(dropna=False).rename_axis("gender").reset_index(name="count").to_string(index=False)
    )
    print(f"\nSaved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
