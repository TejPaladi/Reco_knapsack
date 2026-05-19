"""
Generate fairness tables for meal_project-3 in the style of Tables 2 and 3 from
the fairness paper.

Assumptions for the meal setting:
- Protected attribute: user-side demographics (default: gender).
- Table 2 uses continuous "outcome mass" based on per-user average goodness.
  This avoids the degenerate case where every user receives some recommendation.
- Table 3 uses thresholded best-bundle quality:
  - oracle-positive if any method reaches the oracle threshold for that user
  - method-positive if that method reaches the prediction threshold
- Conditional Statistical Parity (CSP) is conditioned on meal occasion.
- Add-0.5 smoothing is used for Table 3 rate estimates so sparse cells stay
  finite and comparable.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
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
DEFAULT_FAIRNESS_THRESHOLDS = {
    "legacy_meal": {"oracle": 0.95, "prediction": 0.92},
    "meal": {"oracle": 0.95, "prediction": 0.92},
    "teaming": {"oracle": 0.22, "prediction": 0.20},
    "beacon_proxy": {"oracle": 0.90, "prediction": 0.88},
}

METHOD_SPECS = [
    ("M0", "Random baseline", "meal_uc1_m0.csv"),
    ("M1", "Sequential greedy", "meal_uc1_m1.csv"),
    ("M3", "Boosted Bandit", "meal_uc1_m2.csv"),
    ("M6", "Knapsack", "meal_uc1_m3.csv"),
    ("M7", "Marginal Utility", "meal_uc1_m4.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--users-file",
        default=str(PROJECT_ROOT / "data" / "input_data" / "user_meal_requirements.csv"),
        help="User CSV that contains the protected attribute and conditioning attribute.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "output"),
        help="Directory that contains canonical method CSVs.",
    )
    parser.add_argument(
        "--comparisons-dir",
        default=None,
        help="Directory for fairness tables. Defaults to <output-root>/comparisons.",
    )
    parser.add_argument(
        "--protected-attribute",
        default="gender",
        help="Binary protected attribute to evaluate.",
    )
    parser.add_argument(
        "--protected-source",
        choices=["column", "name_inference"],
        default="column",
        help="Use the raw user column or infer gender from first names.",
    )
    parser.add_argument(
        "--name-inference-mode",
        choices=["basic", "paper_ensemble"],
        default="paper_ensemble",
        help="How to infer gender when --protected-source=name_inference.",
    )
    parser.add_argument(
        "--conditioning-attribute",
        default="meal_occasion",
        help="Legitimate attribute used for CSP.",
    )
    parser.add_argument(
        "--table2-score",
        choices=["avg_goodness", "best_goodness"],
        default="avg_goodness",
        help="User-level score used to build Table 2 outcome-mass shares.",
    )
    parser.add_argument(
        "--goodness-profile",
        choices=["legacy_meal", "meal", "teaming", "beacon_proxy"],
        default=os.environ.get("MEAL_GOODNESS_PROFILE", "beacon_proxy"),
        help="Goodness profile used to build the method outputs being evaluated.",
    )
    parser.add_argument(
        "--oracle-threshold",
        type=float,
        default=None,
        help="Oracle best-bundle threshold used for the actual-positive label in Table 3.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=None,
        help="Best-bundle threshold used for method-positive predictions in Table 3.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=4,
        help="Number of decimal places to save in the final tables.",
    )
    return parser.parse_args()


def load_method_outputs(output_root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    frames = []
    method_meta = {}

    for method_label, method_name, filename in METHOD_SPECS:
        path = output_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical method output: {path}")

        df = pd.read_csv(path)
        df["goodness_list"] = df["goodness"].apply(ast.literal_eval)
        df["avg_goodness"] = df["goodness_list"].apply(lambda values: sum(values) / len(values) if values else 0.0)
        df["best_goodness"] = df["goodness_list"].apply(lambda values: max(values) if values else 0.0)
        df["volume"] = df["goodness_list"].apply(len)
        df = df[["user_id", "avg_goodness", "best_goodness", "volume"]].copy()

        # Canonical cross-product exports contain one row per (user, target_item).
        # Fairness evaluation needs one row per user, so collapse duplicates here
        # before merging methods together. For single-row-per-user exports this is
        # a no-op.
        if df["user_id"].duplicated().any():
            df = (
                df.groupby("user_id", as_index=False)
                .agg(
                    {
                        "avg_goodness": "mean",
                        "best_goodness": "max",
                        "volume": "mean",
                    }
                )
            )

        df = df.rename(
            columns={
                "avg_goodness": f"{method_label}_avg_goodness",
                "best_goodness": f"{method_label}_best_goodness",
                "volume": f"{method_label}_volume",
            }
        )
        frames.append(df)
        method_meta[method_label] = {"name": method_name, "filename": filename}

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="user_id", how="inner")

    return merged, method_meta


def extract_first_name(full_name: str) -> str:
    rest = re.sub(r"\([^)]*\)", " ", full_name or "")
    rest = rest.replace(".", " ").replace("-", " ").replace("/", " ")
    tokens = [token for token in re.split(r"[^A-Za-z]+", rest) if token]
    return tokens[0].lower() if tokens else ""


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
    if genderize_vote in {"male", "female"}:
        return genderize_vote, "genderize_fallback"
    return "unknown", "unknown"


def infer_gender_frame(names: pd.Series, mode: str) -> pd.DataFrame:
    detector = gender.Detector(case_sensitive=False)
    sexmachine = build_sexmachine_detector() if mode == "paper_ensemble" else None
    first_name_series = names.fillna("").map(extract_first_name)
    genderize_lookup = lookup_genderize(first_name_series.tolist()) if mode == "paper_ensemble" else {}
    rows = []

    for full_name, first_name in zip(names.fillna(""), first_name_series):
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
            )
        else:
            genderize_vote = None
            genderize_probability = None
            sexmachine_raw = "not_used"
            sexmachine_vote = None
            final_gender = gender_guesser_vote or "unknown"
            source = "gender_guesser_only" if gender_guesser_vote else "unknown"

        rows.append(
            {
                "name": full_name,
                "first_name": first_name,
                "gender_guesser_raw": gender_guesser_raw,
                "gender_guesser_vote": gender_guesser_vote or "unknown",
                "genderize_vote": genderize_vote or "unknown",
                "genderize_probability": genderize_probability,
                "sexmachine_raw": sexmachine_raw,
                "sexmachine_vote": sexmachine_vote or "unknown",
                "gender": final_gender,
                "source": source,
            }
        )

    return pd.DataFrame(rows)


def load_user_cohort(
    users_file: Path,
    user_ids: pd.Series,
    protected_attr: str,
    conditioning_attr: str,
    protected_source: str,
    name_inference_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    users = pd.read_csv(users_file)
    required_cols = {"user_id", conditioning_attr}
    if protected_source == "column":
        required_cols.add(protected_attr)
    elif protected_source == "name_inference":
        required_cols.add("name")
    missing = required_cols - set(users.columns)
    if missing:
        raise ValueError(f"Missing required user columns: {sorted(missing)}")

    keep_cols = ["user_id", conditioning_attr]
    if protected_source == "column":
        keep_cols.append(protected_attr)
    else:
        keep_cols.append("name")

    users = users[keep_cols].copy()
    users = users[users["user_id"].isin(set(user_ids))].copy()
    users[conditioning_attr] = users[conditioning_attr].astype(str).str.strip().str.lower()

    if protected_source == "column":
        users[protected_attr] = users[protected_attr].astype(str).str.strip().str.lower()
        inference_df = None
    else:
        inference_df = infer_gender_frame(users["name"], name_inference_mode)
        users = users.reset_index(drop=True)
        inference_df = inference_df.reset_index(drop=True)
        inference_df = pd.concat([users[["user_id"]], inference_df], axis=1)
        users = pd.concat([users, inference_df.drop(columns=["user_id", "name"])], axis=1)
        users = users.drop(columns=["name"])
        users = users[users[protected_attr].isin(["male", "female"])].copy()

    return users, inference_df


def ordered_groups(values: pd.Series, protected_attr: str) -> list[str]:
    counts = values.value_counts()
    groups = counts.index.tolist()
    if protected_attr == "gender":
        preferred = [group for group in ["male", "female"] if group in groups]
        remaining = [group for group in groups if group not in preferred]
        groups = preferred + remaining

    if len(groups) != 2:
        raise ValueError(
            f"{protected_attr!r} must be binary for these tables; found groups: {groups}"
        )
    return groups


def smooth_rate(positive: int, total: int) -> float:
    return (positive + 0.5) / (total + 1.0)


def dp_gap_smoothed(df: pd.DataFrame, pred_col: str, protected_attr: str, groups: list[str]) -> float:
    rates = []
    for group in groups:
        mask = df[protected_attr] == group
        rates.append(smooth_rate(int(df.loc[mask, pred_col].sum()), int(mask.sum())))
    return abs(rates[0] - rates[1])


def csp_gap_smoothed(
    df: pd.DataFrame,
    pred_col: str,
    protected_attr: str,
    groups: list[str],
    conditioning_attr: str,
) -> float:
    weighted_gap = 0.0
    total_weight = 0

    for _, subset in df.groupby(conditioning_attr):
        if any((subset[protected_attr] == group).sum() == 0 for group in groups):
            continue
        gap = dp_gap_smoothed(subset, pred_col, protected_attr, groups)
        weighted_gap += len(subset) * gap
        total_weight += len(subset)

    return weighted_gap / total_weight if total_weight else float("nan")


def confusion_counts(df: pd.DataFrame, truth_col: str, pred_col: str, protected_attr: str, group: str) -> dict[str, int]:
    mask = df[protected_attr] == group
    tp = int(((df[pred_col] == 1) & (df[truth_col] == 1) & mask).sum())
    fp = int(((df[pred_col] == 1) & (df[truth_col] == 0) & mask).sum())
    fn = int(((df[pred_col] == 0) & (df[truth_col] == 1) & mask).sum())
    tn = int(((df[pred_col] == 0) & (df[truth_col] == 0) & mask).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def eo_gap_smoothed(df: pd.DataFrame, truth_col: str, pred_col: str, protected_attr: str, groups: list[str]) -> float:
    stats = {}
    for group in groups:
        counts = confusion_counts(df, truth_col, pred_col, protected_attr, group)
        stats[group] = {
            "tpr": smooth_rate(counts["tp"], counts["tp"] + counts["fn"]),
            "fpr": smooth_rate(counts["fp"], counts["fp"] + counts["tn"]),
        }
    return 0.5 * (
        abs(stats[groups[0]]["tpr"] - stats[groups[1]]["tpr"])
        + abs(stats[groups[0]]["fpr"] - stats[groups[1]]["fpr"])
    )


def prp_gap_smoothed(df: pd.DataFrame, truth_col: str, pred_col: str, protected_attr: str, groups: list[str]) -> float:
    stats = {}
    for group in groups:
        counts = confusion_counts(df, truth_col, pred_col, protected_attr, group)
        stats[group] = smooth_rate(counts["tp"], counts["tp"] + counts["fp"])
    return abs(stats[groups[0]] - stats[groups[1]])


def te_gap_smoothed(df: pd.DataFrame, truth_col: str, pred_col: str, protected_attr: str, groups: list[str]) -> float:
    ratios = {}
    for group in groups:
        counts = confusion_counts(df, truth_col, pred_col, protected_attr, group)
        ratios[group] = (counts["fn"] + 0.5) / (counts["fp"] + 0.5)
    return abs(ratios[groups[0]] - ratios[groups[1]])


def build_table2(
    df: pd.DataFrame,
    protected_attr: str,
    groups: list[str],
    method_meta: dict[str, dict[str, str]],
    table2_score: str,
) -> pd.DataFrame:
    population = df[protected_attr].value_counts(normalize=True)

    rows = []
    for group in groups:
        row = {
            "Group": group.capitalize(),
            "Population Distribution": float(population.get(group, 0.0)),
        }
        for method_label in method_meta:
            score_col = f"{method_label}_{table2_score}"
            total_score = float(df[score_col].sum())
            if total_score == 0:
                row[method_label] = float("nan")
                continue
            group_score = float(df.loc[df[protected_attr] == group, score_col].sum())
            row[method_label] = group_score / total_score
        rows.append(row)

    sp_row = {"Group": "SP", "Population Distribution": float("nan")}
    for method_label in method_meta:
        score_col = f"{method_label}_{table2_score}"
        total_score = float(df[score_col].sum())
        shares = {}
        for group in groups:
            group_score = float(df.loc[df[protected_attr] == group, score_col].sum())
            shares[group] = group_score / total_score if total_score else float("nan")
        sp_row[method_label] = 0.5 * sum(abs(shares[group] - float(population.get(group, 0.0))) for group in groups)
    rows.append(sp_row)

    return pd.DataFrame(rows)


def build_table3_and_support(
    df: pd.DataFrame,
    protected_attr: str,
    groups: list[str],
    conditioning_attr: str,
    method_meta: dict[str, dict[str, str]],
    oracle_threshold: float,
    prediction_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [f"{method_label}_best_goodness" for method_label in method_meta]
    work = df.copy()
    work["oracle_positive"] = (work[score_cols].max(axis=1) >= oracle_threshold).astype(int)

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
        score_col = f"{method_label}_best_goodness"
        work[pred_col] = (work[score_col] >= prediction_threshold).astype(int)

        table3_rows["SP"][method_label] = dp_gap_smoothed(work, pred_col, protected_attr, groups)
        table3_rows["CSP"][method_label] = csp_gap_smoothed(
            work,
            pred_col,
            protected_attr,
            groups,
            conditioning_attr,
        )
        table3_rows["EO"][method_label] = eo_gap_smoothed(
            work,
            "oracle_positive",
            pred_col,
            protected_attr,
            groups,
        )
        table3_rows["PRP"][method_label] = prp_gap_smoothed(
            work,
            "oracle_positive",
            pred_col,
            protected_attr,
            groups,
        )
        table3_rows["TE"][method_label] = te_gap_smoothed(
            work,
            "oracle_positive",
            pred_col,
            protected_attr,
            groups,
        )

        support_row = {
            "Method": method_label,
            "Method_name": method_meta[method_label]["name"],
            "best_goodness_mean": float(work[score_col].mean()),
            "predicted_positive_rate": float(work[pred_col].mean()),
            "actual_positive_rate": float(work["oracle_positive"].mean()),
            "predicted_positive_count": int(work[pred_col].sum()),
            "actual_positive_count": int(work["oracle_positive"].sum()),
        }
        for group in groups:
            group_mask = work[protected_attr] == group
            support_row[f"{group}_predicted_positive_rate"] = float(work.loc[group_mask, pred_col].mean())
        support_rows.append(support_row)

    table3 = pd.DataFrame([table3_rows[key] for key in ["SP", "CSP", "EO", "PRP", "TE"]])
    support = pd.DataFrame(support_rows)
    return table3, support


def round_frame(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def main():
    args = parse_args()
    fairness_thresholds = DEFAULT_FAIRNESS_THRESHOLDS[args.goodness_profile]
    oracle_threshold = args.oracle_threshold if args.oracle_threshold is not None else fairness_thresholds["oracle"]
    prediction_threshold = (
        args.prediction_threshold
        if args.prediction_threshold is not None
        else fairness_thresholds["prediction"]
    )
    output_root = Path(args.output_root)
    comparisons_dir = Path(args.comparisons_dir) if args.comparisons_dir else output_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    method_outputs, method_meta = load_method_outputs(output_root)
    users, inference_df = load_user_cohort(
        Path(args.users_file),
        method_outputs["user_id"],
        args.protected_attribute,
        args.conditioning_attribute,
        args.protected_source,
        args.name_inference_mode,
    )

    fairness_df = users.merge(method_outputs, on="user_id", how="inner")
    groups = ordered_groups(fairness_df[args.protected_attribute], args.protected_attribute)

    table2 = build_table2(
        fairness_df,
        args.protected_attribute,
        groups,
        method_meta,
        args.table2_score,
    )
    table3, support = build_table3_and_support(
        fairness_df,
        args.protected_attribute,
        groups,
        args.conditioning_attribute,
        method_meta,
        oracle_threshold,
        prediction_threshold,
    )

    table2_path = comparisons_dir / f"fairness_table2_{args.protected_attribute}.csv"
    table3_path = comparisons_dir / f"fairness_table3_{args.protected_attribute}.csv"
    support_path = comparisons_dir / f"fairness_support_{args.protected_attribute}.csv"
    inference_path = comparisons_dir / f"fairness_{args.protected_attribute}_inference.csv"
    metadata_path = comparisons_dir / f"fairness_metadata_{args.protected_attribute}.json"

    round_frame(table2, args.round_digits).to_csv(table2_path, index=False)
    round_frame(table3, args.round_digits).to_csv(table3_path, index=False)
    round_frame(support, args.round_digits).to_csv(support_path, index=False)
    if inference_df is not None:
        inference_df.to_csv(inference_path, index=False)

    metadata = {
        "project": "meal_project-3",
        "cohort_size": int(len(fairness_df)),
        "protected_attribute": args.protected_attribute,
        "protected_source": args.protected_source,
        "name_inference_mode": args.name_inference_mode if args.protected_source == "name_inference" else None,
        "protected_groups": groups,
        "conditioning_attribute": args.conditioning_attribute,
        "goodness_profile": args.goodness_profile,
        "table2_score": args.table2_score,
        "table2_interpretation": (
            "Table 2 reports each group's share of the total per-user outcome mass "
            f"using {args.table2_score}. SP is the total-variation gap between those "
            "shares and the cohort's population distribution."
        ),
        "table3_interpretation": (
            "Table 3 uses thresholded best-bundle quality. A user is oracle-positive "
            f"when any method reaches best_goodness >= {oracle_threshold}. A "
            "method is prediction-positive when its own best_goodness reaches "
            f"{prediction_threshold}. Add-0.5 smoothing is applied to all "
            "rate-based quantities in Table 3."
        ),
        "table3_metrics": {
            "SP": "Smoothed demographic-parity gap between the two protected groups.",
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
        },
    }
    if inference_df is not None:
        metadata["gender_inference"] = {
            "resolved_count": int(inference_df["gender"].isin(groups).sum()),
            "unresolved_count": int((~inference_df["gender"].isin(groups)).sum()),
            "source_counts": inference_df["source"].value_counts().to_dict(),
            "description": (
                "Gender is inferred from first names using a paper-style ensemble of "
                "gender-guesser, Genderize, and SexMachine. Majority vote is used when "
                "the libraries disagree; otherwise the strongest available single-source "
                "guess is kept."
            ),
        }
        metadata["files"]["inference"] = str(inference_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved Table 2 -> {table2_path}")
    print(table2.to_string(index=False))
    print(f"\nSaved Table 3 -> {table3_path}")
    print(table3.to_string(index=False))
    print(f"\nSaved fairness support -> {support_path}")
    print(support.to_string(index=False))
    if inference_df is not None:
        print(f"\nSaved gender inference -> {inference_path}")
        print(
            inference_df["gender"].value_counts(dropna=False).rename_axis("gender").reset_index(name="count").to_string(index=False)
        )
    print(f"\nSaved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
