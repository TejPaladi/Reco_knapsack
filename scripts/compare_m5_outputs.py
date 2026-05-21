#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def domain_output_root(domain: str) -> Path:
    if domain == "teaming":
        return REPO_ROOT / "Teaming" / "data" / "output"
    if domain == "iitr-teaming":
        return REPO_ROOT / "IITR-Teaming" / "data" / "output"
    raise ValueError(f"Unsupported domain: {domain}")


def parse_list_field(value: object) -> list[object]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def coerce_columns(row: dict[str, str]) -> tuple[str, str, str]:
    researcher = row.get("researcher_name", "")
    if not researcher:
        for key in row:
            if "researcher" in key.lower():
                researcher = row[key]
                break

    team = row.get("team", "")
    if not team:
        for key in row:
            if "team" in key.lower():
                team = row[key]
                break

    goodness = row.get("goodness", "")
    if not goodness:
        for key in row:
            if "goodness" in key.lower():
                goodness = row[key]
                break

    return researcher, team, goodness


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def compare_files(pattern: str, source_dir: Path, comparison_dir: Path) -> list[dict[str, object]]:
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {source_dir} for pattern {pattern}")

    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        researcher_scores: dict[str, list[float]] = {}
        researcher_volumes: dict[str, list[int]] = {}

        for row in rows:
            researcher, team, goodness = coerce_columns(row)
            team_list = parse_list_field(team)
            goodness_list = parse_list_field(goodness)
            row_volume = len(team_list)
            row_goodness = mean([float(value) for value in goodness_list]) if goodness_list else 0.0

            researcher_scores.setdefault(researcher, []).append(row_goodness)
            researcher_volumes.setdefault(researcher, []).append(row_volume)

        researcher_stats_path = comparison_dir / f"{path.name}_researcher_stats.csv"
        with researcher_stats_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["researcher_name", "avg_goodness_per_row", "volume", "row_count"],
            )
            writer.writeheader()
            for researcher in sorted(researcher_scores):
                scores = researcher_scores[researcher]
                volumes = researcher_volumes[researcher]
                writer.writerow(
                    {
                        "researcher_name": researcher,
                        "avg_goodness_per_row": round(mean(scores), 6),
                        "volume": round(mean(volumes), 6),
                        "row_count": len(scores),
                    }
                )

        all_row_goodness = [score for scores in researcher_scores.values() for score in scores]
        all_row_volumes = [volume for volumes in researcher_volumes.values() for volume in volumes]
        summary_rows.append(
            {
                "dataset": path.name,
                "G_mean": round(mean(all_row_goodness), 6),
                "G_std": round(pstdev(all_row_goodness), 6),
                "Volume_mean": round(mean(all_row_volumes), 6),
                "n_researchers": len(researcher_scores),
                "n_rows": len(rows),
            }
        )

    summary_path = comparison_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "G_mean", "G_std", "Volume_mean", "n_researchers", "n_rows"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Teaming/IITR output CSVs, including M5.")
    parser.add_argument("--domain", choices=["teaming", "iitr-teaming"], required=True)
    parser.add_argument("--pattern", default="teaming_uc1_m*.csv")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    source_dir = domain_output_root(args.domain)
    comparison_dir = Path(args.output_dir) if args.output_dir else source_dir / "comparisons"
    rows = compare_files(args.pattern, source_dir, comparison_dir)

    print("Comparison summary")
    for row in rows:
        print(
            f"{row['dataset']}: G_mean={row['G_mean']} G_std={row['G_std']} "
            f"Volume_mean={row['Volume_mean']} n_researchers={row['n_researchers']} n_rows={row['n_rows']}"
        )
    print(f"Wrote comparison outputs to {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
