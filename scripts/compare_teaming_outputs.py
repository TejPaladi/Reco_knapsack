#!/usr/bin/env python3
"""Compare Teaming output CSVs across generated methods.

The script mirrors the notebook logic in ``Teaming/code/Results.ipynb``:

- input files are discovered by globbing ``teaming_uc1_m*.csv``
- each row is parsed from ``researcher_name``, ``team``, and ``goodness``
- per-researcher stats and a dataset summary are written to CSV
- two simple bar plots are saved alongside the CSV summary

Supported domains:
  - teaming:      ``Teaming/data/output/``
  - iitr-teaming: ``IITR-Teaming/data/output/``
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - environment-specific fallback
    raise SystemExit(
        "Pillow is required to render PNG bar plots. Install it or run this script in an environment with PIL available."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ROOTS = {
    "teaming": REPO_ROOT / "Teaming",
    "iitr-teaming": REPO_ROOT / "IITR-Teaming",
}
DEFAULT_PATTERN = "teaming_uc1_m*.csv"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def parse_list_field(value: object) -> list[Any]:
    """Parse a list-like CSV field using the notebook's permissive behavior."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (int, float)):
        return [value]

    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (SyntaxError, ValueError):
        pass

    return [part.strip() for part in text.split(",") if part.strip()]


def parse_numeric_list_field(value: object) -> list[float]:
    numbers: list[float] = []
    for item in parse_list_field(value):
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers


def resolve_domain_root(domain: str) -> Path:
    try:
        return DOMAIN_ROOTS[domain]
    except KeyError as exc:
        valid = ", ".join(sorted(DOMAIN_ROOTS))
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {valid}") from exc


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_field(row: dict[str, str], preferred: str, token: str, allow_missing: bool = False) -> str:
    if preferred in row and clean_text(row[preferred]):
        return clean_text(row[preferred])
    for key, value in row.items():
        if token in key.lower() and clean_text(value):
            return clean_text(value)
    if allow_missing:
        return ""
    raise KeyError(f"No '{preferred}'-like column found in row keys: {list(row.keys())}")


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {
                "researcher_name": infer_field(row, "researcher_name", "researcher"),
                "team": infer_field(row, "team", "team"),
                "goodness": infer_field(row, "goodness", "goodness", allow_missing=True),
            }
        )
    return normalized


def compute_metrics(rows: list[dict[str, str]]) -> tuple[dict[str, float | int | str], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, float | int]] = defaultdict(lambda: {"sum_avg": 0.0, "sum_volume": 0.0, "row_count": 0})

    for row in rows:
        researcher = row["researcher_name"]
        team = parse_list_field(row["team"])
        goodness = parse_numeric_list_field(row["goodness"])
        avg_goodness = mean(goodness) if goodness else 0.0
        volume = len(team)

        entry = grouped[researcher]
        entry["sum_avg"] += avg_goodness
        entry["sum_volume"] += volume
        entry["row_count"] += 1

    researcher_stats: list[dict[str, Any]] = []
    for researcher in sorted(grouped):
        entry = grouped[researcher]
        row_count = int(entry["row_count"]) or 1
        researcher_stats.append(
            {
                "researcher_name": researcher,
                "avg_goodness_per_row": entry["sum_avg"] / row_count,
                "volume": entry["sum_volume"] / row_count,
                "row_count": row_count,
            }
        )

    avg_goodness_values = [row["avg_goodness_per_row"] for row in researcher_stats]
    volume_values = [row["volume"] for row in researcher_stats]

    summary_row: dict[str, float | int | str] = {
        "dataset": "",
        "G_mean": mean(avg_goodness_values) if avg_goodness_values else 0.0,
        "G_std": pstdev(avg_goodness_values) if len(avg_goodness_values) > 1 else 0.0,
        "Volume_mean": mean(volume_values) if volume_values else 0.0,
        "n_researchers": len(researcher_stats),
        "n_rows": len(rows),
    }
    return summary_row, researcher_stats


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def text_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def paste_rotated_text(canvas: Image.Image, text: str, x: int, y: int, font: ImageFont.ImageFont, angle: float = 45.0) -> None:
    if not text:
        return
    tmp = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    w, h = text_box(tmp_draw, text, font)
    label = Image.new("RGBA", (w + 6, h + 6), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((3, 3), text, fill="black", font=font)
    rotated = label.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(rotated, (int(x - rotated.width / 2), int(y)), rotated)


def render_bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path, fill: str) -> None:
    width = max(960, 150 * max(1, len(labels)))
    height = 560
    left = 70
    right = 30
    top = 60
    bottom = 150
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title_w, title_h = text_box(draw, title, font)
    draw.text(((width - title_w) / 2, 18), title, fill="black", font=font)

    axis_color = "#444444"
    draw.line((left, top, left, top + plot_height), fill=axis_color, width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill=axis_color, width=2)
    draw.text((8, top + plot_height / 2 - 8), ylabel, fill="black", font=font)

    max_value = max(values) if values else 0.0
    if max_value <= 0:
        max_value = 1.0

    slot_width = plot_width / max(1, len(labels))
    bar_width = max(18, int(slot_width * 0.55))

    for index, (label, value) in enumerate(zip(labels, values)):
        center_x = left + slot_width * index + slot_width / 2
        bar_height = int((value / max_value) * plot_height)
        x0 = int(center_x - bar_width / 2)
        x1 = int(center_x + bar_width / 2)
        y0 = top + plot_height - bar_height
        y1 = top + plot_height
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=axis_color)

        value_text = f"{value:.3f}"
        value_w, value_h = text_box(draw, value_text, font)
        draw.text((center_x - value_w / 2, max(top + 2, y0 - value_h - 2)), value_text, fill="black", font=font)
        paste_rotated_text(image, label, int(center_x), top + plot_height + 12, font=font, angle=45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")


def compare_files(domain: str, output_root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    domain_root = resolve_domain_root(domain)
    input_root = domain_root / "data" / "output"
    comparisons_root = (output_root or input_root) / "comparisons"
    comparisons_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.glob(DEFAULT_PATTERN), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No files found under {input_root} for pattern {DEFAULT_PATTERN}")

    summary_rows: list[dict[str, Any]] = []
    per_dataset_researcher_stats: dict[str, list[dict[str, Any]]] = {}

    for path in files:
        rows = normalize_rows(read_csv_rows(path))
        summary_row, researcher_stats = compute_metrics(rows)
        dataset_name = path.name
        summary_row["dataset"] = dataset_name
        summary_rows.append(summary_row)
        per_dataset_researcher_stats[dataset_name] = researcher_stats

        researcher_stats_path = comparisons_root / f"{dataset_name}_researcher_stats.csv"
        write_csv(
            researcher_stats_path,
            ["researcher_name", "avg_goodness_per_row", "volume", "row_count"],
            researcher_stats,
        )

    summary_rows.sort(key=lambda row: row["dataset"])
    write_csv(
        comparisons_root / "comparison_summary.csv",
        ["dataset", "G_mean", "G_std", "Volume_mean", "n_researchers", "n_rows"],
        summary_rows,
    )

    labels = [row["dataset"] for row in summary_rows]
    g_values = [float(row["G_mean"]) for row in summary_rows]
    volume_values = [float(row["Volume_mean"]) for row in summary_rows]
    render_bar_chart(labels, g_values, "Average Goodness per Dataset", "G mean", comparisons_root / "G_mean_comparison.png", "#4f81bd")
    render_bar_chart(
        labels,
        volume_values,
        "Average Volume per Dataset",
        "Volume mean",
        comparisons_root / "Volume_mean_comparison.png",
        "#c0504d",
    )

    return summary_rows, per_dataset_researcher_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Teaming output CSVs.")
    parser.add_argument(
        "--domain",
        required=True,
        choices=sorted(DOMAIN_ROOTS),
        help="Which domain output folder to scan.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the domain output root. Defaults to the domain's data/output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    summary_rows, _ = compare_files(args.domain, output_root=output_root)
    comparisons_root = (output_root or (resolve_domain_root(args.domain) / "data" / "output")) / "comparisons"

    print(f"Compared {len(summary_rows)} datasets for domain '{args.domain}'.")
    print(f"Saved outputs to {comparisons_root}")


if __name__ == "__main__":
    main()
