"""
Run the canonical Meal pipeline and publish BEACON-style proxy outputs.

The final GitHub-facing output is always written to:
    data/output/

How the pipeline works:
1. Generate BoostSRL train/test facts from data/input_data/.
2. Run BoostSRL train and test in boosted_bandit/.
3. Build source meal outputs with the internal legacy source profile into a
   temporary intermediate directory.
4. Rescore those same returned bundles with the closest BEACON-style proxy and
   publish the final CSVs to data/output/.
5. Generate fairness tables from the published BEACON-style proxy outputs.

The published pipeline therefore exposes only one canonical output root to
future users while keeping the intermediate source-generation step internal.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_INTERMEDIATE_OUTPUT_DIR = PROJECT_ROOT / "data" / "_intermediate_legacy_source"
DEFAULT_BANDIT_DIR = PROJECT_ROOT / "boosted_bandit"
DEFAULT_MEALS_FILE = "meal_categories.csv"
DEFAULT_USERS_FILE = "user_meal_requirements.csv"
SOURCE_GOODNESS_PROFILE = "legacy_meal"
FINAL_GOODNESS_PROFILE = "beacon_proxy"
FINAL_FAIRNESS_THRESHOLDS = {"oracle": 0.90, "prediction": 0.88}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--intermediate-output-dir", default=str(DEFAULT_INTERMEDIATE_OUTPUT_DIR))
    parser.add_argument("--bandit-dir", default=str(DEFAULT_BANDIT_DIR))
    parser.add_argument("--meals-file", default=DEFAULT_MEALS_FILE)
    parser.add_argument("--users-file", default=DEFAULT_USERS_FILE)
    parser.add_argument("--trees", type=int, default=5)
    parser.add_argument("--split", choices=["all", "boosted_test"], default="all")
    parser.add_argument("--match-threshold", type=float, default=None)
    parser.add_argument("--neg-ratio", type=int, default=None)
    parser.add_argument("--skip-fairness", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args()


def describe_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {"rows": len(rows), "columns": reader.fieldnames or []}


def run_command(cmd, *, env, cwd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd))


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    intermediate_output_dir = Path(args.intermediate_output_dir).resolve()
    bandit_dir = Path(args.bandit_dir).resolve()
    meals_path = input_dir / args.meals_file
    users_path = input_dir / args.users_file

    if not meals_path.exists():
        raise FileNotFoundError(f"Missing meals file: {meals_path}")
    if not users_path.exists():
        raise FileNotFoundError(f"Missing users file: {users_path}")
    if not (bandit_dir / "command.sh").exists():
        raise FileNotFoundError(f"Missing BoostSRL launcher: {bandit_dir / 'command.sh'}")

    meal_info = describe_csv(meals_path)
    user_info = describe_csv(users_path)

    print("Canonical BEACON-Proxy Meal Pipeline")
    print("-----------------------------------")
    print(f"Meals file              : {meals_path}")
    print(f"Users file              : {users_path}")
    print(f"Published output root   : {output_dir}")
    print(f"Intermediate output root: {intermediate_output_dir}")
    print(f"Source profile          : {SOURCE_GOODNESS_PROFILE}")
    print(f"Final profile           : {FINAL_GOODNESS_PROFILE}")
    print(f"Meal rows               : {meal_info['rows']}")
    print(f"User rows               : {user_info['rows']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MEAL_INPUT_DIR": str(input_dir),
            "MEAL_CATEGORIES_FILE": args.meals_file,
            "MEAL_USERS_FILE": args.users_file,
            "MEAL_BANDIT_DIR": str(bandit_dir),
        }
    )
    if args.match_threshold is not None:
        env["BOOSTED_MATCH_THRESHOLD"] = str(args.match_threshold)
    if args.neg_ratio is not None:
        env["BOOSTED_NEG_RATIO"] = str(args.neg_ratio)

    # Step 1-3: build the internal source outputs with the legacy source profile.
    env["MEAL_OUTPUT_DIR"] = str(intermediate_output_dir)
    env["MEAL_GOODNESS_PROFILE"] = SOURCE_GOODNESS_PROFILE

    run_command([sys.executable, str(PROJECT_ROOT / "code" / "gen_boosted_data.py")], env=env, cwd=PROJECT_ROOT)
    run_command(["bash", "command.sh", "train", str(args.trees)], env=env, cwd=bandit_dir)
    run_command(["bash", "command.sh", "test", str(args.trees)], env=env, cwd=bandit_dir)
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "build_final_outputs.py"),
            "--split",
            args.split,
            "--goodness-profile",
            SOURCE_GOODNESS_PROFILE,
        ],
        env=env,
        cwd=PROJECT_ROOT,
    )

    # Step 4: rescore the exact same returned bundles with the BEACON-style proxy.
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "rescore_outputs_beacon_proxy.py"),
            "--source-root",
            str(intermediate_output_dir),
            "--output-root",
            str(output_dir),
            "--meals-file",
            str(meals_path),
        ],
        env=env,
        cwd=PROJECT_ROOT,
    )

    # Step 5: build fairness tables from the published proxy outputs.
    if not args.skip_fairness:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "code" / "generate_fairness_tables.py"),
                "--users-file",
                str(users_path),
                "--output-root",
                str(output_dir),
                "--protected-attribute",
                "gender",
                "--protected-source",
                "column",
                "--conditioning-attribute",
                "meal_occasion",
                "--goodness-profile",
                FINAL_GOODNESS_PROFILE,
                "--oracle-threshold",
                str(FINAL_FAIRNESS_THRESHOLDS["oracle"]),
                "--prediction-threshold",
                str(FINAL_FAIRNESS_THRESHOLDS["prediction"]),
            ],
            env=env,
            cwd=PROJECT_ROOT,
        )

    metadata_path = output_dir / "comparisons" / "beacon_proxy_metadata.json"
    if not args.keep_intermediate and intermediate_output_dir.exists():
        shutil.rmtree(intermediate_output_dir)
        print(f"\nRemoved intermediate source outputs -> {intermediate_output_dir}")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_root_retained"] = False
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nPipeline complete.")
    print(f"Published canonical outputs -> {output_dir}")
    if args.keep_intermediate:
        print(f"Intermediate source outputs kept -> {intermediate_output_dir}")


if __name__ == "__main__":
    main()
