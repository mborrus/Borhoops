"""Orchestrate data extraction, transformation, and prediction.

Usage:
  python src/runner.py                              # extract + transform + predict
  python src/runner.py --source kaggle              # just kaggle
  python src/runner.py --source espn                # incremental espn
  python src/runner.py --source espn --backfill     # full season backfill
  python src/runner.py --source barttorvik          # current season T-Rank
  python src/runner.py --source barttorvik --backfill  # all seasons 2008+
  python src/runner.py --source odds                # current season ESPN odds
  python src/runner.py --source odds --backfill     # all seasons 2013+
  python src/runner.py --source polls               # current season AP/Coaches polls
  python src/runner.py --source polls --backfill    # all seasons 2003+
  python src/runner.py --source vegas               # current odds (needs ODDS_API_KEY)
  python src/runner.py --source transform           # ESPN → Kaggle transform
  python src/runner.py --source transform-barttorvik  # Barttorvik → Kaggle IDs
  python src/runner.py --source transform-odds      # Odds → Kaggle IDs + DayNum
  python src/runner.py --source transform-polls     # Polls → Kaggle IDs
  python src/runner.py --source transform-vegas     # Vegas → Kaggle IDs
  python src/runner.py --source predict             # run Elo model + write submission
"""

import argparse
import sys
from pathlib import Path

import yaml

from extract.extract_kaggle import download as kaggle_download
from extract.extract_espn import extract_backfill as espn_backfill
from extract.extract_espn import extract_incremental as espn_incremental
from extract.extract_barttorvik import extract_current as barttorvik_current
from extract.extract_barttorvik import extract_backfill as barttorvik_backfill
from extract.extract_odds import extract_current as odds_current
from extract.extract_odds import extract_backfill as odds_backfill
from extract.extract_polls import extract_current as polls_current
from extract.extract_polls import extract_backfill as polls_backfill
from extract.extract_vegas import extract_current as vegas_current
from extract.extract_vegas import extract_backfill as vegas_backfill
from extract.extract_roster import extract_current as roster_current
from extract.extract_roster import extract_backfill as roster_backfill
from transform.transform_espn import transform_and_union
from transform.transform_barttorvik import transform_barttorvik
from transform.transform_odds import transform_odds
from transform.transform_polls import transform_polls
from transform.transform_vegas import transform_vegas
from transform.transform_roster import transform_roster
from predict.submission import predict as run_predict

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run data extractors")
    parser.add_argument(
        "--source",
        choices=["kaggle", "espn", "barttorvik", "odds", "polls", "vegas", "roster",
                 "transform", "transform-barttorvik", "transform-odds",
                 "transform-polls", "transform-vegas", "transform-roster", "predict"],
        help="Run a single source",
    )
    parser.add_argument("--backfill", action="store_true", help="Full backfill")
    args = parser.parse_args()

    config = load_config()
    data_dir = REPO_ROOT / config["data_dir"]

    sources = [args.source] if args.source else ["kaggle", "espn", "transform", "predict"]
    results = {}

    for source in sources:
        print(f"\n{'='*40}")
        print(f"Running: {source}")
        print(f"{'='*40}")
        try:
            if source == "kaggle":
                results[source] = kaggle_download(data_dir)
            elif source == "espn":
                if args.backfill:
                    results[source] = espn_backfill(data_dir)
                else:
                    results[source] = espn_incremental(data_dir)
            elif source == "barttorvik":
                if args.backfill:
                    results[source] = barttorvik_backfill(data_dir)
                else:
                    results[source] = barttorvik_current(data_dir)
            elif source == "odds":
                if args.backfill:
                    results[source] = odds_backfill(data_dir)
                else:
                    results[source] = odds_current(data_dir)
            elif source == "polls":
                if args.backfill:
                    results[source] = polls_backfill(data_dir)
                else:
                    results[source] = polls_current(data_dir)
            elif source == "vegas":
                if args.backfill:
                    results[source] = vegas_backfill(data_dir)
                else:
                    results[source] = vegas_current(data_dir)
            elif source == "roster":
                if args.backfill:
                    results[source] = roster_backfill(data_dir)
                else:
                    results[source] = roster_current(data_dir)
            elif source == "transform":
                results[source] = transform_and_union(data_dir)
            elif source == "transform-barttorvik":
                results[source] = transform_barttorvik(data_dir)
            elif source == "transform-odds":
                results[source] = transform_odds(data_dir)
            elif source == "transform-polls":
                results[source] = transform_polls(data_dir)
            elif source == "transform-vegas":
                results[source] = transform_vegas(data_dir)
            elif source == "transform-roster":
                results[source] = transform_roster(data_dir)
            elif source == "predict":
                output_dir = REPO_ROOT / config["output_dir"]
                results[source] = run_predict(data_dir, output_dir)
        except Exception as e:
            print(f"FAILED: {source} — {e}", file=sys.stderr)
            results[source] = False

    print(f"\n{'='*40}")
    print("Results:")
    for source, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {source}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
