"""Orchestrate data extraction, transformation, and prediction.

Usage:
  python src/runner.py                              # extract + transform + predict
  python src/runner.py --source kaggle              # just kaggle
  python src/runner.py --source espn                # incremental espn
  python src/runner.py --source espn --backfill     # full season backfill
  python src/runner.py --source transform           # ESPN → Kaggle transform
  python src/runner.py --source predict             # run Elo model + write submission
"""

import argparse
import sys
from pathlib import Path

import yaml

from extract.extract_kaggle import download as kaggle_download
from extract.extract_espn import extract_backfill as espn_backfill
from extract.extract_espn import extract_incremental as espn_incremental
from transform.transform_espn import transform_and_union
from predict.submission import predict as run_predict

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run data extractors")
    parser.add_argument("--source", choices=["kaggle", "espn", "transform", "predict"], help="Run a single source")
    parser.add_argument("--backfill", action="store_true", help="Full season backfill (ESPN only)")
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
            elif source == "transform":
                results[source] = transform_and_union(data_dir)
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
