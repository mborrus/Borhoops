"""Orchestrate data extraction from all sources.

Usage:
  python src/runner.py                              # incremental espn + kaggle
  python src/runner.py --source kaggle              # just kaggle
  python src/runner.py --source espn                # incremental espn (default)
  python src/runner.py --source espn --backfill     # full season backfill
"""

import argparse
import sys
from pathlib import Path

import yaml

from extract_kaggle import download as kaggle_download
from extract_espn import extract_backfill as espn_backfill
from extract_espn import extract_incremental as espn_incremental

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run data extractors")
    parser.add_argument("--source", choices=["kaggle", "espn"], help="Run a single source")
    parser.add_argument("--backfill", action="store_true", help="Full season backfill (ESPN only)")
    args = parser.parse_args()

    config = load_config()
    data_dir = REPO_ROOT / config["data_dir"]

    sources = [args.source] if args.source else ["kaggle", "espn"]
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
