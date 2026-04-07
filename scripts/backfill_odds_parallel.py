"""Parallel ESPN odds backfill — one process per season.

Usage:
  python scripts/backfill_odds_parallel.py [--workers 6] [--start 2013]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract.extract_odds import extract_season, current_season, FIRST_SEASON


def scrape_one_season(args):
    season, data_dir = args
    try:
        new_rows = extract_season(Path(data_dir), season)
        return season, new_rows, None
    except Exception as e:
        return season, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel workers (be polite to ESPN, default 6)")
    parser.add_argument("--start", type=int, default=FIRST_SEASON)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    end_season = current_season()
    seasons = list(range(args.start, end_season + 1))
    print(f"Backfilling {len(seasons)} seasons ({args.start}–{end_season}) with {args.workers} workers")

    tasks = [(s, args.data_dir) for s in seasons]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scrape_one_season, t): t[0] for t in tasks}

        for future in as_completed(futures):
            season, new_rows, err = future.result()
            if err:
                print(f"  {season}: FAILED — {err}")
            else:
                print(f"  {season}: {new_rows} odds rows")

    print("Done.")


if __name__ == "__main__":
    main()
