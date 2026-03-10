"""Scrape ESPN game data via CBBpy.

Two modes:
  - Backfill: scrape full seasons not in Kaggle (get_season_gap)
  - Incremental: scrape only games after Kaggle's last date (get_date_gap)
"""

import pandas as pd
import duckdb as db
import cbbpy.mens_scraper as ms
import cbbpy.womens_scraper as ws
from datetime import datetime, timedelta
from pathlib import Path


def get_season_gap(data_dir: Path, _now=None) -> list[int]:
    """Return full season(s) missing from Kaggle. For bulk backfill."""
    kaggle_csv = f"{data_dir}/kaggle/MRegularSeasonCompactResults.csv"
    max_season = db.sql(f"SELECT MAX(Season) FROM '{kaggle_csv}'").fetchone()[0]

    today = _now or datetime.now()
    current_season = today.year + 1 if today.month >= 5 else today.year

    return list(range(max_season + 1, current_season + 1))


def get_date_gap(data_dir: Path, _now=None) -> tuple[str, str, int]:
    """Return (start_date, end_date, season) for incremental scraping.

    Finds the last game date in Kaggle data using DayZero + max DayNum,
    then returns the range from the day after through yesterday.
    """
    seasons_csv = f"{data_dir}/kaggle/MSeasons.csv"
    results_csv = f"{data_dir}/kaggle/MRegularSeasonCompactResults.csv"

    row = db.sql(f"""
        SELECT s.DayZero, r.Season, MAX(r.DayNum) as last_day
        FROM '{results_csv}' r
        JOIN '{seasons_csv}' s USING (Season)
        WHERE r.Season = (SELECT MAX(Season) FROM '{results_csv}')
        GROUP BY s.DayZero, r.Season
    """).fetchone()

    raw_day_zero = row[0]
    if isinstance(raw_day_zero, str):
        day_zero = datetime.strptime(raw_day_zero, "%m/%d/%Y")
    else:
        day_zero = datetime(raw_day_zero.year, raw_day_zero.month, raw_day_zero.day)
    season = row[1]
    last_day = row[2]

    today = _now or datetime.now()
    last_kaggle_date = day_zero + timedelta(days=last_day)
    start_date = last_kaggle_date + timedelta(days=1)
    end_date = today - timedelta(days=1)  # yesterday

    if start_date > end_date:
        return None, None, season

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), season


def save_to_cbbpy(info_df, box_df, gender: str, cbbpy_dir: Path):
    """Save scraped data, merging with existing files if present."""
    cbbpy_dir.mkdir(parents=True, exist_ok=True)
    info_path = cbbpy_dir / f"{gender}_game_info.csv"
    box_path = cbbpy_dir / f"{gender}_game_boxscore.csv"

    if info_path.exists():
        existing = pd.read_csv(info_path)
        info_df = pd.concat([existing, info_df]).drop_duplicates(
            subset=["game_id"], keep="last"
        )
    if box_path.exists():
        existing = pd.read_csv(box_path)
        box_df = pd.concat([existing, box_df]).drop_duplicates(
            subset=["game_id", "player_id"], keep="last"
        )

    info_df.to_csv(info_path, index=False)
    box_df.to_csv(box_path, index=False)
    print(f"  {gender}: {len(info_df)} games, {len(box_df)} box score rows → {cbbpy_dir}")


def scrape_season(season: int, cbbpy_dir: Path):
    """Scrape a full season. For bulk backfill."""
    for gender, scraper in [("mens", ms), ("womens", ws)]:
        print(f"Scraping {gender} season {season}...")
        try:
            info_df, box_df, _ = scraper.get_games_season(
                season, info=True, box=True, pbp=False
            )
        except Exception as e:
            print(f"  ERROR scraping {gender} {season}: {e}")
            continue
        save_to_cbbpy(info_df, box_df, gender, cbbpy_dir)


def scrape_range(start_date: str, end_date: str, cbbpy_dir: Path):
    """Scrape a date range. For incremental updates."""
    for gender, scraper in [("mens", ms), ("womens", ws)]:
        print(f"Scraping {gender} {start_date} → {end_date}...")
        try:
            info_df, box_df, _ = scraper.get_games_range(
                start_date, end_date, info=True, box=True, pbp=False
            )
        except Exception as e:
            print(f"  ERROR scraping {gender}: {e}")
            continue
        save_to_cbbpy(info_df, box_df, gender, cbbpy_dir)


def extract_backfill(data_dir: Path):
    """Backfill full seasons missing from Kaggle."""
    seasons = get_season_gap(data_dir)
    if not seasons:
        print("No full seasons to backfill.")
        return True
    print(f"Seasons to scrape: {seasons}")
    for season in seasons:
        scrape_season(season, data_dir / "cbbpy")
    return True


def extract_incremental(data_dir: Path):
    """Scrape games between Kaggle's last date and yesterday."""
    start_date, end_date, season = get_date_gap(data_dir)
    if start_date is None:
        print("Kaggle data is up to date — nothing to scrape.")
        return True
    print(f"Incremental scrape: {start_date} → {end_date} (season {season})")
    scrape_range(start_date, end_date, data_dir / "cbbpy")
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    extract_incremental(repo_root / "data")
