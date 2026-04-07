"""Extract T-Rank ratings from barttorvik.com.

Two modes:
  - Current: pull current season only
  - Backfill: pull all available seasons (2008–present)

Data is freely available as CSV at barttorvik.com/{year}_team_results.csv.
Historical data goes back to 2008 (341+ teams per year).
"""

import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://barttorvik.com/{year}_team_results.csv"
FIRST_SEASON = 2008
REQUEST_DELAY = 1.5  # be polite

# The 45 columns in the CSV. We keep the most useful ones.
KEEP_COLS = {
    "rank": "Rank",
    "team": "Team",
    "conf": "Conf",
    "record": "Record",
    "adjoe": "AdjO",
    "adjde": "AdjD",
    "barthag": "Barthag",
    "adjt": "AdjT",
    "oe Rank": "AdjO_Rank",
    "de Rank": "AdjD_Rank",
    "sos": "SOS",
    "ncsos": "NonConSOS",
    "consos": "ConSOS",
    "WAB": "WAB",
    "WAB Rk": "WAB_Rank",
}


def current_season() -> int:
    today = datetime.now()
    return today.year + 1 if today.month >= 5 else today.year


def fetch_season(year: int) -> pd.DataFrame | None:
    url = BASE_URL.format(year=year)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR fetching {year}: {e}")
        return None

    # index_col=False prevents pandas from using the numeric rank column as index
    # (older years like 2008 have sequential rank values that pandas auto-detects as index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
        df = pd.read_csv(StringIO(resp.text), index_col=False)

    if len(df) < 20:
        print(f"  {year}: only {len(df)} teams — skipping (partial data)")
        return None

    # The CSV has two 'rank' columns (overall rank appears twice).
    # pandas suffixes the dupe as 'rank.1'. The first 'rank' is the one we want.
    if "rank.1" in df.columns:
        df = df.drop(columns=["rank.1"])

    rename = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    df = df.rename(columns=rename)
    keep = [v for v in rename.values() if v in df.columns]
    df = df[keep].copy()
    df.insert(0, "Season", year)

    return df


def save_season(df: pd.DataFrame, barttorvik_dir: Path, year: int):
    barttorvik_dir.mkdir(parents=True, exist_ok=True)
    path = barttorvik_dir / f"trank_{year}.csv"
    df.to_csv(path, index=False)
    print(f"  {year}: {len(df)} teams → {path.name}")


def extract_current(data_dir: Path) -> bool:
    year = current_season()
    print(f"Fetching Barttorvik T-Rank for {year}...")
    df = fetch_season(year)
    if df is None:
        return False
    save_season(df, data_dir / "barttorvik", year)
    return True


def extract_backfill(data_dir: Path, start_year: int = FIRST_SEASON) -> bool:
    end_year = current_season()
    barttorvik_dir = data_dir / "barttorvik"
    print(f"Backfilling Barttorvik T-Rank: {start_year}–{end_year}")

    ok = True
    for year in range(start_year, end_year + 1):
        # Skip if already downloaded
        if (barttorvik_dir / f"trank_{year}.csv").exists():
            print(f"  {year}: already exists, skipping")
            continue

        df = fetch_season(year)
        if df is None:
            ok = False
            continue
        save_season(df, barttorvik_dir, year)
        time.sleep(REQUEST_DELAY)

    return ok


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    extract_current(repo_root / "data")
