"""Transform ESPN odds data: map ESPN team IDs to Kaggle TeamIDs, compute DayNum.

Reads per-season odds CSVs from data/odds/, joins with the ESPN→Kaggle crosswalk,
computes DayNum from DayZero, and writes a single combined output.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENDER_MAP = {"mens": "M", "womens": "W"}


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_crosswalk(data_dir: Path) -> dict[int, int]:
    """Return ESPN ID → Kaggle men's ID lookup."""
    cw = pd.read_csv(data_dir / "derived" / "espn_kaggle_crosswalk.csv")
    return cw.set_index("espn_id")["kaggle_m_id"].to_dict()


def load_day_zero(data_dir: Path) -> dict[int, datetime]:
    """Return Season → DayZero datetime for men's seasons."""
    seasons = pd.read_csv(data_dir / "kaggle" / "MSeasons.csv")
    return {
        row.Season: datetime.strptime(row.DayZero, "%m/%d/%Y")
        for row in seasons.itertuples()
    }


def compute_day_num(date_str: str, day_zero_map: dict[int, datetime]) -> int | None:
    dt = pd.to_datetime(date_str)
    season = dt.year + 1 if dt.month >= 5 else dt.year
    dz = day_zero_map.get(season)
    if dz is None:
        return None
    return (dt - dz).days


def transform_odds(data_dir: Path) -> bool:
    odds_dir = data_dir / "odds"
    if not odds_dir.exists():
        print("No odds data directory found.")
        return False

    csv_files = sorted(odds_dir.glob("ncaab_odds_*.csv"))
    if not csv_files:
        print("No odds CSV files found. Run extract_odds first.")
        return False

    id_lookup = load_crosswalk(data_dir)
    day_zero_map = load_day_zero(data_dir)

    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Map ESPN IDs to Kaggle IDs
    combined["HomeTeamID"] = combined["HomeESPN_ID"].map(id_lookup)
    combined["AwayTeamID"] = combined["AwayESPN_ID"].map(id_lookup)

    # Compute DayNum
    combined["DayNum"] = combined["Date"].apply(
        lambda d: compute_day_num(d, day_zero_map)
    )

    # Report missing mappings
    missing_home = combined["HomeTeamID"].isna().sum()
    missing_away = combined["AwayTeamID"].isna().sum()
    missing_day = combined["DayNum"].isna().sum()
    if missing_home or missing_away:
        print(f"  Unmatched teams: {missing_home} home, {missing_away} away")
    if missing_day:
        print(f"  Missing DayNum (no DayZero for season): {missing_day}")

    # Drop rows with missing IDs or DayNum
    before = len(combined)
    combined = combined.dropna(subset=["HomeTeamID", "AwayTeamID", "DayNum"])
    combined["HomeTeamID"] = combined["HomeTeamID"].astype(int)
    combined["AwayTeamID"] = combined["AwayTeamID"].astype(int)
    combined["DayNum"] = combined["DayNum"].astype(int)
    after = len(combined)

    if before != after:
        print(f"  Dropped {before - after} rows with missing mappings")

    # Write output
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "ncaab_odds.csv"
    combined.to_csv(out_path, index=False)
    print(f"  {len(combined)} odds rows → {out_path.name}")
    return True


if __name__ == "__main__":
    config = load_config()
    data_dir = REPO_ROOT / config["data_dir"]
    transform_odds(data_dir)
