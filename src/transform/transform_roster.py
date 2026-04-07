"""Transform roster data: aggregate per-team experience stats, map to Kaggle IDs.

Reads per-season roster CSVs from data/roster/, computes experience metrics,
maps ESPN IDs to Kaggle TeamIDs, and writes data/derived/roster_experience.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_crosswalk(data_dir: Path) -> dict[int, int]:
    path = data_dir / "derived" / "espn_kaggle_crosswalk.csv"
    df = pd.read_csv(path)
    return dict(zip(df["espn_id"], df["kaggle_m_id"]))


def transform_roster(data_dir: Path) -> bool:
    roster_dir = data_dir / "roster"
    if not roster_dir.exists():
        print("No roster data directory found.")
        return False

    csv_files = sorted(roster_dir.glob("roster_*.csv"))
    if not csv_files:
        print("No roster CSV files found. Run extract_roster first.")
        return False

    crosswalk = load_crosswalk(data_dir)

    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["TeamID"] = combined["ESPN_TeamID"].map(crosswalk)

    # Drop unmapped teams
    unmatched = combined["TeamID"].isna().sum()
    if unmatched:
        print(f"  {unmatched} rows with unmapped ESPN IDs dropped")
    combined = combined.dropna(subset=["TeamID"])
    combined["TeamID"] = combined["TeamID"].astype(int)

    # Filter to players with valid class info
    has_class = combined["ClassNum"] > 0

    # Aggregate per team per season
    rows = []
    for (season, team_id), group in combined.groupby(["Season", "TeamID"]):
        valid = group[has_class.loc[group.index]]
        if len(valid) == 0:
            avg_exp = np.nan
            senior_pct = np.nan
        else:
            avg_exp = valid["ClassNum"].mean()
            senior_pct = (valid["ClassNum"] == 4).sum() / len(valid)
        rows.append({
            "Season": int(season),
            "TeamID": int(team_id),
            "AvgExperience": round(avg_exp, 3) if not np.isnan(avg_exp) else np.nan,
            "SeniorPct": round(senior_pct, 3) if not np.isnan(senior_pct) else np.nan,
            "RosterSize": len(group),
        })

    result = pd.DataFrame(rows)
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "roster_experience.csv"
    result.to_csv(out_path, index=False)
    print(f"  {len(result)} team-seasons → {out_path.name}")
    return True


if __name__ == "__main__":
    import yaml
    repo_root = Path(__file__).resolve().parent.parent.parent
    with open(repo_root / "config.yaml") as f:
        config = yaml.safe_load(f)
    data_dir = repo_root / config["data_dir"]
    transform_roster(data_dir)
