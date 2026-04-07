"""Transform Vegas odds data: map team names to Kaggle TeamIDs.

Status: STUB — depends on Vegas extract being implemented with real data.
Currently handles the Odds API output format if available.
"""

from pathlib import Path

import pandas as pd

from transform.team_utils import build_name_to_id, load_kaggle_teams, match_team_name

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def transform_vegas(data_dir: Path) -> bool:
    vegas_dir = data_dir / "vegas"
    if not vegas_dir.exists():
        print("No Vegas data directory found.")
        return False

    csv_files = sorted(vegas_dir.glob("odds_*.csv"))
    if not csv_files:
        print("No Vegas CSV files found. Run extract_vegas first (needs ODDS_API_KEY).")
        return True  # not a failure, just no data yet

    teams_df = load_kaggle_teams(data_dir)
    name_to_id = build_name_to_id(teams_df)

    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["HomeTeamID"] = df["HomeTeam"].apply(lambda n: match_team_name(n, name_to_id, threshold=75))
        df["AwayTeamID"] = df["AwayTeam"].apply(lambda n: match_team_name(n, name_to_id, threshold=75))
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    matched = combined["HomeTeamID"].notna() & combined["AwayTeamID"].notna()
    print(f"  Matched: {matched.sum()}/{len(combined)} odds rows")

    combined = combined[matched].copy()
    combined["HomeTeamID"] = combined["HomeTeamID"].astype(int)
    combined["AwayTeamID"] = combined["AwayTeamID"].astype(int)

    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "vegas_lines.csv"
    combined.to_csv(out_path, index=False)
    print(f"  {len(combined)} rows → {out_path.name}")
    return True


if __name__ == "__main__":
    import yaml
    with open(REPO_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    data_dir = REPO_ROOT / config["data_dir"]
    transform_vegas(data_dir)
