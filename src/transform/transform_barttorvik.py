"""Transform Barttorvik T-Rank data: map team names to Kaggle TeamIDs.

Reads per-season CSVs from data/barttorvik/, fuzzy-matches team names
against MTeams.csv, and writes a unified ratings file with Kaggle IDs.
"""

from pathlib import Path

import pandas as pd

from transform.team_utils import build_name_to_id, load_kaggle_teams, match_team_name

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Manual overrides where fuzzy matching fails.
# Barttorvik name → Kaggle TeamName (from MTeams.csv)
MANUAL_MAP = {
    # Full name → abbreviation mismatches
    "Abilene Christian": "Abilene Chr",
    "Albany": "SUNY Albany",
    "American": "American Univ",
    "Boston University": "Boston Univ",
    "Central Connecticut": "Central Conn",
    "Central Michigan": "C Michigan",
    "Connecticut": "UConn",
    "Detroit Mercy": "Detroit",
    "Eastern Illinois": "E Illinois",
    "Eastern Kentucky": "E Kentucky",
    "Eastern Michigan": "E Michigan",
    "Fairleigh Dickinson": "F Dickinson",
    "Florida Gulf Coast": "FGCU",
    "Houston Christian": "Houston Chr",
    "Illinois Chicago": "IL Chicago",
    "IU Indy": "IUPUI",
    "Kent St.": "Kent",
    "LIU": "LIU Brooklyn",
    "Louisiana Monroe": "ULM",
    "Loyola Chicago": "Loyola-Chicago",
    "Loyola Maryland": "Loyola MD",
    "Maryland Eastern Shore": "MD E Shore",
    "Middle Tennessee": "MTSU",
    "Nebraska Omaha": "NE Omaha",
    "North Carolina Central": "NC Central",
    "Northern Colorado": "N Colorado",
    "Northern Illinois": "N Illinois",
    "Northern Kentucky": "N Kentucky",
    "Purdue Fort Wayne": "PFW",
    "Sacramento St.": "CS Sacramento",
    "Saint Mary's": "St Mary's CA",
    "Southeastern Louisiana": "SE Louisiana",
    "Southern Illinois": "S Illinois",
    "Stephen F. Austin": "SF Austin",
    "Tennessee Martin": "TN Martin",
    "Texas A&M Corpus Chris": "TAM C. Christi",
    "The Citadel": "Citadel",
    "UMKC": "Missouri KC",
    "UT Rio Grande Valley": "UTRGV",
    "UTSA": "UT San Antonio",
    "Western Illinois": "W Illinois",
    "Western Kentucky": "WKU",
    "Western Michigan": "W Michigan",
    "SIU Edwardsville": "SIUE",
    "FIU": "Florida Intl",
    # Common alternate names
    "UConn": "UConn",
    "St. John's": "St John's",
    "Saint John's": "St John's",
    "Saint Joseph's": "St Joseph's PA",
    "Saint Peter's": "St Peter's",
    "Saint Louis": "Saint Louis",
    "Saint Bonaventure": "St Bonaventure",
    "Miami": "Miami FL",
    "Miami FL": "Miami FL",
    "Miami OH": "Miami OH",
    "USC": "Southern California",
    "Ole Miss": "Mississippi",
    "Pitt": "Pittsburgh",
    "Penn": "Pennsylvania",
    "UMass": "Massachusetts",
    "UMass Lowell": "MA Lowell",
    "UCSB": "UC Santa Barbara",
    "Cal St. Fullerton": "CS Fullerton",
    "Cal St. Northridge": "CS Northridge",
    "Cal St. Bakersfield": "CS Bakersfield",
    "ETSU": "East Tennessee St",
    "Loyola Marymount": "Loyola Marymount",
    "Texas A&M Corpus Christi": "TAM C. Christi",
    "McNeese State": "McNeese St",
    "Grambling State": "Grambling",
    "Bethune Cookman": "Bethune-Cookman",
    "Little Rock": "Ark Little Rock",
    "Southeast Missouri State": "SE Missouri St",
    "UNC Greensboro": "NC Greensboro",
    "UNC Wilmington": "NC Wilmington",
    "UIC": "IL Chicago",
    "Winston Salem St.": "W Salem St",
}


def map_teams(barttorvik_df: pd.DataFrame, name_to_id: dict[str, int]) -> pd.DataFrame:
    team_ids = []
    unmatched = []

    for team in barttorvik_df["Team"]:
        tid = match_team_name(team, name_to_id, MANUAL_MAP)
        team_ids.append(tid)
        if tid is None:
            unmatched.append(team)

    barttorvik_df = barttorvik_df.copy()
    barttorvik_df["TeamID"] = team_ids

    if unmatched:
        print(f"  WARNING: {len(unmatched)} teams unmatched: {unmatched[:10]}")

    return barttorvik_df


def transform_barttorvik(data_dir: Path) -> bool:
    barttorvik_dir = data_dir / "barttorvik"
    if not barttorvik_dir.exists():
        print("No Barttorvik data directory found.")
        return False

    csv_files = sorted(barttorvik_dir.glob("trank_*.csv"))
    if not csv_files:
        print("No Barttorvik CSV files found.")
        return False

    teams_df = load_kaggle_teams(data_dir)
    name_to_id = build_name_to_id(teams_df)

    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df = map_teams(df, name_to_id)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    matched = combined["TeamID"].notna()
    print(f"  Matched: {matched.sum()}/{len(combined)} teams across {len(csv_files)} seasons")

    combined = combined[matched].copy()
    combined["TeamID"] = combined["TeamID"].astype(int)

    out_cols = ["Season", "TeamID", "Team", "Conf", "AdjO", "AdjD", "AdjT",
                "Barthag", "Rank", "SOS", "NonConSOS", "ConSOS", "WAB"]
    out_cols = [c for c in out_cols if c in combined.columns]

    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "barttorvik_ratings.csv"
    combined[out_cols].to_csv(out_path, index=False)
    print(f"  {len(combined)} rows → {out_path.name}")
    return True


if __name__ == "__main__":
    import yaml
    with open(REPO_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    data_dir = REPO_ROOT / config["data_dir"]
    transform_barttorvik(data_dir)
