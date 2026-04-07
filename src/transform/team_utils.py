"""Shared team-name matching utilities for transforms.

Maps external team names → Kaggle TeamIDs via:
  1. Manual override map (known mismatches)
  2. Exact match against MTeams.csv
  3. Fuzzy match (fuzzywuzzy, configurable threshold)
"""

from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz, process


def load_kaggle_teams(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "kaggle" / "MTeams.csv")


def build_name_to_id(teams_df: pd.DataFrame) -> dict[str, int]:
    return dict(zip(teams_df["TeamName"], teams_df["TeamID"]))


def fuzzy_match(name: str, choices: list[str], threshold: int = 80) -> str | None:
    result = process.extractOne(name, choices, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def match_team_name(name: str, name_to_id: dict[str, int],
                    manual_map: dict[str, str] | None = None,
                    threshold: int = 80) -> int | None:
    """Resolve a team name to a Kaggle TeamID.

    Tries: manual override → exact match → fuzzy match.
    Returns None if no match found.
    """
    if manual_map and name in manual_map:
        kaggle_name = manual_map[name]
        tid = name_to_id.get(kaggle_name)
        if tid is not None:
            return tid

    tid = name_to_id.get(name)
    if tid is not None:
        return tid

    kaggle_names = list(name_to_id.keys())
    matched = fuzzy_match(name, kaggle_names, threshold)
    if matched:
        return name_to_id[matched]

    return None
