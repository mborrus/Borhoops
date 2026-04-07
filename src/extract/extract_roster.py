"""Extract NCAA basketball roster data from ESPN's public API.

Pulls per-team roster information (class year, injury status) for experience features.

ESPN endpoint (no API key needed):
  https://site.api.espn.com/apis/site/v2/sports/basketball/
    mens-college-basketball/teams/{espn_id}/roster?season={year}

Saves one CSV per season: data/roster/roster_{season}.csv
Then a transform step produces: data/derived/roster_experience.csv
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{espn_id}/roster"
)

FIRST_SEASON = 2015
REQUEST_DELAY = 0.2

CLASS_MAP = {"fr": 1, "so": 2, "jr": 3, "sr": 4}

COLUMNS = [
    "Season", "ESPN_TeamID", "PlayerName", "ClassYear", "ClassNum",
    "Injured", "Position",
]


def _fetch_roster(espn_id, season):
    """Fetch roster for one team/season. Returns list of player dicts or None."""
    url = ROSTER_URL.format(espn_id=espn_id)
    try:
        resp = requests.get(url, params={"season": season}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    athletes = data.get("athletes", [])
    players = []
    for athlete in athletes:
        exp = athlete.get("experience", {})
        class_year = exp.get("abbreviation", "")
        class_num = CLASS_MAP.get(class_year.lower(), 0)

        injuries = athlete.get("injuries", [])
        injured = 1 if injuries else 0

        players.append({
            "Season": season,
            "ESPN_TeamID": espn_id,
            "PlayerName": athlete.get("displayName", ""),
            "ClassYear": class_year,
            "ClassNum": class_num,
            "Injured": injured,
            "Position": athlete.get("position", {}).get("abbreviation", ""),
        })
    return players


def _get_team_ids(data_dir):
    """Load all ESPN team IDs from the crosswalk."""
    cw_path = Path(data_dir) / "derived" / "espn_kaggle_crosswalk.csv"
    if not cw_path.exists():
        return []
    df = pd.read_csv(cw_path)
    return df["espn_id"].dropna().astype(int).tolist()


def extract_season(data_dir, season):
    """Extract roster data for all teams in one season."""
    team_ids = _get_team_ids(data_dir)
    if not team_ids:
        print("No team IDs found in crosswalk.")
        return False

    all_players = []
    for i, espn_id in enumerate(team_ids):
        players = _fetch_roster(espn_id, season)
        if players:
            all_players.extend(players)
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(team_ids)} teams fetched...")
        time.sleep(REQUEST_DELAY)

    if not all_players:
        print(f"  No roster data found for season {season}")
        return False

    df = pd.DataFrame(all_players, columns=COLUMNS)
    roster_dir = Path(data_dir) / "roster"
    roster_dir.mkdir(parents=True, exist_ok=True)
    out_path = roster_dir / f"roster_{season}.csv"
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} players from {df['ESPN_TeamID'].nunique()} teams → {out_path.name}")
    return True


def extract_current(data_dir):
    from datetime import datetime
    current_year = datetime.now().year
    season = current_year if datetime.now().month >= 5 else current_year
    print(f"Extracting roster for season {season}...")
    return extract_season(data_dir, season)


def extract_backfill(data_dir):
    from datetime import datetime
    current_year = datetime.now().year
    end_season = current_year if datetime.now().month >= 5 else current_year
    success = True
    for season in range(FIRST_SEASON, end_season + 1):
        print(f"Extracting roster for season {season}...")
        if not extract_season(data_dir, season):
            success = False
    return success
