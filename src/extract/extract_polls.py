"""Extract AP and Coaches Poll rankings from ESPN Core API.

Historical weekly rankings back to at least 2003. No API key needed.

ESPN Core API endpoint:
  GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
      /seasons/{year}/types/2/weeks/{week}/rankings/{poll_id}

  poll_id: 1 = AP Top 25, 2 = Coaches Poll
"""

import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

RANKINGS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball"
    "/leagues/mens-college-basketball/seasons/{season}/types/2"
    "/weeks/{week}/rankings/{poll_id}"
)

FIRST_SEASON = 2003
MAX_WEEK = 21
REQUEST_DELAY = 0.1

COLUMNS = [
    "Season", "Week", "Poll", "Rank", "ESPN_TeamID", "Previous",
    "Points", "FirstPlaceVotes",
]


def current_season() -> int:
    today = datetime.now()
    return today.year + 1 if today.month >= 5 else today.year


def _extract_espn_id(ref_url: str) -> int | None:
    """Parse ESPN team ID from a $ref URL like .../teams/2509?..."""
    match = re.search(r"/teams/(\d+)", ref_url)
    return int(match.group(1)) if match else None


def fetch_poll(season: int, week: int, poll_id: int = 1) -> list[dict]:
    """Fetch one week's poll. Returns list of rank dicts."""
    url = RANKINGS_URL.format(season=season, week=week, poll_id=poll_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    data = resp.json()
    ranks = data.get("ranks", [])
    if not ranks:
        return []

    poll_name = "AP" if poll_id == 1 else "Coaches"
    rows = []
    for entry in ranks:
        team_ref = entry.get("team", {}).get("$ref", "")
        espn_id = _extract_espn_id(team_ref)
        if espn_id is None:
            continue

        rows.append({
            "Season": season,
            "Week": week,
            "Poll": poll_name,
            "Rank": entry.get("current"),
            "ESPN_TeamID": espn_id,
            "Previous": entry.get("previous"),
            "Points": entry.get("points"),
            "FirstPlaceVotes": entry.get("firstPlaceVotes"),
        })

    return rows


def extract_season(data_dir: Path, season: int) -> int:
    """Extract all weekly polls for one season. Returns row count."""
    polls_dir = data_dir / "polls"
    polls_dir.mkdir(parents=True, exist_ok=True)
    csv_path = polls_dir / f"polls_{season}.csv"

    all_rows = []
    for poll_id, poll_name in [(1, "AP"), (2, "Coaches")]:
        empty_count = 0
        for week in range(1, MAX_WEEK + 1):
            rows = fetch_poll(season, week, poll_id)
            time.sleep(REQUEST_DELAY)

            if not rows:
                empty_count += 1
                if empty_count >= 3:
                    break  # past end of season
                continue

            empty_count = 0
            all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows, columns=COLUMNS)
        df.to_csv(csv_path, index=False)

    return len(all_rows)


def extract_current(data_dir: Path) -> bool:
    season = current_season()
    print(f"Extracting polls for {season}...")
    count = extract_season(data_dir, season)
    print(f"  {season}: {count} poll entries")
    return True


def extract_backfill(data_dir: Path, start_season: int = FIRST_SEASON) -> bool:
    end_season = current_season()
    print(f"Backfilling polls: {start_season}–{end_season}")

    total = 0
    for season in range(start_season, end_season + 1):
        csv_path = data_dir / "polls" / f"polls_{season}.csv"
        if csv_path.exists():
            existing = len(pd.read_csv(csv_path))
            print(f"  {season}: already exists ({existing} entries), skipping")
            total += existing
            continue

        count = extract_season(data_dir, season)
        total += count
        print(f"  {season}: {count} poll entries")

    print(f"Total: {total} poll entries")
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    extract_backfill(repo_root / "data")
