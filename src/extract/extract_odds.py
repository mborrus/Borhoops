"""Extract NCAA basketball odds from ESPN's public API.

Two modes:
  - Current: pull odds for all games in the current season
  - Backfill: pull odds for all seasons from 2012-13 through present

ESPN endpoints (no API key needed):
  - Scoreboard: all games for a given date
  - Odds: spread, over/under, moneyline per event

Saves one CSV per season: data/odds/ncaab_odds_{season}.csv
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball"
    "/leagues/mens-college-basketball/events/{event_id}"
    "/competitions/{event_id}/odds"
)

FIRST_SEASON = 2013
ODDS_DELAY = 0.2  # seconds between odds requests

COLUMNS = [
    "Season", "Date", "ESPN_EventID", "HomeTeam", "AwayTeam",
    "HomeESPN_ID", "AwayESPN_ID", "Spread", "OverUnder",
    "HomeML", "AwayML", "HomeSpreadOdds", "AwaySpreadOdds", "Provider",
]


def current_season() -> int:
    today = datetime.now()
    return today.year + 1 if today.month >= 5 else today.year


def season_date_range(season: int) -> tuple[datetime, datetime]:
    """Return (start, end) dates for a given NCAA season."""
    start = datetime(season - 1, 11, 1)
    end = min(datetime(season, 4, 30), datetime.now() - timedelta(days=1))
    return start, end


def fetch_scoreboard(date: datetime) -> list[dict]:
    """Fetch all games for a given date. Returns list of event dicts."""
    date_str = date.strftime("%Y%m%d")
    params = {"dates": date_str, "limit": 300, "groups": 50}
    try:
        resp = requests.get(SCOREBOARD_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR scoreboard {date_str}: {e}")
        return []

    data = resp.json()
    events = []
    for event in data.get("events", []):
        event_id = event.get("id")
        comps = event.get("competitions", [])
        if not comps:
            continue

        competitors = comps[0].get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        events.append({
            "event_id": event_id,
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_espn_id": int(home.get("team", {}).get("id", 0)),
            "away_espn_id": int(away.get("team", {}).get("id", 0)),
        })

    return events


def fetch_odds(event_id: str) -> dict | None:
    """Fetch odds for a single event. Returns parsed odds dict or None."""
    url = ODDS_URL.format(event_id=event_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    data = resp.json()
    items = data.get("items", [])
    if not items:
        return None

    # Take the first provider (usually consensus/primary)
    item = items[0]
    provider = item.get("provider", {}).get("name", "unknown")

    odds = {"Provider": provider}

    # Spread (negative = favored)
    odds["Spread"] = item.get("spread")
    odds["OverUnder"] = item.get("overUnder")

    # Moneylines and spread odds from homeTeamOdds / awayTeamOdds
    home_odds = item.get("homeTeamOdds", {})
    away_odds = item.get("awayTeamOdds", {})

    odds["HomeML"] = home_odds.get("moneyLine")
    odds["AwayML"] = away_odds.get("moneyLine")
    odds["HomeSpreadOdds"] = home_odds.get("spreadOdds")
    odds["AwaySpreadOdds"] = away_odds.get("spreadOdds")

    return odds


def scrape_date(date: datetime, season: int) -> list[dict]:
    """Scrape all events and odds for a single date. Returns list of row dicts."""
    events = fetch_scoreboard(date)
    if not events:
        return []

    date_str = date.strftime("%Y-%m-%d")
    rows = []

    for event in events:
        odds = fetch_odds(event["event_id"])
        time.sleep(ODDS_DELAY)

        if odds is None:
            continue

        # Skip if no useful data
        if odds.get("Spread") is None and odds.get("OverUnder") is None:
            continue

        rows.append({
            "Season": season,
            "Date": date_str,
            "ESPN_EventID": event["event_id"],
            "HomeTeam": event["home_team"],
            "AwayTeam": event["away_team"],
            "HomeESPN_ID": event["home_espn_id"],
            "AwayESPN_ID": event["away_espn_id"],
            "Spread": odds.get("Spread"),
            "OverUnder": odds.get("OverUnder"),
            "HomeML": odds.get("HomeML"),
            "AwayML": odds.get("AwayML"),
            "HomeSpreadOdds": odds.get("HomeSpreadOdds"),
            "AwaySpreadOdds": odds.get("AwaySpreadOdds"),
            "Provider": odds.get("Provider"),
        })

    return rows


def load_existing_dates(csv_path: Path) -> set[str]:
    """Load dates already scraped for a season."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["Date"])
    return set(df["Date"].unique())


def save_season(rows: list[dict], csv_path: Path):
    """Append new rows to the season CSV (or create it)."""
    new_df = pd.DataFrame(rows, columns=COLUMNS)
    new_df["ESPN_EventID"] = new_df["ESPN_EventID"].astype(str)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["ESPN_EventID"] = existing["ESPN_EventID"].astype(str)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Date", "ESPN_EventID"], keep="last"
        )
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


def extract_season(data_dir: Path, season: int, skip_existing: bool = True) -> int:
    """Scrape odds for an entire season. Returns count of new rows."""
    odds_dir = data_dir / "odds"
    odds_dir.mkdir(parents=True, exist_ok=True)
    csv_path = odds_dir / f"ncaab_odds_{season}.csv"

    existing_dates = load_existing_dates(csv_path) if skip_existing else set()
    start, end = season_date_range(season)

    total_new = 0
    date = start
    while date <= end:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in existing_dates:
            date += timedelta(days=1)
            continue

        rows = scrape_date(date, season)
        if rows:
            save_season(rows, csv_path)
            total_new += len(rows)

        date += timedelta(days=1)

    return total_new


def extract_current(data_dir: Path) -> bool:
    season = current_season()
    print(f"Extracting ESPN odds for {season}...")
    new_rows = extract_season(data_dir, season)
    print(f"  {season}: {new_rows} new odds rows")
    return True


def extract_backfill(data_dir: Path, start_season: int = FIRST_SEASON) -> bool:
    end_season = current_season()
    print(f"Backfilling ESPN odds: {start_season}–{end_season}")

    for season in range(start_season, end_season + 1):
        print(f"  Season {season}...")
        new_rows = extract_season(data_dir, season)
        print(f"    {new_rows} new odds rows")

    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    extract_current(repo_root / "data")
