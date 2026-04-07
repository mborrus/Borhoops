"""Extract historical NCAA basketball Vegas lines.

Status: SKELETON — no free bulk source exists for historical NCAAB spreads.

Research findings (April 2026):
  - The Odds API (the-odds-api.com): best option, NCAAB data from Nov 2020.
    Spreads, moneylines, totals. Paid plans only for historical data.
    Free tier: 500 credits, but historical calls cost 10x (so ~50 calls max).
  - SportsbookReviewOnline: was the go-to free source (Excel files back to ~2007),
    but the site is now returning 404s. Appears defunct.
  - BigDataBall: game-level data with odds, paid per-season ($30+/season).
  - Covers.com: no bulk download, no API. Manual scraping only.
  - Kaggle: no comprehensive NCAAB spreads dataset found.

Recommended path forward:
  1. Sign up for The Odds API paid plan (~$20/mo) for historical data from 2020+.
  2. For pre-2020 data: purchase from BigDataBall or look for Wayback Machine
     captures of SportsbookReviewOnline's old Excel files.
  3. Alternative: use Barttorvik Barthag ratings as a proxy for market expectations.
     Barthag is highly correlated with Vegas implied win probability.

If The Odds API key is set (ODDS_API_KEY env var), this module can pull
current-season lines. Historical pulls require a paid plan.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_ncaab"
REGIONS = "us"
MARKETS = "spreads,totals"
REQUEST_DELAY = 1.0


def get_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY")


def current_season() -> int:
    today = datetime.now()
    return today.year + 1 if today.month >= 5 else today.year


def fetch_current_odds(api_key: str) -> pd.DataFrame | None:
    """Fetch current NCAAB odds (spreads + totals) from The Odds API."""
    url = f"{ODDS_API_BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR fetching odds: {e}")
        return None

    events = resp.json()
    if not events:
        print("  No events returned (offseason?)")
        return None

    rows = []
    for event in events:
        base = {
            "EventID": event["id"],
            "Sport": event["sport_key"],
            "HomeTeam": event["home_team"],
            "AwayTeam": event["away_team"],
            "CommenceTime": event["commence_time"],
        }
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        side = "Home" if outcome["name"] == event["home_team"] else "Away"
                        base[f"Spread_{side}"] = outcome.get("point")
                        base[f"SpreadPrice_{side}"] = outcome.get("price")
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        base[f"Total_{outcome['name']}"] = outcome.get("point")
                        base[f"TotalPrice_{outcome['name']}"] = outcome.get("price")
            base["Bookmaker"] = book["key"]
            rows.append(dict(base))

    return pd.DataFrame(rows)


def extract_current(data_dir: Path) -> bool:
    """Pull current odds if API key is available."""
    api_key = get_api_key()
    if not api_key:
        print("No ODDS_API_KEY set — skipping Vegas extract.")
        print("Set ODDS_API_KEY env var to enable. See module docstring for details.")
        return True  # not a failure, just not configured

    print("Fetching current NCAAB odds from The Odds API...")
    df = fetch_current_odds(api_key)
    if df is None:
        return False

    vegas_dir = data_dir / "vegas"
    vegas_dir.mkdir(parents=True, exist_ok=True)
    season = current_season()
    path = vegas_dir / f"odds_{season}.csv"
    df.to_csv(path, index=False)
    print(f"  {len(df)} odds rows → {path.name}")
    return True


def extract_backfill(data_dir: Path) -> bool:
    """Historical odds require a paid Odds API plan. Not implemented."""
    print("Vegas historical backfill not implemented.")
    print("Requires The Odds API paid plan (~$20/mo) for data from Nov 2020.")
    print("See module docstring for full research notes and alternatives.")
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    extract_current(repo_root / "data")
