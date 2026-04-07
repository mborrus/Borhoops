# ESPN Odds Extractor

Scrapes NCAA men's basketball odds (spreads, over/unders, moneylines) from ESPN's public API. No API key required.

## API Endpoints

### 1. Scoreboard (all games for a date)

```
GET https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard
    ?dates=YYYYMMDD&limit=300&groups=50
```

Returns `events[]`, each with `id` (event_id) and `competitions[0].competitors[]` (home/away teams with ESPN IDs).

### 2. Odds (per event)

```
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/{event_id}/competitions/{event_id}/odds
```

Returns `items[]` with `spread`, `overUnder`, and per-team moneylines from the primary provider.

## Data Coverage

- Seasons 2013 through present (Nov 1 start date per season)
- Not all games have odds (small-conference matchups, early-season games)
- Provider varies by era: Caesar's (pre-2020), ESPN BET (2024+), DraftKings (2026+)
- Pre-2013 seasons return empty odds arrays

## Output Format

One CSV per season at `data/odds/ncaab_odds_{season}.csv`:

| Column | Type | Description |
|--------|------|-------------|
| Season | int | NCAA season year (e.g., 2026 for the 2025-26 season) |
| Date | str | Game date (YYYY-MM-DD) |
| ESPN_EventID | str | ESPN event identifier |
| HomeTeam | str | Home team display name |
| AwayTeam | str | Away team display name |
| HomeESPN_ID | int | ESPN team ID (home) |
| AwayESPN_ID | int | ESPN team ID (away) |
| Spread | float | Point spread (negative = home favored) |
| OverUnder | float | Total points over/under |
| HomeML | int | Home moneyline (American odds) |
| AwayML | int | Away moneyline (American odds) |
| HomeSpreadOdds | int | Home spread price (American odds) |
| AwaySpreadOdds | int | Away spread price (American odds) |
| Provider | str | Odds provider name |

## Pipeline Commands

```bash
# Current season only
python src/runner.py --source odds

# Backfill all seasons (2013-present, sequential)
python src/runner.py --source odds --backfill

# Parallel backfill (recommended, ~30-45 min for 13 seasons)
python scripts/backfill_odds_parallel.py --workers 13

# Transform: map ESPN IDs to Kaggle IDs, compute DayNum
python src/runner.py --source transform-odds
```

## Parallel Backfill

`scripts/backfill_odds_parallel.py` uses `ProcessPoolExecutor` to scrape multiple seasons concurrently. Each season writes to its own CSV file, so there's no contention.

```bash
# Default: 6 workers
python scripts/backfill_odds_parallel.py

# Max parallelism: one worker per season
python scripts/backfill_odds_parallel.py --workers 13 --start 2013
```

With 13 workers and 0.2s delay between requests, aggregate throughput is ~65 requests/second.

## Rate Limiting

- 0.2s delay between odds API calls (per worker, per event)
- No delay on scoreboard calls (one per date, lightweight)
- ESPN's undocumented API has no published rate limits
- 13 parallel workers have been tested without issues

## Idempotency

Backfill checks which dates are already present in each season CSV and skips them. Safe to re-run after interruption — it picks up where it left off.

## Transform Pipeline

`src/transform/transform_odds.py` maps ESPN team IDs to Kaggle TeamIDs using the existing crosswalk at `data/derived/espn_kaggle_crosswalk.csv`, and computes DayNum from game dates using `MSeasons.csv` DayZero values.

Output: `data/derived/ncaab_odds.csv`

## Relationship to extract_vegas.py

`extract_vegas.py` is a separate module for The Odds API (paid, multi-bookmaker data from Nov 2020+). This ESPN odds extractor is free and covers longer history (2013+), but provides only one provider's line per game. Both can coexist — ESPN odds for breadth, Odds API for depth on recent seasons.

## Airflow/BQ Notes

- `extract_current()` is designed for daily scheduling (one scoreboard call per date)
- Idempotent by design — safe for retry-on-failure DAGs
- Season CSVs are append-only, suitable for incremental BigQuery loads
- Transform step is a simple ID join, trivial as a BQ SQL view
