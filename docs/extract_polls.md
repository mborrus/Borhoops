# AP & Coaches Poll Extractor

Extracts weekly AP Top 25 and Coaches Poll rankings from ESPN's Core API. Historical data back to 2003. No API key required.

## API Endpoint

```
GET https://sports.core.api.espn.com/v2/sports/basketball
    /leagues/mens-college-basketball/seasons/{year}/types/2
    /weeks/{week}/rankings/{poll_id}
```

- `poll_id=1`: AP Top 25
- `poll_id=2`: Coaches Poll
- `week`: 1-21 (NCAA season weeks)
- Returns ranked teams with `$ref` URLs to team objects (ESPN team IDs parsed from URL)

**Important**: The site API (`site.api.espn.com`) always returns *current* rankings regardless of season/week params. The Core API (`sports.core.api.espn.com`) returns true historical data.

## Data Coverage

- **Seasons**: 2003 through present
- **Polls**: AP Top 25, Coaches Poll
- **Granularity**: Weekly (up to 21 weeks per season)
- **Fields**: Rank, previous rank, points, first-place votes, ESPN team ID

## Output Format

One CSV per season at `data/polls/polls_{season}.csv`:

| Column | Type | Description |
|--------|------|-------------|
| Season | int | NCAA season year |
| Week | int | Poll week (1-21) |
| Poll | str | "AP" or "Coaches" |
| Rank | int | Current rank (1-25) |
| ESPN_TeamID | int | ESPN team identifier |
| Previous | int | Previous week's rank (null if new entry) |
| Points | int | Poll points received |
| FirstPlaceVotes | int | Number of first-place votes |

## Pipeline Commands

```bash
# Current season
python src/runner.py --source polls

# Backfill all seasons (2003-present, ~17 min)
python src/runner.py --source polls --backfill

# Transform: map ESPN IDs to Kaggle TeamIDs
python src/runner.py --source transform-polls
```

## Feature Derivation

From the raw poll data, these features can be derived per (Season, TeamID):

- **poll_rank**: Current AP rank (unranked = 26 or NaN)
- **poll_rank_change**: Rank change from previous week (positive = improving)
- **weeks_ranked**: Number of weeks in the Top 25 this season
- **highest_rank**: Best rank achieved this season
- **poll_points**: Normalized poll points (captures "how ranked", not just "if ranked")

## Idempotency

Backfill skips seasons where the CSV already exists. Delete the CSV to re-fetch.

## Rate Limiting

- 0.1s delay between requests
- ~840 requests per season (2 polls × 21 weeks × 25 teams, but team resolution not needed — ESPN IDs parsed from $ref URLs)
- Full backfill: ~10K requests total, ~17 minutes

## Airflow/BQ Notes

- Weekly schedule: run once per week during the season (Nov-Mar)
- Append-only per season: new weeks get added to the season CSV
- Transform is a simple ESPN→Kaggle ID join via crosswalk
