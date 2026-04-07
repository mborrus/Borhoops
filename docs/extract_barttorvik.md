# Barttorvik T-Rank Extractor

## Data Source

[barttorvik.com](https://barttorvik.com) — Free college basketball analytics site with tempo-free efficiency metrics (KenPom alternative).

Data is published as CSV at `barttorvik.com/{year}_team_results.csv`. No API key required.

## What's Available

| Metric | Column | Description |
|--------|--------|-------------|
| AdjO | `AdjO` | Adjusted offensive efficiency (pts per 100 possessions) |
| AdjD | `AdjD` | Adjusted defensive efficiency |
| AdjT | `AdjT` | Adjusted tempo (possessions per 40 min) |
| Barthag | `Barthag` | Power rating (win prob vs avg D1 team, neutral court) |
| Rank | `Rank` | Overall T-Rank ranking |
| SOS | `SOS` | Strength of schedule |
| NonConSOS | `NonConSOS` | Non-conference SOS |
| ConSOS | `ConSOS` | Conference SOS |
| WAB | `WAB` | Wins above bubble |

## Historical Coverage

- **2008–present**: Full data (341-365 teams/year)
- **Pre-2008**: Partial data (~14 teams), auto-skipped by extractor

## Pipeline

```
Extract: python src/runner.py --source barttorvik [--backfill]
  → data/barttorvik/trank_{year}.csv (one per season)

Transform: python src/runner.py --source transform-barttorvik
  → data/derived/barttorvik_ratings.csv (all seasons, Kaggle TeamIDs)
```

## Team Name Matching

Barttorvik team names differ from Kaggle names. The transform uses:
1. Manual override map (76 known mismatches like "Connecticut" → "UConn")
2. Fuzzy matching (fuzzywuzzy, 80% threshold) as fallback

Current match rate: 365/365 for 2026.

## Rate Limiting

- `REQUEST_DELAY = 1.5s` between requests during backfill
- Be polite — it's a free resource

## Airflow/BQ Notes

- Extract is idempotent (skips already-downloaded seasons on backfill)
- Parameterized by season year
- Output schema: `Season, TeamID, Team, Conf, AdjO, AdjD, AdjT, Barthag, Rank, SOS, NonConSOS, ConSOS, WAB`
- `Season` is the natural partition key, `TeamID` is the join key to all other tables
