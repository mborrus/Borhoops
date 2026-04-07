# Vegas Lines Extractor

## Status: SKELETON

No free, comprehensive source for historical NCAAB spreads exists. The extractor supports The Odds API if a key is configured.

## Research Findings (April 2026)

| Source | Data | Cost | Coverage |
|--------|------|------|----------|
| **The Odds API** | Spreads, moneylines, totals | ~$20/mo (historical) | Nov 2020+ |
| BigDataBall | Game stats + odds | ~$30/season | Varies |
| SportsbookReviewOnline | Was free Excel files | Free (defunct) | Site now 404 |
| Covers.com | Browse only | Free (no bulk) | No API |
| Kaggle | No NCAAB spreads dataset | — | — |

## Recommended Path

1. **Best option**: The Odds API paid plan (~$20/mo) for data from Nov 2020+
2. **Pre-2020**: Purchase from BigDataBall, or search Wayback Machine for SBRO archives
3. **Free alternative**: Use Barttorvik's Barthag rating as proxy — highly correlated with Vegas implied win probability, available 2008+

## Pipeline (when configured)

```
# Requires: export ODDS_API_KEY=your_key_here
Extract: python src/runner.py --source vegas
  → data/vegas/odds_{year}.csv

Transform: python src/runner.py --source transform-vegas
  → data/derived/vegas_lines.csv (Kaggle TeamIDs)
```

Without `ODDS_API_KEY`, the extract gracefully skips (not a failure).

## Airflow/BQ Notes

- Extract is idempotent, parameterized by season
- Vegas data would partition by Season with TeamID join keys
- Consider Barthag as a zero-cost substitute until Vegas data justifies the spend
