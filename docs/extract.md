# Data Extraction

## Setup

```bash
source ballenvy/bin/activate
source secrets.sh            # sets KAGGLE_API_TOKEN
pip install -e ./CBBpy       # local modified version
```

## Usage

```bash
python src/runner.py                              # all sources (kaggle + espn + transform + predict)
python src/runner.py --source kaggle              # just kaggle
python src/runner.py --source espn                # incremental espn
python src/runner.py --source espn --backfill     # full season backfill
python src/runner.py --source barttorvik          # current season T-Rank
python src/runner.py --source barttorvik --backfill  # all seasons 2008+
python src/runner.py --source odds                # current season ESPN odds
python src/runner.py --source odds --backfill     # all seasons 2013+
python src/runner.py --source polls               # current season AP/Coaches polls
python src/runner.py --source polls --backfill    # all seasons 2003+
python src/runner.py --source vegas               # current odds (needs ODDS_API_KEY)
python src/runner.py --source transform           # ESPN → Kaggle transform
python src/runner.py --source transform-barttorvik
python src/runner.py --source transform-odds
python src/runner.py --source transform-polls
python src/runner.py --source transform-vegas
python src/runner.py --source predict             # run Elo model + write submission
```

## Sources

| Source | Script | Output | Auth | Coverage |
|--------|--------|--------|------|----------|
| Kaggle | `extract_kaggle.py` | `data/kaggle/` (35 CSVs) | `KAGGLE_API_TOKEN` | 1985+ |
| ESPN | `extract_espn.py` | `data/cbbpy/` (game info + box scores) | None | 2002+ |
| Barttorvik | `extract_barttorvik.py` | `data/barttorvik/trank_{year}.csv` | None | 2008+ |
| ESPN Odds | `extract_odds.py` | `data/odds/ncaab_odds_{year}.csv` | None | 2013+ |
| AP/Coaches Polls | `extract_polls.py` | `data/polls/polls_{year}.csv` | None | 2003+ |
| Vegas (Odds API) | `extract_vegas.py` | `data/vegas/odds_{year}.csv` | `ODDS_API_KEY` | 2020+ |

## Transforms

| Transform | Input | Output | Method |
|-----------|-------|--------|--------|
| ESPN | `data/cbbpy/` | `data/derived/M_combined.csv`, `W_combined.csv` | ESPN→Kaggle ID crosswalk |
| Barttorvik | `data/barttorvik/` | `data/derived/barttorvik_ratings.csv` | Team name fuzzy matching |
| Odds | `data/odds/` | `data/derived/ncaab_odds.csv` | ESPN→Kaggle ID crosswalk |
| Polls | `data/polls/` | `data/derived/poll_rankings.csv` | ESPN→Kaggle ID crosswalk |
| Vegas | `data/vegas/` | `data/derived/vegas_lines.csv` | Team name fuzzy matching |

## Shared Utilities

- `src/transform/team_utils.py` — shared team-name matching for Barttorvik, Vegas, and any future name-based source. See [docs/transform_team_utils.md](transform_team_utils.md).
- `data/derived/espn_kaggle_crosswalk.csv` — ESPN ID ↔ Kaggle ID mapping, used by ESPN, odds, and polls transforms.

## Parallel Backfill

For odds (the slowest backfill), use the parallel script:

```bash
python scripts/backfill_odds_parallel.py --workers 13
```

## Detailed Docs

- [extract_barttorvik.md](extract_barttorvik.md) — T-Rank data, manual team name overrides
- [extract_odds.md](extract_odds.md) — ESPN Core API endpoints, rate limiting, parallelization
- [extract_polls.md](extract_polls.md) — AP/Coaches polls, Core API vs Site API differences
- [extract_vegas.md](extract_vegas.md) — The Odds API research, paid plan details
