# Borhoops

March Madness prediction model for the Kaggle March Machine Learning Mania competition. Predicts win probabilities for every possible NCAA tournament matchup (men's and women's).

**Read [`docs/golden_rules.md`](docs/golden_rules.md) before writing any code.** These are non-negotiable.

## Project Status

The v1 notebooks have been extracted into a working Python pipeline (`src/`). The extract → transform → predict flow runs end-to-end via `python src/runner.py`. Next phase is hooking it up to BigQuery + Airflow for automation, then improving the model.

## Repo Structure

```
Borhoops/
├── src/                      # v2 Python pipeline
│   ├── extract/              # extract_kaggle.py, extract_espn.py
│   ├── transform/            # transform_espn.py (ESPN → Kaggle format)
│   ├── predict/              # elo.py (Elo engine), submission.py (I/O + predictions)
│   └── runner.py             # Orchestrates all steps (--source kaggle|espn|transform|predict)
├── scripts/                  # One-time data precomputation
│   ├── build_home_lookup.py  # Geocode team home cities → Home_Lookup.csv
│   └── build_hfa.py          # Compute home field advantage → HomeFieldAdvantage.csv
├── tests/                    # 65 tests (pytest)
├── original_notebooks/       # v1 Jupyter notebooks (reference, no longer primary)
├── data/                     # All input data (gitignored), flat by source
│   ├── kaggle/               # Kaggle competition CSVs (men's M* and women's W*)
│   ├── nate/                 # Nate Silver SBCB ratings (scraped from Substack)
│   ├── derived/              # Generated lookup tables (Home_Lookup.csv, HomeFieldAdvantage.csv, espn_kaggle_crosswalk.csv)
│   └── cbbpy/                # CBBpy ESPN scraper downloads (daily game results + locations)
├── Output/                   # Model prediction CSVs in Kaggle submission format
├── config.yaml               # Paths config (data_dir, output_dir)
├── requirements.txt          # Python dependencies
├── secrets.sh                # Credentials (gitignored)
└── ballenvy/                 # Python 3.11 virtual environment (gitignored)
```

## Key Data Files for the Elo Model

The Complicated_Elo notebook (the primary model) uses these Kaggle source files:

| File | Purpose |
|------|---------|
| `MTeams.csv` / `WTeams.csv` | Team ID ↔ name lookup |
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | Game results: scores, H/A/N, overtime (core Elo input) |
| `MGameCities.csv` / `WGameCities.csv` | Which city each game was played in (travel distance calc) |
| `Cities.csv` | City lat/lon lookup |
| `MTeamConferences.csv` / `WTeamConferences.csv` | Conference membership per season (mean reversion target) |
| `SampleSubmissionStage2.csv` | Kaggle submission template — all possible matchup pairs |

For the v2 pipeline, **RegularSeasonCompactResults** and **GameCities** are the two tables that need live/daily updates. Everything else is static or seasonal.

## Elo Model Details (Complicated_Elo)

The model calculates Elo ratings from scratch using all regular season games back to 1985 (men) / 1998 (women), incorporating:

- **Home field advantage**: Per-team HFA calculated from historical home vs away scoring margins, scaled to Elo points (26 pts/point for men, ~25 for women)
- **Travel distance**: Geocoded team home cities → game cities, converted to Elo impact via `8 * miles^(1/3)`
- **Point differential**: Margin of victory scales the K-factor via a log function (above league-average MOV of 12)
- **Variable K-factor**: Starts at 56 for early-season games, decays linearly to 38 by game 20
- **Conference mean reversion**: 30% reversion toward conference Elo mean between seasons (not league-wide mean)
- **Tournament boost**: 1.07x multiplier on Elo differences for tournament win probability predictions

Final output is `calc_elo_win_tourney(A, B, boost=1.07)` = `1 / (1 + 10^((B - A) * 1.07 / 400))`

## v2 Pipeline Plan (in progress)

Goal: Automate the entire Extract → Load → Transform → Predict flow.

**Done:**
- Extract: Kaggle CLI download + ESPN scraper (incremental + backfill)
- Transform: ESPN → Kaggle format, union with dedup (Kaggle wins)
- Predict: Elo engine extracted from notebook, runs via `python src/runner.py --source predict`
- ID crosswalk: ESPN team IDs → Kaggle TeamIDs (`data/derived/espn_kaggle_crosswalk.csv`)

**Next:**
- **Warehouse**: BigQuery
- **Orchestration**: Airflow (automate extract + predict)
- **Model improvements**: TBD (composites, additional features, etc.)
- **Live data source**: CBBpy (ESPN scraper) for daily game results + locations
- **Static data**: Kaggle dataset loaded once, refreshed annually
- **Output**: Publicly accessible predictions

## Tech Stack

- Python 3.11
- pandas, numpy, duckdb (for SQL-in-notebook queries)
- fuzzywuzzy (team name matching between data sources)
- geopy (geocoding for travel distance)
- matplotlib (visualization)
- Kaggle CLI for data downloads (`kaggle competitions download`)

## Branches

- `main` — original notebook-based project
- `restructure-bball-fball` — reorganized into bball/fball directories (has shared/ config)
- `v1` — original notebook-based working branch
- `v2` — current working branch, extracted pipeline in src/

## Kaggle Competition

- **2025**: https://www.kaggle.com/competitions/march-machine-learning-mania-2025
- **2026**: https://www.kaggle.com/competitions/march-machine-learning-mania-2026
- Submission deadline: March 19, 2026
- Format: CSV with columns `ID` (YEAR_TEAMID1_TEAMID2) and `Pred` (win probability for lower TeamID)
- Scored on Brier score (mean squared error of probabilities vs outcomes)
