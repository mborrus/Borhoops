# Borhoops

March Madness prediction model for the Kaggle March Machine Learning Mania competition. Predicts win probabilities for every possible NCAA tournament matchup (men's and women's).

**Read [`docs/golden_rules.md`](docs/golden_rules.md) before writing any code.** These are non-negotiable.

## Project Status

Transitioning from Jupyter notebook prototypes to a productionized data pipeline. The v1 notebooks are working and were used for the 2025 Kaggle competition. The next phase (v2) is building an automated ELT pipeline with Airflow, dbt, and Snowflake.

## Repo Structure

```
Borhoops/
├── original_notebooks/       # v1 Jupyter notebooks (the working prototypes)
│   ├── Complicated_Elo.ipynb # Main model — custom Elo with travel distance, HFA, MOV, conference mean reversion
│   ├── Nates_ELO.ipynb       # Benchmark using Nate Silver's published SBCB Elo ratings
│   ├── Basic_Elo.ipynb       # Simple Elo baseline
│   ├── EDA.ipynb             # Exploratory data analysis + data wrangling
│   └── Random_Choice_Input.ipynb  # Random baseline
├── data/                     # All input data (gitignored), flat by source
│   ├── kaggle/               # Kaggle competition CSVs (men's M* and women's W*)
│   ├── nate/                 # Nate Silver SBCB ratings (scraped from Substack)
│   ├── derived/              # Generated lookup tables (Home_Lookup.csv, HomeFieldAdvantage.csv, espn_kaggle_crosswalk.csv)
│   └── cbbpy/                # CBBpy ESPN scraper downloads (daily game results + locations)
├── data_exploration/         # Scripts for data exploration and crosswalk building
├── Output/                   # Model prediction CSVs in Kaggle submission format
├── config.yaml               # Paths config (data_dir, output_dir)
├── requirements.txt          # Python dependencies
├── secrets.sh                # Credentials (gitignored)
└── ballenv/                  # Python 3.11 virtual environment (gitignored)
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

- **Orchestration**: Airflow
- **Transformations**: dbt
- **Warehouse**: Snowflake
- **Live data source**: CBBpy (ESPN scraper) for daily game results + locations
- **Static data**: Kaggle dataset loaded once, refreshed annually
- **Output**: Publicly accessible predictions

The ID crosswalk from ESPN team IDs to Kaggle TeamIDs needs to be built as a lookup table.

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
- `v1` — current working branch, notebooks in original_notebooks/

## Kaggle Competition

- **2025**: https://www.kaggle.com/competitions/march-machine-learning-mania-2025
- **2026**: https://www.kaggle.com/competitions/march-machine-learning-mania-2026
- Submission deadline: March 19, 2026
- Format: CSV with columns `ID` (YEAR_TEAMID1_TEAMID2) and `Pred` (win probability for lower TeamID)
- Scored on Brier score (mean squared error of probabilities vs outcomes)
