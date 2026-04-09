# Borhoops

March Madness prediction model for the Kaggle March Machine Learning Mania competition. Predicts win probabilities for every possible NCAA tournament matchup (men's and women's).

**Read [`docs/golden_rules.md`](docs/golden_rules.md) before writing any code.** These are non-negotiable.

## Project Status

The v1 notebooks have been extracted into a working Python pipeline (`src/`). The extract → transform → predict flow runs end-to-end via `python src/runner.py`. Next phase is hooking it up to BigQuery + Airflow for automation, then improving the model.

## Repo Structure

```
Borhoops/
├── src/                          # Core Python pipeline
│   ├── extract/                  # Data extraction
│   │   ├── extract_kaggle.py     # Kaggle competition data download
│   │   ├── extract_espn.py       # ESPN game scraper (CBBpy)
│   │   ├── extract_odds.py       # ESPN odds scraper (2013+)
│   │   ├── extract_polls.py      # AP/Coaches poll scraper (2003+)
│   │   ├── extract_barttorvik.py # T-Rank ratings
│   │   ├── extract_vegas.py      # The Odds API
│   │   └── extract_roster.py     # ESPN roster API (2025+ only)
│   ├── transform/                # Data transformation
│   │   ├── transform_espn.py     # ESPN → Kaggle format
│   │   ├── transform_odds.py     # Odds → Kaggle IDs + DayNum
│   │   ├── transform_polls.py    # Polls → Kaggle IDs
│   │   ├── transform_barttorvik.py
│   │   ├── transform_roster.py
│   │   └── team_utils.py         # Fuzzy team name matching
│   ├── predict/                  # Prediction pipeline
│   │   ├── elo.py                # Elo engine (HFA, travel, MOV, K-factor)
│   │   ├── submission.py         # load_data() + Kaggle submission generation
│   │   └── ml_submit.py          # ML model submission generation
│   ├── train/                    # Model training & evaluation
│   │   ├── features.py           # Feature pipeline (40 features, Elo loop)
│   │   ├── extended_features.py  # Odds, polls, roster, seed feature loaders
│   │   ├── custom_ratings.py     # Colley Matrix + SRS rating systems
│   │   ├── evaluate.py           # LOYO CV framework, Brier/log-loss
│   │   ├── cache_features.py     # Precompute & cache Elo + per-season snapshots
│   │   ├── train.py              # Train 4 base models (RF, LR, SVM, XGB)
│   │   ├── train_gbm.py          # GBM hyperparameter sweep (Optuna/grid)
│   │   ├── train_tournament.py   # Tournament-specific model sweep
│   │   ├── train_ensemble.py     # Stacking ensemble with meta-learner
│   │   ├── train_nn.py           # PyTorch feed-forward neural net
│   │   ├── train_lstm.py         # Siamese LSTM temporal model
│   │   ├── train_bayesian.py     # PyMC Bayesian hierarchical
│   │   ├── train_bart.py         # BART (Bayesian trees)
│   │   ├── train_gp.py           # Gaussian Process classifier
│   │   ├── train_edge.py         # Edge model (model vs Vegas)
│   │   ├── sweep_edge.py         # Edge model feature/model sweep
│   │   ├── alpha_analysis.py     # Alpha analysis vs closing lines
│   │   └── mlflow_utils.py       # MLflow logging helpers
│   ├── evaluate/                 # Evaluation utilities
│   │   ├── backtest.py           # Tournament & season backtesting
│   │   └── metrics.py            # Brier score, calibration
│   └── runner.py                 # CLI orchestrator (--source kaggle|espn|predict|...)
├── autoresearch/                 # Autonomous ML experimentation
│   ├── prepare.py                # Evaluation harness (LOYO CV, 40 features)
│   ├── experiment.py             # Model config (agent modifies this)
│   ├── tournament_prepare.py     # Tournament-specific LOYO harness
│   ├── tournament_experiment.py  # Tournament model config
│   └── results.tsv               # Experiment log
├── scripts/                      # Utility scripts
│   ├── hpc/                      # HPC (Gaia cluster) job scripts
│   │   ├── setup_env.sh          # Create borhoops conda env
│   │   ├── launch_phase2.sh      # Submit daytime model zoo jobs
│   │   ├── launch_phase2_full.sh # Submit full-cluster overnight run
│   │   ├── train_gbm_*.sh        # GBM sweep jobs (XGB/LGB/CatBoost)
│   │   ├── train_tournament.sh   # Tournament model sweep
│   │   ├── sweep_edge.sh         # Edge model sweep
│   │   └── run_*.sh              # Various experiment jobs
│   ├── build_home_lookup.py      # Geocode team cities
│   ├── build_hfa.py              # Home field advantage computation
│   └── backfill_mlflow.py        # Backfill results to MLflow
├── tests/                        # pytest test suite
│   ├── test_train.py             # Feature pipeline tests
│   ├── test_phase1_features.py   # Phase 1 feature tests (22 tests)
│   └── ...                       # Elo, transform, submission tests
├── docs/                         # Documentation (see docs/README.md)
├── data/                         # All input data (gitignored)
│   ├── kaggle/                   # Kaggle competition CSVs
│   ├── odds/                     # ESPN odds per season (2013-2026)
│   ├── polls/                    # AP/Coaches polls per season (2003-2026)
│   ├── roster/                   # ESPN roster data (2025-2026)
│   ├── barttorvik/               # T-Rank ratings per season
│   ├── derived/                  # Generated files (crosswalk, odds, polls, HFA, etc.)
│   ├── cache/                    # Precomputed Elo features + per-season snapshots
│   ├── nate/                     # Nate Silver SBCB ratings
│   └── cbbpy/                    # CBBpy ESPN scraper downloads
├── results/                      # Model results (gitignored)
│   ├── gbm/                      # GBM grid search (576 configs)
│   ├── tournament/               # Tournament model sweep (180 configs)
│   ├── edge_sweep/               # Edge model sweep (30 configs)
│   ├── bayesian/                 # Bayesian model results
│   ├── nn/                       # Neural net sweep results
│   └── ...                       # alpha, ensemble, gp, lstm, bart
├── mlruns/                       # MLflow experiment tracking (gitignored)
├── Output/                       # Kaggle submission CSVs
├── config.yaml                   # Paths config
├── requirements.txt              # Python dependencies
└── logs/                         # Slurm job logs (gitignored)
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
