# Session: 2026-03-08 — Docs Setup & Planning

## Goal
Create living documentation to track work sessions and project progress.

## What was done
- Created `docs/` folder with per-session log files
- Backfilled 2025 competition history from git log
- Documented 2026-02-27 session (v2 planning, CBBpy review)

## Current project state
- **Branch**: v1
- **Working model**: Complicated Elo in `original_notebooks/Complicated_Elo.ipynb`
- **v2 pipeline**: Planned but not yet started (Airflow + dbt + Snowflake)
- **2026 competition**: Deadline March 19, 2026 — 11 days away
- **Open blocker**: ESPN→Kaggle team ID crosswalk not yet built

## Proposal: "Mining March Madness" (Data Mining class)
Shared the formal project proposal. Key additions beyond what v1 already does:

### New data sources to integrate
| Source | What it provides | Format |
|--------|-----------------|--------|
| sportsdataverse / espn_api | Game results + box scores (daily updates) | Python package |
| henrygd/ncaa-api | Team stats, NET rankings, tournament seeds | REST API |
| Massey Ordinals | Composite rankings from many professional raters | Web scraping |
| KenPom | Per-possession efficiency adjusted for opponent strength | Python wrapper |

### Modeling plan (beyond Elo)
1. Correlation analysis to find strongest predictors, remove redundant vars
2. **Logistic regression** as baseline (probability estimates per matchup)
3. **Random forest** and **SVM** classifiers for non-linear relationships
4. Compare all three via k-fold cross-validation and log-loss scoring
5. Feed custom Elo ratings as engineered features into the ML models to see if they improve over KenPom/NET/seeds/Massey alone

### Pipeline goal
- Full ETL pipeline → database → up-to-date forecasts at any moment
- Model evaluation system that updates as new game results come in
- Daily data ingestion (not just one-time Kaggle dump)

## Roadmap (user-defined, in order)
1. **Data availability** — Make sure all training data is downloaded and usable
2. **Data pulling** — Clean, simple code for pulling/refreshing data
3. **Modular training** — Refactor training code to be modular and adaptable
4. **Orchestration** — Wire it together with dbt/Airflow/etc (download → save → retrain → predict)
5. **Hosting** — Public-facing predictions people can see, usable for March Madness picks
6. **Experimentation** — Tune knobs, add new features/models per the proposal
