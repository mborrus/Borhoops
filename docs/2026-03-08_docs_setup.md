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

---

## Step 1 findings: Data Availability

### 2026 Kaggle data
- Downloaded at `SourceData/kaggle/2026/`
- Zip was not fully extracted — **extracted it this session**, all 35 CSVs now available
- Contains all files the Complicated Elo notebook needs (game results, cities, teams, conferences, sample submission)

### Data files the Complicated Elo notebook reads
| File | Source | Status |
|------|--------|--------|
| `SampleSubmissionStage2.csv` | Kaggle | Available (2026) |
| `MTeams.csv` / `WTeams.csv` | Kaggle | Available (2026) |
| `MRegularSeasonCompactResults.csv` / `W...` | Kaggle | Available (2026) |
| `MGameCities.csv` / `WGameCities.csv` | Kaggle | Available (2026) |
| `Cities.csv` | Kaggle | Available (2026) |
| `MTeamConferences.csv` / `WTeamConferences.csv` | Kaggle | Available (2026) |
| `derived/Home_Lookup.csv` | Generated (geocoding) | Exists from 2025, **needs regeneration for any new teams in 2026** |
| `derived/HomeFieldAdvantage.csv` | Generated (from results) | Exists from 2025, **regenerated each run** |
| `NateEloProbs_SCBC.csv` | Nate Silver scrape | Exists from 2025, **need to re-scrape for 2026** |

### Path mess (to fix in step 2)
Data is scattered across 3 locations with inconsistent casing:
- `bball/SourceData/Kaggle/` — where the notebooks currently read from (capital K)
- `SourceData/kaggle/2026/` — where the 2026 download lives (lowercase k, year subfolder)
- `SourceData/Nate/`, `SourceData/derived/` — supplementary data

The notebook also has hardcoded duckdb paths like `./SourceData/Kaggle/MTeams.csv` that bypass `config.yaml`.

### Decision: future data layout
Going with **flat by source**: `data/kaggle/`, `data/nate/`, `data/derived/`
- No year subfolders — Kaggle CSVs already have a Season column
- Derived files regenerate each run
- This aligns with the eventual goal of a continuously-updated source (Snowflake or similar)
- One canonical location, one config entry, no casing issues

---

## Next session: Get a 2026 submission working

Priority is getting on the board for the competition (deadline March 19). Steps:
1. Consolidate data into `data/kaggle/`, `data/nate/`, `data/derived/`
2. Update `config.yaml` and notebook paths to point there
3. Re-scrape Nate Silver's 2026 Elo ratings (if published)
4. Run Complicated_Elo notebook end-to-end on 2026 data
5. Submit to Kaggle

Once that's done, steps 4-6 of the roadmap (orchestration, hosting, experimentation) build on a working baseline.
