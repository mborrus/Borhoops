# Session: 2026-03-12 — Notebook Logic Extraction

## What was done
- Extracted Complicated_Elo notebook into importable Python modules
- Created `src/predict/elo.py` — pure Elo engine (~140 lines):
  - `point_differential_scaler()`, `k_factor()`, `distance_to_elo_impact()`, `get_advantage()`
  - `calc_elo_win_tourney()`, `update_elo()`, `add_game_counts()`
  - `run_elo()` — main season→game loop, returns `{TeamID: final_elo}`
- Created `src/predict/submission.py` — I/O orchestrator:
  - `load_data()` prefers combined files (v2 pipeline), falls back to raw Kaggle
  - `predict()` runs Elo for M+W, writes Kaggle submission CSV
- Created `scripts/build_home_lookup.py` and `scripts/build_hfa.py` for one-time precomputation
- Updated `src/runner.py` — added `--source predict`
- 29 new tests (`test_elo.py`, `test_submission.py`), 65 total all passing
- Ran end-to-end against real data: 365 M teams, 363 W teams, 132,133 predictions

## Bugs fixed from notebook
- **`row1` bug**: `travel_distance()` referenced a global `row1` (hardcoded to `mens_results.loc[8]`) instead of the actual game row. Fixed by passing `w_loc` explicitly to `_travel_miles()`
- **HFA overwrite bug**: HFA assignment overwrote travel adjustment instead of being additive. Fixed to `+=`

## Design decisions
- All Elo functions take primitives, not DataFrame rows — cleaner and testable
- Pre-build `{TeamID: hfa}` and `{TeamID: (lat, lon)}` dicts before the game loop (notebook did DataFrame filter per row)
- Conference mean reversion uses pandas merge+groupby instead of DuckDB SQL strings
- No global state — `run_elo()` takes all data as arguments

## CLAUDE.md updates
- Updated project status to reflect working pipeline
- Repo structure updated with `src/predict/`, `scripts/`, `tests/`
- v2 pipeline plan: Snowflake → BigQuery, added Done/Next sections
- Added `v2` branch to branch list

## Next up
- BigQuery setup
- Airflow to automate extract + predict
- Model improvements
