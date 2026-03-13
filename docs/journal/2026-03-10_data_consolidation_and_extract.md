# Session: 2026-03-10 — Data Consolidation & Extract Scripts

## What was done
- Consolidated all data into `data/` directory (flat by source: `kaggle/`, `nate/`, `derived/`, `cbbpy/`)
- Updated `config.yaml` to `data_dir: ./data/`, removed `venv_dir`
- Fixed all 5 notebooks: `Kaggle/` → `kaggle/`, `Nate/` → `nate/`, hardcoded duckdb paths → config-driven f-strings
- Fixed `build_team_crosswalk.py` paths: `SourceData/static_data/` → `data/kaggle/`
- Updated `.gitignore` to include `data/`
- Built extract layer (`src/`):
  - `extract_kaggle.py` — downloads Kaggle competition CSVs via CLI
  - `extract_espn.py` — scrapes ESPN via CBBpy for seasons not in Kaggle
  - `runner.py` — orchestrates with `--source` flag
- Installed CBBpy as editable (`pip install -e ./CBBpy`)
- Created `docs/extract.md` with setup/usage instructions
- Moved dated docs into `docs/journal/`

## Issues hit
- ESPN scraper returns 3 values (info, box, pbp) even with `pbp=False` — fixed unpack to `info_df, box_df, _`
- `data/kaggle/` was copied from older `bball/SourceData/Kaggle/` (ends at Season=2025). Newer data with 2026 lives at `SourceData/kaggle/2026/` — needs re-download via extract script

## ESPN → Kaggle transform (`src/transform/transform_espn.py`)
- Built `transform()`: reads CBBpy game_info → crosswalk lookup → flip home/away to W/L → compute DayNum from DayZero → categorize (regular/ncaa/secondary)
- Built `union_and_save()`: splits by category, stacks Kaggle (first) + ESPN, dedupes on `(Season, DayNum, WTeamID, LTeamID)` keeping Kaggle, adds `_source` column, maps ESPN `tournament` → `SecondaryTourney`
- `runner.py` updated: `--source transform` invokes `transform_and_union()`
- `tests/test_transform_espn.py`: 25 tests (19 transform + 6 union_and_save)

## Restructured `src/` into subfolders
- `src/extract/` — `extract_kaggle.py`, `extract_espn.py`
- `src/transform/` — `transform_espn.py`
- `src/runner.py` — updated imports to `extract.*` / `transform.*`
- All tests updated to match new paths, 36/36 passing

## Next up
- Re-run Kaggle extract with `source secrets.sh` to get 2026 season data
- Re-run ESPN extract (men's needs redo after unpack fix, women's needs to finish)
- End-to-end test: run `python src/runner.py --source transform` against real data, spot-check output
- Incremental ESPN downloader (diff between played and downloaded games)
- Future games downloader (scheduled/upcoming)
- Learn Airflow + BigQuery, check dbt-bigquery adapter
