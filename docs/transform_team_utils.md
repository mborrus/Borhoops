# Team Name Matching Utilities

## Module: `src/transform/team_utils.py`

Shared functions for mapping external team names to Kaggle TeamIDs. Used by the Barttorvik and Vegas transforms (and any future external source that uses team names rather than ID-based crosswalks).

## Functions

| Function | Purpose |
|----------|---------|
| `load_kaggle_teams(data_dir)` | Read `data/kaggle/MTeams.csv` |
| `build_name_to_id(teams_df)` | Build `{TeamName: TeamID}` dict |
| `fuzzy_match(name, choices, threshold)` | fuzzywuzzy best match above threshold |
| `match_team_name(name, name_to_id, manual_map, threshold)` | Full resolution: manual → exact → fuzzy |

## Resolution Order

`match_team_name()` tries three strategies in order:

1. **Manual override** — known mismatches (e.g., "Connecticut" → "UConn")
2. **Exact match** — name appears in `MTeams.csv` as-is
3. **Fuzzy match** — fuzzywuzzy `ratio` scorer above threshold (default 80)

## Usage

Each transform provides its own `MANUAL_MAP` dict for source-specific overrides:

```python
from transform.team_utils import load_kaggle_teams, build_name_to_id, match_team_name

teams_df = load_kaggle_teams(data_dir)
name_to_id = build_name_to_id(teams_df)
tid = match_team_name("Connecticut", name_to_id, MANUAL_MAP)
```

## Adding a New External Source

1. Create `src/extract/extract_<source>.py` and `src/transform/transform_<source>.py`
2. Import `load_kaggle_teams`, `build_name_to_id`, `match_team_name` from `team_utils`
3. Define a `MANUAL_MAP` dict for source-specific name mismatches
4. Add source to `src/runner.py`

## Airflow/BQ Notes

- Team matching is deterministic (no randomness)
- `MTeams.csv` is static (updated annually with new teams)
- The manual maps should be reviewed each season for new team names
