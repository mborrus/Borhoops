# Feature Pipeline

## Overview

The ML feature pipeline extracts 30 features from the Elo rating loop. All features are computed as `low_team - high_team` differences (lower Kaggle TeamID perspective), matching the Kaggle submission format.

Features are recorded **before** each Elo update — no data leakage within the loop.

## Feature Groups (30 total)

### Base (5) — always available
| Feature | Description |
|---------|-------------|
| `elo_diff` | Pre-game Elo rating difference |
| `elo_pred` | Elo win probability (no tournament boost) |
| `home` | Location: +1 home, 0 neutral, -1 away (for lower-ID team) |
| `day_num` | Calendar day in season (1-153) |
| `game_count_avg` | Average cumulative games played by both teams |

### Compact Rolling (5) — from game results, `deque(maxlen=10)`
| Feature | Description |
|---------|-------------|
| `sos_diff` | Strength of schedule (mean opponent Elo) |
| `win_streak_diff` | Current win streak |
| `rest_days_diff` | Days since last game |
| `margin_mean_diff` | Mean margin of victory (last 10 games) |
| `margin_std_diff` | Margin variance (consistency) |

### Detail Rolling (7) — from box scores, NaN when DetailedResults unavailable
| Feature | Description |
|---------|-------------|
| `off_eff_r10_diff` | Offensive efficiency (points/100 possessions, last 10) |
| `def_eff_r10_diff` | Defensive efficiency |
| `efg_pct_r10_diff` | Effective field goal % |
| `opp_efg_pct_r10_diff` | Opponent EFG% (defensive) |
| `to_rate_r10_diff` | Turnover rate |
| `or_pct_r10_diff` | Offensive rebound % |
| `pace_r10_diff` | Pace (possessions/game) |

### Massey Ordinals (6) — men's only, end-of-season ranks
| Feature | Description |
|---------|-------------|
| `massey_pom_diff` | KenPom (POM) rank difference |
| `massey_sag_diff` | Sagarin (SAG) rank difference |
| `massey_mor_diff` | Massey (MOR) rank difference |
| `massey_dok_diff` | Dokter Entropy (DOK) rank difference |
| `massey_col_diff` | Colley (COL) rank difference |
| `massey_avg_diff` | Average across all 5 systems |

### Coach (2) — men's only
| Feature | Description |
|---------|-------------|
| `coach_tenure_diff` | Years with current coach |
| `coach_change_diff` | New coach indicator (1 = new this season) |

### Derived (2) — computed within the Elo loop
| Feature | Description |
|---------|-------------|
| `close_win_pct_diff` | Win % in games decided by ≤5 points (min 3 games) |
| `conf_elo_diff` | Prior season's conference mean Elo |

### Barttorvik (3) — men's only, 2008+
| Feature | Description |
|---------|-------------|
| `barthag_diff` | Barttorvik power rating |
| `trank_adjO_diff` | Adjusted offensive efficiency |
| `trank_adjD_diff` | Adjusted defensive efficiency |

## Leakage Strategy

| Feature Group | Training (Elo loop) | Tournament Predictions |
|---------------|--------------------|-----------------------|
| Rolling (compact + detail) | Uses only past games (deque) | End-of-season state |
| Massey/Coach/Barttorvik | End-of-season values (mild within-season leakage) | Current season values |
| Barttorvik | **Previous season** (`season - 1`) for leak-free training | **Current season** at prediction time |
| Seeds | Not used (unavailable during regular season) | Current season seed |

## Data Dependencies

| Feature Group | Source File | Loaded By |
|---------------|-----------|-----------|
| Base + Compact + Derived | `MRegularSeasonCompactResults.csv` | `load_data()` |
| Detail Rolling | `MRegularSeasonDetailedResults.csv` | `load_data()` |
| Massey | `MMasseyOrdinals.csv` (5.8M rows) | `load_massey_rankings()` |
| Coach | `MTeamCoaches.csv` | `load_coach_data()` |
| Barttorvik | `data/derived/barttorvik_ratings.csv` | `load_data()` |
| Seeds | `MNCAATourneySeeds.csv` | `load_seeds()` |

## Code Location

- `src/train/features.py` — `build_features()`, `matchup_features()`, all column constants
- `src/train/extended_features.py` — Massey, coach, seed loaders and matchup feature functions
- `src/train/train.py` — trains 4 models per gender using `build_features()`
- `src/predict/ml_submit.py` — generates submission CSVs using `matchup_features()`

## Model Configuration

All models wrapped in `StandardScaler` pipelines. NaN values replaced with 0.0 before training.

| Model | Key Hyperparameters |
|-------|-------------------|
| Random Forest | `n_estimators=300, max_depth=4, min_samples_leaf=20` |
| Logistic Regression | `C=0.01, max_iter=1000` |
| SVM (LinearSVC + calibration) | `C=0.01, max_iter=5000, cv=3` |
| XGBoost | `n_estimators=500, max_depth=2, lr=0.03, reg_lambda=5.0` |

## Backtest Results (2025 Tournament, 67 games each)

| Model | Men's Brier | Women's Brier |
|-------|------------|--------------|
| Elo (baseline) | **0.144** | 0.136 |
| Random Forest | 0.151 | 0.156 |
| Logistic | 0.155 | **0.123** |
| SVM | 0.150 | **0.125** |
| XGBoost | 0.147 | **0.127** |

Women's models strongly beat Elo. Men's XGBoost is close. More features (odds, polls) and tuning expected to improve men's results.
