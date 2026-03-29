# 2026-03-19: ML Models — Train, Tune, Backtest

## What we did

Built a full ML training pipeline on top of the Elo engine. Four models (Random Forest, Logistic Regression, SVM, XGBoost) trained on Elo-derived features for both genders.

### New files
- `src/train/__init__.py` — package init
- `src/train/features.py` — feature extraction from Elo loop (`build_features`, `matchup_features`)
- `src/train/train.py` — trains 4 models per gender, saves `.joblib` files
- `src/train/tune.py` — hyperparameter sweep script (train on <2024, validate on 2024 tourney)
- `src/predict/ml_submit.py` — generates per-model + ensemble Kaggle submission CSVs
- `tests/test_train.py` — 8 tests (feature shape, no-leakage, matchup features, roundtrip)

### Modified files
- `src/evaluate/backtest.py` — added `--model-dir` flag, `backtest_tourney_ml()`, `backtest_season_ml()`, side-by-side Elo vs ML output
- `requirements.txt` — added scikit-learn, xgboost, joblib

### Features (5, oriented to lower-TeamID)
| Feature | Description |
|---------|-------------|
| `elo_diff` | Pre-game base Elo difference (lower minus higher ID) |
| `elo_pred` | Raw Elo win probability (boost=1.0) |
| `home` | 1 if lower-ID home, -1 if away, 0 neutral |
| `day_num` | Day number in season |
| `game_count_avg` | Average games played by both teams |

Key design: features recorded *before* each Elo update (same temporal discipline as `_run_elo_collecting_preds`). Strict temporal train/test split — no cross-season leakage.

## Tuning

Ran hyperparameter sweep on 2024 tournament (67 games per gender). Key finding: **tree depth constraints are critical**. Unconstrained RF/XGBoost massively overfit, especially women's.

### Tuned configs (selected via 2024 validation)
| Model | Key params |
|-------|-----------|
| RF | `max_depth=4, min_samples_leaf=20, n_estimators=300` |
| Logistic | `C=0.01` |
| SVM | `C=0.01` |
| XGBoost | `max_depth=3, n_estimators=200` |

## Results — 2025 Tournament Backtest

```
=== Men's (67 games) ===
                Before tuning    After tuning
Elo:            0.144            0.144
Random Forest:  0.154            0.148
Logistic:       0.152            0.151
SVM:            0.151            0.151
XGBoost:        0.145            0.136  ★ beats Elo

=== Women's (67 games) ===
                Before tuning    After tuning
Elo:            0.136            0.136
Random Forest:  0.190            0.141  (massive fix)
Logistic:       0.136            0.136  (ties Elo)
SVM:            0.137            0.137
XGBoost:        0.189            0.149  (massive fix)
```

XGBoost beat Elo on men's (0.136 vs 0.144). With only 5 Elo-derived features, the ML models are mostly learning recalibrations of Elo — the real gains will come from features Elo can't capture.

## Training data
- Men's: 187,289 games (1985–2024), 48.9% lower-ID win rate
- Women's: 131,584 games (1998–2024), 50.1% lower-ID win rate

## Lessons
1. With few features, linear models (Logistic, SVM) match Elo trivially — they're just recalibrating the sigmoid
2. Tree models need strong regularization when features are low-dimensional
3. The 2024→2025 generalization gap suggests model performance is noisy on 67-game samples
4. More/richer features are needed to consistently beat Elo

## Next: additional features to explore
See recommendations discussed in session — detailed box score stats, tournament seeds, Massey ordinals, KenPom efficiency, coach tenure.
