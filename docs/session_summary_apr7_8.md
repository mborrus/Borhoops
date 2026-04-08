# Session Summary: April 7-8, 2026

## What We Built

### Phase 1: Feature Integration (30 → 40 features)
- **Odds** (4 features): spread, spread_abs, implied_prob, over_under from ESPN (2013+, 68K rows)
- **Polls** (3): AP poll rank, momentum, weeks ranked (2003+, 22K rows)
- **Roster** (2): avg experience, senior pct (ESPN API, 2025+ only — limited utility)
- **Seeds** (1): seed_diff (tournament only)
- Files: `extended_features.py`, `features.py`, `submission.py`, `ml_submit.py`, `extract_roster.py`, `transform_roster.py`

### Phase 2: Model Zoo
- **GBM grid search**: 576 configs (XGB/LGB/CatBoost), all completed
  - Best: LightGBM 0.1946 (men's only LOYO)
- **Bayesian logistic (PyMC)**: 0.2035 — linear model can't compete
- **Gaussian Process**: 0.2006 — 5K subset too small
- **BART**: 0.2229 — interaction terms insufficient
- **NN sweep**: 54 configs, best 0.1976
- **LSTM**: Precompute fix applied, NaN output (PyTorch bug)

### Phase 3: Stacking Ensemble
- Isotonic meta-learner: 0.2006 — overfits on small tournament samples

### Phase 4: Alpha/Edge Analysis
- Model vs Vegas comparison on 60K regular season games
- **Sweet spot discovery**: 74.4% win rate on pick'em games (verified with real ML odds)
- **Clean out-of-sample 2025**: 67.3%, +29.4% ROI
- Kelly sizing backtest: 0.25x Kelly turns 100u → 130-620u per season

### Autoresearch
- 24 experiments over 18 hours
- **Best game model**: XGBoost, 11 features, Brier 0.1649 (M+W combined LOYO)
- Feature selection was the key insight: 30 → 16 → 11 features, each cut improved

### Competition Analysis
- Analyzed 5 top Kaggle solutions (3rd, 10th, 19th, 21st, 49th place)
- Identified gap: we train on regular season, they train on tournament games with seeds

### Infrastructure
- MLflow logging (585+ runs)
- HPC job scripts with proper node management
- Elo feature caching module (30 min → 30 sec iterations)
- Edge model framework
- Backfill scripts

## Key Findings

### Game Model
1. **Feature selection > model choice** — dropping noisy features improved more than any model swap
2. **11 optimal features**: elo_diff, elo_pred, home, day_num, sos_diff, margin_mean_diff, massey_avg_diff, barthag_diff, trank_adjO_diff, trank_adjD_diff, conf_elo_diff
3. **Massey avg alone** beats 5 individual Massey systems combined
4. **XGBoost beats all other models** (LGB, CatBoost, NN, Bayesian, GP, BART) on M+W combined
5. **LightGBM wins men's only** — different optimal model per gender
6. Our model (0.1337) vs Kaggle winner (0.1097): gap is seeds + tournament training + BartTorvik + market blend

### Edge/Betting Model
1. **Model beats Vegas on Brier** (0.162 vs 0.177 regular season) but has **no directional edge on disagreement games**
2. **Sweet spot exists**: Vegas pick'em games where our model sees a clear favorite → 75% win rate, +45% ROI with real odds
3. **Hypothesis**: market overprices home court advantage in near-even games
4. **Edge is declining**: 53% ROI (2016-19) → 32% ROI (2022-25), still profitable
5. **Tournament: no edge** — market is too efficient, all approaches lose money
6. **Vegas odds are redundant** with Massey + Elo — same information

### Competition Insights (from top solutions)
1. **Train on tournament games, not regular season** — fundamental paradigm difference
2. **Seeds are the #1 feature** for tournament prediction (0.1671 Brier alone)
3. **LR beats XGBoost** on ~650 tournament rows (lower variance)
4. **BartTorvik as direct features** = 3.1% gain (proven by 2 independent solutions)
5. **Point-diff regression > classification** — richer training signal
6. **Market blend** worth ~0.01+ Brier for Round 1 games
7. **Feature interactions** critical for LR (seed × massey, etc.)

## What to Do Next (Priority Order)

### For Kaggle 2027
1. Build tournament-specific model (train on 2015-2026 tournament games)
2. Seeds as primary feature
3. BartTorvik as direct XGBoost/LR features
4. Test LR vs XGBoost on tournament data
5. Point-diff regression + spline calibration
6. Add Colley Matrix + SRS ratings
7. Market blend for R1 games
8. Feature interactions
9. Multi-seed ensemble
10. Gender-specific regularization

### For Betting (Next Season)
1. Build automated pipeline to flag sweet spot games daily
2. Run the Elo cache → model prediction → odds comparison pipeline
3. Paper trade first month (November) before committing real money
4. Monitor edge trend — if ROI drops below 15%, pause and reassess
5. Consider expanding to wider Vegas ranges (40-60%) if R1 edge holds

### Infrastructure
1. Run `cache_features.py` to enable 30-second autoresearch iterations
2. Integrate competition learnings into autoresearch experiment.py
3. Set up automated data pipeline for next season (Kaggle download + ESPN scraper + BartTorvik)

## Files Created/Modified This Session

### New Files
- `src/train/evaluate.py` — LOYO CV framework
- `src/train/train_gbm.py` — GBM sweep (Optuna + grid)
- `src/train/train_bayesian.py` — PyMC hierarchical
- `src/train/train_nn.py` — PyTorch feed-forward
- `src/train/train_lstm.py` — Siamese LSTM
- `src/train/train_gp.py` — Gaussian Process
- `src/train/train_ensemble.py` — Stacking ensemble
- `src/train/train_edge.py` — Edge model (residual + detection)
- `src/train/train_bart.py` — BART model
- `src/train/alpha_analysis.py` — Model vs Vegas
- `src/train/mlflow_utils.py` — MLflow logging
- `src/train/cache_features.py` — Elo feature caching
- `src/extract/extract_roster.py` — ESPN roster scraper
- `src/transform/transform_roster.py` — Roster aggregation
- `scripts/hpc/` — 12 HPC job scripts
- `scripts/backfill_mlflow.py` — MLflow backfill
- `tests/test_phase1_features.py` — 22 tests
- `docs/phase1_implementation.md`
- `docs/phase2_overnight_log.md`
- `docs/sweet_spot_discovery.md`
- `docs/edge_model_analysis.md`
- `docs/competition_analysis.md`
- `docs/session_summary_apr7_8.md`

### Modified Files
- `src/train/features.py` — 30→40 features, new column groups
- `src/train/extended_features.py` — odds, polls, roster, seed loaders
- `src/predict/submission.py` — load new data sources
- `src/predict/ml_submit.py` — pass new data through
- `src/runner.py` — roster + transform-roster commands
- `autoresearch/prepare.py` — 40-feature pipeline integration
- `autoresearch/experiment.py` — iterated through 24 experiments
- `autoresearch/results.tsv` — full experiment log
- `docs/features.md` — updated to 40 features
