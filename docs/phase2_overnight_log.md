# Phase 2: Overnight Run Log (April 7-8, 2026)

## Cluster Layout

| Node Pool | Nodes | Jobs |
|---|---|---|
| GBM sweep | brussels, chicago, athena, paris, singapore, tokyo, cronus, ares | XGB/LGB/CAT grid (192 configs × 3) |
| Bayesian logistic | beijing, nyc, shanghai + others | 10 folds, 32 chains, MKL BLAS |
| BART | amsterdam, dubai, frankfurt, istanbul, jakarta, london, madrid, mexicocity, milan, mumbai | 10 folds, baseline experiment |
| NN/LSTM/GP | athena, singapore (shared with GBM) | Single jobs, still running |

## GBM Grid Search

192 pre-defined hyperparameter configs per library (XGBoost, LightGBM, CatBoost).
Grid: n_estimators × max_depth × learning_rate × reg_lambda × subsample × colsample_bytree × min_child_weight = 4×4×4×3×2×2×2 = 768 total, split into 192 per library segment.

Each config runs full 10-fold LOYO CV (men's only). ~20 min per config.

### Leaderboard (updated as results come in)

| Library | Best Brier | Configs Tested | Best Params |
|---|---|---|---|
| LightGBM | 0.1946 | 135 | lr=?, depth=?, n=? |
| XGBoost | 0.1949 | 109 | lr=0.01, depth=3, n=200, lambda=5.0, sub=0.9 |
| CatBoost | 0.1970 | 131 | — |

Note: These Brier scores are from `train/evaluate.py` (men's only, strict LOYO). The autoresearch harness scores are different (men's + women's combined, different train/test split).

## Bayesian Logistic (PyMC)

Simple model: `logit(P) = alpha + X @ beta` with Normal priors.

### Issues encountered
1. **pip install → no BLAS**: PyMC installed via pip lacked BLAS, making MCMC ~50x slower. Fixed by reinstalling via `conda install -c conda-forge pymc` which linked MKL.
2. **4 cores instead of 168**: `SLURM_CPUS_PER_TASK` not set with `--exclusive`, defaulting to 4. Fixed by falling back to `os.cpu_count()`.
3. **Array task grabbed all nodes**: `--exclusive` without `--nodes=1` made each task claim all listed nodes. Fixed by adding `--nodes=1`.

### Results
Per-fold Brier scores (men's only):
- 2015: 0.1822
- 2017: 0.1881
- 2018: 0.2141
- 2022: 0.2104
- 2023: 0.2183
- 2024: 0.1883
- 2025: 0.1611

Mean: ~0.195. Not competitive with XGBoost (0.166 in autoresearch harness). Expected — linear model can't learn feature interactions.

## BART (Bayesian Additive Regression Trees)

BART = Bayesian XGBoost. Uses `pymc-bart` for tree-based nonlinear interactions with posterior uncertainty.

Also includes a fallback model (`BARTFallback`) that adds explicit interaction terms (elo_diff × barthag_diff, etc.) to the Bayesian logistic.

Submitted on 10 nodes, 1 fold each. Results pending.

## Autoresearch Loop (Local)

Running on the login node. Each iteration modifies `experiment.py`, runs LOYO CV (~30 min), keeps or reverts.

### Results

| Commit | Brier | Status | Description |
|---|---|---|---|
| baseline | **0.1662** | keep | XGBoost (n=500, depth=2, lr=0.03, λ=5.0) all 30 features |
| b12c9fb | 0.1699 | discard | XGBoost grid-best, all 40 features — new features add noise |
| 963a7b6 | 0.1672 | discard | LightGBM, 30 features — worse than XGB |
| 9f49fb6 | 0.1663 | discard | XGBoost + polls (33 features) — polls neutral |
| 8b85552 | 0.1733 | discard | XGB+LGB ensemble — LGB drags XGB down |
| 1dac2b3 | 0.1722 | discard | XGBoost + isotonic calibration — overfits small folds |
| 51f9378 | pending | — | Power ratings only (16 features) |

### Key findings so far
- The original baseline config is remarkably hard to beat
- Adding new features (odds, polls) hurts rather than helps — too many NaNs in early years
- LightGBM is worse than XGBoost in the autoresearch harness (different from HPC grid search where LGB leads — likely due to different CV methodology)
- Ensembling XGB+LGB is worse than XGB alone
- Isotonic calibration overfits on the small tournament test sets

### Full experiment log (17 experiments)

| # | Brier | Status | Description |
|---|---|---|---|
| 1 | 0.1662 | keep | Baseline: XGB n=500 d=2 lr=0.03 λ=5.0, 30 features |
| 2 | 0.1699 | discard | All 40 features — new features add noise |
| 3 | 0.1672 | discard | LightGBM, 30 features |
| 4 | 0.1663 | discard | XGB + polls (33 features) — neutral |
| 5 | 0.1733 | discard | XGB+LGB ensemble — LGB drags down |
| 6 | 0.1722 | discard | Isotonic calibration — overfits small folds |
| 7 | 0.1656 | keep | Power ratings only (16 features) |
| 8 | 0.1664 | discard | +off/def efficiency (18) — adds noise |
| 9 | 0.1785 | discard | Dart boosting — much worse, 2+ hours |
| 10 | 0.1662 | discard | CatBoost 16 features — matches baseline |
| 11 | 0.1659 | discard | XGB n=1000 lr=0.01 |
| **12** | **0.1649** | **keep** | **Drop individual Massey → avg only (11 features) — BEST** |
| 13 | 0.1693 | discard | Ultra-minimal 6 features — cut too deep |
| 14 | 0.1652 | discard | 11 feat + higher reg (λ=10, α=0.5) |
| 15 | 0.1650 | discard | 11 feat + game_count_avg |
| 16 | 0.1655 | discard | 11 feat + depth=3 — overfits |
| 17 | 0.1650 | discard | 11 feat + sigmoid calibration — already well-calibrated |

### Key insight
Feature selection matters more than model choice or hyperparameter tuning. The optimal 11-feature set (Elo, Massey avg, Barttorvik, SOS, margin, conf Elo, context) improved Brier from 0.1662 → 0.1649. Every other change (model, params, calibration, ensemble) made things worse or was neutral.

## Slurm Issues

### Stale NFS file handle
After reinstalling PyMC via conda, compute nodes got stale file handles for `libcrypto.so.3`. Jobs failed with `ImportError: libcrypto.so.3: cannot stat shared object: Stale file handle`. Self-resolved after a few minutes.

### --export=ALL,VAR=val breaks jobs
Using `sbatch --export=ALL,BART_EXPERIMENT=baseline` causes "user env retrieval failed requeued held". Known Gaia issue. Fix: set env vars inside the script body, use `--export=ALL` only.

### --exclusive without --nodes=1
Array jobs with `--exclusive` and no `--nodes=1` cause each task to claim ALL listed nodes instead of one. Fix: always include `#SBATCH --nodes=1` with `--exclusive`.

## GBM Grid Search — COMPLETE

Final results (576 configs, all evaluated):

| Library | Best Brier | Best Config |
|---|---|---|
| LightGBM | **0.1946** | — |
| XGBoost | 0.1949 | lr=0.01, depth=3, n=200, λ=5.0 |
| CatBoost | 0.1964 | — |

All results logged to MLflow (585 runs total).

## BART Results

Baseline (Bayesian logistic + interaction terms): **0.2229** mean (9/10 folds).
Not competitive. 2016 fold is an outlier at 0.32. Interaction terms don't capture enough nonlinearity.

## Autoresearch Results (Local XGBoost Loop)

| Commit | Brier | Status | Description |
|---|---|---|---|
| baseline | 0.1662 | keep | XGBoost (n=500, depth=2, lr=0.03, λ=5.0) all 30 features |
| 51f9378 | **0.1656** | keep | **Power ratings only (16 features) — BEST** |
| 5e6d523 | 0.1664 | discard | + off/def efficiency (18 feat) — efficiency adds noise |
| b3799bb | pending | — | Dart boosting + 16 features |
| b12c9fb | 0.1699 | discard | All 40 features — new features add noise |
| 963a7b6 | 0.1672 | discard | LightGBM 30 features |
| 9f49fb6 | 0.1663 | discard | + polls (33 features) — neutral |
| 8b85552 | 0.1733 | discard | XGB+LGB ensemble |
| 1dac2b3 | 0.1722 | discard | Isotonic calibration — overfits |

**Key finding: Feature selection > model choice.** Dropping 14 noisy features improved Brier by 0.0006. The optimal set is 16 power-rating features (Elo, Massey, Barttorvik, SOS, margin, conf Elo).

## Timeline

- 4:30 PM: Daytime GBM grid search running (5 nodes)
- 5:30 PM: Launched full after-hours run (13 nodes total)
- 5:45 PM: Fixed Bayesian cores issue (4→32 parallel chains)
- 6:00 PM: Started autoresearch loop
- 8:30 PM: Submitted BART on 10 nodes
- 9:00 PM: GBM sweep complete (576 configs)
- 10:00 PM: BART baseline results (0.2229, not competitive)
- 10:30 PM: Autoresearch hits 0.1656 (16-feature power ratings)
- 12:30 AM: Autoresearch hits 0.1649 (11-feature — dropped individual Massey)
- 1:00 AM: GP complete (0.2006)
- 1:30 AM: Bayesian complete (0.2035)
- 2:00 AM: Autoresearch converged — 17 experiments, all variations around 0.1649 are worse
- 7:00 AM cutoff: Stop submitting new jobs

## Final Model Zoo Leaderboard

| Model | Mean Brier | Features | Notes |
|---|---|---|---|
| **XGBoost (autoresearch best)** | **0.1649** | 11 | M+W combined LOYO, d=2, lr=0.03 |
| GBM grid LightGBM | 0.1946 | 40 | Men's only LOYO |
| GBM grid XGBoost | 0.1949 | 40 | Men's only LOYO |
| GBM grid CatBoost | 0.1964 | 40 | Men's only LOYO |
| Bayesian logistic | 0.2035 | 40 | Linear, can't learn interactions |
| Gaussian Process | 0.2006 | 40 | 5K subset, too few samples |
| BART (interactions) | 0.2229 | 40 | Interaction terms insufficient |
| NN sweep | pending | 40 | 8.5+ hours, still running |
| LSTM temporal | pending | 40 | 8.5+ hours, still running |

Note: Autoresearch (M+W, different CV split) and HPC grid search (M only) use different evaluation harnesses, so scores aren't directly comparable.

## Overnight Summary

**18 autoresearch experiments** tested. Key progression: 30 features (0.1662) → 16 features (0.1656) → 11 features (0.1649). Every other change (model type, hyperparams, calibration, ensemble) was worse or neutral.

**Best model**: XGBoost, depth=2, lr=0.03, λ=5.0, n=500, 11 features:
`elo_diff, elo_pred, home, day_num, sos_diff, margin_mean_diff, massey_avg_diff, barthag_diff, trank_adjO_diff, trank_adjD_diff, conf_elo_diff`

**576 GBM grid configs** completed. Best: LightGBM 0.1946 (men's only eval).

**Bayesian**: 0.2035 — linear model can't compete with trees.

**GP**: 0.2006 — 5K subset too small.

**BART**: 0.2229 — interaction terms not enough; actual BART trees might help but weren't tested due to time.

**NN/LSTM**: Still running at cutoff. Will produce results eventually.

**MLflow**: 585+ runs logged. View with `mlflow ui --backend-store-uri file:///home/mborrus/Borhoops/mlruns --port 5001`

## What to try next
1. Train the 11-feature XGBoost on full data for 2026 submission
2. Phase 3: stacking ensemble (if NN/LSTM produce competitive results)
3. Phase 4: alpha analysis vs Vegas
4. Try actual PyMC BART trees (not the interaction fallback)
5. Investigate why women's predictions are consistently better than men's
