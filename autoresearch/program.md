# autoresearch — Borhoops

Autonomous ML research for NCAA tournament prediction. Minimize Brier score.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr8`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `autoresearch/prepare.py` — immutable evaluation harness. DO NOT MODIFY.
   - `autoresearch/experiment.py` — the file you modify. Model, features, hyperparameters.
   - `docs/features.md` — describes all 30 available features.
   - `docs/hpc_plan.md` — model architecture ideas and research directions.
4. **Verify data exists**: Check that `data/kaggle/`, `data/derived/`, `data/barttorvik/` contain CSVs.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row.
6. **Confirm and go**: Run baseline, then begin the loop.

## Experimentation

Each experiment modifies `experiment.py` and runs `prepare.py` to evaluate. A single run takes 2-5 minutes (11-fold leave-one-year-out CV on tournament data, 2015-2025).

**What you CAN do:**
- Modify `experiment.py` — this is the only file you edit. Everything is fair game:
  - Model algorithm (XGBoost, LightGBM, CatBoost, Logistic, SVM, RandomForest, neural net, Bayesian, GP)
  - Hyperparameters (any and all knobs)
  - Feature subset selection (drop features, combine features, add interaction terms)
  - Ensemble/stacking (return a VotingClassifier, StackingClassifier, or custom ensemble)
  - Preprocessing (scaling, PCA, polynomial features, target encoding)
  - Custom sklearn-compatible wrappers for non-sklearn models

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness.
- Change the evaluation metric (Brier score on tournament games).
- Change the CV folds (2015-2025, leave-one-year-out).
- Install packages not already in requirements.txt (but see allowed list below).

**The goal is simple: get the lowest mean Brier score.**

**Allowed packages**: scikit-learn, xgboost, lightgbm, catboost, numpy, pandas, scipy, torch, pymc, arviz, optuna. If you need something else, ask.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 improvement with 50 lines of hacky code? Probably not worth it. A 0.001 improvement from *removing* complexity? Definitely keep.

## Running an experiment

```bash
cd /path/to/Borhoops
python autoresearch/prepare.py > run.log 2>&1
```

Output format at the end of run.log:
```
---
mean_brier:      0.142000
n_games:         1340
elapsed_seconds: 180.5
```

Extract the key metric:
```bash
grep "^mean_brier:" run.log
```

## Logging results

Log every experiment to `autoresearch/results.tsv` (tab-separated):

```
commit	mean_brier	elapsed_s	status	description
a1b2c3d	0.142000	180.5	keep	baseline XGBoost
b2c3d4e	0.139500	195.2	keep	LightGBM with dart boosting
c3d4e5f	0.145000	160.1	discard	drop Massey features
d4e5f6g	0.000000	0.0	crash	PyMC model OOM
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and results.tsv: what's been tried, what worked
2. Modify `experiment.py` with a new idea
3. `git commit -am "description of change"`
4. Run: `python autoresearch/prepare.py > run.log 2>&1`
5. Read results: `grep "^mean_brier:\|^elapsed_seconds:" run.log`
6. If grep is empty, it crashed. Read `tail -n 50 run.log` for the traceback.
7. Log to results.tsv
8. If mean_brier improved (lower): keep the commit, advance the branch
9. If mean_brier is equal or worse: `git reset --hard HEAD~1` to revert

**NEVER STOP**: Once the loop begins, do not pause to ask. The human may be asleep. Run until manually interrupted. If you run out of ideas, re-read `docs/hpc_plan.md` and `docs/features.md` for inspiration.

## Research directions (roughly ordered by expected impact)

### Tier 1 — High impact, try first
1. **LightGBM/CatBoost**: Often better calibrated than XGBoost on tabular data
2. **Feature selection**: Try top-10, top-15, top-20 features by importance. Remove noise.
3. **Regularization sweep**: Explore L1/L2 penalties, dropout equivalents
4. **Ensemble of GBMs**: Blend XGBoost + LightGBM + CatBoost (simple average)

### Tier 2 — Medium impact
5. **Logistic regression with feature engineering**: Polynomial interactions on top-5 features
6. **Isotonic calibration**: Wrap any model in CalibratedClassifierCV(method='isotonic')
7. **Per-gender models**: Currently training one model on both genders. Try separate models.
8. **Drop early-season games**: Train only on games after DayNum > 50 (mid-season form)

### Tier 3 — Exploratory
9. **Stacking**: Train 3-5 diverse base models, logistic meta-learner on OOF predictions
10. **Bayesian logistic regression** (sklearn BayesianRidge or PyMC): Natural calibration
11. **Neural network**: Feed-forward with dropout, trained with BCELoss
12. **Gaussian Process**: Naturally calibrated, expensive but HPC has the memory
13. **LSTM temporal model**: Team season as sequence → embedding → matchup prediction

### Tier 4 — Long shots
14. **Interaction features**: elo_diff × barthag_diff, massey_avg × coach_tenure
15. **Conference-aware features**: One-hot conference encoding, power-5 indicator
16. **Transfer learning from regular season**: Pre-train on regular season, fine-tune on tournament

## Feature reference (38 features available)

Base (5): elo_diff, elo_pred, home, day_num, game_count_avg
Compact (5): sos_diff, win_streak_diff, rest_days_diff, margin_mean_diff, margin_std_diff
Detail (7): off_eff_r10_diff, def_eff_r10_diff, efg_pct_r10_diff, opp_efg_pct_r10_diff, to_rate_r10_diff, or_pct_r10_diff, pace_r10_diff
Massey (6): massey_pom_diff, massey_sag_diff, massey_mor_diff, massey_dok_diff, massey_col_diff, massey_avg_diff
Coach (2): coach_tenure_diff, coach_change_diff
Derived (2): close_win_pct_diff, conf_elo_diff
Barttorvik (3): barthag_diff, trank_adjO_diff, trank_adjD_diff
Odds (3): spread, over_under, implied_prob — NaN during training, available per-game if odds data matched
Polls (5): poll_rank_diff, poll_momentum_diff, weeks_ranked_diff, poll_points_diff, seed_diff

Note: odds features are NaN during regular-season training (no per-game ESPN ID matching yet).
Poll and seed features are populated from end-of-season AP poll and tournament seeds.
Models should handle NaN gracefully (XGBoost/LightGBM do natively, others need imputation).
