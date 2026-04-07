# HPC Model Plan

## Objective

Minimize Brier score on NCAA tournament predictions. Secondary objective: find alpha against Vegas closing lines — identify where the model disagrees with the market and is correct more often than not.

## Current Baseline

| Model | Men's 2025 Brier | Women's 2025 Brier |
|-------|-----------------|-------------------|
| Elo (pure) | 0.144 | 0.136 |
| XGBoost (30 features) | 0.147 | 0.127 |
| Logistic (30 features) | 0.155 | 0.123 |
| Vegas closing line | ~0.12-0.13 (estimated) | N/A |

Target: consistently beat 0.13 on men's, 0.12 on women's.

## Setup

```bash
# Clone repo, install deps
git clone <repo> && cd Borhoops
python -m venv ballenvy && source ballenvy/bin/activate
pip install -r requirements.txt

# Copy data/ directory from local machine (gitignored)
# Should contain: data/kaggle/, data/barttorvik/, data/odds/, data/polls/, data/derived/

# Verify
python -m pytest tests/ -q          # 131 tests should pass
PYTHONPATH=src python -c "from predict.submission import load_data; d = load_data('data'); print({k: type(v).__name__ for k, v in d.items()})"
```

## Phase 1: Feature Integration (before training anything)

### 1a. Integrate ESPN odds as features

The odds CSVs are in `data/odds/ncaab_odds_{season}.csv`. Need to:

1. Run `python src/runner.py --source transform-odds` to produce `data/derived/ncaab_odds.csv` with Kaggle TeamIDs
2. Build an odds lookup: `{(Season, DayNum, TeamID_low, TeamID_high): closing_spread}` 
3. Add to `features.py`:
   - `spread` — raw closing spread (from low-team perspective)
   - `spread_abs` — absolute spread (game expected closeness)
   - `implied_prob` — spread converted to win probability via logistic function
4. For tournament predictions via `matchup_features()`: use the season's average spread differential between the two teams as a proxy (no game-specific line available)

**This is the single highest-signal feature.** Vegas lines incorporate injury reports, travel, motivation, matchup-specific info — everything our model can't see.

### 1b. Integrate AP/Coaches poll as features

Poll data in `data/polls/polls_{season}.csv`. Need to:

1. Run `python src/runner.py --source transform-polls` to produce `data/derived/poll_rankings.csv`
2. Build lookup: `{(Season, Week, TeamID): {rank, previous, points}}`
3. Map game DayNum → poll week (approximately DayNum // 7)
4. Add to `features.py`:
   - `poll_rank_diff` — AP rank difference (unranked = 30)
   - `poll_momentum_diff` — rank change from previous week
   - `weeks_ranked_diff` — cumulative weeks in Top 25 this season

### 1c. Integrate roster experience features

ESPN roster endpoint: `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{espn_id}/roster?season={year}`

Build `extract_roster.py` to pull per-team experience distribution:
- `avg_experience` — weighted average class year (Fr=1, So=2, Jr=3, Sr=4)
- `senior_pct` — fraction of roster that are seniors
- `injury_count` — number of players with injury status

### 1d. Add seed_diff to tournament features

Currently excluded from `FEATURE_COLS`. For tournament predictions only:
- Add `seed_diff` to `matchup_features()` output
- Train a separate tournament-specific model that includes seeds
- Or: add seeds as a feature but use 0/NaN during regular season training

**After Phase 1, expect ~38-40 features.**

---

## Phase 2: Model Zoo (embarrassingly parallel across HPC nodes)

Train and evaluate each model architecture independently. Each can run on a separate node.

### 2a. Tuned Gradient Boosting (baseline improvement)

```python
# Massive hyperparameter sweep — XGBoost + LightGBM + CatBoost
# ~500 configurations × 5-fold CV = 2,500 fits
# Use optuna or sklearn RandomizedSearchCV

param_space = {
    "n_estimators": [100, 200, 500, 1000, 2000],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
}
# Also try LightGBM (often better calibration) and CatBoost (handles categoricals)
```

**HPC parallelism**: Each hyperparameter config is independent. Distribute across nodes.

### 2b. Bayesian Hierarchical Model (PyMC)

The natural model for this problem. Hierarchy: Conference → Team → Game.

```python
import pymc as pm

with pm.Model() as tournament_model:
    # Conference-level priors
    conf_mu = pm.Normal("conf_mu", mu=0, sigma=1, shape=n_conferences)
    conf_sigma = pm.HalfNormal("conf_sigma", sigma=0.5)
    
    # Team-level (nested within conference)
    team_strength = pm.Normal("team_strength", mu=conf_mu[team_conf_idx], 
                              sigma=conf_sigma, shape=n_teams)
    
    # Feature coefficients
    beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)
    
    # Game outcome
    logit_p = team_strength[team_a_idx] - team_strength[team_b_idx] + X @ beta
    y = pm.Bernoulli("y", logit_p=logit_p, observed=outcomes)
    
    # MCMC sampling — use all available CPUs
    trace = pm.sample(2000, chains=min(n_cpus, 32), cores=n_cpus)
```

**Why this is promising for Brier score**: MCMC produces full posterior distributions. The posterior mean is a naturally calibrated probability — exactly what Brier score rewards. Tree models output poorly calibrated probabilities that need post-hoc calibration.

**HPC parallelism**: Each MCMC chain is independent. 168 CPUs = up to 168 parallel chains. More chains = better convergence diagnostics.

**Dependencies**: `pip install pymc arviz`

### 2c. Neural Network (PyTorch)

Feed-forward network with dropout for calibration.

```python
# Architecture: Input(40) → 128 → Dropout(0.3) → 64 → Dropout(0.3) → 32 → 1(sigmoid)
# Loss: Binary cross-entropy (equivalent to minimizing Brier score at convergence)
# Key: use temperature scaling on the final layer for calibration
```

**HPC parallelism**: Hyperparameter sweep (architecture, dropout rate, learning rate, weight decay). Each config on a separate node.

### 2d. Temporal Sequence Model (LSTM)

Represent each team's season as a **sequence** of per-game feature vectors:

```
Team A: [game1_features, game2_features, ..., game30_features]  → LSTM → team_A_embedding
Team B: [game1_features, game2_features, ..., game30_features]  → LSTM → team_B_embedding

P(A wins) = sigmoid(MLP(team_A_embedding - team_B_embedding))
```

This captures **form trajectory** — a team peaking in March vs one that peaked in January. Current features only see the last 10 games; an LSTM sees the entire arc.

**Data**: Build per-team game sequences from the Elo loop (features already computed per-game).

**HPC parallelism**: Architecture/hyperparameter sweep across nodes.

### 2e. Gaussian Process Classifier

GP with RBF kernel on the feature space. Naturally outputs calibrated probabilities with uncertainty estimates.

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Expensive to train (O(n³)) — but with HPC memory (1.5TB), 
# can handle the full 187K training set
kernel = ConstantKernel() * RBF(length_scale_bounds=(1e-3, 1e3))
gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=20)
```

**HPC parallelism**: Kernel hyperparameter optimization. Each restart is independent.

---

## Phase 3: Ensemble / Stacking

After Phase 2, combine the best models:

### 3a. Simple Average Ensemble
Arithmetic mean of the top-K model probabilities. Often surprisingly hard to beat.

### 3b. Stacked Meta-Learner
Train a logistic regression on out-of-fold predictions from Phase 2 models:

```python
# For each model in Phase 2:
#   - Generate out-of-fold predictions via 5-fold CV on training data
#   - Stack into X_meta = [model1_pred, model2_pred, ..., modelK_pred]
# Train meta-learner (logistic or isotonic regression) on X_meta → outcomes
```

**Isotonic regression** as the meta-learner is particularly good for Brier score — it's a non-parametric calibration method that monotonically maps predictions to calibrated probabilities.

### 3c. Bayesian Model Averaging
Weight each model by its posterior predictive likelihood (not just Brier score). Models that are well-calibrated across the full probability range get more weight.

---

## Phase 4: Alpha Analysis (Model vs Vegas)

Once you have a model that beats the Elo baseline, compare against Vegas:

### Setup
```python
# For each historical tournament game with odds data:
model_pred = model.predict_proba(features)  # our probability
vegas_implied = spread_to_prob(closing_spread)  # market probability

disagreement = model_pred - vegas_implied
```

### Key Questions
1. **Calibration comparison**: Is our model better calibrated than Vegas across probability bins?
2. **Edge detection**: Where `abs(disagreement) > threshold`, who is right more often?
3. **Profitable strategy**: If we bet every game where `disagreement > 0.05`, what's the simulated ROI?
4. **Regime analysis**: Does our edge concentrate in certain matchup types? (e.g., mid-major vs power conference, early rounds vs late rounds)

### Simulation Framework
```python
def simulate_betting(model_preds, vegas_lines, outcomes, threshold=0.05):
    """Simulate flat-betting on games where model disagrees with market."""
    bets = []
    for pred, line, actual in zip(model_preds, vegas_lines, outcomes):
        vegas_implied = spread_to_prob(line)
        edge = pred - vegas_implied
        if abs(edge) > threshold:
            # Bet on our model's side
            bet_on_low = edge > 0  # we think low team is undervalued
            won = (actual == 1) == bet_on_low
            # Standard -110 vig
            profit = 1.0 if won else -1.1
            bets.append({"edge": edge, "won": won, "profit": profit})
    
    total_profit = sum(b["profit"] for b in bets)
    roi = total_profit / (len(bets) * 1.1) if bets else 0
    return {"n_bets": len(bets), "win_pct": mean(b["won"] for b in bets),
            "total_profit": total_profit, "roi": roi}
```

Run this across multiple thresholds (0.02, 0.05, 0.10) and multiple years to assess stability.

---

## Execution Order on HPC

```
Node 1:  Phase 1 (feature integration) — single node, ~1 hour
Node 2:  Phase 2a (XGBoost/LightGBM sweep) — needs Phase 1 output
Node 3:  Phase 2a (CatBoost sweep) — needs Phase 1 output  
Node 4:  Phase 2b (PyMC Bayesian) — needs Phase 1 output, all CPUs for MCMC
Node 5:  Phase 2c (Neural network sweep) — needs Phase 1 output
Node 6:  Phase 2d (LSTM temporal) — needs Phase 1 output
Node 7:  Phase 2e (Gaussian Process) — needs Phase 1 output, high memory
...
After Phase 2 completes:
Node 1:  Phase 3 (stacking/ensemble)
Node 2:  Phase 4 (alpha analysis)
```

Phase 1 is the bottleneck. After that, Phase 2 models are embarrassingly parallel.

## Dependencies (add to requirements.txt)

```
# Existing
scikit-learn
xgboost
pandas
numpy
joblib

# New for HPC
pymc>=5.0          # Bayesian hierarchical
arviz              # MCMC diagnostics
torch              # Neural network + LSTM
lightgbm           # Alternative gradient boosting
catboost            # Another GBM variant
optuna             # Hyperparameter optimization
```

## Evaluation Protocol

**All models evaluated on the same held-out data using the same metric.**

- **Primary metric**: Brier score on tournament games (lower = better)
- **Secondary metrics**: Log loss, calibration curve, ROI against Vegas
- **Validation**: Leave-one-year-out CV on tournaments 2015-2025 (10 folds)
  - Train on all years except the target year
  - Predict that year's tournament
  - Report mean ± std Brier across all 10 folds
- **Test**: 2026 tournament (final holdout, evaluate once)

This gives a robust estimate of model quality rather than relying on a single 67-game sample.
