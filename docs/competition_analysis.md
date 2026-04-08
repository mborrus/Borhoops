# Competition Analysis: March Machine Learning Mania 2026

## Leaderboard Context

| Place | Team | Brier | Notes |
|---|---|---|---|
| 1st | harry | 0.1097 | goto-conversion package |
| 2nd | Brendan Carlin | 0.1150 | |
| 3rd | Kevin E R MILLE | 0.1160 | LR + Triple-Market Blend |
| 4th | Yi Zhang (Xroxa) | 0.1173 | |
| 5th | alphawave9 | 0.1175 | |
| **Ours** | | **0.1283** | Pure Elo model |

Our Elo submission placed in approximately the top 15-20% (estimated from the 3,485 entries). The gap to 1st place is 0.019 — significant but not enormous.

---

## Solution 1: goto-conversion (1st/2nd place, 0.1097-0.1150)

**Source**: `kaito510/goto-conversion-winning-solution` (Kaggle notebook)

**Approach**: Uses a prepackaged `goto-conversion` Python library (version 4.0.1). The notebook is essentially just:
```python
pip install goto-conversion==4.0.1
submission_df = pd.read_csv(files(of) / 'gotoConversion.csv')
```

The actual model is inside the package — not open-sourced in the notebook. However, the notebook also includes code to:
- Import seeds and compute round of match
- Implement an "optimal strategy" based on seed matchups

**Key technique**: Appears to be a sophisticated pre-built model that leverages historical seed-based win probabilities with corrections.

**What we can learn**: The winning approach likely uses historical seed matchup win rates as a strong prior, something our model ignores entirely (seed_diff is NaN in regular season training).

**Testable ideas**:
- [ ] Build historical seed-vs-seed win rate lookup (e.g., 1-seed beats 16-seed 99% of the time)
- [ ] Use seed matchup as a direct feature for tournament prediction
- [ ] Blend our Elo predictions with seed-based probabilities

---

## Solution 2: LightGBM (public notebook, competitive)

**Source**: `jiaoyouzhang/ncaa-2026-final-lightgbm` (261 votes)
**Based on**: Theo Viel's recurring solution template

**Approach**: LightGBM classifier with very simple features.

**Features (9 total)**:
- SeedA, SeedB, SeedDiff
- WinRatioA, WinRatioB, WinRatioDiff
- GapAvgA, GapAvgB, GapAvgDiff

Where:
- WinRatio = wins / total games (season aggregate)
- GapAvg = weighted average of win margin and loss margin (season aggregate)

**Model**: LightGBM with `num_leaves=31, lr=0.05, n_estimators=1000, early_stopping=50`

**CV**: Leave-one-season-out on last 3 seasons. Train on tournament games (not regular season!).

**Key differences from our approach**:
1. **Trains on tournament games** — target is tournament outcome, not regular season
2. **Seeds are a primary feature** — available for all tournament games
3. **No Elo or power ratings** — just seeds, win ratio, score gap
4. **Season-level aggregates** — no per-game rolling features

**What we can learn**:
- Tournament prediction benefits from tournament-specific training data
- Seeds contain massive signal for tournament predictions that we're not using
- Very simple features can be competitive

**Testable ideas**:
- [ ] Train a separate tournament-specific model on historical tournament games
- [ ] Add seed-based features to tournament matchup predictions
- [ ] Test LightGBM on our feature set vs their simpler feature set
- [ ] Blend tournament-trained model with our regular-season-trained model

---

## Solution 3: LR + Triple-Market Blend (3rd place, 0.1160)

**Source**: [Full writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania) | [GitHub](https://github.com/kevin1000/march-mania-2026-3rd-place)
**Author**: Kevin E R MILLE (solo, first entry in March Mania)
**Score**: 0.1160374 (#3/3,485)
**Final CV (LOSO Brier)**: Men's 0.112, Women's 0.135, Combined 0.124

### Core Architecture

**Model**: Logistic Regression (NOT XGBoost). LR CV Brier 0.124 vs XGB 0.157. With ~650 tournament training rows, LR has lower variance and generalizes better. XGB won Stage 1 (memorized known games) but lost Stage 2 (unseen games).

**Training data**: Tournament games 2015-2025 only (~669 M, ~646 W). NOT regular season.

**Post-processing**: Triple-market probability blend applied to 2026 predictions only (not in CV).

### Features

**Men's (25 features, all as differentials):**
- Seeds: SeedNum
- Efficiency: adjoe, adjde (Barttorvik), efg_pct, tov_pct, oreb_pct, oreb_pct_d, fg3_pct
- Ratings: WAB, sos, adjt (Barttorvik), massey composite (avg of ~28 systems), ap_rank, colley_rank, srs, glm_quality, ncsos
- Trajectory: em_change (preseason → end efficiency change), win_pct
- Coach: coach_pase (Performance Above Seed Expectation)
- **Interactions**: SeedNum × pt_diff, massey × barthag, close_win_pct × pt_diff, SeedNum × massey, ncsos × massey, srs × ap_rank

**Women's (11 features):** SeedNum, win_pct, oreb_pct_d, blk_pct, last_n_win_pct, elo_slope, net × elo, elo × colley, srs, 3-way interactions

Fewer features for women's because higher seed predictability (76.2% vs 68.9%) and same ~646 rows means more features = more overfitting.

### Custom Rating Systems
- **Carry-over Elo**: K=32, HCA=100pts, MOV multiplier, **75% carryover** (vs our 70% mean reversion). Captures multi-year program strength.
- **Colley Matrix**: Solves `(2I + C)r = 1 + (w-l)/2` from the win/loss graph. Orthogonal to efficiency-based ratings.
- **SRS**: `rating = avg_margin + mean(opponent_ratings)`, iterated to convergence.
- **GLM Quality**: MLE team strength from game graph (Raddar method).
- **Massey Composite**: Average of ALL ~28 available systems (not a curated 5-system subset like ours).

### Triple-Market Blend (Post-Processing, 2026 Only)

| Tier | Source | Coverage | Men's Weight | Women's Weight |
|---|---|---|---|---|
| 1 (R1 games) | Vegas ML (60%) + ESPN BPI (40%) | 30M + 28W R1 games | 90% market / 10% LR | 75% market / 25% LR |
| 2 (non-R1, has BPI) | BPI championship → Bradley-Terry | Both teams have BPI odds | 25% BT / 75% LR | 15% BT / 85% LR |
| 3 (has Kalshi) | Kalshi futures → Bradley-Terry | Teams with futures odds | 15% Kalshi / 85% LR | 15% Kalshi / 85% LR |
| 4 (fallback) | Pure LR | Everything else | 100% LR | 100% LR |

**Critical insight from comments**: Market blend has ZERO CV impact. CV 0.124 is purely the LR model. Blend weights were chosen by judgment, not backtested. But R1 market data is extremely high signal because it captures injuries, travel, and matchup factors that pre-tournament models miss.

### What Worked (Their Findings)
1. **LR over XGBoost** — biggest single improvement. ~650 training rows is too few for trees.
2. **Aggressive feature pruning** — 33→23 men's, 19→11 women's. Single biggest CV gain (-0.007 Brier).
3. **Triple market blend** — markets aggregate real-money information
4. **75% Elo carryover** — captures multi-year program strength
5. **Colley + SRS** as orthogonal rating systems beyond Barttorvik/Massey
6. **Feature interactions** — compensate for LR's inability to learn nonlinear relationships
7. **Separate M/W models** with different feature counts and regularization

### What Didn't Work (Their Findings)
1. **XGBoost/LightGBM for Stage 2** — great for memorization, terrible for generalization on 650 rows
2. **Ensembling LR + XGB** — worsened both CV and LB
3. **Stacking individually-helpful features** — multicollinearity in small-data regime
4. **Recency weighting for LR** — reduces effective sample size
5. **Data pre-2015** — Barttorvik NaN creates distributional shift
6. **Isotonic calibration** — trained on OOF distributions ≠ full-model predictions
7. **MSE objective for XGBoost** — binary:logistic produces better calibration
8. **Polynomial features, ElasticNet, Bagging, KNN** — all noise

### Key Lessons for Us

**For our Game Model (Kaggle):**
1. **Switch to LR for tournament predictions** — we're using XGBoost on ~650 rows, exactly the scenario where LR wins
2. **Train on tournament games, not regular season** — seeds become available, sample is appropriate
3. **Use ALL Massey systems** (~28), not just 5 — their composite was strictly better
4. **Add Colley Matrix + SRS** as custom rating systems (we have none of these)
5. **Feature interactions are critical for LR** — SeedNum × massey, close_wins × pt_diff, etc.
6. **Prune aggressively** — they went from 33 to 23 features as their biggest improvement
7. **75% Elo carryover** instead of our 70% mean reversion (they found this helps)
8. **Clip predictions** — [0.03, 0.97] for men's, [0.005, 0.995] for women's
9. **Separate M/W models with different regularization** — C=100 for men's, C=0.15 for women's

**For our Edge Model (Betting):**
1. **Market data is game-specific gold for R1** — ESPN BPI predictions account for injuries/travel
2. **The market blend concept could work for regular season** — if we can get game-specific BPI or other predictions
3. **Their LR model without markets scored 0.124** — still much better than our 0.128 Elo. The feature engineering (Colley, SRS, interactions) is the difference.
4. **Carry-over Elo captures information** our current mean-reversion approach partially discards

**Testable Ideas (Priority Order):**
- [ ] Train LR on tournament games (2015-2025) with seeds + our features
- [ ] Add Colley Matrix ranking to feature pipeline
- [ ] Add SRS (Simple Rating System) to feature pipeline
- [ ] Use all Massey systems instead of just avg
- [ ] Add feature interactions (seed × massey, elo × barttorvik, etc.)
- [ ] Test 75% Elo carryover (change mean_reversion from 0.3 to 0.25)
- [ ] Implement prediction clipping
- [ ] Blend our model with Vegas moneyline for tournament R1 games
- [ ] Test separate M/W regularization
- [ ] Add coach PASE (Performance Above Seed Expectation)

---

## Solution 4: 10th Place — Point Differential Regression + Spline Calibration (0.1193)

**Source**: [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups) by Cyrus (jamalsaeedi)
**Score**: 0.1193 (#10/3,485)
**OOF Brier**: 0.1616

### Approach
- **Predict point differential, not win/loss.** XGBoost regression on margin. A 20-point blowout carries more signal than a 1-point win. Probabilities recovered via calibration.
- **Joint M/W model** with gender indicator feature (4th most important feature)
- **Tournament games 2003-2025** for training (~1,300 games, doubled to ~2,600 via symmetric augmentation)
- **LOSO validation** over 22 seasons. ALL 22 models kept for inference — averaged for 2026 predictions (free ensemble).

### Features (53 total)
- Seeds (3): each team + diff — strongest single predictor (0.1671 Brier alone)
- Box scores (18): season-avg points, rebounds, blocks, fouls (team + opponent), OT-normalized
- Late-season form (2): avg point diff in final 2 weeks
- Elo (3): margin-weighted, 60% season carry-over
- GLM quality (2): team fixed effects from point-diff regression
- Massey Ordinals (4): KenPom rank + avg of ~60 systems
- **BartTorvik (18)**: AdjOE, AdjDE, Barthag, WAB, SOS, Tempo + diffs — **single biggest improvement (+3.1%)**
- Gender (1)

### Key Findings
- **BartTorvik was the breakthrough** — 0.1650→0.1599 Brier, biggest single jump. Once Torvik is in, Four Factors become completely redundant.
- **Point-diff regression > binary classification** — richer training signal
- **Spline calibration** (degree-5) for margin→probability conversion
- **80/20 blend**: 80% full model + 20% simpler model (without Torvik) regularizes extreme predictions
- **LOSO ensemble of 22 models** — free variance reduction
- Seeds alone get 0.1671, Elo alone 0.1809. Power comes from combining complementary signals.

### What Didn't Work
- Conference strength (redundant with Elo/Massey)
- Four Factors (redundant with BartTorvik)
- Probability clipping (spline already well-calibrated)
- Clutch factor (low importance without play-by-play data)

### Testable Ideas
- [ ] Try point-differential regression instead of classification
- [ ] Spline calibration from margin→probability
- [ ] LOSO model ensemble (average all fold models for prediction)
- [ ] 80/20 blend of full model + simpler model for regularization
- [ ] Late-season form features (last 2 weeks point diff)
- [ ] Symmetric data augmentation (each game from both perspectives)

---

## Solution 5: 19th Place — Silver Medal — XGBoost + BartTorvik + Spline (0.1201)

**Source**: [Writeup + GitHub](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups) by Joe Swanson (jswanson)
**Score**: 0.1201 (#19/3,462)
**First Kaggle competition.** Built with Claude Code as coding assistant.

### Approach
Same core idea as 10th place: **predict point differential, not win/loss.** Joint M/W with gender indicator. 2003-2025 tournament games, symmetric augmentation.

### Features (53)
Seeds (3), box scores (18), late-season form (2), Elo (3, 60% carry-over), GLM quality (2), Massey (4), **BartTorvik (18)**, gender (1).

### Key Findings
- **BartTorvik as training features** was the single biggest gain (3.1%), same finding as 10th place
- Feature importance: seed_diff #1, elo_diff #2, gender #4, barthag_diff #5, GLM quality #6
- **Box score Four Factors completely redundant** once BartTorvik is included
- **Spline calibration** (degree-5) for margin→probability
- 80/20 blend of full model + simpler (no-Torvik) model

### Model Evolution (Instructive!)
| Level | Brier | What | Learning |
|---|---|---|---|
| 1 | 0.1671 | Seed lookup | Hard to beat with anything simple |
| 2 | 0.2273 | Raw win% | WORSE. Win% without opponent quality is misleading |
| 4 | 0.1809 | Elo alone | WORSE than seeds alone |
| 8 | 0.1655 | Replicated prior 1st place | Gave a baseline |
| 9 | 0.1650 | Better Elo + Massey | Small stacking gains |
| 12 | 0.1599 | + BartTorvik | Big jump from external data |

**Seeds alone = 0.1671. Our best model = 0.1337. We're better, but the gap to top solutions (0.11-0.12) is in BartTorvik + calibration + tournament-specific training.**

### Testable Ideas
- [ ] Use BartTorvik as direct XGBoost features (we have the data!)
- [ ] GLM team quality from point-diff regression
- [ ] 60% Elo season carry-over (they found this optimal vs our 70% reversion)

---

## Solution 6: 21st Place — 5-Model Ensemble + Optuna Weights (0.1203)

**Source**: [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups) by SQLRockstar
**Score**: 0.1203 (#21/3,485)

### Approach
5-model ensemble with gender-specific Optuna weight optimization.

### Models
| Model | Men's Weight | Women's Weight |
|---|---|---|
| LightGBM | 58.4% | 34.0% |
| Logistic Regression | 21.9% | 8.5% |
| XGBoost | 18.4% | 10.7% |
| Neural Network | 0.1% | **43.9%** |
| HistGradientBoosting | 1.3% | 2.8% |

### Features
- MOV-adjusted Elo (FiveThirtyEight-style)
- Pythagorean win% + last-10-game momentum
- SOS, consistency, volatility
- **Historical seed matchup win rates** (overall, recent 5-year, recent 10-year windows)
- SeedDiff × Gender interaction
- Gap average
- Symmetric augmentation

### Key Findings
- **NN weight divergence**: 0.1% for men's, 43.9% for women's. Women's tournament behavior is fundamentally different — NNs capture this better than trees.
- **Gender-specific weights matter** — combined model leaves points on the table
- **"Don't chase upsets"** — mathematically proven that post-hoc upset adjustment increases Brier score. Cost is exactly `(p̂ - p)²` per game.
- **Historical seed matchup rates** across different time windows (all-time, 5yr, 10yr) add signal
- HGB was redundant at 1-3% weight

### Testable Ideas
- [ ] Gender-specific ensemble weights (different model mix for M vs W)
- [ ] Historical seed matchup win rates as features
- [ ] Neural network for women's predictions specifically
- [ ] Pythagorean win% as a feature
- [ ] SeedDiff × Gender interaction term
- [ ] "Don't chase upsets" — avoid post-hoc probability adjustments

---

## Solution 7: 2025 1st Place Adaptation (modeh7)

**Source**: `kacchanwriting/2025-1st-place-solution-modeh7-2026-adaptation` (137 votes)

**Not yet analyzed** — need to pull and review.

---

## Key Themes Across All Top Solutions

### Universal Findings
1. **Seeds are the #1 feature** — every solution uses them. Seeds alone = 0.1671 Brier. Hard to beat.
2. **Train on tournament games** — all top solutions train on ~650-1,300 tournament games, NOT regular season
3. **BartTorvik is the biggest feature improvement** — 3.1% Brier gain, makes Four Factors redundant (found independently by 10th AND 19th place)
4. **Point-differential regression > binary classification** — richer signal, better calibration (10th, 19th)
5. **LR beats XGBoost for tournament prediction** — with ~650 rows, variance kills trees (3rd place)
6. **Market data is extremely valuable** for R1 games (3rd place: 90% market weight)
7. **Gender-specific models/weights** — M and W tournaments behave very differently (21st: NN 0.1% M vs 43.9% W)
8. **Symmetric augmentation** — double training data by including each game from both perspectives
9. **Feature pruning > feature addition** — multicollinearity kills small-data models (3rd: -0.007 Brier from pruning alone)
10. **Multiple rating systems** add orthogonal signal — Elo + Massey + Colley + SRS + GLM (3rd place)

### Calibration Matters
- 3rd place: prediction clipping [0.03, 0.97]
- 10th place: degree-5 spline from margin → probability
- 19th place: same spline approach
- 21st place: "don't chase upsets" — mathematically guaranteed to worsen Brier

### What Consistently Doesn't Work
- Raw win% without opponent adjustment
- Elo alone (worse than seeds)
- Stacking features that each help individually (multicollinearity)
- Isotonic calibration on small tournament data
- Conference strength (redundant)
- Ensembling LR + XGB (3rd place: worsened both)

## Implications for Our Models

### Game Model (Kaggle/Brier) — Gap Analysis: 0.1337 → 0.11-0.12

Our 11-feature XGBoost trained on regular season data scored 0.1337 on 2026. The top solutions score 0.11-0.12. The gap is ~0.02 Brier and comes from:

| Change | Est. Impact | Source |
|---|---|---|
| **Train on tournament games** (not regular season) | Major | All top solutions |
| **Add seeds as primary feature** | 0.1671 → much lower | Universal |
| **Switch to LR** (or at minimum test it) | -0.013 Brier (3rd place) | 3rd place |
| **Add BartTorvik as direct features** | -3.1% Brier | 10th, 19th place |
| **Point-diff regression + spline calibration** | Significant | 10th, 19th place |
| **Market blend for R1 games** | -0.01+ est. | 3rd place |
| **Feature interactions** (seed × massey, etc.) | -0.007 Brier | 3rd place |
| **Add Colley Matrix + SRS** | Meaningful | 3rd place |
| **Use all ~28 Massey systems** (not just 5 or avg) | Small | 3rd place |
| **Prediction clipping** | Small | 3rd place |
| **Gender-specific regularization** | Small | 3rd, 21st place |

**Priority implementation order:**
1. Tournament-specific training with seeds (biggest paradigm shift)
2. BartTorvik as direct features (we have the data, proven 3.1% gain)
3. LR vs XGBoost comparison on tournament data
4. Point-diff regression + calibration
5. Market blend for R1
6. Colley + SRS + feature interactions

### Edge Model (Betting)
The top solutions' heavy use of market data confirms our finding: publicly available power ratings don't beat the market **for tournament games**. Our sweet spot edge exists in **regular season** where lines are softer.

However, several ideas from the competition could improve the edge model:
- **BartTorvik features** may capture signal our current Massey-only approach misses
- **Point-diff regression** could give better edge sizing (not just direction)
- **Colley Matrix / SRS** as orthogonal rating systems may find edges Massey/Elo miss

## Links
- Leaderboard: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/leaderboard
- 1st place notebook: `kaito510/goto-conversion-winning-solution`
- LightGBM solution: `jiaoyouzhang/ncaa-2026-final-lightgbm`
- 3rd place writeup: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania
