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

**Source**: Kaggle writeup (content not fully accessible)

**Approach**: Logistic Regression combined with a "Triple-Market Blend"

**What we know**:
- Model: Logistic Regression (not GBM!)
- Key technique: Blending predictions from 3 market sources
- Score: 0.1160 on 3,485 entries (#3)

**Hypothesis**: "Triple-Market" likely means blending:
1. Vegas closing spreads → implied probabilities
2. Moneyline odds → implied probabilities
3. A third source (futures/power ratings like KenPom, or a second bookmaker)

Then using LR to learn optimal weights for the blend.

**What we can learn**:
- Market data is extremely powerful for tournament predictions
- A simple model (LR) with market features can beat complex models
- Blending multiple market views captures different information

**Testable ideas**:
- [ ] Blend our model predictions with Vegas-implied probabilities using LR
- [ ] Test if LR on market features alone beats our XGBoost
- [ ] If we can get multiple odds sources (ESPN, The Odds API), test multi-source blending

---

## Solution 4: 2025 1st Place Adaptation (modeh7)

**Source**: `kacchanwriting/2025-1st-place-solution-modeh7-2026-adaptation` (137 votes)

**Not yet analyzed** — need to pull and review.

---

## Key Themes Across Top Solutions

1. **Seeds matter enormously** for tournament prediction — every top solution uses them
2. **Market data** (Vegas lines) provides very strong signal
3. **Simple models** (LR, basic LightGBM) can compete with complex ones
4. **Tournament-specific training** beats regular-season training for tournament predictions
5. **Blending/ensembling** market sources is a winning strategy

## Implications for Our Models

### Game Model (Kaggle/Brier)
Our 11-feature XGBoost trained on regular season data scored 0.1337 on 2026. To get to 0.11-0.12 range, we likely need:
1. A tournament-specific model with seeds as primary features
2. Market data (Vegas lines) as features or blend targets
3. Possibly a simpler model (LR) that's naturally better calibrated

### Edge Model (Betting)
The top solutions' heavy use of market data confirms our finding: publicly available power ratings don't beat the market. The edge, if it exists, is in the regular season where lines are softer — not in tournaments where every sharp bettor and model is competing.

## Links
- Leaderboard: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/leaderboard
- 1st place notebook: `kaito510/goto-conversion-winning-solution`
- LightGBM solution: `jiaoyouzhang/ncaa-2026-final-lightgbm`
- 3rd place writeup: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania
