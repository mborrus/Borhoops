# Edge Model Analysis: Model vs Vegas

## Summary

Our game model (XGBoost, 11 features, Brier 0.1649) has **better calibrated probabilities** than Vegas spread-implied probabilities on regular season games. But it has **no directional edge** — when our model and Vegas disagree on who wins, Vegas picks correctly just as often as we do.

**Conclusion: Our model is good at predicting who wins but cannot identify mispriced games for profitable betting.**

## Methodology

### Game Model
- XGBoost, 11 features: elo_diff, elo_pred, home, day_num, sos_diff, margin_mean_diff, massey_avg_diff, barthag_diff, trank_adjO_diff, trank_adjD_diff, conf_elo_diff
- Leave-one-year-out CV: train on years < target, predict target year
- Evaluated on regular season games with ESPN odds data (2016-2025)

### Vegas Baseline
- Closing spread from ESPN odds, converted to win probability via `P = 1/(1+10^(spread/K))` where K=15
- K=15 was verified as optimal by minimizing Brier score against actual outcomes
- Calibration is accurate: 7-point favorites win 82.1% (implied 81.7%), 14+ point favorites win 95.4% (implied 94.6%)

### Spread-to-Probability Calibration
| K | Brier |
|---|---|
| 8 | 0.1880 |
| 10 | 0.1816 |
| 12 | 0.1783 |
| **15** | **0.1769** |
| 18 | 0.1778 |
| 20 | 0.1791 |

K=15 minimizes Brier. The conversion is well-calibrated across spread ranges:
| Spread Range | Actual Fav Win % | Implied Fav Win % |
|---|---|---|
| 0-3 pts | 55.2% | 56.7% |
| 3-7 pts | 65.8% | 67.4% |
| 7-14 pts | 82.1% | 81.7% |
| 14-30 pts | 95.4% | 94.6% |

## Results

### Brier Score: Model Beats Vegas
| Year | Games | Model Brier | Vegas Brier |
|---|---|---|---|
| 2016 | 3,837 | **0.155** | 0.173 |
| 2017 | 4,006 | **0.159** | 0.176 |
| 2018 | 4,287 | **0.157** | 0.177 |
| 2019 | 5,434 | **0.157** | 0.174 |
| 2020 | 5,288 | **0.158** | 0.177 |
| 2021 | 3,840 | **0.163** | 0.183 |
| 2022 | 5,278 | **0.157** | 0.174 |
| 2023 | 5,573 | **0.162** | 0.179 |
| 2024 | 5,601 | **0.163** | 0.179 |
| 2025 | 5,641 | **0.159** | 0.173 |

Our model has ~0.015 better Brier every year. This means our probability estimates are more accurate on average.

### Directional Accuracy: No Edge
On games where model and Vegas disagree by >5% (who they pick to win):

| Year | Disagree Games | Model Right | Vegas Right |
|---|---|---|---|
| 2016 | 2,317 | 69.1% | **70.1%** |
| 2017 | 2,443 | 66.4% | **70.0%** |
| 2018 | 2,599 | 68.8% | **69.6%** |
| 2019 | 3,481 | 69.3% | **70.3%** |
| 2020 | 3,365 | **69.6%** | 69.4% |
| 2021 | 2,441 | **69.9%** | 68.9% |
| 2022 | 3,362 | 70.8% | **70.9%** |
| 2023 | 3,599 | **71.5%** | 69.6% |
| 2024 | 3,498 | **69.7%** | 68.9% |
| 2025 | 3,581 | 65.6% | **68.8%** |

**Vegas wins 5 years, model wins 3, ties 2.** No systematic directional edge.

### Bet Decomposition
Of 30,686 "bets" at the 5% threshold:
- **79% (24,284)** are same-side-as-Vegas — model is just more confident in the favorite
- **21% (6,402)** are contrarian — model picks the opposite side

The 70% overall "hit rate" is simply the favorite base rate (favorites win 73.2% of all games). Not alpha.

### Naive Strategies (No Model Needed)
| Strategy | Games | Win Rate | ROI |
|---|---|---|---|
| Always bet favorite | 48,785 | 73.2% | N/A (odds vary) |
| Always bet underdog | 48,785 | 26.8% | -48.8% |
| Underdog, spread 1-3 | 9,612 | 44.2% | -15.5% |
| Underdog, spread 3-7 | 16,365 | 34.2% | -34.7% |
| Underdog, spread 7-14 | 15,023 | 17.7% | -66.3% |
| Underdog, spread 14+ | 7,290 | 4.4% | -91.5% |

## Why Our Model Can't Beat Vegas

1. **Same information**: Our features (Massey ordinals, Elo, Barttorvik) are publicly available power ratings. Vegas uses the same (and more) to set lines. KenPom, Sagarin, etc. are literally inputs to the line-making process.

2. **Better calibration ≠ better direction**: Our model assigns more accurate probabilities (lower Brier) but doesn't identify which games the market misprices. Better Brier comes from less overconfidence in moderate favorites, not from knowing something Vegas doesn't.

3. **Market efficiency**: The NCAA basketball betting market is efficient enough that publicly available power ratings don't contain exploitable edges. The market aggregates information from thousands of sharp bettors, injury reports, travel schedules, motivation factors, and insider knowledge that our features don't capture.

## What Would Be Needed for a Profitable Edge Model

To find real alpha, we'd need features that contain **information the market doesn't have or systematically underweights**:

### Potentially Unpriced Information
1. **Live injury/availability data** — Injuries announced after line setting but before tipoff
2. **Referee assignments** — Some referees call more fouls, affecting pace/style matchups
3. **Travel fatigue modeling** — Detailed travel schedules, time zone changes, back-to-back detection
4. **Style matchup features** — Pace mismatches, zone defense effectiveness vs specific offensive styles
5. **Weather/altitude** — For outdoor or high-altitude venues
6. **Line movement data** — Sharp money indicators, reverse line movement
7. **Social media / news sentiment** — Early signals of team chemistry issues, coaching changes

### Market Inefficiency Hypotheses to Test
1. **Small conference games** — Less betting volume = less efficient lines
2. **Early season** — Teams haven't established form, ratings are noisy
3. **Post-coaching-change** — Market may be slow to adjust to new coaching schemes
4. **Conference tournament underdogs** — Motivation asymmetry (must-win vs already-qualified)
5. **Fatigue in back-to-backs** — Market may not fully price rest differentials

### Alternative Approaches
1. **Don't bet the spread** — Our model is better calibrated on moneyline probabilities. Moneyline betting (not spread betting) might be profitable if we can find soft moneyline odds.
2. **Props and totals** — Our pace/efficiency features might find edges in over/under markets even if the spread market is efficient.
3. **Live betting** — Use our pre-game model as a baseline and bet in-game when the live line diverges from our pre-game estimate.
4. **Tournament-specific model** — Train only on tournament data where dynamics are different (neutral sites, elimination pressure).

## Nomenclature

| Model Type | Purpose | Metric |
|---|---|---|
| **Game model** | Predict P(team A wins) | Brier score |
| **Edge model** | Predict where Vegas is wrong | Directional accuracy on disagreement games, ROI |
| **Confidence model** | Estimate certainty of our edge | P(our edge is correct) when we disagree |

## Fair Model Comparison (11 features, men's only LOYO)

| Model | Brier |
|---|---|
| **LightGBM** | **0.1995** |
| XGBoost | 0.2032 |
| Logistic | 0.2046 |
| Neural Net | 0.2128 |
| CatBoost | 0.2022 |

LightGBM beats XGBoost by 0.004 on the same features. Previous autoresearch found XGBoost winning, but that was on M+W combined — women's results favored XGBoost's configuration. For men's-only betting applications, LightGBM is the better base model.

## Files
- `src/train/train_edge.py` — Edge model implementation (residual + detection approaches)
- `src/train/alpha_analysis.py` — Alpha analysis vs Vegas (tournament)
- `results/edge/` — Edge model results
