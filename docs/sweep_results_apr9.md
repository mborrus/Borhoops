# Sweep Results: April 9, 2026

## Tournament v2 Sweep (84 configs)

**NEW BEST: LR b8 + SRS, M=100 W=0.15 = 0.1610**

Top 5:
| Brier | Model | Features | Colley | SRS | Regularization |
|---|---|---|---|---|---|
| **0.1610** | LR | b8 | No | Yes | M=100, W=0.15 |
| 0.1611 | LR | b8 | No | Yes | M=50, W=0.3 |
| 0.1611 | LR | b8 | No | Yes | M=10, W=0.5 |
| 0.1612 | LR | b11 | No | Yes | M=100, W=0.15 |
| 0.1612 | LR | b8 | Yes | Yes | M=100, W=0.15 |

**Key findings:**
- b8 (8 features, dropping elo_diff/elo_pred/day_num) beats b11. Less is more, again.
- SRS is essential (+0.003 when dropped in ablation)
- Colley adds nothing when SRS is present (0.1610 without vs 0.1612 with)
- Separate M/W regularization still matters

## Regression Sweep (156 configs, 18 done so far)

**NEW BEST OVERALL: Ridge + logistic calibration = 0.1592**

Top 5:
| Brier | Model | Features | Calibration | Colley | SRS |
|---|---|---|---|---|---|
| **0.1592** | Ridge | b11 | logistic | Yes | Yes |
| 0.1592 | Ridge | b11 | spline5 | Yes | Yes |
| 0.1638 | LGB reg | b11 | spline5 | Yes | Yes |
| 0.1639 | XGB reg | b11 | spline5 | Yes | Yes |
| 0.1642 | XGB reg | b11 | spline5 | Yes | Yes |

**Key findings:**
- Ridge regression (linear) CRUSHES tree-based regression (0.159 vs 0.164)
- Same pattern as classification: linear models win on small data
- Logistic and spline5 calibration perform identically
- Both Colley AND SRS help for regression (unlike classification where Colley was neutral)
- Only 18/156 configs done — more results coming

## Edge Deep Sweep (1,651 configs, COMPLETE)

**NEW BEST: 84.3% win rate on 89 bets**

Top 5:
| Win Rate | Bets | ROI | Features |
|---|---|---|---|
| **84.3%** | 89 | **+63.7%** | home, massey_avg, margin_mean, conf_elo |
| 83.7% | 86 | +62.5% | home, massey_avg, sos, conf_elo |
| 83.3% | (varies) | +61%+ | various 4-feature combos with conf_elo |

**Key findings:**
- conf_elo_diff is in EVERY top edge model — it's the key edge signal
- 84% win rate is up from 78.6% in the first edge sweep
- These are tighter vegas ranges / higher thresholds that filter to highest-confidence bets

## Feature Ablation (COMPLETE)

14 features tested by dropping one at a time from the best tournament LR model:

**Critical features (dropping hurts by >0.003):**
- trank_adjO_diff (+0.0049 when dropped)
- barthag_diff (+0.0047)
- trank_adjD_diff (+0.0035)
- srs_diff (+0.0032)

**Important (dropping hurts by 0.0002-0.001):**
- massey_avg_diff (+0.0005)
- margin_mean_diff (+0.0002)

**Can drop (dropping helps or neutral):**
- conf_elo_diff (-0.0016 — IMPROVES model when dropped!)
- colley_rank_diff (-0.0003)
- elo_pred (-0.0001)
- seed_diff (-0.0001)
- day_num (-0.0001)
- elo_diff (-0.0001)
- home (0.0000)
- sos_diff (+0.0000)

**Surprise: conf_elo_diff HURTS the tournament model** (but is the BEST feature for the edge model). Different objectives need different features.

## Progress Tracker

| Date | Model | Brier | What Changed |
|---|---|---|---|
| Apr 7 | XGB 30 features, regular season | 0.1662 | Baseline |
| Apr 7 | XGB 11 features | 0.1649 | Feature selection |
| Apr 8 | LR tournament + seeds | 0.1646 | Tournament training |
| Apr 8 | + Colley + SRS | 0.1620 | Orthogonal ratings |
| Apr 8 | + M=100 W=0.15 | 0.1616 | Gender-specific regularization |
| Apr 9 | LR b8 + SRS (no Colley) | 0.1610 | Drop redundant features |
| **Apr 9** | **Ridge regression + logistic cal** | **0.1592** | **Point-diff regression** |

**Total improvement: 0.0072 Brier from baseline.** Ridge regression is the clear winner.

## Regression Sweep FINAL (156/156 configs)

| Brier | Model | Features | Calibration | Colley | SRS |
|---|---|---|---|---|---|
| **0.1590** | Ridge | b11 | logistic | No | Yes |
| 0.1591 | Ridge | b11 | spline5 | No | Yes |
| 0.1592 | Ridge | b11 | logistic | Yes | Yes |
| 0.1595 | Ridge | b8 | logistic | No | Yes |
| 0.1623 | Ridge | b11 | logistic | Yes | No SRS |
| 0.1628 | XGB reg | b11 | spline5 | Yes | Yes |
| 0.1630 | LGB reg | b11 | spline5 | Yes | Yes |
| 0.1635 | Ridge | b5 | logistic | No | Yes |

**Ridge dominates** — 0.004 better than tree-based regression. Same finding as classification: linear models win on small tournament data (~1,300 games).

More features help for regression (b11 > b8 > b5) — opposite of classification where b8 won. Regression has a richer training signal (continuous margin vs binary outcome) so it can use more features without overfitting.
