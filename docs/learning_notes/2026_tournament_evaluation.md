# 2026 Tournament Evaluation — What We Learned

## Overall Results

| Metric | Men's | Women's | Combined |
|--------|-------|---------|----------|
| **Brier Score** | 0.1623 | 0.1143 | **0.1383** |
| **Correct Picks** | 50/67 (74.6%) | 57/67 (85.1%) | 107/134 (79.9%) |
| **vs Coin Flip** | 35.1% better | 54.3% better | 44.7% better |
| **vs Seed Baseline** | 12.6% better | 3.1% better | — |
| **Log Loss** | 0.4747 | 0.3648 | — |

## Round-by-Round Brier Scores

### Men's
| Round | Brier | Correct |
|-------|-------|---------|
| First Four | 0.4239 | 1/4 |
| Round of 64 | 0.1230 | 25/32 |
| Round of 32 | 0.1335 | 14/16 |
| Sweet 16 | 0.2454 | 4/8 |
| Elite 8 | 0.1577 | 3/4 |
| Final Four | 0.1999 | 2/2 |
| Championship | 0.1114 | 1/1 |

### Women's
| Round | Brier | Correct |
|-------|-------|---------|
| First Four | 0.1664 | 2/4 |
| Round of 64 | 0.0655 | 30/32 |
| Round of 32 | 0.1739 | 12/16 |
| Sweet 16 | 0.1651 | 6/8 |
| Elite 8 | 0.0721 | 4/4 |
| Final Four | 0.1507 | 2/2 |
| Championship | 0.2020 | 1/1 |

## Bracket Simulation Accuracy

- **Men's bracket**: 69.2% (27/39). Correctly picked Michigan as champion but missed Arizona→Michigan in the Final Four.
- **Women's bracket**: 90.0% (45/50). Correctly picked UCLA as champion through the entire bracket — only 5 wrong picks total.

## Key Findings

### Strengths
1. **Early rounds are excellent** — R64 Brier of 0.12 (men) and 0.065 (women) shows the Elo model captures talent gaps well.
2. **Women's model outperforms men's** — probably because the women's game has more dominant teams (UConn, UCLA, South Carolina, Texas), making outcomes more predictable.
3. **Champion picks**: Got both champions right (Michigan men's, UCLA women's).
4. **Better than seeds** — 12.6% improvement over seed-based model for men, 3.1% for women. The Elo model adds value beyond just seeding.

### Weaknesses
1. **First Four is a blind spot** — Men's First Four Brier of 0.42 is *worse than coin flip* (0.25). These teams have thin D1 histories and the Elo model doesn't capture play-in dynamics well.
2. **Sweet 16 regression (men's)** — Brier jumps to 0.245 (nearly coin-flip). When talent gaps shrink in later rounds, the model's confidence doesn't hold up. Key misses: Houston over Illinois, Tennessee over Iowa State.
3. **Calibration** — Men's model is slightly overconfident in the 40-60% range (see calibration plot). The mid-range predictions could be sharpened.
4. **Upset detection** — Missed the biggest upset: 12-seed High Point over 5-seed Wisconsin (model gave Wisconsin 92.4%). Also missed VCU over UNC, Iowa over Florida in R32.

### Notable Upsets Missed
| Game | Model Prob (Favorite) | Round |
|------|----------------------|-------|
| High Point over Wisconsin | 92.4% Wisconsin | R64 |
| Iowa over Florida | 73.3% Florida | R32 |
| VCU over North Carolina | 60.1% UNC | R64 |
| Texas over BYU | 63.7% BYU | R64 |
| St. Louis over Georgia | 69.8% Georgia | R64 |

### What Worked
- **Michigan's run**: Model had Michigan as the highest Elo (2170) and correctly picked them throughout.
- **UCLA women's dominance**: Model had UCLA at the top (Elo 2355) and correctly simulated the entire path to the championship.
- **Close-game accuracy**: Several 50-55% predictions on close matchups were correct (Michigan over Arizona in Final Four at 48.3%, UConn over Illinois at 56%).

## Ideas for Improvement

1. **First Four fix**: Consider excluding or downweighting play-in predictions, or use a separate model for those games.
2. **Late-round adjustment**: The Sweet 16+ model could benefit from:
   - Strength of schedule adjustments
   - Tempo/pace corrections
   - Fatigue or momentum factors
3. **Composite ratings**: Incorporate other signals (efficiency metrics, pace, SOS) alongside Elo.
4. **Upset model**: Build a separate model or feature to detect upset potential (e.g., seed vs. efficiency mismatch).
5. **Calibration tuning**: Apply Platt scaling or isotonic regression to recalibrate probabilities in the 40-60% range.
6. **Compare to Nate Silver's model**: See below — Silver's SBCB Elo outperforms Borhoops on Brier by ~18%.

## Head-to-Head: Borhoops vs Nate Silver (SBCB Elo)

Silver's pre-tournament Elo ratings were used with a standard `P = 1 / (1 + 10^((B-A)/400))` formula (no tournament boost). Borhoops uses a 1.07x tournament boost on the Elo difference.

### Combined Results (134 games)

| Metric | Borhoops | Silver | Diff |
|--------|----------|--------|------|
| **Brier Score** | 0.1383 | **0.1136** | +0.0247 |
| **Correct Picks** | 107/134 | **109/134** | -2 |
| **Log Loss (Men)** | 0.4747 | **0.4108** | +0.0639 |
| **Log Loss (Women)** | 0.3648 | **0.3007** | +0.0641 |

**Silver wins overall by ~18% on Brier.** The gap is consistent across both tournaments.

### Where Borhoops Beats Silver

| Round | Borhoops | Silver | Notes |
|-------|----------|--------|-------|
| Men's R32 | **0.1335** | 0.1368 | Borhoops 14/16 vs Silver 13/16 |
| Men's Final Four | **0.1999** | 0.2089 | Borhoops 2/2, Silver 1/2 (Silver picked Illinois over UConn) |
| Women's Final Four | **0.1507** | 0.2635 | Borhoops 2/2, Silver 1/2 (Silver picked UConn over South Carolina) |

### Where Silver Beats Borhoops

| Round | Borhoops | Silver | Notes |
|-------|----------|--------|-------|
| Men's R64 | 0.1230 | **0.0950** | Silver 27/32 vs Borhoops 25/32 |
| Men's Sweet 16 | 0.2454 | **0.1819** | Silver 5/8 vs Borhoops 4/8 |
| Women's Elite 8 | 0.0721 | **0.0106** | Both 4/4, but Silver far more confident in correct picks |
| Women's R32 | 0.1739 | **0.1149** | Silver 14/16 vs Borhoops 12/16 |

### Key Disagreements (7 men's, 7 women's)

Models picked **different favorites** in 14 games. Score: **Borhoops 6, Silver 8**.

Notable:
- Silver correctly picked UConn over Michigan State in men's Sweet 16 (Borhoops had MSU)
- Borhoops correctly picked UConn over Illinois in men's Final Four (Silver had Illinois)
- Borhoops correctly picked South Carolina over UConn in women's Final Four (Silver had UConn)
- Silver got all 4 women's First Four games right vs Borhoops' 2/4

### Why Silver Outperforms (observations, not confirmed methodology)

Note: We don't know Silver's exact methodology. These are observations from the data and results.

1. **More confident on lopsided games** — The scatter plots show Silver stretches further toward 0 and 1 for mismatches, while Borhoops clusters more in the 0.1-0.9 range. Silver's higher confidence on correct calls yields lower Brier.
2. **The gap is in probability calibration, not pick accuracy** — Borhoops is only 2 picks behind (107 vs 109). The Brier difference comes from *how confident* each model is on correct picks.
3. **Silver's CSV includes `b_xelo_n` alongside `b_pppg_n`, `b_ppag_n`, `b_netrating_n`, `sos`** — Whether these feed into the Elo or are tracked separately is unknown without reading his methodology on Silver Bulletin.

### Implications for Model Improvement

1. **Investigate Silver's methodology**: Read his Substack posts to understand how xElo is actually computed before copying his approach.
2. **Better calibration**: Borhoops probabilities are too conservative on clear mismatches. Consider widening the prediction range.
3. **Revisit the 1.07x tournament boost**: It compresses predictions toward 0.5, which hurts when the model is right about mismatches. Test whether reducing or removing it improves Brier.
4. **Explore composite ratings**: Silver tracks efficiency metrics (PPG/PAPG/net rating/SOS). Whether or not he uses them in Elo, they're worth investigating as additional inputs.
5. **Strength of schedule**: Borhoops' conference mean reversion partially captures SOS, but a direct adjustment could help.
