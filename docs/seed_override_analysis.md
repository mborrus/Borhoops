# Seed Override Analysis

## Concept

Post-processing step: override model predictions for top seeds in early rounds with high-confidence values. 1-seeds historically almost never lose in R1/R2, so being maximally confident there should reduce Brier score.

## Best Override Strategy (on 2026)

| Seed | Round 1 | Round 2 | Sweet 16 | Elite 8 |
|---|---|---|---|---|
| 1-seed | 99% | 99% | 95% | — |
| 2-seed | 99% | 85% | — | — |
| 3-seed | 99% | — | — | — |
| 4-seed | 99% | — | — | — |

**2026 Result: Elo 0.1283 → 0.1222 with overrides (+0.006 improvement)**

## 2026 Seed Performance

| Seed | R1 | R2 | S16 | E8 | F4 | Championship |
|---|---|---|---|---|---|---|
| 1-seed | 8/8 (100%) | 7/8 (88%) | 7/7 (100%) | 6/7 (86%) | 3/3 (100%) | 2/2 (100%) |
| 2-seed | 8/8 (100%) | 7/8 (88%) | | | | |
| 3-seed | 8/8 (100%) | 5/8 (62%) | | | | |
| 4-seed | 8/8 (100%) | 6/8 (75%) | | | | |

2026 was a particularly chalky year — 1-seeds dominated through the Sweet 16.

## Backtest 2015-2025 (Partial)

Override config: 1s 99% R1+R2 + 95% S16, 2s 99% R1 + 85% R2, 3s+4s 99% R1

| Year | Base Elo | With Override | Delta | 1-seeds in S16 | Verdict |
|---|---|---|---|---|---|
| 2015 | 0.1592 | 0.1604 | +0.0012 | 7/8 | HURTS |
| 2016 | 0.1770 | 0.1744 | -0.0026 | 8/8 | HELPS |
| 2017 | 0.1558 | 0.1566 | +0.0008 | 7/8 | HURTS |
| 2018 | 0.1737 | 0.1770 | +0.0033 | 6/7 | HURTS |
| 2019 | 0.1493 | 0.1478 | -0.0015 | 8/8 | HELPS |
| 2021+ | pending | | | | |

**Pattern: Override helps in chalky years (1-seeds 8/8 in S16), hurts in upset years.**

Full 10-year backtest (using cached Elo snapshots — instant):

| Year | Base | Override | Delta | 1-seeds S16 | Verdict |
|---|---|---|---|---|---|
| 2015 | 0.1592 | 0.1604 | +0.0012 | 7/8 | Hurts |
| 2016 | 0.1770 | 0.1744 | -0.0026 | 8/8 | Helps |
| 2017 | 0.1558 | 0.1566 | +0.0008 | 7/8 | Hurts |
| 2018 | 0.1737 | 0.1770 | +0.0033 | 6/7 | Hurts |
| 2019 | 0.1493 | 0.1478 | -0.0015 | 8/8 | Helps |
| 2021 | 0.1892 | 0.1884 | -0.0008 | 7/8 | Slight help |
| 2022 | 0.1932 | 0.1925 | -0.0007 | 7/8 | Slight help |
| 2023 | 0.1914 | 0.1947 | +0.0033 | 5/7 | Hurts |
| 2024 | 0.1631 | 0.1612 | -0.0019 | 8/8 | Helps |
| 2025 | 0.1398 | 0.1391 | -0.0007 | 8/8 | Slight help |
| **TOTAL** | **0.1692** | **0.1692** | **+0.0001** | | **Dead even** |

**Overall: exactly neutral across 1,315 games.** Helps 6 years, hurts 4, but the hurts are bigger when they happen (2018 and 2023: +0.0033 each from 1-seed S16 upsets).

## Conclusion

**The seed override is dead neutral over 10 years (+0.0001).** Not a default strategy.

It's a bet on chalk — helps in chalky years (2016, 2019, 2024: +0.002 avg), hurts in upset years (2018, 2023: -0.003 avg). Over 1,315 games, these cancel out exactly.

**For Kaggle**: skip it. You can't predict in advance whether it'll be a chalky year, and Stage 2 submissions are locked before games start.

**For betting**: the market already prices 1-seed dominance correctly. No edge here.

## Models Tested

| Model | Base | With Override | Improvement |
|---|---|---|---|
| Elo (original) | 0.1283 | 0.1222 | +0.0061 |
| XGB 11-feature | 0.1337 | 0.1311 | +0.0026 |

Override helps XGB more because XGB was less confident on top seeds than Elo.
