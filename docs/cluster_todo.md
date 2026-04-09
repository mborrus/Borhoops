# Cluster TODO: Ready-to-Submit Jobs

## Currently Running
- **Tournament v2 sweep** (31282301): 84 configs on 6 nodes, ~3-4 hours. Tests Colley/SRS/feature subsets/M-W regularization.
- **Feature ablation** (31282150): 14 drop-one experiments on chicago, ~2 hours remaining.

## Job 2: Point-Diff Regression Sweep

The 10th and 19th place Kaggle solutions both used point-differential regression instead of classification. Our initial attempt scored 0.22 (broken spline calibration). This sweep fixes the calibration and tests properly.

**What it tests:**
- XGBoost regression (`reg:squarederror`) predicting margin of victory
- Spline calibration (degree 3, 5, 7) from predicted margin → win probability
- Same feature sets as tournament v2 (b5, b8, b11 + colley/srs)
- Comparison: is predicting margin better than predicting win/loss?

**To submit:**
```bash
sbatch --nodelist=<NODES> scripts/hpc/sweep_regression.sh
```

**Script needed:** `scripts/hpc/sweep_regression.sh` + `src/train/sweep_regression.py`
**Status:** Not yet written. Need to fix spline calibration first. Estimate: 1 hour to write, 2-3 hours to run on 4 nodes.

## Job 3: Edge Model Deep Sweep with Cached Elo

The edge sweep found LR with [home, massey_avg, sos, conf_elo] gets 78.6% win rate. With cached Elo snapshots, we can now run hundreds of feature/threshold/range combos in seconds each.

**What it tests:**
- Every 3, 4, 5-feature combination from the top 12 features
- Vegas range: [0.43,0.57], [0.45,0.55], [0.47,0.53]
- Edge thresholds: 0.10, 0.15, 0.20, 0.25
- LR C values: 0.1, 1.0, 10.0
- ~2,000 total configs but each takes <5 seconds with cached Elo

**To submit:**
```bash
sbatch --nodelist=<NODES> scripts/hpc/sweep_edge_deep.sh
```

**Script needed:** `scripts/hpc/sweep_edge_deep.sh` + `src/train/sweep_edge_deep.py`
**Status:** Not yet written. The cached Elo makes this fast — could run on 1-2 nodes in 1-2 hours. Estimate: 30 min to write.

## Job 4: Full Seed Override Backtest (bonus)

Already done with cached Elo (10 years in 2 seconds). Result: dead neutral. No cluster needed.

## Priority Order
1. **Tournament v2** (running) — directly improves Kaggle model
2. **Point-diff regression** — fundamentally different approach, could be breakthrough
3. **Edge deep sweep** — optimizes betting model, less urgent
