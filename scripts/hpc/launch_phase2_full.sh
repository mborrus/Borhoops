#!/bin/bash
# After-hours launch: 10 city nodes on scratch partition.
# Run when cluster is free: bash scripts/hpc/launch_phase2_full.sh
#
# Resource budget (10 nodes max):
#   Bayesian:   5 exclusive nodes, 2 folds each (london,nyc,beijing,shanghai,dubai)
#   GBM arrays: %10 throttle × 3 on remaining 5 nodes
#   NN:         shares with GBM nodes
#   LSTM:       shares with GBM nodes
#   GP:         shares with GBM nodes
#   Total:      10 nodes
#
# Cancel daytime jobs first: scancel -u $USER

set -euo pipefail

cd /home/mborrus/Borhoops
mkdir -p logs results/gbm results/bayesian results/nn results/lstm results/gp

BAY_NODES=london,nyc,beijing,shanghai,dubai
GBM_NODES=hongkong,paris,singapore,tokyo,amsterdam

echo "=== Phase 2: Full Cluster Run (10 nodes) ==="
echo ""

# 2b: Bayesian — 5 exclusive nodes, 2 folds each, 32 chains per node
echo "Submitting Bayesian hierarchical (5 nodes, 2 folds each)..."
BAY_JOB=$(sbatch --nodelist=$BAY_NODES --array=0-4 scripts/hpc/train_bayesian_full.sh | awk '{print $NF}')
echo "  Bayesian: $BAY_JOB"

# 2a: GBM sweeps — 10 concurrent each across 5 city nodes
echo "Submitting XGBoost grid sweep (192 tasks, %10)..."
XGB_JOB=$(sbatch --nodelist=$GBM_NODES --array=0-191%10 scripts/hpc/train_gbm_xgboost.sh | awk '{print $NF}')
echo "  XGBoost: $XGB_JOB"

echo "Submitting LightGBM grid sweep (192 tasks, %10)..."
LGB_JOB=$(sbatch --nodelist=$GBM_NODES --array=0-191%10 scripts/hpc/train_gbm_lightgbm.sh | awk '{print $NF}')
echo "  LightGBM: $LGB_JOB"

echo "Submitting CatBoost grid sweep (192 tasks, %10)..."
CAT_JOB=$(sbatch --nodelist=$GBM_NODES --array=0-191%10 scripts/hpc/train_gbm_catboost.sh | awk '{print $NF}')
echo "  CatBoost: $CAT_JOB"

# 2c: Neural net sweep
echo "Submitting Neural net sweep..."
NN_JOB=$(sbatch --nodelist=$GBM_NODES scripts/hpc/train_nn.sh | awk '{print $NF}')
echo "  Neural net: $NN_JOB"

# 2d: LSTM
echo "Submitting LSTM temporal..."
LSTM_JOB=$(sbatch --nodelist=$GBM_NODES scripts/hpc/train_lstm.sh | awk '{print $NF}')
echo "  LSTM: $LSTM_JOB"

# 2e: GP
echo "Submitting Gaussian Process..."
GP_JOB=$(sbatch --nodelist=$GBM_NODES scripts/hpc/train_gp.sh | awk '{print $NF}')
echo "  GP: $GP_JOB"

echo ""
echo "All jobs submitted."
echo "  Bayesian: $BAY_NODES (5 exclusive)"
echo "  GBM/NN/LSTM/GP: $GBM_NODES (5 shared)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Expected completion:"
echo "  GBM sweeps:  3-5 hours (576 configs at %10)"
echo "  Bayesian:    1-3 hours (2 folds per node, 32 chains, MKL BLAS)"
echo "  NN sweep:    2-4 hours (54 configs)"
echo "  LSTM:        2-4 hours (10 folds)"
echo "  GP:          4-12 hours (10 folds, 5K subset)"
