#!/bin/bash
# Launch all Phase 2 model training jobs.
# All jobs are independent — no dependencies between them.
#
# Usage: bash scripts/hpc/launch_phase2.sh
#
# Estimated resource usage:
#   GBM (3 array jobs × 192 tasks, %20 concurrent): 3 nodes steady
#   Bayesian: 1 node (32 CPUs)
#   Neural net: 1 node
#   LSTM: 1 node
#   GP: 1 node (32 CPUs, 256G)
#   Total peak: ~7 nodes (well within 30-node budget)

set -euo pipefail

cd /home/mborrus/Borhoops
mkdir -p logs results/gbm results/bayesian results/nn results/lstm results/gp

echo "=== Phase 2: Model Zoo ==="
echo ""

# 2a: GBM sweeps (192 configs each, 20 concurrent)
echo "Submitting XGBoost grid sweep (192 tasks)..."
XGB_JOB=$(sbatch scripts/hpc/train_gbm_xgboost.sh | awk '{print $NF}')
echo "  XGBoost: $XGB_JOB"

echo "Submitting LightGBM grid sweep (192 tasks)..."
LGB_JOB=$(sbatch scripts/hpc/train_gbm_lightgbm.sh | awk '{print $NF}')
echo "  LightGBM: $LGB_JOB"

echo "Submitting CatBoost grid sweep (192 tasks)..."
CAT_JOB=$(sbatch scripts/hpc/train_gbm_catboost.sh | awk '{print $NF}')
echo "  CatBoost: $CAT_JOB"

# 2b: Bayesian
echo "Submitting Bayesian hierarchical (1 job, 16 chains)..."
BAY_JOB=$(sbatch scripts/hpc/train_bayesian.sh | awk '{print $NF}')
echo "  Bayesian: $BAY_JOB"

# 2c: Neural net
echo "Submitting Neural net sweep (1 job, 54 configs)..."
NN_JOB=$(sbatch scripts/hpc/train_nn.sh | awk '{print $NF}')
echo "  Neural net: $NN_JOB"

# 2d: LSTM
echo "Submitting LSTM temporal (1 job)..."
LSTM_JOB=$(sbatch scripts/hpc/train_lstm.sh | awk '{print $NF}')
echo "  LSTM: $LSTM_JOB"

# 2e: GP
echo "Submitting Gaussian Process (1 job)..."
GP_JOB=$(sbatch scripts/hpc/train_gp.sh | awk '{print $NF}')
echo "  GP: $GP_JOB"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/gbm_xgb_*.out"
echo ""
echo "Results will appear in:"
echo "  results/gbm/        — GBM grid search JSONs"
echo "  results/bayesian/   — Bayesian LOYO results"
echo "  results/nn/         — Neural net sweep results"
echo "  results/lstm/       — LSTM LOYO results"
echo "  results/gp/         — GP LOYO results"
