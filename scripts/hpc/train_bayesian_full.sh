#!/bin/bash
#SBATCH --job-name=bayesian
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/bayesian_%A_%a.out
#SBATCH --error=logs/bayesian_%A_%a.err
#
# Usage — adjust array size to available nodes:
#   sbatch --nodelist=london,nyc,beijing,shanghai,dubai --array=0-4 scripts/hpc/train_bayesian_full.sh
#     → 5 nodes, 2 folds each
#   sbatch --nodelist=london,nyc --array=0-1 scripts/hpc/train_bayesian_full.sh
#     → 2 nodes, 5 folds each
#   sbatch --nodelist=london --array=0-0 scripts/hpc/train_bayesian_full.sh
#     → 1 node, all 10 folds
#
# Array size MUST match number of nodes in --nodelist.

set -eo pipefail

# Derive n_workers from array range
N_WORKERS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

cd $LOCAL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

# 32 chains per node, 5 OMP threads each = 160 threads (fits 168-CPU city node)
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -m train.train_bayesian \
    --gender M \
    --chains 32 \
    --samples 1000 \
    --fold-index $SLURM_ARRAY_TASK_ID \
    --n-workers $N_WORKERS \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/bayesian
