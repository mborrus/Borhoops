#!/bin/bash
#SBATCH --job-name=bart
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/bart_%A_%a.out
#SBATCH --error=logs/bart_%A_%a.err
#
# Usage:
#   sbatch --nodelist=london,nyc,beijing,shanghai,dubai --array=0-4 scripts/hpc/train_bart.sh
#   sbatch --nodelist=<10 nodes> --array=0-9 scripts/hpc/train_bart.sh

set -eo pipefail

N_WORKERS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
# Change this line to switch experiments:
EXPERIMENT=baseline

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -m train.train_bart \
    --gender M \
    --experiment $EXPERIMENT \
    --fold-index $SLURM_ARRAY_TASK_ID \
    --n-workers $N_WORKERS \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/bart
