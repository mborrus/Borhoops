#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/run_reg_%A_%a.out
#SBATCH --error=logs/run_reg_%A_%a.err
#SBATCH --array=0-155%6

set -eo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -u -m train.sweep_regression \
    --config-index $SLURM_ARRAY_TASK_ID \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/regression
