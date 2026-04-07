#!/bin/bash
#SBATCH --job-name=gbm_xgb
#SBATCH --partition=scratch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/gbm_xgb_%A_%a.out
#SBATCH --error=logs/gbm_xgb_%A_%a.err
#SBATCH --nodelist=london,nyc,beijing,shanghai,dubai
#SBATCH --array=0-191%5

set -euo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

# Stage data to local scratch
cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

cd $LOCAL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -m train.train_gbm \
    --library xgboost \
    --grid-index $SLURM_ARRAY_TASK_ID \
    --gender M \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/gbm
