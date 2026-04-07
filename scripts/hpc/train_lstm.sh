#!/bin/bash
#SBATCH --job-name=lstm
#SBATCH --partition=scratch
#SBATCH --nodelist=london,nyc,beijing,shanghai,dubai
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/lstm_%j.out
#SBATCH --error=logs/lstm_%j.err

set -euo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

cd $LOCAL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -m train.train_lstm \
    --gender M \
    --hidden-size 64 \
    --epochs 100 \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/lstm
