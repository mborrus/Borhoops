#!/bin/bash
#SBATCH --job-name=nn_sweep
#SBATCH --partition=scratch
#SBATCH --nodelist=london,nyc,beijing,shanghai,dubai
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/nn_%j.out
#SBATCH --error=logs/nn_%j.err

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
PYTHONPATH=$LOCAL/src python -m train.train_nn \
    --gender M \
    --sweep \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/nn
