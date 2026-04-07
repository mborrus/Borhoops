#!/bin/bash
#SBATCH --job-name=gp
#SBATCH --partition=scratch
#SBATCH --nodelist=london,nyc,beijing,shanghai,dubai
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/gp_%j.out
#SBATCH --error=logs/gp_%j.err

set -euo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

cd $LOCAL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

# GP is O(n³) — using subset training. Increase max-train if memory allows.
cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -m train.train_gp \
    --gender M \
    --max-train 10000 \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/gp
