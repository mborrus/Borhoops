#!/bin/bash
#SBATCH --job-name=opt
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/opt_tv2_%A_%a.out
#SBATCH --error=logs/opt_tv2_%A_%a.err
#SBATCH --array=0-83%12

set -eo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -u -m train.sweep_tournament_v2 \
    --config-index $SLURM_ARRAY_TASK_ID \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/tournament_v2
