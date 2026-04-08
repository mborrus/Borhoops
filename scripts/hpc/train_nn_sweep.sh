#!/bin/bash
#SBATCH --job-name=nn_sweep
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/nn_%A_%a.out
#SBATCH --error=logs/nn_%A_%a.err
#SBATCH --array=0-53%10

set -eo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -u -m train.train_nn \
    --grid-index $SLURM_ARRAY_TASK_ID \
    --gender M \
    --data-dir $LOCAL/data \
    --output-dir /home/mborrus/Borhoops/results/nn
