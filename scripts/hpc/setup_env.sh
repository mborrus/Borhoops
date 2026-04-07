#!/bin/bash
# Create the borhoops conda environment on Gaia for Phase 2 training.
#
# Run this once on zeus or leto:
#   bash scripts/hpc/setup_env.sh
#
# The environment will be available on all compute nodes via shared home dir.

set -euo pipefail

ENV_NAME=borhoops
PYTHON_VERSION=3.11

echo "=== Creating conda environment: $ENV_NAME ==="

# Use miniforge/mamba if available, fall back to conda
if command -v mamba &>/dev/null; then
    INSTALLER=mamba
else
    INSTALLER=conda
fi

# Remove existing env if present
$INSTALLER env remove -n $ENV_NAME -y 2>/dev/null || true

$INSTALLER create -n $ENV_NAME python=$PYTHON_VERSION -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "=== Installing core dependencies ==="

# Core data stack
pip install \
    numpy==2.2.3 \
    pandas==2.2.3 \
    scikit-learn>=1.4 \
    scipy \
    joblib \
    pyyaml \
    requests \
    tqdm

# Gradient boosting (Phase 2a)
echo ""
echo "=== Installing gradient boosting libraries ==="
pip install \
    xgboost>=2.0 \
    lightgbm>=4.0 \
    catboost>=1.2 \
    optuna>=3.0

# Bayesian (Phase 2b)
echo ""
echo "=== Installing PyMC ==="
pip install \
    pymc>=5.0 \
    arviz

# Neural net + LSTM (Phase 2c, 2d)
echo ""
echo "=== Installing PyTorch ==="
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Geopy + fuzzywuzzy (existing deps)
echo ""
echo "=== Installing remaining project deps ==="
pip install \
    geopy \
    fuzzywuzzy \
    python-Levenshtein \
    duckdb \
    matplotlib

echo ""
echo "=== Verifying installation ==="
python -c "
import xgboost; print(f'  xgboost {xgboost.__version__}')
import lightgbm; print(f'  lightgbm {lightgbm.__version__}')
import catboost; print(f'  catboost {catboost.__version__}')
import optuna; print(f'  optuna {optuna.__version__}')
import pymc; print(f'  pymc {pymc.__version__}')
import torch; print(f'  torch {torch.__version__}')
import sklearn; print(f'  sklearn {sklearn.__version__}')
import pandas; print(f'  pandas {pandas.__version__}')
"

echo ""
echo "=== Done ==="
echo "Activate with: conda activate $ENV_NAME"
echo ""
echo "To use in sbatch scripts:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
