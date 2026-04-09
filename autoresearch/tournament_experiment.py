"""Tournament model autoresearch experiment file.

Same contract as experiment.py but for tournament-specific models.
Run with: python autoresearch/tournament_prepare.py
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def get_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
    ])


# Base 11 + seed_diff + interactions (best from HPC sweep so far)
FEATURE_SUBSET = [
    "elo_diff", "elo_pred", "home", "day_num",
    "sos_diff", "margin_mean_diff",
    "massey_avg_diff",
    "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]

INCLUDE_SEEDS = True
INCLUDE_INTERACTIONS = True
CLIP = (0.03, 0.97)
