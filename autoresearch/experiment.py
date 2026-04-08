"""THE ONLY FILE THE AGENT MODIFIES.

Define the model, feature subset, and any preprocessing.
The harness (prepare.py) calls get_model() and uses FEATURE_SUBSET.

Contract:
  - get_model() returns a fresh sklearn-compatible estimator
    with .fit(X, y) and .predict_proba(X) -> array of shape (n, 2)
  - FEATURE_SUBSET is None (use all features) or a list of feature names
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=500,
            max_depth=2,
            learning_rate=0.03,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric="logloss",
        )),
    ])


# Minimal: drop individual Massey systems, keep only avg
# Hypothesis: 5 individual systems are redundant when avg is present
FEATURE_SUBSET = [
    "elo_diff", "elo_pred", "home", "day_num",
    "sos_diff", "margin_mean_diff",
    "massey_avg_diff",
    "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]
