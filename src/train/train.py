"""Train ML models on Elo-derived features. One model per (gender, algorithm)."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from train.features import build_features, FEATURE_COLS


# ── Model configuration ─────────────────────────────────────────────────────
# TODO: Tune these. Each entry is (short_name, sklearn-compatible estimator).
# Considerations:
#   - RF: more trees = smoother but slower. max_depth controls overfitting.
#   - Logistic: already well-calibrated; max_iter needs to be enough to converge.
#   - SVM: LinearSVC is fast but needs CalibratedClassifierCV for probabilities.
#     cv=3 vs cv=5 trades calibration quality for speed.
#   - XGBoost: learning_rate + n_estimators is the main knob. Lower LR + more
#     trees = better but slower. max_depth controls tree complexity.
#
# These defaults are reasonable starting points. Adjust and re-run.

def get_models():
    """Return list of (name, estimator) tuples."""
    return [
        ("random_forest", RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42,
        )),
        ("logistic", LogisticRegression(
            max_iter=1000, C=0.01, random_state=42,
        )),
        ("svm", CalibratedClassifierCV(
            LinearSVC(max_iter=5000, C=0.01, random_state=42), cv=3,
        )),
        ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=3, random_state=42, eval_metric="logloss",
        )),
    ]


# ── Training pipeline ────────────────────────────────────────────────────────

def train_gender(data, gender, test_year, output_dir):
    """Train all models for one gender, save to disk."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    all_seasons = sorted(results["Season"].unique())
    train_seasons = [s for s in all_seasons if s < test_year]

    print(f"\n{gender}: building features for seasons {train_seasons[0]}-{train_seasons[-1]}")

    features_df, _, _ = build_features(
        results=results,
        conferences=data[conf_key],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(train_seasons),
    )

    X = features_df[FEATURE_COLS].values
    y = features_df["win"].values
    print(f"{gender}: {len(X)} training samples, {y.mean():.1%} lower-ID win rate")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in get_models():
        print(f"  Training {name}...")
        model.fit(X, y)
        path = output_dir / f"{gender}_{name}.joblib"
        joblib.dump(model, path)
        print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML models on Elo features")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--test-year", type=int, default=2025,
                        help="Hold out this year (train on all prior)")
    args = parser.parse_args()

    data = load_data(args.data_dir)

    for gender in ("M", "W"):
        train_gender(data, gender, args.test_year, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
