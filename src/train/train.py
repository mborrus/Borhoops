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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from train.features import build_features, FEATURE_COLS


def get_models():
    """Return list of (name, estimator) tuples. All wrapped in StandardScaler pipelines."""
    return [
        ("random_forest", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42,
            )),
        ])),
        ("logistic", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=0.01, random_state=42,
            )),
        ])),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=5000, C=0.01, random_state=42), cv=3,
            )),
        ])),
        ("xgboost", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=2, learning_rate=0.03,
                reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
                random_state=42, eval_metric="logloss",
            )),
        ])),
    ]


def _get_extra_data(data, gender):
    """Extract all extra feature data for the given gender."""
    detailed_key = "mens_detailed" if gender == "M" else "womens_detailed"
    detailed = data.get(detailed_key)
    massey_per = data.get("massey_per_system", {}) if gender == "M" else {}
    massey_avg = data.get("massey_avg", {}) if gender == "M" else {}
    coach_tenure = data.get("coach_tenure", {}) if gender == "M" else {}
    coach_changed = data.get("coach_changed", {}) if gender == "M" else {}
    barttorvik = data.get("barttorvik") if gender == "M" else None
    return detailed, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik


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

    detailed, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik = _get_extra_data(data, gender)

    print(f"\n{gender}: building features for seasons {train_seasons[0]}-{train_seasons[-1]}")

    features_df, _, _, _, _ = build_features(
        results=results,
        conferences=data[conf_key],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(train_seasons),
        detailed_results=detailed,
        massey_per_system=massey_per,
        massey_avg=massey_avg,
        coach_tenure=coach_tenure,
        coach_changed=coach_changed,
        barttorvik=barttorvik,
    )

    X = features_df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0)
    y = features_df["win"].values
    print(f"{gender}: {len(X)} training samples, {len(FEATURE_COLS)} features, {y.mean():.1%} lower-ID win rate")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in get_models():
        print(f"  Training {name}...")
        model.fit(X, y)
        path = output_dir / f"{gender}_{name}.joblib"
        joblib.dump(model, path)
        print(f"  Saved -> {path}")


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
