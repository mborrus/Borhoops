"""Quick hyperparameter sweep: train on <2024, validate on 2024 tournament."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from evaluate.backtest import load_tourney_results
from evaluate.metrics import brier_score
from train.features import build_features, matchup_features, FEATURE_COLS


def sweep(data_dir="../data", val_year=2024):
    data = load_data(data_dir)

    configs = {
        "random_forest": [
            {"n_estimators": 200, "random_state": 42},
            {"n_estimators": 200, "max_depth": 3, "random_state": 42},
            {"n_estimators": 200, "max_depth": 4, "random_state": 42},
            {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10, "random_state": 42},
            {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 20, "random_state": 42},
        ],
        "logistic": [
            {"max_iter": 1000, "C": 1.0, "random_state": 42},
            {"max_iter": 1000, "C": 0.1, "random_state": 42},
            {"max_iter": 1000, "C": 0.01, "random_state": 42},
            {"max_iter": 1000, "C": 10.0, "random_state": 42},
        ],
        "svm": [
            {"max_iter": 5000, "C": 1.0, "random_state": 42},
            {"max_iter": 5000, "C": 0.1, "random_state": 42},
            {"max_iter": 5000, "C": 0.01, "random_state": 42},
        ],
        "xgboost": [
            {"n_estimators": 200, "random_state": 42, "eval_metric": "logloss"},
            {"n_estimators": 200, "max_depth": 3, "random_state": 42, "eval_metric": "logloss"},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "random_state": 42, "eval_metric": "logloss"},
            {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1, "random_state": 42, "eval_metric": "logloss"},
            {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.05, "reg_lambda": 2.0, "random_state": 42, "eval_metric": "logloss"},
        ],
    }

    model_classes = {
        "random_forest": lambda p: RandomForestClassifier(**p),
        "logistic": lambda p: LogisticRegression(**p),
        "svm": lambda p: CalibratedClassifierCV(LinearSVC(**p), cv=3),
        "xgboost": lambda p: XGBClassifier(**p),
    }

    for gender, label in [("M", "Men's"), ("W", "Women's")]:
        results_key = "mens_results" if gender == "M" else "womens_results"
        conf_key = "mens_conf" if gender == "M" else "womens_conf"

        results = data[results_key].copy()
        results = results[results["Season"] <= val_year]
        results = add_game_counts(results)

        hfa_dict = _build_hfa_dict(data["hfa"], gender)
        location_dict = _build_location_dict(data["home_lookup"], gender)
        conferences = data[conf_key][data[conf_key]["Season"] <= val_year]

        all_seasons = sorted(results["Season"].unique())
        train_seasons = [s for s in all_seasons if s < val_year]

        # Build training features
        features_df, elo_ratings, game_counts, _, _ = build_features(
            results=results, conferences=conferences,
            hfa_dict=hfa_dict, location_dict=location_dict,
            seasons=set(train_seasons),
        )
        X_train = features_df[FEATURE_COLS].values
        y_train = features_df["win"].values

        # Build validation features from tournament
        data[results_key].attrs["data_dir"] = data_dir
        tourney = load_tourney_results(data_dir, gender, val_year)
        if tourney.empty:
            print(f"No {val_year} tournament data for {gender}")
            continue

        feat_rows, outcomes = [], []
        for _, game in tourney.iterrows():
            low = min(game["WTeamID"], game["LTeamID"])
            high = max(game["WTeamID"], game["LTeamID"])
            feat = matchup_features(
                elo_ratings.get(low, 1500), elo_ratings.get(high, 1500),
                game_counts, low, high,
            )
            feat_rows.append(feat)
            outcomes.append(1 if game["WTeamID"] == low else 0)

        X_val = pd.DataFrame(feat_rows)[FEATURE_COLS].values
        y_val = np.array(outcomes)

        print(f"\n{'='*60}")
        print(f" {label} — {val_year} Tournament Validation ({len(y_val)} games)")
        print(f"{'='*60}")

        for model_name, param_list in configs.items():
            print(f"\n  {model_name}:")
            for params in param_list:
                model = model_classes[model_name](params)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                score = brier_score(preds, y_val)
                # Show key params (skip random_state, eval_metric, max_iter)
                show = {k: v for k, v in params.items()
                        if k not in ("random_state", "eval_metric", "max_iter")}
                print(f"    {show} → Brier {score:.4f}")


if __name__ == "__main__":
    sweep()
