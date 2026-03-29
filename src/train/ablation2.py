"""Follow-up ablation: try Logistic + targeted feature subsets."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from evaluate.backtest import load_tourney_results
from evaluate.metrics import brier_score
from train.features import build_features, matchup_features, FEATURE_COLS
from train.extended_features import (
    team_season_stats, stat_matchup_features, STAT_COLS,
    load_seeds, seed_matchup_features,
    load_massey_rankings, massey_matchup_features,
)
from train.ablation import build_extended_tourney_features, _build_training_with_extended

DATA_DIR = "../data"


def run():
    data = load_data(DATA_DIR)
    m_detailed = pd.read_csv(Path(DATA_DIR) / "kaggle" / "MRegularSeasonDetailedResults.csv")
    w_detailed = pd.read_csv(Path(DATA_DIR) / "kaggle" / "WRegularSeasonDetailedResults.csv")
    m_seeds = load_seeds(DATA_DIR, "M")
    w_seeds = load_seeds(DATA_DIR, "W")
    massey_ranks = load_massey_rankings(DATA_DIR)

    STAT_DIFF_COLS = [f"{c}_diff" for c in STAT_COLS]

    # Targeted subsets
    feature_groups = {
        "Elo only":                         FEATURE_COLS,
        "Elo + seed":                       FEATURE_COLS + ["seed_diff"],
        "Elo + off/def eff":                FEATURE_COLS + ["off_eff_diff", "def_eff_diff"],
        "Elo + efg + to":                   FEATURE_COLS + ["efg_pct_diff", "opp_efg_pct_diff", "to_rate_diff"],
        "Elo + seed + eff":                 FEATURE_COLS + ["seed_diff", "off_eff_diff", "def_eff_diff"],
        "Elo + massey":                     FEATURE_COLS + ["massey_rank_diff"],
        "Elo + seed + massey":              FEATURE_COLS + ["seed_diff", "massey_rank_diff"],
        "Elo + seed + eff + massey":        FEATURE_COLS + ["seed_diff", "off_eff_diff", "def_eff_diff", "massey_rank_diff"],
    }

    models = {
        "XGB(d=3)": lambda: XGBClassifier(n_estimators=200, max_depth=3, random_state=42, eval_metric="logloss"),
        "XGB(d=2)": lambda: XGBClassifier(n_estimators=200, max_depth=2, random_state=42, eval_metric="logloss"),
        "Logistic":  lambda: LogisticRegression(max_iter=1000, C=0.01, random_state=42),
    }

    for gender, label in [("M", "Men's"), ("W", "Women's")]:
        results_key = "mens_results" if gender == "M" else "womens_results"
        conf_key = "mens_conf" if gender == "M" else "womens_conf"
        detailed = m_detailed if gender == "M" else w_detailed
        seeds = m_seeds if gender == "M" else w_seeds
        massey = massey_ranks if gender == "M" else {}

        results = data[results_key].copy()
        results = add_game_counts(results)
        hfa_dict = _build_hfa_dict(data["hfa"], gender)
        location_dict = _build_location_dict(data["home_lookup"], gender)
        data[results_key].attrs["data_dir"] = DATA_DIR

        for eval_year in [2024, 2025]:
            results_up_to = results[results["Season"] <= eval_year]
            conferences = data[conf_key][data[conf_key]["Season"] <= eval_year]
            train_seasons = [s for s in sorted(results_up_to["Season"].unique()) if s < eval_year]

            features_df, elo_ratings, game_counts = build_features(
                results=results_up_to, conferences=conferences,
                hfa_dict=hfa_dict, location_dict=location_dict,
                seasons=set(train_seasons),
            )

            train_rows = _build_training_with_extended(
                features_df, results_up_to, detailed, seeds, massey, train_seasons,
            )
            train_df = pd.DataFrame(train_rows)

            test_rows, test_outcomes = build_extended_tourney_features(
                data, gender, eval_year, elo_ratings, game_counts,
                detailed, seeds, massey,
            )
            if not test_rows:
                continue
            test_df = pd.DataFrame(test_rows)
            y_test = np.array(test_outcomes)

            print(f"\n{'='*70}")
            print(f" {label} — {eval_year} Tournament")
            print(f"{'='*70}")
            print(f"  {'Features':<30} {'XGB(d=3)':>10} {'XGB(d=2)':>10} {'Logistic':>10}")
            print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

            for group_name, cols in feature_groups.items():
                if gender == "W" and "massey_rank_diff" in cols:
                    continue
                available = [c for c in cols if c in train_df.columns]

                scores = []
                for model_name, model_fn in models.items():
                    X_tr = train_df[available].values
                    y_tr = train_df["win"].values
                    X_te = test_df[available].values

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    model = model_fn()
                    model.fit(X_tr_s, y_tr)
                    preds = model.predict_proba(X_te_s)[:, 1]
                    scores.append(brier_score(preds, y_test))

                print(f"  {group_name:<30} {scores[0]:>10.4f} {scores[1]:>10.4f} {scores[2]:>10.4f}")


if __name__ == "__main__":
    run()
