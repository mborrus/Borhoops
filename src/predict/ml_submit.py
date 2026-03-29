"""Generate Kaggle submission CSVs from trained ML models."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predict.elo import add_game_counts
from predict.submission import load_data, extract_game_info, _build_hfa_dict, _build_location_dict
from train.features import build_features, matchup_features, FEATURE_COLS


MODEL_NAMES = ["random_forest", "logistic", "svm", "xgboost"]
DISPLAY_NAMES = {"random_forest": "RF", "logistic": "Logistic", "svm": "SVM", "xgboost": "XGBoost"}


def predict_gender(data, gender, model_dir, submission_df):
    """Run Elo for one gender, predict all submission matchups with each model."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)
    all_seasons = sorted(results["Season"].unique())

    # Run Elo on all data to get end-of-season ratings + game counts
    _, elo_ratings, game_counts = build_features(
        results=results,
        conferences=data[conf_key],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(),  # don't collect features, just run the loop
    )

    # Build feature matrix for submission matchups
    prefix = "M" if gender == "M" else "W"
    gender_rows = submission_df[submission_df["ID"].str.startswith(f"{prefix[0]}") |
                                submission_df["Season"].between(1000, 9999)]

    feature_rows = []
    for _, row in submission_df.iterrows():
        team_a, team_b = row["TeamID1"], row["TeamID2"]
        elo_a = elo_ratings.get(team_a, 1500)
        elo_b = elo_ratings.get(team_b, 1500)
        feature_rows.append(matchup_features(elo_a, elo_b, game_counts, team_a, team_b))

    X = pd.DataFrame(feature_rows)[FEATURE_COLS].values

    # Predict with each model
    model_dir = Path(model_dir)
    preds_by_model = {}
    for name in MODEL_NAMES:
        path = model_dir / f"{gender}_{name}.joblib"
        if not path.exists():
            print(f"  Skipping {name} — {path} not found")
            continue
        model = joblib.load(path)
        preds_by_model[name] = model.predict_proba(X)[:, 1]
        print(f"  {gender} {DISPLAY_NAMES[name]}: {len(X)} matchups predicted")

    return preds_by_model


def main():
    parser = argparse.ArgumentParser(description="Generate ML submission CSVs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="Output")
    args = parser.parse_args()

    data = load_data(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission = data["submission"].copy()
    submission[["Season", "TeamID1", "TeamID2"]] = (
        submission["ID"].apply(extract_game_info).tolist()
    )

    # Split submission by gender based on team ID ranges
    # Men's teams: 1100-1499, Women's teams: 3100-3499
    mens_mask = submission["TeamID1"] < 3000
    womens_mask = submission["TeamID1"] >= 3000

    all_preds = {}  # {model_name: full pred array}

    for gender, mask in [("M", mens_mask), ("W", womens_mask)]:
        sub = submission[mask]
        if sub.empty:
            continue

        print(f"\n=== {gender} ===")
        preds = predict_gender(data, gender, args.model_dir, sub)

        for name, pred_arr in preds.items():
            if name not in all_preds:
                all_preds[name] = np.full(len(submission), np.nan)
            all_preds[name][mask.values] = pred_arr

    # Write one CSV per model + ensemble
    ensemble_preds = []
    for name in MODEL_NAMES:
        if name not in all_preds:
            continue
        out = submission[["ID"]].copy()
        out["Pred"] = all_preds[name]
        path = output_dir / f"{DISPLAY_NAMES[name]}_Probs.csv"
        out.to_csv(path, index=False)
        print(f"Wrote {path}")
        ensemble_preds.append(all_preds[name])

    if ensemble_preds:
        out = submission[["ID"]].copy()
        out["Pred"] = np.nanmean(ensemble_preds, axis=0)
        path = output_dir / f"Ensemble_Probs.csv"
        out.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
