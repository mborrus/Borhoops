"""Generate Kaggle submission CSVs from trained ML models."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predict.elo import add_game_counts
from predict.submission import load_data, extract_game_info, _build_hfa_dict, _build_location_dict
from train.features import build_features, matchup_features, FEATURE_COLS, _build_barttorvik_lookup


MODEL_NAMES = ["random_forest", "logistic", "svm", "xgboost"]
DISPLAY_NAMES = {"random_forest": "RF", "logistic": "Logistic", "svm": "SVM", "xgboost": "XGBoost"}


def _get_extra_data(data, gender):
    """Extract all extra feature data for the given gender."""
    detailed_key = "mens_detailed" if gender == "M" else "womens_detailed"
    detailed = data.get(detailed_key)
    seeds = data.get("m_seeds", {}) if gender == "M" else data.get("w_seeds", {})
    massey_per = data.get("massey_per_system", {}) if gender == "M" else {}
    massey_avg = data.get("massey_avg", {}) if gender == "M" else {}
    coach_tenure = data.get("coach_tenure", {}) if gender == "M" else {}
    coach_changed = data.get("coach_changed", {}) if gender == "M" else {}
    barttorvik = data.get("barttorvik") if gender == "M" else None
    # Odds/polls/roster are men's only for now (ESPN data source is men's)
    odds_lookup = data.get("odds_lookup", {}) if gender == "M" else {}
    odds_team_avg = data.get("odds_team_avg", {}) if gender == "M" else {}
    poll_lookup = data.get("poll_lookup", {}) if gender == "M" else {}
    weeks_ranked = data.get("weeks_ranked", {}) if gender == "M" else {}
    roster_lookup = data.get("roster_lookup", {}) if gender == "M" else {}
    return (detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed,
            barttorvik, odds_lookup, odds_team_avg, poll_lookup, weeks_ranked, roster_lookup)


def predict_gender(data, gender, model_dir, submission_df):
    """Run Elo for one gender, predict all submission matchups with each model."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    (detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed,
     barttorvik, odds_lookup, odds_team_avg, poll_lookup, weeks_ranked,
     roster_lookup) = _get_extra_data(data, gender)
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)

    # Run Elo on all data to get end-of-season ratings + states
    _, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[conf_key],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(),
        detailed_results=detailed,
        massey_per_system=massey_per,
        massey_avg=massey_avg,
        coach_tenure=coach_tenure,
        coach_changed=coach_changed,
        barttorvik=barttorvik,
        odds_lookup=odds_lookup,
        odds_team_avg=odds_team_avg,
        poll_lookup=poll_lookup,
        weeks_ranked=weeks_ranked,
        roster_lookup=roster_lookup,
        seeds=seeds,
    )

    # Determine the prediction season from submission IDs
    season = submission_df["Season"].iloc[0] if "Season" in submission_df.columns else 2026

    feature_rows = []
    for _, row in submission_df.iterrows():
        team_a, team_b = row["TeamID1"], row["TeamID2"]
        elo_a = elo_ratings.get(team_a, 1500)
        elo_b = elo_ratings.get(team_b, 1500)
        feature_rows.append(matchup_features(
            elo_a, elo_b, game_counts, team_a, team_b,
            team_states=team_states, season=season, seeds=seeds,
            massey_per_system=massey_per, massey_avg=massey_avg,
            coach_tenure=coach_tenure, coach_changed=coach_changed,
            conf_elo_means=conf_elo_means, conferences=data[conf_key],
            barttorvik_lookup=barttorvik_lookup,
            odds_team_avg=odds_team_avg,
            poll_lookup=poll_lookup, weeks_ranked=weeks_ranked,
            roster_lookup=roster_lookup,
        ))

    X = pd.DataFrame(feature_rows)[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0)

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

    mens_mask = submission["TeamID1"] < 3000
    womens_mask = submission["TeamID1"] >= 3000

    all_preds = {}

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
