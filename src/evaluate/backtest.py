"""Backtest Elo predictions against historical NCAA tournament results and regular season games."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predict.elo import (
    run_elo, add_game_counts, calc_elo_win_tourney,
    _travel_miles, distance_to_elo_impact, get_advantage,
    k_factor, point_differential_scaler, update_elo,
)
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from evaluate.metrics import brier_score, calibration_table
from train.features import build_features, matchup_features, FEATURE_COLS, _build_barttorvik_lookup


def load_tourney_results(data_dir, gender, year):
    prefix = "M" if gender == "M" else "W"
    path = Path(data_dir) / "kaggle" / f"{prefix}NCAATourneyCompactResults.csv"
    df = pd.read_csv(path)
    return df[df["Season"] == year]


def _run_elo_collecting_preds(results, conferences, hfa_dict, location_dict,
                               predict_seasons, initial_elo=1500, mean_reversion=0.3,
                               k_start=56, k_end=38, hfa_scalar=26, mov_avg=12):
    """Run Elo loop and collect pre-game predictions for specified seasons."""
    seasons = sorted(results["Season"].unique())
    elo_ratings = {}
    preds, outcomes = [], []

    for season in seasons:
        season_games = results[results["Season"] == season]
        active_teams = set(season_games["WTeamID"]).union(season_games["LTeamID"])

        prev_ratings = elo_ratings.copy()
        elo_ratings = {t: prev_ratings.get(t, initial_elo) for t in active_teams}
        collecting = season in predict_seasons

        for _, row in season_games.iterrows():
            winner, loser = row["WTeamID"], row["LTeamID"]
            w_loc = row["WLoc"]

            w_miles, l_miles = _travel_miles(winner, loser, w_loc, location_dict)
            winner_elo = elo_ratings[winner] + distance_to_elo_impact(w_miles)
            loser_elo = elo_ratings[loser] + distance_to_elo_impact(l_miles)

            if w_loc == "H":
                winner_elo += get_advantage(winner, hfa_dict, hfa_scalar)
            elif w_loc == "A":
                loser_elo += get_advantage(loser, hfa_dict, hfa_scalar)

            if collecting:
                low = min(winner, loser)
                high = max(winner, loser)
                low_elo = winner_elo if winner == low else loser_elo
                high_elo = loser_elo if winner == low else winner_elo
                pred = calc_elo_win_tourney(low_elo, high_elo, boost=1.0)
                preds.append(pred)
                outcomes.append(1 if winner == low else 0)

            game_num = (row["WTeam_Game_Count"] + row["LTeam_Game_Count"]) / 2
            k = k_factor(game_num, k_start, k_end)
            k_coeff = point_differential_scaler(row["WScore"], row["LScore"], mov_avg)

            elo_change = update_elo(winner_elo, loser_elo, k * k_coeff)
            elo_ratings[winner] += elo_change
            elo_ratings[loser] -= elo_change

        df = pd.DataFrame(list(elo_ratings.items()), columns=["TeamID", "Elo"])
        league_mean = df["Elo"].mean()
        divergence = initial_elo - league_mean

        season_conf = conferences[conferences["Season"] == season][["TeamID", "ConfAbbrev"]]
        df = df.merge(season_conf, on="TeamID", how="left")
        df["ConfAbbrev"] = df["ConfAbbrev"].fillna("unknown")
        conf_means = df.groupby("ConfAbbrev")["Elo"].mean().to_dict()

        elo_ratings = {
            row.TeamID: (
                (1 - mean_reversion) * row.Elo
                + mean_reversion * conf_means.get(row.ConfAbbrev, initial_elo)
            ) * (1 + divergence / initial_elo)
            for row in df.itertuples()
        }

    return elo_ratings, preds, outcomes


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
    return detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik


def backtest_tourney(data, gender, year):
    """Run Elo through season `year`, predict that year's tournament."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key]
    results = results[results["Season"] <= year].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    elo = run_elo(
        results=results,
        conferences=data[conf_key][data[conf_key]["Season"] <= year],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
    )

    tourney = load_tourney_results(
        Path(data[results_key].attrs.get("data_dir", "data")),
        gender, year
    )
    if tourney.empty:
        print(f"No tournament data for {year} {gender}")
        return None

    preds, outcomes = [], []
    for _, game in tourney.iterrows():
        low = min(game["WTeamID"], game["LTeamID"])
        high = max(game["WTeamID"], game["LTeamID"])
        pred = calc_elo_win_tourney(elo.get(low, 1500), elo.get(high, 1500))
        outcome = 1 if game["WTeamID"] == low else 0
        preds.append(pred)
        outcomes.append(outcome)

    score = brier_score(preds, outcomes)
    return {"preds": preds, "outcomes": outcomes, "brier": score, "games": len(preds)}


def backtest_season(data, gender, year):
    """Predict each regular season game using pre-game Elo ratings."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key]
    results = results[results["Season"] <= year].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    _, preds, outcomes = _run_elo_collecting_preds(
        results=results,
        conferences=data[conf_key][data[conf_key]["Season"] <= year],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        predict_seasons={year},
    )

    if not preds:
        print(f"No season data for {year} {gender}")
        return None

    score = brier_score(preds, outcomes)
    return {"preds": preds, "outcomes": outcomes, "brier": score, "games": len(preds)}


MODEL_NAMES = ["random_forest", "logistic", "svm", "xgboost"]
DISPLAY_NAMES = {"random_forest": "Random Forest", "logistic": "Logistic",
                 "svm": "SVM", "xgboost": "XGBoost"}


def _load_models(model_dir, gender):
    model_dir = Path(model_dir)
    models = {}
    for name in MODEL_NAMES:
        path = model_dir / f"{gender}_{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def backtest_tourney_ml(data, gender, year, model_dir):
    """Predict tournament games with ML models using end-of-season Elo features."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key]
    results = results[results["Season"] <= year].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik = _get_extra_data(data, gender)
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)

    _, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[conf_key][data[conf_key]["Season"] <= year],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(),
        detailed_results=detailed,
        massey_per_system=massey_per,
        massey_avg=massey_avg,
        coach_tenure=coach_tenure,
        coach_changed=coach_changed,
        barttorvik=barttorvik,
    )

    tourney = load_tourney_results(
        Path(data[results_key].attrs.get("data_dir", "data")),
        gender, year,
    )
    if tourney.empty:
        return {}

    feature_rows, outcomes = [], []
    for _, game in tourney.iterrows():
        low = min(game["WTeamID"], game["LTeamID"])
        high = max(game["WTeamID"], game["LTeamID"])
        feat = matchup_features(
            elo_ratings.get(low, 1500), elo_ratings.get(high, 1500),
            game_counts, low, high,
            team_states=team_states, season=year, seeds=seeds,
            massey_per_system=massey_per, massey_avg=massey_avg,
            coach_tenure=coach_tenure, coach_changed=coach_changed,
            conf_elo_means=conf_elo_means,
            conferences=data[conf_key],
            barttorvik_lookup=barttorvik_lookup,
        )
        feature_rows.append(feat)
        outcomes.append(1 if game["WTeamID"] == low else 0)

    X = pd.DataFrame(feature_rows)[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0)
    outcomes = np.array(outcomes)

    models = _load_models(model_dir, gender)
    results_dict = {}
    for name, model in models.items():
        preds = model.predict_proba(X)[:, 1]
        score = brier_score(preds, outcomes)
        results_dict[name] = {"preds": preds.tolist(), "outcomes": outcomes.tolist(),
                              "brier": score, "games": len(preds)}
    return results_dict


def backtest_season_ml(data, gender, year, model_dir):
    """Predict regular season games with ML models using pre-game Elo features."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key]
    results = results[results["Season"] <= year].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    detailed, _, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik = _get_extra_data(data, gender)

    features_df, _, _, _, _ = build_features(
        results=results,
        conferences=data[conf_key][data[conf_key]["Season"] <= year],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons={year},
        detailed_results=detailed,
        massey_per_system=massey_per,
        massey_avg=massey_avg,
        coach_tenure=coach_tenure,
        coach_changed=coach_changed,
        barttorvik=barttorvik,
    )

    if features_df.empty:
        return {}

    X = features_df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0)
    outcomes = features_df["win"].values

    models = _load_models(model_dir, gender)
    results_dict = {}
    for name, model in models.items():
        preds = model.predict_proba(X)[:, 1]
        score = brier_score(preds, outcomes)
        results_dict[name] = {"preds": preds.tolist(), "outcomes": outcomes.tolist(),
                              "brier": score, "games": len(preds)}
    return results_dict


def _print_results(label, result):
    print(f"\n=== {label} ===")
    print(f"Games: {result['games']} | Brier Score: {result['brier']:.3f}")

    try:
        cal = calibration_table(result["preds"], result["outcomes"])
        print("\nCalibration:")
        print(f"{'Bin':<14}| {'Games':>5} | {'Actual':>7} | {'Expected':>8}")
        for _, row in cal.iterrows():
            print(f"{row['bin_label']:<14}| {row['count']:>5} | "
                  f"{row['actual_win_pct']:>6.1%} | {row['expected_win_pct']:>7.1%}")
    except NotImplementedError:
        print("(calibration_table not yet implemented)")


def _print_ml_results(label, elo_result, ml_results):
    print(f"\n=== {label} ===")
    if elo_result:
        print(f"{'Elo:':<20} Brier {elo_result['brier']:.3f} ({elo_result['games']} games)")
    for name in MODEL_NAMES:
        if name in ml_results:
            r = ml_results[name]
            print(f"{DISPLAY_NAMES[name] + ':':<20} Brier {r['brier']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Elo against past tournaments")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mode", choices=["tourney", "season", "both"], default="tourney",
                        help="tourney = NCAA tournament only, season = regular season, both = compare")
    parser.add_argument("--model-dir", default=None,
                        help="Path to trained ML models. If provided, evaluates ML alongside Elo.")
    args = parser.parse_args()

    data = load_data(args.data_dir)
    for key in ("mens_results", "womens_results"):
        data[key].attrs["data_dir"] = args.data_dir

    all_tourney = {"preds": [], "outcomes": []}
    all_season = {"preds": [], "outcomes": []}

    for year in args.years:
        for gender, label in [("M", "Men's"), ("W", "Women's")]:

            if args.mode in ("tourney", "both"):
                result = backtest_tourney(data, gender, year)
                if result:
                    if args.model_dir:
                        ml = backtest_tourney_ml(data, gender, year, args.model_dir)
                        _print_ml_results(f"{year} {label} Tournament", result, ml)
                    else:
                        _print_results(f"{year} {label} Tournament", result)
                    all_tourney["preds"].extend(result["preds"])
                    all_tourney["outcomes"].extend(result["outcomes"])

            if args.mode in ("season", "both"):
                result = backtest_season(data, gender, year)
                if result:
                    if args.model_dir:
                        ml = backtest_season_ml(data, gender, year, args.model_dir)
                        _print_ml_results(f"{year} {label} Regular Season", result, ml)
                    else:
                        _print_results(f"{year} {label} Regular Season", result)
                    all_season["preds"].extend(result["preds"])
                    all_season["outcomes"].extend(result["outcomes"])

    print("\n=== Overall ===")
    if all_tourney["preds"]:
        b = brier_score(all_tourney["preds"], all_tourney["outcomes"])
        print(f"Tournament Brier:      {b:.3f} ({len(all_tourney['preds'])} games)")
    if all_season["preds"]:
        b = brier_score(all_season["preds"], all_season["outcomes"])
        print(f"Regular Season Brier:  {b:.3f} ({len(all_season['preds'])} games)")


if __name__ == "__main__":
    main()
