"""Leave-one-year-out cross-validation framework for tournament prediction.

Shared evaluation protocol for all Phase 2 models. Each fold:
  1. Build features from regular season games (all years except target)
  2. Build tournament matchup features for target year
  3. Train model, predict, compute Brier score

Usage:
  from train.evaluate import prepare_data, run_loyo_cv
  data = prepare_data()
  results = run_loyo_cv(lambda: XGBClassifier(), data, gender="M")
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from train.features import build_features, matchup_features, FEATURE_COLS, _build_barttorvik_lookup


# 2020 cancelled (COVID). 10 folds.
TOURNEY_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def prepare_data(data_dir="data"):
    return load_data(data_dir)


def _load_tourney_results(data_dir, gender):
    """Load tournament game outcomes as (LowTeam, HighTeam, LowTeamWon) rows."""
    prefix = "M" if gender == "M" else "W"
    path = Path(data_dir) / "kaggle" / f"{prefix}NCAATourneyCompactResults.csv"
    df = pd.read_csv(path)
    rows = []
    for _, g in df.iterrows():
        low = min(g["WTeamID"], g["LTeamID"])
        high = max(g["WTeamID"], g["LTeamID"])
        rows.append({
            "Season": g["Season"],
            "LowTeam": low,
            "HighTeam": high,
            "LowTeamWon": 1 if g["WTeamID"] == low else 0,
        })
    return pd.DataFrame(rows)


def _get_gender_data(data, gender):
    """Extract all gender-specific lookups from the data dict."""
    is_mens = gender == "M"
    return {
        "results_key": "mens_results" if is_mens else "womens_results",
        "conf_key": "mens_conf" if is_mens else "womens_conf",
        "detailed": data.get("mens_detailed" if is_mens else "womens_detailed"),
        "seeds": data.get("m_seeds", {}) if is_mens else data.get("w_seeds", {}),
        "massey_per": data.get("massey_per_system", {}) if is_mens else {},
        "massey_avg": data.get("massey_avg", {}) if is_mens else {},
        "coach_tenure": data.get("coach_tenure", {}) if is_mens else {},
        "coach_changed": data.get("coach_changed", {}) if is_mens else {},
        "barttorvik": data.get("barttorvik") if is_mens else None,
        "odds_lookup": data.get("odds_lookup", {}) if is_mens else {},
        "odds_team_avg": data.get("odds_team_avg", {}) if is_mens else {},
        "poll_lookup": data.get("poll_lookup", {}) if is_mens else {},
        "weeks_ranked": data.get("weeks_ranked", {}) if is_mens else {},
        "roster_lookup": data.get("roster_lookup", {}) if is_mens else {},
    }


def prepare_fold(data, gender, target_year, data_dir="data"):
    """Build training features and tournament test set for one LOYO fold.

    Training: regular season features from all years except target_year.
    Test: tournament matchup features for target_year games.

    Returns (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    gd = _get_gender_data(data, gender)
    results = add_game_counts(data[gd["results_key"]].copy())
    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)
    barttorvik_lookup = _build_barttorvik_lookup(gd["barttorvik"])

    all_seasons = sorted(results["Season"].unique())
    train_seasons = set(s for s in all_seasons if s != target_year)

    # Build training features (Elo processes all seasons for continuity,
    # but only collects feature rows for train_seasons)
    features_df, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[gd["conf_key"]],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=train_seasons,
        detailed_results=gd["detailed"],
        massey_per_system=gd["massey_per"],
        massey_avg=gd["massey_avg"],
        coach_tenure=gd["coach_tenure"],
        coach_changed=gd["coach_changed"],
        barttorvik=gd["barttorvik"],
        odds_lookup=gd["odds_lookup"],
        odds_team_avg=gd["odds_team_avg"],
        poll_lookup=gd["poll_lookup"],
        weeks_ranked=gd["weeks_ranked"],
        roster_lookup=gd["roster_lookup"],
        seeds=gd["seeds"],
    )

    X_train = np.nan_to_num(features_df[FEATURE_COLS].values, nan=0.0)
    y_train = features_df["win"].values

    # Build tournament test set
    tourney = _load_tourney_results(data_dir, gender)
    tourney_year = tourney[tourney["Season"] == target_year]

    test_rows = []
    for _, g in tourney_year.iterrows():
        a, b = g["LowTeam"], g["HighTeam"]
        feat = matchup_features(
            elo_ratings.get(a, 1500), elo_ratings.get(b, 1500),
            game_counts, a, b,
            team_states=team_states, season=target_year, seeds=gd["seeds"],
            massey_per_system=gd["massey_per"], massey_avg=gd["massey_avg"],
            coach_tenure=gd["coach_tenure"], coach_changed=gd["coach_changed"],
            conf_elo_means=conf_elo_means, conferences=data[gd["conf_key"]],
            barttorvik_lookup=barttorvik_lookup,
            odds_team_avg=gd["odds_team_avg"],
            poll_lookup=gd["poll_lookup"], weeks_ranked=gd["weeks_ranked"],
            roster_lookup=gd["roster_lookup"],
        )
        test_rows.append(feat)

    X_test = np.nan_to_num(
        pd.DataFrame(test_rows)[FEATURE_COLS].values, nan=0.0
    )
    y_test = tourney_year["LowTeamWon"].values

    return X_train, y_train, X_test, y_test


def brier_score(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def run_loyo_cv(model_factory, data, gender="M", years=None, data_dir="data", verbose=True):
    """Run full leave-one-year-out CV.

    Args:
        model_factory: callable() returning a fresh model with fit() and predict_proba().
        data: dict from prepare_data().
        gender: "M" or "W".
        years: list of tournament years to evaluate. Defaults to TOURNEY_YEARS.

    Returns:
        dict with per-year results and aggregate metrics.
    """
    if years is None:
        years = TOURNEY_YEARS

    results = {}
    all_brier = []

    for year in years:
        t0 = time.time()
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

        bs = brier_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        elapsed = time.time() - t0

        results[year] = {
            "brier": round(bs, 4),
            "log_loss": round(ll, 4),
            "n_games": len(y_test),
            "n_train": len(y_train),
            "elapsed_s": round(elapsed, 1),
        }
        all_brier.append(bs)

        if verbose:
            print(f"  {year}: Brier={bs:.4f}  LogLoss={ll:.4f}  "
                  f"({len(y_test)} games, {elapsed:.1f}s)")

    results["overall"] = {
        "mean_brier": round(float(np.mean(all_brier)), 4),
        "std_brier": round(float(np.std(all_brier)), 4),
        "median_brier": round(float(np.median(all_brier)), 4),
        "n_folds": len(years),
        "n_total_games": sum(r["n_games"] for y, r in results.items() if y != "overall"),
    }

    if verbose:
        o = results["overall"]
        print(f"\n  LOYO CV: Brier = {o['mean_brier']:.4f} ± {o['std_brier']:.4f} "
              f"({o['n_total_games']} games across {o['n_folds']} folds)")

    return results


def save_results(results, path, model_name=None, extra=None):
    """Save evaluation results to JSON."""
    out = {"model": model_name, "results": results}
    if extra:
        out.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
