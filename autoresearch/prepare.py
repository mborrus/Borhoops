"""Immutable evaluation harness for autoresearch experiments.

DO NOT MODIFY THIS FILE. The agent only modifies experiment.py.

This module:
  1. Loads all data (Kaggle, Massey, coach, Barttorvik, DetailedResults, odds, polls)
  2. Runs leave-one-year-out CV on tournament predictions
  3. Reports Brier score — the single metric to minimize
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from train.features import build_features, matchup_features, FEATURE_COLS, _build_barttorvik_lookup
from evaluate.backtest import load_tourney_results
from evaluate.metrics import brier_score


# -- Constants (immutable) -----------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
CV_YEARS = list(range(2015, 2026))  # 11-fold leave-one-year-out
GENDERS = ["M", "W"]


# -- Data loading (cached across experiments) ----------------------------------

_DATA_CACHE = {}


def get_data():
    """Load all data once, cache for reuse across CV folds."""
    if not _DATA_CACHE:
        data = load_data(DATA_DIR)
        for key in ("mens_results", "womens_results"):
            data[key].attrs["data_dir"] = str(DATA_DIR)
        _DATA_CACHE.update(data)
    return _DATA_CACHE


def _get_extra_data(data, gender):
    is_mens = gender == "M"
    detailed_key = "mens_detailed" if is_mens else "womens_detailed"
    detailed = data.get(detailed_key)
    seeds = data.get("m_seeds", {}) if is_mens else data.get("w_seeds", {})
    massey_per = data.get("massey_per_system", {}) if is_mens else {}
    massey_avg = data.get("massey_avg", {}) if is_mens else {}
    coach_tenure = data.get("coach_tenure", {}) if is_mens else {}
    coach_changed = data.get("coach_changed", {}) if is_mens else {}
    barttorvik = data.get("barttorvik") if is_mens else None
    odds_lookup = data.get("odds_lookup", {}) if is_mens else {}
    odds_team_avg = data.get("odds_team_avg", {}) if is_mens else {}
    poll_lookup = data.get("poll_lookup", {}) if is_mens else {}
    weeks_ranked = data.get("weeks_ranked", {}) if is_mens else {}
    roster_lookup = data.get("roster_lookup", {}) if is_mens else {}
    return (detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed,
            barttorvik, odds_lookup, odds_team_avg, poll_lookup, weeks_ranked, roster_lookup)


# -- Core evaluation -----------------------------------------------------------

def build_fold(data, gender, eval_year):
    """Build train features + tournament test features for one CV fold.

    Returns (X_train, y_train, X_test, y_test, feature_names) or None if no tourney data.
    """
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"

    results = data[results_key]
    results = results[results["Season"] <= eval_year].copy()
    results = add_game_counts(results)

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    (detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed,
     barttorvik, odds_lookup, odds_team_avg, poll_lookup, weeks_ranked,
     roster_lookup) = _get_extra_data(data, gender)
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)

    all_seasons = sorted(results["Season"].unique())
    train_seasons = [s for s in all_seasons if s < eval_year]

    # Build training features
    features_df, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[conf_key][data[conf_key]["Season"] <= eval_year],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(train_seasons),
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

    X_train = features_df[FEATURE_COLS].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = features_df["win"].values

    # Build tournament test features
    tourney = load_tourney_results(str(DATA_DIR), gender, eval_year)
    if tourney.empty:
        return None

    feature_rows, outcomes = [], []
    for _, game in tourney.iterrows():
        low = min(game["WTeamID"], game["LTeamID"])
        high = max(game["WTeamID"], game["LTeamID"])
        feat = matchup_features(
            elo_ratings.get(low, 1500), elo_ratings.get(high, 1500),
            game_counts, low, high,
            team_states=team_states, season=eval_year, seeds=seeds,
            massey_per_system=massey_per, massey_avg=massey_avg,
            coach_tenure=coach_tenure, coach_changed=coach_changed,
            conf_elo_means=conf_elo_means,
            conferences=data[conf_key],
            barttorvik_lookup=barttorvik_lookup,
            odds_team_avg=odds_team_avg,
            poll_lookup=poll_lookup, weeks_ranked=weeks_ranked,
            roster_lookup=roster_lookup,
        )
        feature_rows.append(feat)
        outcomes.append(1 if game["WTeamID"] == low else 0)

    X_test = np.nan_to_num(
        pd.DataFrame(feature_rows)[FEATURE_COLS].values, nan=0.0
    )
    y_test = np.array(outcomes)

    return X_train, y_train, X_test, y_test, FEATURE_COLS


def evaluate(get_model_fn, feature_subset=None):
    """Run leave-one-year-out CV and return Brier score.

    Args:
        get_model_fn: callable that returns a fresh sklearn-compatible model
                      (must have .fit(X, y) and .predict_proba(X))
        feature_subset: list of feature names to use, or None for all

    Returns:
        dict with keys: mean_brier, per_year, n_games, elapsed_seconds
    """
    data = get_data()
    all_preds, all_outcomes = [], []
    per_year = {}
    start = time.time()

    if feature_subset is not None:
        col_idx = [FEATURE_COLS.index(c) for c in feature_subset]
    else:
        col_idx = None

    for year in CV_YEARS:
        year_preds, year_outcomes = [], []

        for gender in GENDERS:
            fold = build_fold(data, gender, year)
            if fold is None:
                continue

            X_train, y_train, X_test, y_test, _ = fold

            if col_idx is not None:
                X_train = X_train[:, col_idx]
                X_test = X_test[:, col_idx]

            model = get_model_fn()
            try:
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_test)[:, 1]
            except Exception as e:
                print(f"  CRASH {year} {gender}: {e}")
                continue

            year_preds.extend(preds)
            year_outcomes.extend(y_test)

        if year_preds:
            year_brier = brier_score(year_preds, year_outcomes)
            per_year[year] = {"brier": year_brier, "games": len(year_preds)}
            all_preds.extend(year_preds)
            all_outcomes.extend(year_outcomes)

    elapsed = time.time() - start
    mean_brier = brier_score(all_preds, all_outcomes) if all_preds else 1.0

    return {
        "mean_brier": mean_brier,
        "per_year": per_year,
        "n_games": len(all_preds),
        "elapsed_seconds": elapsed,
    }


def print_results(results):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  Mean Brier: {results['mean_brier']:.6f}")
    print(f"  Games:      {results['n_games']}")
    print(f"  Time:       {results['elapsed_seconds']:.1f}s")
    print(f"  Features:   {len(FEATURE_COLS)}")
    print(f"{'='*50}")
    for year, r in sorted(results["per_year"].items()):
        print(f"  {year}: {r['brier']:.4f} ({r['games']} games)")
    print()


# -- Standalone run: evaluate experiment.py ------------------------------------

if __name__ == "__main__":
    from experiment import get_model, FEATURE_SUBSET

    print(f"Loading data ({len(FEATURE_COLS)} features available)...")
    get_data()

    print("Running leave-one-year-out CV...")
    results = evaluate(get_model, FEATURE_SUBSET)
    print_results(results)

    # Output in grep-friendly format
    print("---")
    print(f"mean_brier:      {results['mean_brier']:.6f}")
    print(f"n_games:         {results['n_games']}")
    print(f"elapsed_seconds: {results['elapsed_seconds']:.1f}")
    print(f"n_features:      {len(FEATURE_SUBSET) if FEATURE_SUBSET else len(FEATURE_COLS)}")
