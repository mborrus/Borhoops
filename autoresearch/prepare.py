"""Immutable evaluation harness for autoresearch experiments.

DO NOT MODIFY THIS FILE. The agent only modifies experiment.py.

This module:
  1. Loads all data (Kaggle, Massey, coach, Barttorvik, DetailedResults, odds, polls)
  2. Enriches features with odds (spread) and polls (AP rank, momentum)
  3. Runs leave-one-year-out CV on tournament predictions
  4. Reports Brier score — the single metric to minimize
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
from train.features import (
    build_features, matchup_features, FEATURE_COLS as ELO_FEATURE_COLS,
    _build_barttorvik_lookup,
)
from evaluate.backtest import load_tourney_results
from evaluate.metrics import brier_score


# -- Constants (immutable) -----------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
CV_YEARS = list(range(2015, 2026))  # 11-fold leave-one-year-out
GENDERS = ["M", "W"]

# Extra features added on top of the 30 Elo-loop features
ODDS_COLS = ["spread", "over_under", "implied_prob"]
POLL_COLS = ["poll_rank_diff", "poll_momentum_diff", "weeks_ranked_diff",
             "poll_points_diff", "seed_diff"]

FEATURE_COLS = list(ELO_FEATURE_COLS) + ODDS_COLS + POLL_COLS  # 38 total


# -- Data loading (cached across experiments) ----------------------------------

_DATA_CACHE = {}


def get_data():
    """Load all data once, cache for reuse across CV folds."""
    if not _DATA_CACHE:
        data = load_data(DATA_DIR)
        for key in ("mens_results", "womens_results"):
            data[key].attrs["data_dir"] = str(DATA_DIR)

        # Load odds
        odds_path = DATA_DIR / "derived" / "ncaab_odds.csv"
        if not odds_path.exists():
            # Try loading raw odds and building a simple lookup
            odds_dir = DATA_DIR / "odds"
            if odds_dir.exists():
                odds_files = sorted(odds_dir.glob("ncaab_odds_*.csv"))
                if odds_files:
                    data["odds"] = pd.concat(
                        [pd.read_csv(f) for f in odds_files], ignore_index=True
                    )
                else:
                    data["odds"] = pd.DataFrame()
            else:
                data["odds"] = pd.DataFrame()
        else:
            data["odds"] = pd.read_csv(odds_path)

        # Load polls
        polls_path = DATA_DIR / "derived" / "poll_rankings.csv"
        if not polls_path.exists():
            polls_dir = DATA_DIR / "polls"
            if polls_dir.exists():
                polls_files = sorted(polls_dir.glob("polls_*.csv"))
                if polls_files:
                    data["polls"] = pd.concat(
                        [pd.read_csv(f) for f in polls_files], ignore_index=True
                    )
                else:
                    data["polls"] = pd.DataFrame()
            else:
                data["polls"] = pd.DataFrame()
        else:
            data["polls"] = pd.read_csv(polls_path)

        # Build odds lookup: {(Season, ESPN_EventID): {Spread, OverUnder}}
        data["odds_lookup"] = _build_odds_lookup(data["odds"])

        # Build polls lookup: {(Season, TeamID): {rank, points, weeks_ranked, momentum}}
        data["polls_lookup"] = _build_polls_lookup(data["polls"])

        # Build seeds lookup: {(Season, TeamID): seed_num}
        data["seeds_lookup"] = {}
        for prefix, key in [("m", "m_seeds"), ("w", "w_seeds")]:
            seeds = data.get(key, {})
            data["seeds_lookup"].update(seeds)

        _DATA_CACHE.update(data)
    return _DATA_CACHE


def _build_odds_lookup(odds_df):
    """Build {(Season, HomeESPN_ID, AwayESPN_ID): {Spread, OverUnder}} from raw odds."""
    if odds_df.empty or "Season" not in odds_df.columns:
        return {}
    lookup = {}
    for _, row in odds_df.iterrows():
        key = (int(row.get("Season", 0)),
               int(row.get("HomeESPN_ID", 0)),
               int(row.get("AwayESPN_ID", 0)))
        lookup[key] = {
            "Spread": row.get("Spread"),
            "OverUnder": row.get("OverUnder"),
        }
    return lookup


def _build_polls_lookup(polls_df):
    """Build per-team season-end poll features.

    Returns {(Season, TeamID): {rank, points, weeks_ranked, momentum}}
    where TeamID is ESPN_TeamID (or Kaggle TeamID if poll_rankings.csv has it).
    """
    if polls_df.empty:
        return {}

    # Use TeamID if available (transformed), otherwise ESPN_TeamID
    id_col = "TeamID" if "TeamID" in polls_df.columns else "ESPN_TeamID"
    if id_col not in polls_df.columns:
        return {}

    # Only use AP poll
    ap = polls_df[polls_df["Poll"] == "AP"].copy() if "Poll" in polls_df.columns else polls_df.copy()
    if ap.empty:
        return {}

    lookup = {}
    for season in ap["Season"].unique():
        season_df = ap[ap["Season"] == season].sort_values("Week")

        # Per-team aggregates for the season
        for team_id in season_df[id_col].unique():
            team_df = season_df[season_df[id_col] == team_id].sort_values("Week")
            if team_df.empty:
                continue

            last = team_df.iloc[-1]
            first = team_df.iloc[0]
            lookup[(int(season), int(team_id))] = {
                "rank": int(last.get("Rank", 30)),
                "points": float(last.get("Points", 0)) if pd.notna(last.get("Points")) else 0.0,
                "weeks_ranked": len(team_df),
                "momentum": int(first.get("Rank", 30)) - int(last.get("Rank", 30)),  # positive = improving
            }

    return lookup


def _spread_to_implied_prob(spread):
    """Convert point spread to implied win probability via logistic function."""
    if spread is None or np.isnan(spread):
        return np.nan
    # Empirical: ~2.7 points per 10% probability shift
    return 1.0 / (1.0 + 10 ** (spread / 8.0))


def _get_extra_data(data, gender):
    detailed_key = "mens_detailed" if gender == "M" else "womens_detailed"
    detailed = data.get(detailed_key)
    seeds = data.get("m_seeds", {}) if gender == "M" else data.get("w_seeds", {})
    massey_per = data.get("massey_per_system", {}) if gender == "M" else {}
    massey_avg = data.get("massey_avg", {}) if gender == "M" else {}
    coach_tenure = data.get("coach_tenure", {}) if gender == "M" else {}
    coach_changed = data.get("coach_changed", {}) if gender == "M" else {}
    barttorvik = data.get("barttorvik") if gender == "M" else None
    return detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik


# -- Feature enrichment (odds + polls) -----------------------------------------

def _enrich_training_features(features_df, data):
    """Add odds/polls/seed columns to training feature rows.

    features_df has team_low, team_high, season columns from build_features().
    """
    polls_lookup = data.get("polls_lookup", {})
    seeds_lookup = data.get("seeds_lookup", {})

    n = len(features_df)
    extra = {col: np.full(n, np.nan) for col in ODDS_COLS + POLL_COLS}

    # Odds: not available per-game in training (would need ESPN ID mapping per game).
    # Leave as NaN for training — models learn to handle missing odds.
    # The signal comes at tournament prediction time.

    if "team_low" in features_df.columns and "season" in features_df.columns:
        for i, (_, row) in enumerate(features_df.iterrows()):
            season = int(row["season"])
            low = int(row["team_low"])
            high = int(row["team_high"])

            # Polls
            low_poll = polls_lookup.get((season, low), {})
            high_poll = polls_lookup.get((season, high), {})
            extra["poll_rank_diff"][i] = low_poll.get("rank", 30) - high_poll.get("rank", 30)
            extra["poll_momentum_diff"][i] = low_poll.get("momentum", 0) - high_poll.get("momentum", 0)
            extra["weeks_ranked_diff"][i] = low_poll.get("weeks_ranked", 0) - high_poll.get("weeks_ranked", 0)
            extra["poll_points_diff"][i] = low_poll.get("points", 0) - high_poll.get("points", 0)

            # Seeds (0 during regular season, populated for tournament teams)
            low_seed = seeds_lookup.get((season, low), 17)
            high_seed = seeds_lookup.get((season, high), 17)
            extra["seed_diff"][i] = low_seed - high_seed

    for col in ODDS_COLS + POLL_COLS:
        features_df[col] = extra[col]

    return features_df


def _enrich_matchup_features(feat_dict, season, team_a, team_b, data):
    """Add odds/polls/seed features to a tournament matchup feature dict."""
    polls_lookup = data.get("polls_lookup", {})
    seeds_lookup = data.get("seeds_lookup", {})

    # Odds: leave as NaN (no game-specific line for hypothetical matchups)
    feat_dict["spread"] = np.nan
    feat_dict["over_under"] = np.nan
    feat_dict["implied_prob"] = np.nan

    # Polls
    a_poll = polls_lookup.get((season, team_a), {})
    b_poll = polls_lookup.get((season, team_b), {})
    feat_dict["poll_rank_diff"] = a_poll.get("rank", 30) - b_poll.get("rank", 30)
    feat_dict["poll_momentum_diff"] = a_poll.get("momentum", 0) - b_poll.get("momentum", 0)
    feat_dict["weeks_ranked_diff"] = a_poll.get("weeks_ranked", 0) - b_poll.get("weeks_ranked", 0)
    feat_dict["poll_points_diff"] = a_poll.get("points", 0) - b_poll.get("points", 0)

    # Seeds
    seed_a = seeds_lookup.get((season, team_a), 17)
    seed_b = seeds_lookup.get((season, team_b), 17)
    feat_dict["seed_diff"] = seed_a - seed_b

    return feat_dict


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

    detailed, seeds, massey_per, massey_avg, coach_tenure, coach_changed, barttorvik = _get_extra_data(data, gender)
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)

    all_seasons = sorted(results["Season"].unique())
    train_seasons = [s for s in all_seasons if s < eval_year]

    # Build training features (30 Elo-loop features)
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
    )

    # Enrich with odds/polls/seeds (8 extra features → 38 total)
    features_df = _enrich_training_features(features_df, data)

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
        )
        feat = _enrich_matchup_features(feat, eval_year, low, high, data)
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
        feature_subset: list of feature names to use, or None for all 38

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
