"""Tournament model autoresearch harness.

Evaluates tournament_experiment.py using LOSO CV on tournament games.
Same loop pattern as prepare.py but tournament-specific.

Usage:
  python autoresearch/tournament_prepare.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from train.features import (
    build_features, matchup_features, FEATURE_COLS,
    _build_barttorvik_lookup,
)
from evaluate.metrics import brier_score

DATA_DIR = REPO_ROOT / "data"
TOURNEY_YEARS = [y for y in range(2003, 2026) if y != 2020]
EVAL_YEARS = [y for y in range(2015, 2026) if y != 2020]
GENDERS = ["M", "W"]


def _get_gender_data(data, gender):
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


def _build_interactions(X, feat_list, has_seeds):
    """Build feature interactions."""
    n_base = len(feat_list)
    seed_col = n_base if has_seeds else None
    massey_idx = feat_list.index("massey_avg_diff") if "massey_avg_diff" in feat_list else None
    barthag_idx = feat_list.index("barthag_diff") if "barthag_diff" in feat_list else None
    elo_idx = feat_list.index("elo_diff") if "elo_diff" in feat_list else None

    pairs = []
    if seed_col is not None and massey_idx is not None:
        pairs.append((seed_col, massey_idx))
    if seed_col is not None and barthag_idx is not None:
        pairs.append((seed_col, barthag_idx))
    if massey_idx is not None and barthag_idx is not None:
        pairs.append((massey_idx, barthag_idx))
    if seed_col is not None and elo_idx is not None:
        pairs.append((seed_col, elo_idx))

    if not pairs:
        return X
    ints = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
    return np.hstack([X, ints])


def build_fold(data, gender, eval_year, feat_list, include_seeds, include_interactions):
    """Build tournament train/test data for one LOSO fold."""
    gd = _get_gender_data(data, gender)
    results = add_game_counts(data[gd["results_key"]].copy())
    results = results[results["Season"] <= eval_year]

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)
    barttorvik_lookup = _build_barttorvik_lookup(gd["barttorvik"])

    _, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[gd["conf_key"]][data[gd["conf_key"]]["Season"] <= eval_year],
        hfa_dict=hfa_dict, location_dict=location_dict,
        seasons=set(),
        detailed_results=gd["detailed"],
        massey_per_system=gd["massey_per"], massey_avg=gd["massey_avg"],
        coach_tenure=gd["coach_tenure"], coach_changed=gd["coach_changed"],
        barttorvik=gd["barttorvik"],
        odds_lookup=gd["odds_lookup"], odds_team_avg=gd["odds_team_avg"],
        poll_lookup=gd["poll_lookup"], weeks_ranked=gd["weeks_ranked"],
        roster_lookup=gd["roster_lookup"], seeds=gd["seeds"],
    )

    prefix = "M" if gender == "M" else "W"
    tourney = pd.read_csv(DATA_DIR / "kaggle" / f"{prefix}NCAATourneyCompactResults.csv")

    feat_idx = [FEATURE_COLS.index(c) for c in feat_list]
    sd_idx = FEATURE_COLS.index("seed_diff")
    train_rows, test_rows = [], []
    train_y, test_y = [], []

    for year in [y for y in TOURNEY_YEARS if y <= eval_year]:
        year_games = tourney[tourney["Season"] == year]
        is_test = (year == eval_year)

        for _, g in year_games.iterrows():
            low = min(g["WTeamID"], g["LTeamID"])
            high = max(g["WTeamID"], g["LTeamID"])

            feat = matchup_features(
                elo_ratings.get(low, 1500), elo_ratings.get(high, 1500),
                game_counts, low, high,
                team_states=team_states, season=year, seeds=gd["seeds"],
                massey_per_system=gd["massey_per"], massey_avg=gd["massey_avg"],
                coach_tenure=gd["coach_tenure"], coach_changed=gd["coach_changed"],
                conf_elo_means=conf_elo_means, conferences=data[gd["conf_key"]],
                barttorvik_lookup=barttorvik_lookup,
                odds_team_avg=gd["odds_team_avg"],
                poll_lookup=gd["poll_lookup"], weeks_ranked=gd["weeks_ranked"],
                roster_lookup=gd["roster_lookup"],
            )
            row = [feat.get(c, 0) for c in FEATURE_COLS]
            outcome = 1 if g["WTeamID"] == low else 0

            if is_test:
                test_rows.append(row)
                test_y.append(outcome)
            else:
                train_rows.append(row)
                train_y.append(outcome)
                # Symmetric augmentation
                train_rows.append([-v for v in row])
                train_y.append(1 - outcome)

    if not test_rows or not train_rows:
        return None

    X_train = np.array(train_rows)[:, feat_idx]
    X_test = np.array(test_rows)[:, feat_idx]

    if include_seeds:
        X_train = np.hstack([X_train, np.array(train_rows)[:, sd_idx:sd_idx+1]])
        X_test = np.hstack([X_test, np.array(test_rows)[:, sd_idx:sd_idx+1]])

    if include_interactions:
        X_train = _build_interactions(X_train, feat_list, include_seeds)
        X_test = _build_interactions(X_test, feat_list, include_seeds)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    return X_train, np.array(train_y), X_test, np.array(test_y)


def evaluate(get_model_fn, feature_subset, include_seeds, include_interactions, clip):
    """Run LOSO tournament CV."""
    data = load_data(DATA_DIR)
    all_preds, all_outcomes = [], []
    per_year = {}

    for year in EVAL_YEARS:
        year_preds, year_outcomes = [], []
        for gender in GENDERS:
            fold = build_fold(data, gender, year, feature_subset, include_seeds, include_interactions)
            if fold is None:
                continue
            X_train, y_train, X_test, y_test = fold

            model = get_model_fn()
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            preds = np.clip(preds, clip[0], clip[1])

            year_preds.extend(preds)
            year_outcomes.extend(y_test)

        if year_preds:
            yb = brier_score(year_preds, year_outcomes)
            per_year[year] = {"brier": yb, "games": len(year_preds)}
            all_preds.extend(year_preds)
            all_outcomes.extend(year_outcomes)

    mean_brier = brier_score(all_preds, all_outcomes) if all_preds else 1.0
    return {
        "mean_brier": mean_brier,
        "per_year": per_year,
        "n_games": len(all_preds),
    }


if __name__ == "__main__":
    from tournament_experiment import get_model, FEATURE_SUBSET, INCLUDE_SEEDS, INCLUDE_INTERACTIONS, CLIP

    print(f"Loading data...")
    start = time.time()

    results = evaluate(get_model, FEATURE_SUBSET, INCLUDE_SEEDS, INCLUDE_INTERACTIONS, CLIP)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"  Mean Brier: {results['mean_brier']:.6f}")
    print(f"  Games:      {results['n_games']}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Features:   {len(FEATURE_SUBSET)} + seeds={INCLUDE_SEEDS} + int={INCLUDE_INTERACTIONS}")
    print(f"{'='*50}")
    for year, r in sorted(results["per_year"].items()):
        print(f"  {year}: {r['brier']:.4f} ({r['games']} games)")
    print()
    print("---")
    print(f"mean_brier:      {results['mean_brier']:.6f}")
    print(f"n_games:         {results['n_games']}")
    print(f"elapsed_seconds: {elapsed:.1f}")
