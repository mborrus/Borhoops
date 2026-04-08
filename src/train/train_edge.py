"""Edge model: predict where Vegas is wrong.

Two approaches:
  1. Residual model: predict (outcome - vegas_prob) from features
  2. Edge detection: predict outcome from features + spread, then compare to spread alone

Evaluation metrics (not Brier):
  - Hit rate at threshold: when |model_edge| > T, what % are correct?
  - Expected profit: simulated flat-betting ROI at -110 vig
  - Kelly growth: log-wealth growth rate

Usage:
  PYTHONPATH=src python -m train.train_edge
  PYTHONPATH=src python -m train.train_edge --approach residual
  PYTHONPATH=src python -m train.train_edge --approach detection
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from train.evaluate import (
    prepare_data, prepare_fold, TOURNEY_YEARS, save_results,
    _load_tourney_results,
)
from train.features import FEATURE_COLS
from train.extended_features import _spread_to_prob, SPREAD_TO_PROB_K

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


# -- Feature sets to explore ---------------------------------------------------

# Game model best (for comparison)
GAME_FEATURES = [
    "elo_diff", "elo_pred", "home", "day_num", "sos_diff", "margin_mean_diff",
    "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]

# Edge model candidates: features the market might underweight
# Include everything — let the model figure out what Vegas misses
EDGE_FEATURES_FULL = [
    "elo_diff", "elo_pred", "home", "day_num", "game_count_avg",
    "sos_diff", "win_streak_diff", "rest_days_diff", "margin_mean_diff", "margin_std_diff",
    "off_eff_r10_diff", "def_eff_r10_diff", "efg_pct_r10_diff", "opp_efg_pct_r10_diff",
    "to_rate_r10_diff", "or_pct_r10_diff", "pace_r10_diff",
    "massey_avg_diff",
    "coach_tenure_diff", "coach_change_diff",
    "close_win_pct_diff", "conf_elo_diff",
    "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
]

# Edge-specific: features that might be underpriced by market
EDGE_FEATURES_BEHAVIORAL = [
    "win_streak_diff",      # momentum / recency bias
    "rest_days_diff",       # fatigue
    "margin_std_diff",      # consistency (market may overprice volatile teams)
    "coach_change_diff",    # market slow to adjust to new coaches
    "close_win_pct_diff",   # clutch performance
    "off_eff_r10_diff",     # recent form vs season-long rating
    "def_eff_r10_diff",
    "pace_r10_diff",        # tempo mismatches
    "home",                 # HFA mispricing
    "day_num",              # early vs late season
]


def _get_spread_for_fold(data, gender, year, data_dir):
    """Get Vegas spread for tournament games in a LOYO fold.
    Returns array of spreads (NaN where unavailable), aligned with y_test.
    """
    odds_lookup = data.get("odds_lookup", {}) if gender == "M" else {}
    tourney = _load_tourney_results(data_dir, gender)
    tourney_year = tourney[tourney["Season"] == year]

    spreads = []
    for _, g in tourney_year.iterrows():
        low, high = g["LowTeam"], g["HighTeam"]
        # Try multiple DayNums around tournament time (134-154)
        spread = np.nan
        for dn in range(134, 155):
            entry = odds_lookup.get((year, dn, low, high))
            if entry is not None:
                spread = entry["spread"]
                break
        spreads.append(spread)
    return np.array(spreads)


# -- Betting evaluation metrics ------------------------------------------------

def evaluate_betting(y_true, model_prob, vegas_prob, thresholds=(0.02, 0.05, 0.08, 0.10)):
    """Evaluate edge model on betting metrics.

    Returns dict with metrics at each threshold.
    """
    results = {}

    for t in thresholds:
        edge = model_prob - vegas_prob
        bet_mask = np.abs(edge) >= t
        if not bet_mask.any():
            results[f"t{t:.2f}"] = {"n_bets": 0}
            continue

        bet_on_low = edge[bet_mask] > 0
        actual = y_true[bet_mask]
        correct = (actual == 1) == bet_on_low

        n_bets = int(bet_mask.sum())
        wins = int(correct.sum())
        win_rate = wins / n_bets
        profit_per_bet = np.where(correct, 1.0, -1.1)
        total_profit = float(profit_per_bet.sum())
        roi = total_profit / (n_bets * 1.1)

        # Kelly: avg log-wealth growth
        kelly_growth = float(np.mean(np.log(1 + np.where(correct, 1/1.1, -1.0) * 0.05)))

        results[f"t{t:.2f}"] = {
            "n_bets": n_bets,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "total_profit": round(total_profit, 2),
            "roi": round(roi, 4),
            "kelly_growth": round(kelly_growth, 6),
            "avg_edge": round(float(np.mean(np.abs(edge[bet_mask]))), 4),
        }

    return results


# -- Approach 1: Residual model ------------------------------------------------

def run_residual_model(data, feature_names, gender="M", data_dir="data"):
    """Train model to predict (outcome - vegas_prob).
    Then: model_edge = model_prediction, bet when |edge| > threshold.
    """
    feat_idx = [FEATURE_COLS.index(c) for c in feature_names]

    all_model_prob = []
    all_vegas_prob = []
    all_outcomes = []

    for year in TOURNEY_YEARS:
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)
        spreads = _get_spread_for_fold(data, gender, year, data_dir)

        # Train: use only games with odds
        # Build training spreads from the odds lookup for regular season games
        # For simplicity: train a standard classifier on our features,
        # then compute edge as (our_prob - vegas_prob) at test time
        X_tr = X_train[:, feat_idx]
        X_te = X_test[:, feat_idx]

        model = XGBClassifier(
            n_estimators=500, max_depth=2, learning_rate=0.03,
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, eval_metric="logloss",
        )
        model.fit(X_tr, y_train)
        model_prob = model.predict_proba(X_te)[:, 1]

        # Vegas prob from spread
        vegas_prob = np.array([_spread_to_prob(s) for s in spreads])

        # Only keep games with odds
        valid = ~np.isnan(spreads)
        if valid.any():
            all_model_prob.extend(model_prob[valid])
            all_vegas_prob.extend(vegas_prob[valid])
            all_outcomes.extend(y_test[valid])

    all_model_prob = np.array(all_model_prob)
    all_vegas_prob = np.array(all_vegas_prob)
    all_outcomes = np.array(all_outcomes)

    print(f"  {len(all_outcomes)} tournament games with odds")

    # Evaluate
    betting = evaluate_betting(all_outcomes, all_model_prob, all_vegas_prob)
    from train.evaluate import brier_score
    model_brier = brier_score(all_outcomes, all_model_prob)
    vegas_brier = brier_score(all_outcomes, all_vegas_prob)

    return {
        "n_games": len(all_outcomes),
        "model_brier": round(model_brier, 4),
        "vegas_brier": round(vegas_brier, 4),
        "betting": betting,
        "features": feature_names,
    }


# -- Approach 2: Edge detection model -----------------------------------------

def run_detection_model(data, feature_names, gender="M", data_dir="data"):
    """Train model with spread as a feature. The model learns to correct the market.
    Edge = model_prob - vegas_prob.
    """
    feat_idx = [FEATURE_COLS.index(c) for c in feature_names]
    spread_idx = FEATURE_COLS.index("spread")
    implied_idx = FEATURE_COLS.index("implied_prob")

    all_model_prob = []
    all_vegas_prob = []
    all_outcomes = []

    for year in TOURNEY_YEARS:
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)
        spreads = _get_spread_for_fold(data, gender, year, data_dir)

        # Augment features with spread (NaN-native for XGBoost)
        X_tr = np.column_stack([X_train[:, feat_idx], X_train[:, spread_idx], X_train[:, implied_idx]])
        X_te = np.column_stack([X_test[:, feat_idx], X_test[:, spread_idx], X_test[:, implied_idx]])

        model = XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, eval_metric="logloss",
        )
        model.fit(X_tr, y_train)
        model_prob = model.predict_proba(X_te)[:, 1]

        vegas_prob = np.array([_spread_to_prob(s) for s in spreads])

        valid = ~np.isnan(spreads)
        if valid.any():
            all_model_prob.extend(model_prob[valid])
            all_vegas_prob.extend(vegas_prob[valid])
            all_outcomes.extend(y_test[valid])

    all_model_prob = np.array(all_model_prob)
    all_vegas_prob = np.array(all_vegas_prob)
    all_outcomes = np.array(all_outcomes)

    print(f"  {len(all_outcomes)} tournament games with odds")

    betting = evaluate_betting(all_outcomes, all_model_prob, all_vegas_prob)
    from train.evaluate import brier_score
    model_brier = brier_score(all_outcomes, all_model_prob)
    vegas_brier = brier_score(all_outcomes, all_vegas_prob)

    return {
        "n_games": len(all_outcomes),
        "model_brier": round(model_brier, 4),
        "vegas_brier": round(vegas_brier, 4),
        "betting": betting,
        "features": feature_names + ["spread", "implied_prob"],
    }


def print_betting_results(label, results):
    print(f"\n=== {label} ===")
    print(f"  Games with odds: {results['n_games']}")
    print(f"  Model Brier: {results['model_brier']:.4f}")
    print(f"  Vegas Brier: {results['vegas_brier']:.4f}")
    print(f"  Betting results:")
    for threshold, stats in sorted(results["betting"].items()):
        if stats["n_bets"] == 0:
            print(f"    {threshold}: 0 bets")
            continue
        print(f"    {threshold}: {stats['n_bets']} bets, "
              f"win {stats['win_rate']:.1%}, "
              f"ROI {stats['roi']:.1%}, "
              f"profit {stats['total_profit']:.1f}u")


def main():
    parser = argparse.ArgumentParser(description="Edge model: predict where Vegas is wrong")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/edge")
    parser.add_argument("--approach", default="both",
                        choices=["residual", "detection", "both"])
    args = parser.parse_args()

    data = prepare_data(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    if args.approach in ("residual", "both"):
        # Test residual model with different feature sets
        for name, features in [
            ("game_11", GAME_FEATURES),
            ("full_25", EDGE_FEATURES_FULL),
            ("behavioral_10", EDGE_FEATURES_BEHAVIORAL),
        ]:
            print(f"\nResidual model: {name} ({len(features)} features)")
            r = run_residual_model(data, features, args.gender, args.data_dir)
            print_betting_results(f"Residual {name}", r)
            all_results[f"residual_{name}"] = r

    if args.approach in ("detection", "both"):
        # Test edge detection with different feature sets
        for name, features in [
            ("game_11", GAME_FEATURES),
            ("full_25", EDGE_FEATURES_FULL),
            ("behavioral_10", EDGE_FEATURES_BEHAVIORAL),
        ]:
            print(f"\nEdge detection: {name} ({len(features)} + spread)")
            r = run_detection_model(data, features, args.gender, args.data_dir)
            print_betting_results(f"Detection {name}", r)
            all_results[f"detection_{name}"] = r

    path = output_dir / f"edge_{args.gender}_{timestamp}.json"
    save_results(all_results, path, model_name="edge_model")
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
