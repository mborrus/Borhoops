"""Phase 4: Alpha analysis — model vs Vegas closing lines.

Compares model predictions against Vegas-implied probabilities.
Identifies systematic edges and simulates flat-betting ROI.

Usage:
  PYTHONPATH=src python -m train.alpha_analysis
  PYTHONPATH=src python -m train.alpha_analysis --threshold 0.05
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from train.evaluate import (
    prepare_data, prepare_fold, brier_score,
    TOURNEY_YEARS, save_results, _load_tourney_results,
)
from train.features import FEATURE_COLS
from train.extended_features import _spread_to_prob, SPREAD_TO_PROB_K

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Best 11 features from autoresearch
BEST_FEATURES = [
    "elo_diff", "elo_pred", "home", "day_num",
    "sos_diff", "margin_mean_diff",
    "massey_avg_diff",
    "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]
BEST_IDX = [FEATURE_COLS.index(c) for c in BEST_FEATURES]


def get_best_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=500, max_depth=2, learning_rate=0.03,
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, eval_metric="logloss",
        )),
    ])


def _load_tourney_odds(data_dir, gender="M"):
    """Load odds for tournament games. Returns {(Season, low, high): spread}."""
    odds_path = Path(data_dir) / "derived" / "ncaab_odds.csv"
    if not odds_path.exists():
        return {}

    df = pd.read_csv(odds_path)
    tourney_path = Path(data_dir) / "kaggle" / ("M" if gender == "M" else "W") + "NCAATourneyCompactResults.csv"

    # Build set of tournament game keys (Season, DayNum)
    tourney = pd.read_csv(tourney_path)
    tourney_keys = set()
    for _, g in tourney.iterrows():
        low = min(g["WTeamID"], g["LTeamID"])
        high = max(g["WTeamID"], g["LTeamID"])
        tourney_keys.add((g["Season"], g["DayNum"], low, high))

    # Match odds to tournament games
    lookup = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("HomeTeamID")) or pd.isna(row.get("AwayTeamID")):
            continue
        home = int(row["HomeTeamID"])
        away = int(row["AwayTeamID"])
        low, high = min(home, away), max(home, away)
        season = int(row["Season"])
        day_num = int(row["DayNum"])

        if (season, day_num, low, high) in tourney_keys:
            spread = row["Spread"]
            # Convert to low-team perspective
            spread_low = spread if home == low else -spread
            lookup[(season, low, high)] = spread_low

    return lookup


def simulate_betting(model_preds, vegas_spreads, outcomes, threshold=0.05):
    """Simulate flat-betting on games where model disagrees with market.

    Args:
        model_preds: array of model P(low team wins)
        vegas_spreads: array of closing spreads from low-team perspective
        outcomes: array of 1 (low team won) or 0
        threshold: minimum edge to bet (|model_prob - vegas_prob|)

    Returns dict with betting stats.
    """
    bets = []
    for pred, spread, actual in zip(model_preds, vegas_spreads, outcomes):
        if np.isnan(spread):
            continue
        vegas_prob = _spread_to_prob(spread)
        edge = pred - vegas_prob

        if abs(edge) >= threshold:
            bet_on_low = edge > 0  # model thinks low team is undervalued
            won = (actual == 1) == bet_on_low
            # Standard -110 vig
            profit = 1.0 if won else -1.1
            bets.append({
                "edge": round(edge, 4),
                "vegas_prob": round(vegas_prob, 4),
                "model_prob": round(pred, 4),
                "bet_on_low": bet_on_low,
                "won": won,
                "profit": profit,
                "actual": actual,
            })

    if not bets:
        return {"n_bets": 0, "win_pct": 0, "total_profit": 0, "roi": 0}

    n_bets = len(bets)
    wins = sum(b["won"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    total_risked = n_bets * 1.1  # risking 1.1 per bet (to win 1.0)
    roi = total_profit / total_risked if total_risked > 0 else 0

    return {
        "n_bets": n_bets,
        "wins": wins,
        "win_pct": round(wins / n_bets, 4),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "avg_edge": round(np.mean([abs(b["edge"]) for b in bets]), 4),
        "bets": bets,
    }


def calibration_comparison(model_preds, vegas_spreads, outcomes, n_bins=5):
    """Compare model vs Vegas calibration across probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    results = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (model_preds >= lo) & (model_preds < hi)
        if not mask.any():
            continue

        model_avg = model_preds[mask].mean()
        actual_avg = outcomes[mask].mean()
        n = mask.sum()

        # Vegas comparison for same games
        vegas_probs = np.array([_spread_to_prob(s) if not np.isnan(s) else np.nan
                                for s in vegas_spreads[mask]])
        vegas_valid = ~np.isnan(vegas_probs)
        vegas_avg = vegas_probs[vegas_valid].mean() if vegas_valid.any() else np.nan

        results.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n_games": int(n),
            "actual_win_pct": round(actual_avg, 3),
            "model_avg_prob": round(model_avg, 3),
            "vegas_avg_prob": round(vegas_avg, 3) if not np.isnan(vegas_avg) else None,
            "model_error": round(abs(model_avg - actual_avg), 4),
            "vegas_error": round(abs(vegas_avg - actual_avg), 4) if not np.isnan(vegas_avg) else None,
        })

    return results


def run_alpha_analysis(data, gender="M", data_dir="data", thresholds=None):
    """Full alpha analysis across LOYO folds."""
    if thresholds is None:
        thresholds = [0.02, 0.05, 0.10, 0.15]

    tourney_odds = _load_tourney_odds(data_dir, gender)
    print(f"Tournament odds available: {len(tourney_odds)} games")

    all_model_preds = []
    all_vegas_spreads = []
    all_outcomes = []
    per_year = {}

    for year in TOURNEY_YEARS:
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)

        model = get_best_model()
        model.fit(X_train[:, BEST_IDX], y_train)
        y_pred = model.predict_proba(X_test[:, BEST_IDX])[:, 1]

        # Get tournament games for this year to match with odds
        tourney = _load_tourney_results(data_dir, gender)
        tourney_year = tourney[tourney["Season"] == year]

        spreads = []
        for _, g in tourney_year.iterrows():
            low = min(g["WTeamID"], g["LTeamID"])
            high = max(g["WTeamID"], g["LTeamID"])
            spread = tourney_odds.get((year, low, high), np.nan)
            spreads.append(spread)
        spreads = np.array(spreads)

        n_with_odds = (~np.isnan(spreads)).sum()
        bs_model = brier_score(y_test, y_pred)

        # Vegas Brier (where odds available)
        valid = ~np.isnan(spreads)
        if valid.any():
            vegas_probs = np.array([_spread_to_prob(s) for s in spreads[valid]])
            bs_vegas = brier_score(y_test[valid], vegas_probs)
        else:
            bs_vegas = None

        per_year[year] = {
            "n_games": len(y_test),
            "n_with_odds": n_with_odds,
            "model_brier": round(bs_model, 4),
            "vegas_brier": round(bs_vegas, 4) if bs_vegas is not None else None,
        }
        print(f"  {year}: Model={bs_model:.4f}, Vegas={bs_vegas:.4f if bs_vegas else 'N/A'} "
              f"({n_with_odds}/{len(y_test)} games with odds)")

        all_model_preds.extend(y_pred)
        all_vegas_spreads.extend(spreads)
        all_outcomes.extend(y_test)

    all_model_preds = np.array(all_model_preds)
    all_vegas_spreads = np.array(all_vegas_spreads)
    all_outcomes = np.array(all_outcomes)

    # Betting simulation at different thresholds
    print(f"\n=== Betting Simulation ===")
    betting = {}
    for t in thresholds:
        result = simulate_betting(all_model_preds, all_vegas_spreads, all_outcomes, t)
        betting[str(t)] = {k: v for k, v in result.items() if k != "bets"}
        print(f"  Threshold {t:.2f}: {result['n_bets']} bets, "
              f"win {result['win_pct']:.1%}, ROI {result['roi']:.1%}, "
              f"profit {result['total_profit']:.1f} units")

    # Calibration comparison
    print(f"\n=== Calibration Comparison ===")
    calibration = calibration_comparison(all_model_preds, all_vegas_spreads, all_outcomes)
    for row in calibration:
        vegas_err = f"Vegas={row['vegas_error']:.3f}" if row.get('vegas_error') else "Vegas=N/A"
        print(f"  {row['bin']}: Model={row['model_error']:.3f}, {vegas_err} "
              f"({row['n_games']} games)")

    # Overall
    valid = ~np.isnan(all_vegas_spreads)
    overall_model = brier_score(all_outcomes, all_model_preds)
    if valid.any():
        vegas_probs = np.array([_spread_to_prob(s) for s in all_vegas_spreads[valid]])
        overall_vegas = brier_score(all_outcomes[valid], vegas_probs)
    else:
        overall_vegas = None

    print(f"\n=== Overall ===")
    print(f"  Model Brier:  {overall_model:.4f} ({len(all_outcomes)} games)")
    if overall_vegas:
        print(f"  Vegas Brier:  {overall_vegas:.4f} ({valid.sum()} games with odds)")

    return {
        "per_year": per_year,
        "betting": betting,
        "calibration": calibration,
        "overall_model_brier": round(overall_model, 4),
        "overall_vegas_brier": round(overall_vegas, 4) if overall_vegas else None,
        "n_games": len(all_outcomes),
        "n_with_odds": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Alpha analysis: model vs Vegas")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/alpha")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10, 0.15])
    args = parser.parse_args()

    data = prepare_data(args.data_dir)
    print("Running alpha analysis...")
    results = run_alpha_analysis(data, args.gender, args.data_dir, args.thresholds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"alpha_{args.gender}_{timestamp}.json"
    save_results(results, path, model_name="alpha_analysis")
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
