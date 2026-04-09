"""Edge model sweep: find the best feature set and model for detecting mispriced games.

Tests different feature combinations, model types, and sweet spot thresholds
on regular season games with odds data.

Usage:
  PYTHONPATH=src python -m train.sweep_edge
  PYTHONPATH=src python -m train.sweep_edge --config-index 0  # HPC array
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from train.evaluate import prepare_data, save_results
from train.features import FEATURE_COLS, build_features
from train.extended_features import _spread_to_prob
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Feature sets to test
FEATURE_SETS = {
    "base11": [
        "elo_diff", "elo_pred", "home", "day_num", "sos_diff", "margin_mean_diff",
        "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
        "conf_elo_diff",
    ],
    "behavioral": [
        "win_streak_diff", "rest_days_diff", "margin_std_diff",
        "coach_change_diff", "close_win_pct_diff",
        "off_eff_r10_diff", "def_eff_r10_diff", "pace_r10_diff",
        "home", "day_num",
    ],
    "base11_plus_behavioral": [
        "elo_diff", "elo_pred", "home", "day_num", "sos_diff", "margin_mean_diff",
        "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
        "conf_elo_diff",
        "win_streak_diff", "rest_days_diff", "margin_std_diff",
        "coach_change_diff", "close_win_pct_diff",
    ],
    "massey_detail": [
        "elo_diff", "elo_pred", "home",
        "massey_pom_diff", "massey_sag_diff", "massey_mor_diff",
        "massey_dok_diff", "massey_col_diff", "massey_avg_diff",
        "barthag_diff", "conf_elo_diff",
    ],
    "efficiency_focus": [
        "elo_diff", "home", "massey_avg_diff",
        "off_eff_r10_diff", "def_eff_r10_diff", "efg_pct_r10_diff",
        "opp_efg_pct_r10_diff", "to_rate_r10_diff", "or_pct_r10_diff",
        "pace_r10_diff", "barthag_diff",
    ],
    "minimal5": [
        "elo_diff", "elo_pred", "home", "massey_avg_diff", "margin_mean_diff",
    ],
}


def _build_configs():
    configs = []
    for feat_name in FEATURE_SETS:
        # XGBoost configs
        for depth in [2, 3]:
            configs.append({
                "model": "xgb",
                "features": feat_name,
                "max_depth": depth,
                "learning_rate": 0.03,
            })
        # LightGBM
        configs.append({
            "model": "lgb",
            "features": feat_name,
            "max_depth": 3,
            "learning_rate": 0.03,
        })
        # Logistic Regression
        for C in [0.1, 1.0]:
            configs.append({
                "model": "lr",
                "features": feat_name,
                "C": C,
            })
    return configs


def _make_model(cfg):
    if cfg["model"] == "xgb":
        return XGBClassifier(
            n_estimators=500, max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
    elif cfg["model"] == "lgb":
        return LGBMClassifier(
            n_estimators=500, max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbose=-1,
        )
    elif cfg["model"] == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=cfg["C"], max_iter=1000)),
        ])


def ml_to_payout(ml):
    if ml > 0:
        return ml / 100
    else:
        return 100 / abs(ml)


def evaluate_edge(cfg, data, data_dir="data"):
    """Evaluate one edge model config on regular season sweet spot."""
    feat_list = FEATURE_SETS[cfg["features"]]
    feat_idx = [FEATURE_COLS.index(c) for c in feat_list]

    results_df = add_game_counts(data["mens_results"].copy())
    hfa = _build_hfa_dict(data["hfa"], "M")
    loc = _build_location_dict(data["home_lookup"], "M")

    features_df, _, _, _, _ = build_features(
        results=results_df, conferences=data["mens_conf"],
        hfa_dict=hfa, location_dict=loc,
        seasons=set(range(2013, 2026)),
        detailed_results=data.get("mens_detailed"),
        massey_per_system=data.get("massey_per_system", {}),
        massey_avg=data.get("massey_avg", {}),
        coach_tenure=data.get("coach_tenure", {}),
        coach_changed=data.get("coach_changed", {}),
        barttorvik=data.get("barttorvik"),
        odds_lookup=data.get("odds_lookup", {}),
        odds_team_avg=data.get("odds_team_avg", {}),
        poll_lookup=data.get("poll_lookup", {}),
        weeks_ranked=data.get("weeks_ranked", {}),
        roster_lookup=data.get("roster_lookup", {}),
        seeds=data.get("m_seeds", {}),
    )

    X = features_df[FEATURE_COLS].values
    y = features_df["win"].values
    spreads = features_df["spread"].values
    has_odds = ~np.isnan(spreads)
    seasons = features_df["season"].values

    # Load ML odds
    odds = pd.read_csv(f"{data_dir}/derived/ncaab_odds.csv")
    odds = odds[(odds["HomeML"] != 0) & (odds["AwayML"] != 0) &
                odds["HomeML"].notna() & odds["AwayML"].notna()]
    ml_lookup = {}
    for _, row in odds.iterrows():
        home, away = int(row["HomeTeamID"]), int(row["AwayTeamID"])
        low, high = min(home, away), max(home, away)
        key = (int(row["Season"]), int(row["DayNum"]), low, high)
        if home == low:
            ml_lookup[key] = {"low_ml": row["HomeML"], "high_ml": row["AwayML"]}
        else:
            ml_lookup[key] = {"low_ml": row["AwayML"], "high_ml": row["HomeML"]}

    daynums = features_df["day_num"].values
    team_lows = features_df["team_low"].values
    team_highs = features_df["team_high"].values

    # Per-year evaluation
    yearly = {}
    all_bets = []

    for year in range(2016, 2026):
        train_mask = (seasons < year) & has_odds
        test_mask = (seasons == year) & has_odds
        if test_mask.sum() == 0:
            continue

        m = _make_model(cfg)
        m.fit(np.nan_to_num(X[train_mask][:, feat_idx], nan=0.0), y[train_mask])
        mp = m.predict_proba(np.nan_to_num(X[test_mask][:, feat_idx], nan=0.0))[:, 1]
        vp = np.array([_spread_to_prob(s) for s in spreads[test_mask]])
        yt = y[test_mask]
        dn = daynums[test_mask]
        tl = team_lows[test_mask]
        th = team_highs[test_mask]

        # Sweet spot: Vegas 45-55%, model 15%+ edge
        sweet = (vp >= 0.45) & (vp <= 0.55) & (np.abs(mp - vp) >= 0.15)
        if sweet.sum() == 0:
            yearly[year] = {"n_bets": 0}
            continue

        year_profit = 0
        year_wins = 0
        year_n = 0
        for i in np.where(sweet)[0]:
            key = (year, int(dn[i]), int(tl[i]), int(th[i]))
            ml = ml_lookup.get(key)
            if ml is None:
                continue
            bet_on_low = mp[i] > vp[i]
            odds_used = ml["low_ml"] if bet_on_low else ml["high_ml"]
            won = (yt[i] == 1) == bet_on_low
            payout = ml_to_payout(odds_used)
            profit = payout if won else -1.0
            year_profit += profit
            year_wins += int(won)
            year_n += 1
            all_bets.append({"year": year, "won": won, "profit": profit})

        if year_n > 0:
            yearly[year] = {
                "n_bets": year_n,
                "win_rate": round(year_wins / year_n, 4),
                "profit": round(year_profit, 1),
                "roi": round(year_profit / year_n, 4),
            }

    # Also evaluate wider sweet spot (40-60%)
    wide_bets = []
    for year in range(2016, 2026):
        train_mask = (seasons < year) & has_odds
        test_mask = (seasons == year) & has_odds
        if test_mask.sum() == 0:
            continue
        m = _make_model(cfg)
        m.fit(np.nan_to_num(X[train_mask][:, feat_idx], nan=0.0), y[train_mask])
        mp = m.predict_proba(np.nan_to_num(X[test_mask][:, feat_idx], nan=0.0))[:, 1]
        vp = np.array([_spread_to_prob(s) for s in spreads[test_mask]])
        yt = y[test_mask]
        wide = (vp >= 0.40) & (vp <= 0.60) & (np.abs(mp - vp) >= 0.15)
        for i in np.where(wide)[0]:
            key = (year, int(daynums[test_mask][i]), int(team_lows[test_mask][i]), int(team_highs[test_mask][i]))
            ml = ml_lookup.get(key)
            if ml is None:
                continue
            bet_on_low = mp[i] > vp[i]
            won = (yt[i] == 1) == bet_on_low
            payout = ml_to_payout(ml["low_ml"] if bet_on_low else ml["high_ml"])
            wide_bets.append({"won": won, "profit": payout if won else -1.0})

    # Summary
    if all_bets:
        total_wins = sum(b["won"] for b in all_bets)
        total_profit = sum(b["profit"] for b in all_bets)
        n = len(all_bets)
        sweet_result = {
            "n_bets": n, "win_rate": round(total_wins / n, 4),
            "profit": round(total_profit, 1), "roi": round(total_profit / n, 4),
        }
    else:
        sweet_result = {"n_bets": 0}

    if wide_bets:
        w_wins = sum(b["won"] for b in wide_bets)
        w_profit = sum(b["profit"] for b in wide_bets)
        w_n = len(wide_bets)
        wide_result = {
            "n_bets": w_n, "win_rate": round(w_wins / w_n, 4),
            "profit": round(w_profit, 1), "roi": round(w_profit / w_n, 4),
        }
    else:
        wide_result = {"n_bets": 0}

    # Conference analysis: split by whether teams are from power conferences
    # (approximation: high-Elo teams = power conference)

    return {
        "sweet_spot_45_55": sweet_result,
        "sweet_spot_40_60": wide_result,
        "per_year": yearly,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Edge model sweep")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/edge_sweep")
    args = parser.parse_args()

    configs = _build_configs()
    print(f"Total configs: {len(configs)}")

    data = prepare_data(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_index is not None:
        if args.config_index >= len(configs):
            print(f"Index {args.config_index} out of range")
            return
        cfg = configs[args.config_index]
        print(f"Config {args.config_index}: {cfg}")
        t0 = time.time()
        result = evaluate_edge(cfg, data, args.data_dir)
        elapsed = time.time() - t0
        ss = result["sweet_spot_45_55"]
        print(f"  Sweet 45-55%: {ss.get('n_bets',0)} bets, "
              f"win={ss.get('win_rate',0):.1%}, ROI={ss.get('roi',0):+.1%} ({elapsed:.0f}s)")
        result["elapsed_s"] = round(elapsed, 1)
        path = output_dir / f"edge_{args.config_index}.json"
        save_results(result, path, model_name=f"edge_{args.config_index}")
    else:
        for i, cfg in enumerate(configs):
            t0 = time.time()
            result = evaluate_edge(cfg, data, args.data_dir)
            elapsed = time.time() - t0
            ss = result["sweet_spot_45_55"]
            wr = ss.get("win_rate", 0)
            marker = " ***" if wr > 0.76 else ""
            print(f"[{i+1}/{len(configs)}] {cfg['model']:3s} {cfg['features']:25s} "
                  f"bets={ss.get('n_bets',0):3d} win={wr:.1%} "
                  f"ROI={ss.get('roi',0):+.1%} ({elapsed:.0f}s){marker}")
            result["elapsed_s"] = round(elapsed, 1)
            path = output_dir / f"edge_{i}.json"
            save_results(result, path, model_name=f"edge_{i}")


if __name__ == "__main__":
    main()
