"""Deep edge model sweep: test all feature combos with cached Elo.

Uses cached Elo snapshots for instant evaluation. Tests every 3-5 feature
combination from the top candidates across multiple thresholds and vegas ranges.

Usage:
  PYTHONPATH=src python -m train.sweep_edge_deep --config-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train.evaluate import prepare_data, save_results
from train.features import FEATURE_COLS, build_features
from train.extended_features import _spread_to_prob
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from evaluate.metrics import brier_score


TOP_FEATURES = [
    "elo_diff", "elo_pred", "home", "massey_avg_diff", "margin_mean_diff",
    "barthag_diff", "sos_diff", "conf_elo_diff", "trank_adjO_diff", "trank_adjD_diff",
    "off_eff_r10_diff", "def_eff_r10_diff",
]


def _build_configs():
    configs = []
    config_id = 0

    # All 3-feature combos
    for combo in combinations(TOP_FEATURES, 3):
        configs.append({"features": list(combo), "C": 1.0, "vegas_lo": 0.45, "vegas_hi": 0.55, "edge_thresh": 0.15, "id": config_id})
        config_id += 1

    # All 4-feature combos
    for combo in combinations(TOP_FEATURES, 4):
        configs.append({"features": list(combo), "C": 1.0, "vegas_lo": 0.45, "vegas_hi": 0.55, "edge_thresh": 0.15, "id": config_id})
        config_id += 1

    # All 5-feature combos
    for combo in combinations(TOP_FEATURES, 5):
        configs.append({"features": list(combo), "C": 1.0, "vegas_lo": 0.45, "vegas_hi": 0.55, "edge_thresh": 0.15, "id": config_id})
        config_id += 1

    # Best feature sets from prior sweep with varied thresholds
    BEST_SETS = [
        ["home", "massey_avg_diff", "sos_diff", "conf_elo_diff"],
        ["home", "massey_avg_diff", "barthag_diff", "conf_elo_diff"],
        ["elo_diff", "elo_pred", "home", "massey_avg_diff", "margin_mean_diff"],
        ["home", "massey_avg_diff", "margin_mean_diff", "conf_elo_diff"],
    ]
    for feats in BEST_SETS:
        for C in [0.1, 1.0, 10.0]:
            for vlo, vhi in [(0.43, 0.57), (0.45, 0.55), (0.47, 0.53)]:
                for thresh in [0.10, 0.15, 0.20, 0.25]:
                    configs.append({"features": feats, "C": C, "vegas_lo": vlo, "vegas_hi": vhi, "edge_thresh": thresh, "id": config_id})
                    config_id += 1

    return configs


def ml_to_payout(ml):
    if ml > 0: return ml / 100
    else: return 100 / abs(ml)


def evaluate_edge(cfg, X, y, spreads, has_odds, seasons, daynums, team_lows, team_highs, ml_lookup):
    feat_idx = [FEATURE_COLS.index(c) for c in cfg["features"]]
    all_bets = []

    for year in range(2016, 2026):
        train_mask = (seasons < year) & has_odds
        test_mask = (seasons == year) & has_odds
        if test_mask.sum() == 0:
            continue

        m = Pipeline([("scl", StandardScaler()), ("clf", LogisticRegression(C=cfg["C"], max_iter=1000))])
        m.fit(np.nan_to_num(X[train_mask][:, feat_idx], nan=0.0), y[train_mask])
        mp = m.predict_proba(np.nan_to_num(X[test_mask][:, feat_idx], nan=0.0))[:, 1]
        vp = np.array([_spread_to_prob(s) for s in spreads[test_mask]])
        yt = y[test_mask]
        dn = daynums[test_mask]
        tl = team_lows[test_mask]
        th = team_highs[test_mask]

        sweet = (vp >= cfg["vegas_lo"]) & (vp <= cfg["vegas_hi"]) & (np.abs(mp - vp) >= cfg["edge_thresh"])
        for i in np.where(sweet)[0]:
            key = (year, int(dn[i]), int(tl[i]), int(th[i]))
            ml = ml_lookup.get(key)
            if ml is None:
                continue
            bet_on_low = mp[i] > vp[i]
            won = (yt[i] == 1) == bet_on_low
            payout = ml_to_payout(ml["low_ml"] if bet_on_low else ml["high_ml"])
            all_bets.append({"won": won, "profit": payout if won else -1.0})

    if not all_bets:
        return {"n_bets": 0, "win_rate": 0, "roi": 0, "profit": 0}
    n = len(all_bets)
    wins = sum(b["won"] for b in all_bets)
    profit = sum(b["profit"] for b in all_bets)
    return {"n_bets": n, "win_rate": round(wins / n, 4), "roi": round(profit / n, 4), "profit": round(profit, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/edge_deep")
    args = parser.parse_args()

    configs = _build_configs()
    print(f"Total configs: {len(configs)}")

    data = prepare_data(args.data_dir)
    results_df = add_game_counts(data["mens_results"].copy())

    print("Building features...")
    features_df, _, _, _, _ = build_features(
        results=results_df, conferences=data["mens_conf"],
        hfa_dict=_build_hfa_dict(data["hfa"], "M"),
        location_dict=_build_location_dict(data["home_lookup"], "M"),
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
    daynums = features_df["day_num"].values
    team_lows = features_df["team_low"].values
    team_highs = features_df["team_high"].values

    print("Loading ML odds...")
    odds = pd.read_csv(f"{args.data_dir}/derived/ncaab_odds.csv")
    odds = odds[(odds["HomeML"] != 0) & (odds["AwayML"] != 0) & odds["HomeML"].notna() & odds["AwayML"].notna()]
    ml_lookup = {}
    for _, row in odds.iterrows():
        home, away = int(row["HomeTeamID"]), int(row["AwayTeamID"])
        low, high = min(home, away), max(home, away)
        key = (int(row["Season"]), int(row["DayNum"]), low, high)
        if home == low:
            ml_lookup[key] = {"low_ml": row["HomeML"], "high_ml": row["AwayML"]}
        else:
            ml_lookup[key] = {"low_ml": row["AwayML"], "high_ml": row["HomeML"]}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_index is not None:
        # Run a batch of configs (each is fast, so do 50 per task)
        batch_size = 50
        start = args.config_index * batch_size
        end = min(start + batch_size, len(configs))
        if start >= len(configs):
            print(f"Index {args.config_index} out of range")
            return

        results = []
        for i in range(start, end):
            cfg = configs[i]
            r = evaluate_edge(cfg, X, y, spreads, has_odds, seasons, daynums, team_lows, team_highs, ml_lookup)
            r["config"] = cfg
            results.append(r)
            if r["n_bets"] > 0:
                print(f"  [{i}] win={r['win_rate']:.1%} bets={r['n_bets']:3d} ROI={r['roi']:+.1%}  {cfg['features']}")

        path = output_dir / f"edge_deep_{args.config_index}.json"
        save_results({"batch": results}, path, model_name=f"edge_deep_{args.config_index}")
    else:
        best_wr = 0
        for i, cfg in enumerate(configs):
            r = evaluate_edge(cfg, X, y, spreads, has_odds, seasons, daynums, team_lows, team_highs, ml_lookup)
            if r["n_bets"] > 20 and r["win_rate"] > best_wr:
                best_wr = r["win_rate"]
                print(f"  [{i}] win={r['win_rate']:.1%} bets={r['n_bets']:3d} ROI={r['roi']:+.1%}  {cfg['features']} ***")


if __name__ == "__main__":
    main()
