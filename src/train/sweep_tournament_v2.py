"""Tournament model sweep v2: includes Colley, SRS, and competition-inspired features.

Tests all combinations of:
  - Model: LR (various C), XGB, LGB
  - Base features: base11, base8 (drop low-importance), base5 (minimal)
  - Add-ons: seeds, colley, srs, interactions
  - M/W regularization: combined vs separate
  - Prediction clipping

Usage:
  PYTHONPATH=src python -m train.sweep_tournament_v2 --config-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from train.evaluate import prepare_data, save_results
from train.features import (
    build_features, matchup_features, FEATURE_COLS,
    _build_barttorvik_lookup, get_all_snapshots,
)
from train.custom_ratings import compute_colley, compute_srs
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from evaluate.metrics import brier_score
from train.cache_features import load_all_elo_snapshots, load_cached_features


EVAL_YEARS = [y for y in range(2015, 2026) if y != 2020]

# Feature sets
FEAT_BASE11 = [
    "elo_diff", "elo_pred", "home", "day_num", "sos_diff", "margin_mean_diff",
    "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]

# Drop low-importance from ablation: elo_pred, day_num, elo_diff contribute nothing
FEAT_BASE8 = [
    "home", "sos_diff", "margin_mean_diff",
    "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]

FEAT_BASE5 = [
    "home", "margin_mean_diff", "massey_avg_diff", "barthag_diff", "conf_elo_diff",
]

FEATURE_SETS = {
    "b11": FEAT_BASE11,
    "b8": FEAT_BASE8,
    "b5": FEAT_BASE5,
}


def _build_configs():
    configs = []

    for feat_name in ["b11", "b8", "b5"]:
        for add_seeds in [True]:  # seeds always help
            for add_colley in [True, False]:
                for add_srs in [True, False]:
                    for interactions in [False]:  # proven to hurt
                        # LR configs with separate M/W
                        for C_m, C_w in [(1.0, 1.0), (100.0, 0.15), (50.0, 0.3), (10.0, 0.5)]:
                            configs.append({
                                "model": "lr",
                                "features": feat_name,
                                "seeds": add_seeds,
                                "colley": add_colley,
                                "srs": add_srs,
                                "interactions": interactions,
                                "C_m": C_m,
                                "C_w": C_w,
                            })

                        # XGB configs
                        for depth in [2, 3]:
                            configs.append({
                                "model": "xgb",
                                "features": feat_name,
                                "seeds": add_seeds,
                                "colley": add_colley,
                                "srs": add_srs,
                                "interactions": interactions,
                                "max_depth": depth,
                                "C_m": None,
                                "C_w": None,
                            })

                        # LGB
                        configs.append({
                            "model": "lgb",
                            "features": feat_name,
                            "seeds": add_seeds,
                            "colley": add_colley,
                            "srs": add_srs,
                            "interactions": interactions,
                            "max_depth": 3,
                            "C_m": None,
                            "C_w": None,
                        })

    return configs


def _precompute_ratings(data):
    """Precompute Colley + SRS for all seasons, both genders."""
    colley_all, srs_all = {}, {}
    for gk, rk in [("M", "mens_results"), ("W", "womens_results")]:
        for season in range(2003, 2027):
            if season in data[rk]["Season"].values:
                for t, r in compute_colley(data[rk], season).items():
                    colley_all[(gk, season, t)] = r
                for t, r in compute_srs(data[rk], season).items():
                    srs_all[(gk, season, t)] = r
    return colley_all, srs_all


def evaluate_config(cfg, data, colley_all, srs_all, data_dir="data"):
    """Evaluate one config with LOSO tournament CV."""
    feat_list = FEATURE_SETS[cfg["features"]]
    feat_idx = [FEATURE_COLS.index(c) for c in feat_list]

    all_preds, all_outcomes = [], []

    for year in EVAL_YEARS:
        for gender in ["M", "W"]:
            is_mens = gender == "M"

            # Determine C for this gender
            if cfg["model"] == "lr":
                C = cfg["C_m"] if is_mens else cfg["C_w"]
            else:
                C = None

            rk = "mens_results" if is_mens else "womens_results"
            ck = "mens_conf" if is_mens else "womens_conf"
            results = add_game_counts(data[rk].copy())
            results = results[results["Season"] <= year]
            hfa = _build_hfa_dict(data["hfa"], gender)
            loc = _build_location_dict(data["home_lookup"], gender)

            _, elo, gc, ts, cem = build_features(
                results=results,
                conferences=data[ck][data[ck]["Season"] <= year],
                hfa_dict=hfa, location_dict=loc, seasons=set(),
                detailed_results=data.get("mens_detailed" if is_mens else "womens_detailed"),
                massey_per_system=data.get("massey_per_system", {}) if is_mens else {},
                massey_avg=data.get("massey_avg", {}) if is_mens else {},
                coach_tenure=data.get("coach_tenure", {}) if is_mens else {},
                coach_changed=data.get("coach_changed", {}) if is_mens else {},
                barttorvik=data.get("barttorvik") if is_mens else None,
            )
            bt_lk = _build_barttorvik_lookup(data.get("barttorvik") if is_mens else None)
            seeds = data.get("m_seeds", {}) if is_mens else data.get("w_seeds", {})

            prefix = "M" if is_mens else "W"
            tourney = pd.read_csv(f"{data_dir}/kaggle/{prefix}NCAATourneyCompactResults.csv")
            train_rows, test_rows, train_y, test_y = [], [], [], []

            for ty in [y for y in range(2003, year + 1) if y != 2020]:
                for _, g in tourney[tourney["Season"] == ty].iterrows():
                    low = min(g["WTeamID"], g["LTeamID"])
                    high = max(g["WTeamID"], g["LTeamID"])

                    feat = matchup_features(
                        elo.get(low, 1500), elo.get(high, 1500), gc, low, high,
                        team_states=ts, season=ty, seeds=seeds,
                        massey_per_system=data.get("massey_per_system", {}) if is_mens else {},
                        massey_avg=data.get("massey_avg", {}) if is_mens else {},
                        coach_tenure=data.get("coach_tenure", {}) if is_mens else {},
                        coach_changed=data.get("coach_changed", {}) if is_mens else {},
                        conf_elo_means=cem, conferences=data[ck],
                        barttorvik_lookup=bt_lk,
                    )

                    rb = [feat.get(c, 0) for c in FEATURE_COLS]
                    row = [rb[i] for i in feat_idx]

                    if cfg["seeds"]:
                        sd = seeds.get((ty, low), 17) - seeds.get((ty, high), 17)
                        row.append(sd)
                    if cfg["colley"]:
                        cd = colley_all.get((gender, ty, low), 175) - colley_all.get((gender, ty, high), 175)
                        row.append(cd)
                    if cfg["srs"]:
                        sr = srs_all.get((gender, ty, low), 0) - srs_all.get((gender, ty, high), 0)
                        row.append(sr)

                    outcome = 1 if g["WTeamID"] == low else 0
                    if ty == year:
                        test_rows.append(row)
                        test_y.append(outcome)
                    else:
                        train_rows.append(row)
                        train_y.append(outcome)
                        train_rows.append([-v for v in row])
                        train_y.append(1 - outcome)

            if not test_rows or not train_rows:
                continue

            X_tr = np.nan_to_num(np.array(train_rows), nan=0.0)
            X_te = np.nan_to_num(np.array(test_rows), nan=0.0)

            if cfg["model"] == "lr":
                model = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scl", StandardScaler()),
                    ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=1000)),
                ])
            elif cfg["model"] == "xgb":
                model = XGBClassifier(
                    n_estimators=500, max_depth=cfg["max_depth"],
                    learning_rate=0.03, reg_lambda=5.0, subsample=0.8,
                    colsample_bytree=0.7, random_state=42,
                    eval_metric="logloss", verbosity=0,
                )
            elif cfg["model"] == "lgb":
                model = LGBMClassifier(
                    n_estimators=500, max_depth=cfg["max_depth"],
                    learning_rate=0.03, reg_lambda=5.0, subsample=0.8,
                    colsample_bytree=0.7, random_state=42, verbose=-1,
                )

            model.fit(X_tr, np.array(train_y))
            preds = model.predict_proba(X_te)[:, 1]
            all_preds.extend(preds)
            all_outcomes.extend(test_y)

    if not all_preds:
        return {"brier": 1.0}

    brier = float(np.mean((np.array(all_outcomes) - np.array(all_preds)) ** 2))
    return {"brier": round(brier, 6), "n_games": len(all_preds), "config": cfg}


def main():
    parser = argparse.ArgumentParser(description="Tournament model sweep v2")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/tournament_v2")
    args = parser.parse_args()

    configs = _build_configs()
    print(f"Total configs: {len(configs)}")

    data = prepare_data(args.data_dir)
    print("Precomputing Colley + SRS...")
    colley_all, srs_all = _precompute_ratings(data)
    print(f"  Colley: {len(colley_all)}, SRS: {len(srs_all)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_index is not None:
        if args.config_index >= len(configs):
            print(f"Index {args.config_index} out of range (max {len(configs)-1})")
            return
        cfg = configs[args.config_index]
        print(f"Config {args.config_index}: {cfg}")
        t0 = time.time()
        result = evaluate_config(cfg, data, colley_all, srs_all, args.data_dir)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        c_str = f"C_m={cfg.get('C_m')} C_w={cfg.get('C_w')}" if cfg["model"] == "lr" else f"d={cfg.get('max_depth')}"
        print(f"  Brier: {result['brier']:.6f} ({elapsed:.0f}s) "
              f"{cfg['model']} {cfg['features']} colley={cfg['colley']} srs={cfg['srs']} {c_str}")
        path = output_dir / f"tv2_{args.config_index}.json"
        save_results(result, path, model_name=f"tv2_{args.config_index}")
    else:
        best = 1.0
        for i, cfg in enumerate(configs):
            t0 = time.time()
            result = evaluate_config(cfg, data, colley_all, srs_all, args.data_dir)
            elapsed = time.time() - t0
            marker = " ***" if result["brier"] < best else ""
            if result["brier"] < best:
                best = result["brier"]
            c_str = f"C={cfg.get('C_m')}/{cfg.get('C_w')}" if cfg["model"] == "lr" else f"d={cfg.get('max_depth')}"
            print(f"[{i+1}/{len(configs)}] {result['brier']:.6f} "
                  f"{cfg['model']:3s} {cfg['features']:3s} col={cfg['colley']} srs={cfg['srs']} {c_str}{marker}")
            result["elapsed_s"] = round(elapsed, 1)
            path = output_dir / f"tv2_{i}.json"
            save_results(result, path, model_name=f"tv2_{i}")


if __name__ == "__main__":
    main()
