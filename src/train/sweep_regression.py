"""Point-diff regression sweep: predict margin of victory, calibrate to probability.

Inspired by 10th and 19th place Kaggle solutions.
Tests XGB/LGB regression with spline calibration across feature sets.

Usage:
  PYTHONPATH=src python -m train.sweep_regression --config-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.special import expit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from train.evaluate import prepare_data, save_results
from train.features import FEATURE_COLS, build_features, matchup_features, _build_barttorvik_lookup
from train.custom_ratings import compute_colley, compute_srs
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from evaluate.metrics import brier_score


EVAL_YEARS = [y for y in range(2015, 2026) if y != 2020]

FEAT_B11 = ["elo_diff", "elo_pred", "home", "day_num", "sos_diff", "margin_mean_diff",
            "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff", "conf_elo_diff"]
FEAT_B8 = ["home", "sos_diff", "margin_mean_diff", "massey_avg_diff",
           "barthag_diff", "trank_adjO_diff", "trank_adjD_diff", "conf_elo_diff"]
FEAT_B5 = ["home", "margin_mean_diff", "massey_avg_diff", "barthag_diff", "conf_elo_diff"]

FEATURE_SETS = {"b11": FEAT_B11, "b8": FEAT_B8, "b5": FEAT_B5}


def _spline_calibrate(train_margins, train_outcomes, test_margins, degree=5):
    """Fit spline from margin → P(win), apply to test margins."""
    idx = np.argsort(train_margins)
    sm = train_margins[idx]
    so = train_outcomes[idx].astype(float)

    n_bins = min(50, max(10, len(sm) // 20))
    edges = np.percentile(sm, np.linspace(0, 100, n_bins + 1))
    centers, probs = [], []
    for i in range(n_bins):
        mask = (sm >= edges[i]) & (sm < edges[i + 1])
        if mask.sum() >= 3:
            centers.append(sm[mask].mean())
            probs.append(so[mask].mean())

    if len(centers) < 4:
        return expit(test_margins * 0.04)

    k = min(degree, len(centers) - 1)
    spline = UnivariateSpline(centers, probs, k=k, s=len(centers) * 0.02)
    result = spline(test_margins)
    return np.clip(result, 0.01, 0.99)


def _logistic_calibrate(train_margins, train_outcomes, test_margins):
    """Simple logistic calibration: P = 1/(1+exp(-a*margin - b))."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(train_margins.reshape(-1, 1), train_outcomes)
    return lr.predict_proba(test_margins.reshape(-1, 1))[:, 1]


def _build_configs():
    configs = []
    for feat_name in ["b11", "b8", "b5"]:
        for colley in [True, False]:
            for srs in [True, False]:
                # XGB regression
                for depth in [2, 3, 4]:
                    for cal in ["spline3", "spline5", "logistic"]:
                        configs.append({
                            "model": "xgb_reg", "features": feat_name,
                            "seeds": True, "colley": colley, "srs": srs,
                            "max_depth": depth, "calibration": cal,
                        })
                # LGB regression
                for cal in ["spline5", "logistic"]:
                    configs.append({
                        "model": "lgb_reg", "features": feat_name,
                        "seeds": True, "colley": colley, "srs": srs,
                        "max_depth": 3, "calibration": cal,
                    })
                # Ridge regression
                for cal in ["spline5", "logistic"]:
                    configs.append({
                        "model": "ridge", "features": feat_name,
                        "seeds": True, "colley": colley, "srs": srs,
                        "calibration": cal,
                    })
    return configs


def evaluate_config(cfg, data, colley_all, srs_all, data_dir="data"):
    feat_list = FEATURE_SETS[cfg["features"]]
    feat_idx = [FEATURE_COLS.index(c) for c in feat_list]

    all_preds, all_outcomes = [], []

    for year in EVAL_YEARS:
        for gender in ["M", "W"]:
            is_mens = gender == "M"
            rk = "mens_results" if is_mens else "womens_results"
            ck = "mens_conf" if is_mens else "womens_conf"
            results = add_game_counts(data[rk].copy())
            results = results[results["Season"] <= year]

            _, elo, gc, ts, cem = build_features(
                results=results, conferences=data[ck][data[ck]["Season"] <= year],
                hfa_dict=_build_hfa_dict(data["hfa"], gender),
                location_dict=_build_location_dict(data["home_lookup"], gender),
                seasons=set(),
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
            train_rows, test_rows = [], []
            train_margins, test_margins_true = [], []
            train_y, test_y = [], []

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
                        conf_elo_means=cem, conferences=data[ck], barttorvik_lookup=bt_lk,
                    )
                    rb = [feat.get(c, 0) for c in FEATURE_COLS]
                    row = [rb[i] for i in feat_idx]
                    if cfg["seeds"]:
                        row.append(seeds.get((ty, low), 17) - seeds.get((ty, high), 17))
                    if cfg["colley"]:
                        row.append(colley_all.get((gender, ty, low), 175) - colley_all.get((gender, ty, high), 175))
                    if cfg["srs"]:
                        row.append(srs_all.get((gender, ty, low), 0) - srs_all.get((gender, ty, high), 0))

                    outcome = 1 if g["WTeamID"] == low else 0
                    margin = (g["WScore"] - g["LScore"]) * (1 if g["WTeamID"] == low else -1)

                    if ty == year:
                        test_rows.append(row)
                        test_y.append(outcome)
                    else:
                        train_rows.append(row)
                        train_y.append(outcome)
                        train_margins.append(margin)
                        # Symmetric augmentation
                        train_rows.append([-v for v in row])
                        train_y.append(1 - outcome)
                        train_margins.append(-margin)

            if not test_rows or not train_rows:
                continue

            X_tr = np.nan_to_num(np.array(train_rows), nan=0.0)
            X_te = np.nan_to_num(np.array(test_rows), nan=0.0)
            y_margins = np.array(train_margins)
            y_outcomes = np.array(train_y)

            # Train regression model
            if cfg["model"] == "xgb_reg":
                model = XGBRegressor(
                    n_estimators=500, max_depth=cfg["max_depth"],
                    learning_rate=0.03, reg_lambda=5.0, subsample=0.8,
                    colsample_bytree=0.7, random_state=42, verbosity=0,
                )
            elif cfg["model"] == "lgb_reg":
                model = LGBMRegressor(
                    n_estimators=500, max_depth=cfg["max_depth"],
                    learning_rate=0.03, reg_lambda=5.0, subsample=0.8,
                    colsample_bytree=0.7, random_state=42, verbose=-1,
                )
            elif cfg["model"] == "ridge":
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                model = Pipeline([("scl", StandardScaler()), ("reg", Ridge(alpha=1.0))])

            model.fit(X_tr, y_margins)
            pred_margins = model.predict(X_te)

            # Also get training predictions for calibration
            train_pred_margins = model.predict(X_tr)

            # Calibrate
            cal = cfg["calibration"]
            if cal == "logistic":
                preds = _logistic_calibrate(train_pred_margins, y_outcomes, pred_margins)
            elif cal.startswith("spline"):
                degree = int(cal.replace("spline", ""))
                preds = _spline_calibrate(train_pred_margins, y_outcomes, pred_margins, degree)
            else:
                preds = expit(pred_margins * 0.04)

            all_preds.extend(preds)
            all_outcomes.extend(test_y)

    if not all_preds:
        return {"brier": 1.0}
    return {
        "brier": round(brier_score(all_outcomes, all_preds), 6),
        "n_games": len(all_preds),
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/regression")
    args = parser.parse_args()

    configs = _build_configs()
    print(f"Total configs: {len(configs)}")

    data = prepare_data(args.data_dir)
    print("Precomputing Colley + SRS...")
    colley_all, srs_all = {}, {}
    for gk, rk in [("M", "mens_results"), ("W", "womens_results")]:
        for season in range(2003, 2027):
            if season in data[rk]["Season"].values:
                for t, r in compute_colley(data[rk], season).items(): colley_all[(gk, season, t)] = r
                for t, r in compute_srs(data[rk], season).items(): srs_all[(gk, season, t)] = r

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_index is not None:
        if args.config_index >= len(configs):
            return
        cfg = configs[args.config_index]
        print(f"Config {args.config_index}: {cfg}")
        t0 = time.time()
        result = evaluate_config(cfg, data, colley_all, srs_all, args.data_dir)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        print(f"  Brier: {result['brier']:.6f} ({elapsed:.0f}s)")
        path = output_dir / f"reg_{args.config_index}.json"
        save_results(result, path, model_name=f"reg_{args.config_index}")
    else:
        best = 1.0
        for i, cfg in enumerate(configs):
            t0 = time.time()
            result = evaluate_config(cfg, data, colley_all, srs_all, args.data_dir)
            marker = " ***" if result["brier"] < best else ""
            if result["brier"] < best: best = result["brier"]
            print(f"[{i+1}/{len(configs)}] {result['brier']:.6f} {cfg['model']:7s} {cfg['features']:3s} "
                  f"col={cfg['colley']} srs={cfg['srs']} cal={cfg['calibration']}{marker}")
            path = output_dir / f"reg_{i}.json"
            save_results(result, path)


if __name__ == "__main__":
    main()
