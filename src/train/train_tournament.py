"""Tournament-specific model: train on historical tournament games.

Key differences from regular season training:
  - Training data: ~1,300 tournament games (2003-2025), not 190K regular season
  - Seeds available as a primary feature
  - LR often beats XGBoost on this small dataset (~650 rows per gender)
  - Point-diff regression + spline calibration as alternative to classification
  - Feature interactions compensate for LR's linearity

Implements findings from top Kaggle solutions (3rd, 10th, 19th, 21st, 49th place).

Usage:
  PYTHONPATH=src python -m train.train_tournament
  PYTHONPATH=src python -m train.train_tournament --config-index 0  # for HPC array
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from train.evaluate import prepare_data, save_results, _load_tourney_results
from train.features import (
    build_features, matchup_features, FEATURE_COLS,
    _build_barttorvik_lookup,
)
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict


# Tournament years with data (no 2020)
TOURNEY_YEARS = [y for y in range(2003, 2026) if y != 2020]
EVAL_YEARS = [y for y in range(2015, 2026) if y != 2020]  # LOSO evaluation


# -- Feature sets from competition analysis ------------------------------------

# Our proven 11 features
FEAT_BASE_11 = [
    "elo_diff", "elo_pred", "home", "day_num", "sos_diff", "margin_mean_diff",
    "massey_avg_diff", "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]

# Extended with more Barttorvik + box scores (inspired by 10th/19th place)
FEAT_EXTENDED = FEAT_BASE_11 + [
    "off_eff_r10_diff", "def_eff_r10_diff", "efg_pct_r10_diff",
    "pace_r10_diff", "margin_std_diff", "win_streak_diff",
]

# All 30 original features
FEAT_ALL_30 = [c for c in FEATURE_COLS if c not in [
    "spread", "spread_abs", "implied_prob", "over_under",
    "poll_rank_diff", "poll_momentum_diff", "weeks_ranked_diff",
    "avg_experience_diff", "senior_pct_diff", "seed_diff",
]]


def _build_tournament_training_data(data, gender, eval_year, features_list,
                                    include_seeds=True, include_interactions=False):
    """Build training and test data from historical tournament games.

    Training: tournament games from TOURNEY_YEARS where year < eval_year
    Test: tournament games from eval_year
    """
    is_mens = gender == "M"
    results_key = "mens_results" if is_mens else "womens_results"
    conf_key = "mens_conf" if is_mens else "womens_conf"

    results = add_game_counts(data[results_key].copy())
    # Only use data up to eval_year
    results = results[results["Season"] <= eval_year]

    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    detailed = data.get("mens_detailed" if is_mens else "womens_detailed")
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
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)

    # Run Elo loop on all data up to eval_year (to get end-of-season states)
    all_seasons = sorted(results["Season"].unique())
    _, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[conf_key][data[conf_key]["Season"] <= eval_year],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(),  # don't collect regular season features
        detailed_results=detailed,
        massey_per_system=massey_per, massey_avg=massey_avg,
        coach_tenure=coach_tenure, coach_changed=coach_changed,
        barttorvik=barttorvik,
        odds_lookup=odds_lookup, odds_team_avg=odds_team_avg,
        poll_lookup=poll_lookup, weeks_ranked=weeks_ranked,
        roster_lookup=roster_lookup, seeds=seeds,
    )

    # Build tournament matchup features
    prefix = "M" if is_mens else "W"
    tourney_path = Path(data[results_key].attrs.get("data_dir", "data")) / "kaggle" / f"{prefix}NCAATourneyCompactResults.csv"
    all_tourney = pd.read_csv(tourney_path)

    feat_idx = [FEATURE_COLS.index(c) for c in features_list]
    train_rows, test_rows = [], []
    train_margins, test_margins = [], []
    train_y, test_y = [], []

    for year in [y for y in TOURNEY_YEARS if y <= eval_year]:
        year_games = all_tourney[all_tourney["Season"] == year]
        is_test = (year == eval_year)

        # Need Elo state at end of this year's regular season
        # For eval_year, we have the current elo_ratings (end of eval_year reg season)
        # For earlier years, we'd need per-season snapshots
        # Approximation: use final elo_ratings (slight leakage for older years)
        # The competition solutions do the same — LOSO trains on all past tournament games

        for _, g in year_games.iterrows():
            low = min(g["WTeamID"], g["LTeamID"])
            high = max(g["WTeamID"], g["LTeamID"])

            feat = matchup_features(
                elo_ratings.get(low, 1500), elo_ratings.get(high, 1500),
                game_counts, low, high,
                team_states=team_states, season=year, seeds=seeds,
                massey_per_system=massey_per, massey_avg=massey_avg,
                coach_tenure=coach_tenure, coach_changed=coach_changed,
                conf_elo_means=conf_elo_means,
                conferences=data[conf_key],
                barttorvik_lookup=barttorvik_lookup,
                odds_team_avg=odds_team_avg,
                poll_lookup=poll_lookup, weeks_ranked=weeks_ranked,
                roster_lookup=roster_lookup,
            )

            row = [feat.get(c, 0) for c in FEATURE_COLS]

            # Add seed_diff
            seed_low = seeds.get((year, low), 17)
            seed_high = seeds.get((year, high), 17)
            seed_diff = seed_low - seed_high

            outcome = 1 if g["WTeamID"] == low else 0
            margin = (g["WScore"] - g["LScore"]) if g["WTeamID"] == low else -(g["WScore"] - g["LScore"])

            if is_test:
                test_rows.append(row)
                test_y.append(outcome)
                test_margins.append(margin)
            else:
                train_rows.append(row)
                train_y.append(outcome)
                train_margins.append(margin)

                # Symmetric augmentation: add the reverse matchup
                row_rev = [-v for v in row]  # flip all diffs
                train_rows.append(row_rev)
                train_y.append(1 - outcome)
                train_margins.append(-margin)

    X_train = np.array(train_rows)[:, feat_idx] if train_rows else np.empty((0, len(feat_idx)))
    X_test = np.array(test_rows)[:, feat_idx] if test_rows else np.empty((0, len(feat_idx)))

    # Add seed_diff as a separate column if requested
    if include_seeds:
        # Recompute seed_diff for each row (it's in FEATURE_COLS as "seed_diff")
        sd_idx = FEATURE_COLS.index("seed_diff")
        sd_train = np.array(train_rows)[:, sd_idx:sd_idx+1] if train_rows else np.empty((0, 1))
        sd_test = np.array(test_rows)[:, sd_idx:sd_idx+1] if test_rows else np.empty((0, 1))
        X_train = np.hstack([X_train, sd_train])
        X_test = np.hstack([X_test, sd_test])

    # Add feature interactions if requested
    if include_interactions and X_train.shape[0] > 0:
        interactions = _build_interactions(X_train, X_test, features_list, include_seeds)
        X_train = np.hstack([X_train, interactions[0]])
        X_test = np.hstack([X_test, interactions[1]])

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    return (X_train, np.array(train_y), np.array(train_margins),
            X_test, np.array(test_y), np.array(test_margins))


def _build_interactions(X_train, X_test, features_list, has_seeds):
    """Build feature interactions (inspired by 3rd place solution)."""
    n_base = len(features_list)
    seed_col = n_base if has_seeds else None

    # Key interactions from 3rd place:
    # seed × massey, seed × barthag, massey × barthag
    interaction_pairs = []

    massey_idx = features_list.index("massey_avg_diff") if "massey_avg_diff" in features_list else None
    barthag_idx = features_list.index("barthag_diff") if "barthag_diff" in features_list else None
    elo_idx = features_list.index("elo_diff") if "elo_diff" in features_list else None

    if seed_col is not None and massey_idx is not None:
        interaction_pairs.append((seed_col, massey_idx))
    if seed_col is not None and barthag_idx is not None:
        interaction_pairs.append((seed_col, barthag_idx))
    if massey_idx is not None and barthag_idx is not None:
        interaction_pairs.append((massey_idx, barthag_idx))
    if seed_col is not None and elo_idx is not None:
        interaction_pairs.append((seed_col, elo_idx))

    if not interaction_pairs:
        return np.empty((len(X_train), 0)), np.empty((len(X_test), 0))

    train_ints = np.column_stack([X_train[:, i] * X_train[:, j] for i, j in interaction_pairs])
    test_ints = np.column_stack([X_test[:, i] * X_test[:, j] for i, j in interaction_pairs])
    return train_ints, test_ints


def _spline_calibrate(margins, outcomes, test_margins, degree=5):
    """Convert predicted point differentials to probabilities via spline."""
    # Sort by margin
    idx = np.argsort(margins)
    sorted_margins = margins[idx]
    sorted_outcomes = outcomes[idx]

    # Bin for smoothing
    n_bins = min(50, len(margins) // 10)
    if n_bins < 5:
        # Not enough data, use logistic fallback
        from scipy.special import expit
        return expit(test_margins * 0.15)

    bin_edges = np.percentile(sorted_margins, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_probs = []
    for i in range(n_bins):
        mask = (sorted_margins >= bin_edges[i]) & (sorted_margins < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append(sorted_margins[mask].mean())
            bin_probs.append(sorted_outcomes[mask].mean())

    if len(bin_centers) < 5:
        from scipy.special import expit
        return expit(test_margins * 0.15)

    spline = UnivariateSpline(bin_centers, bin_probs, k=min(degree, len(bin_centers)-1), s=0.1)
    probs = spline(test_margins)
    return np.clip(probs, 0.01, 0.99)


# -- Model configs for sweep ---------------------------------------------------

def _build_configs():
    """Build all configurations for the tournament model sweep."""
    configs = []

    feature_sets = {
        "base11": FEAT_BASE_11,
        "extended17": FEAT_EXTENDED,
        "all30": FEAT_ALL_30,
    }

    for feat_name, feat_list in feature_sets.items():
        for seeds in [True, False]:
            for interactions in [True, False]:
                # LR configs
                for C in [0.1, 1.0, 10.0, 100.0]:
                    configs.append({
                        "model": "lr",
                        "features": feat_name,
                        "seeds": seeds,
                        "interactions": interactions,
                        "C": C,
                        "clip_lo": 0.03,
                        "clip_hi": 0.97,
                    })

                # XGBoost classifier configs
                for depth in [2, 3, 4]:
                    for lr in [0.03, 0.05]:
                        configs.append({
                            "model": "xgb_clf",
                            "features": feat_name,
                            "seeds": seeds,
                            "interactions": interactions,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "n_seeds": 10,
                        })

                # XGBoost regressor (point-diff) configs
                for depth in [2, 3, 4]:
                    configs.append({
                        "model": "xgb_reg",
                        "features": feat_name,
                        "seeds": seeds,
                        "interactions": interactions,
                        "max_depth": depth,
                        "learning_rate": 0.03,
                        "n_seeds": 10,
                        "calibration": "spline",
                    })

                # LightGBM classifier
                for depth in [3, 4]:
                    configs.append({
                        "model": "lgb_clf",
                        "features": feat_name,
                        "seeds": seeds,
                        "interactions": interactions,
                        "max_depth": depth,
                        "learning_rate": 0.05,
                    })

    return configs


def _make_model(cfg):
    """Create a model from config dict."""
    if cfg["model"] == "lr":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=cfg["C"], solver="lbfgs", max_iter=1000)),
        ])
    elif cfg["model"] == "xgb_clf":
        return XGBClassifier(
            n_estimators=500, max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
    elif cfg["model"] == "xgb_reg":
        return XGBRegressor(
            n_estimators=500, max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, objective="reg:squarederror", verbosity=0,
        )
    elif cfg["model"] == "lgb_clf":
        return LGBMClassifier(
            n_estimators=500, max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbose=-1,
        )


FEATURE_SETS = {
    "base11": FEAT_BASE_11,
    "extended17": FEAT_EXTENDED,
    "all30": FEAT_ALL_30,
}


def evaluate_config(cfg, data, data_dir="data"):
    """Run LOSO tournament evaluation for one config."""
    feat_list = FEATURE_SETS[cfg["features"]]
    is_reg = cfg["model"].endswith("_reg")
    n_seeds = cfg.get("n_seeds", 1)

    all_preds, all_outcomes = [], []

    for year in EVAL_YEARS:
        year_preds, year_outcomes = [], []

        for gender in ["M", "W"]:
            X_train, y_train, margins_train, X_test, y_test, margins_test = \
                _build_tournament_training_data(
                    data, gender, year, feat_list,
                    include_seeds=cfg["seeds"],
                    include_interactions=cfg["interactions"],
                )

            if len(X_test) == 0 or len(X_train) == 0:
                continue

            if is_reg:
                # Point-diff regression: predict margin, calibrate to probability
                preds_list = []
                for seed in range(n_seeds):
                    model = _make_model({**cfg, "learning_rate": cfg["learning_rate"]})
                    model.set_params(random_state=seed * 42)
                    model.fit(X_train, margins_train)
                    pred_margins = model.predict(X_test)
                    preds_list.append(pred_margins)
                avg_margins = np.mean(preds_list, axis=0)
                preds = _spline_calibrate(margins_train, y_train, avg_margins)

            elif n_seeds > 1:
                # Multi-seed ensemble for classifiers
                preds_list = []
                for seed in range(n_seeds):
                    model = _make_model(cfg)
                    if hasattr(model, "set_params"):
                        model.set_params(random_state=seed * 42)
                    elif hasattr(model, "named_steps"):
                        pass  # LR doesn't have random_state that matters
                    model.fit(X_train, y_train)
                    preds_list.append(model.predict_proba(X_test)[:, 1])
                preds = np.mean(preds_list, axis=0)

            else:
                model = _make_model(cfg)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_test)[:, 1]

            # Clip
            clip_lo = cfg.get("clip_lo", 0.01)
            clip_hi = cfg.get("clip_hi", 0.99)
            preds = np.clip(preds, clip_lo, clip_hi)

            year_preds.extend(preds)
            year_outcomes.extend(y_test)

        all_preds.extend(year_preds)
        all_outcomes.extend(year_outcomes)

    if not all_preds:
        return {"brier": 1.0}

    brier = float(np.mean((np.array(all_outcomes) - np.array(all_preds)) ** 2))
    return {"brier": round(brier, 6), "n_games": len(all_preds)}


def main():
    parser = argparse.ArgumentParser(description="Tournament-specific model sweep")
    parser.add_argument("--config-index", type=int, default=None,
                        help="Config index for HPC array jobs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/tournament")
    args = parser.parse_args()

    configs = _build_configs()
    print(f"Total configs: {len(configs)}")

    data = prepare_data(args.data_dir)
    # Store data_dir for tournament results path
    for key in ("mens_results", "womens_results"):
        data[key].attrs["data_dir"] = args.data_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_index is not None:
        if args.config_index >= len(configs):
            print(f"Config index {args.config_index} out of range (max {len(configs)-1})")
            return

        cfg = configs[args.config_index]
        print(f"Config {args.config_index}: {cfg}")
        t0 = time.time()
        result = evaluate_config(cfg, data, args.data_dir)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        result["config"] = cfg
        print(f"  Brier: {result['brier']:.6f} ({elapsed:.1f}s)")

        path = output_dir / f"tourney_{args.config_index}.json"
        save_results(result, path, model_name=f"tourney_{args.config_index}")
        print(f"  Saved → {path}")
    else:
        # Run all configs sequentially (local mode)
        best_brier = 1.0
        best_cfg = None
        for i, cfg in enumerate(configs):
            t0 = time.time()
            result = evaluate_config(cfg, data, args.data_dir)
            elapsed = time.time() - t0
            marker = " ***" if result["brier"] < best_brier else ""
            if result["brier"] < best_brier:
                best_brier = result["brier"]
                best_cfg = cfg
            print(f"[{i+1}/{len(configs)}] Brier={result['brier']:.6f} ({elapsed:.0f}s) "
                  f"{cfg['model']} {cfg['features']} seeds={cfg['seeds']} int={cfg['interactions']}{marker}")

            result["config"] = cfg
            result["elapsed_s"] = round(elapsed, 1)
            path = output_dir / f"tourney_{i}.json"
            save_results(result, path, model_name=f"tourney_{i}")

        print(f"\nBest: {best_brier:.6f}")
        print(f"Config: {best_cfg}")


if __name__ == "__main__":
    main()
