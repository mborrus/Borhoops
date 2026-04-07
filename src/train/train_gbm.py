"""Phase 2a: Gradient boosting hyperparameter sweep (XGBoost, LightGBM, CatBoost).

Runs Optuna-driven hyperparameter optimization with LOYO CV as the objective.
Each trial trains on 10 LOYO folds and reports mean Brier score.

Usage (local):
  PYTHONPATH=src python src/train/train_gbm.py --library xgboost --n-trials 50
  PYTHONPATH=src python src/train/train_gbm.py --library lightgbm --n-trials 50
  PYTHONPATH=src python src/train/train_gbm.py --library catboost --n-trials 50

Usage (HPC — grid search mode for array jobs):
  PYTHONPATH=src python src/train/train_gbm.py --library xgboost --grid-index $SLURM_ARRAY_TASK_ID

Results saved to: results/gbm/{library}_{timestamp}.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from train.evaluate import prepare_data, prepare_fold, brier_score, TOURNEY_YEARS, save_results


def _make_xgboost(params):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )


def _make_lightgbm(params):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        random_state=42,
        verbose=-1,
    )


def _make_catboost(params):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=params["n_estimators"],
        depth=min(params["max_depth"], 10),  # CatBoost max depth is 16
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["reg_lambda"],
        subsample=params["subsample"],
        random_seed=42,
        verbose=0,
    )


LIBRARY_FACTORY = {
    "xgboost": _make_xgboost,
    "lightgbm": _make_lightgbm,
    "catboost": _make_catboost,
}


def _suggest_params(trial):
    """Optuna parameter suggestions shared across all GBM libraries."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }


def evaluate_params(params, library, data, gender="M", data_dir="data", years=None):
    """Evaluate one hyperparameter config across LOYO folds. Returns mean Brier."""
    if years is None:
        years = TOURNEY_YEARS

    factory = LIBRARY_FACTORY[library]
    brier_scores = []

    for year in years:
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)
        model = factory(params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        brier_scores.append(brier_score(y_test, y_pred))

    return float(np.mean(brier_scores)), brier_scores


def run_optuna(library, data, n_trials=100, gender="M", data_dir="data"):
    """Run Optuna hyperparameter optimization."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_result = {"brier": 1.0, "params": {}, "per_year": []}

    def objective(trial):
        params = _suggest_params(trial)
        mean_brier, per_year = evaluate_params(params, library, data, gender, data_dir)

        if mean_brier < best_result["brier"]:
            best_result["brier"] = mean_brier
            best_result["params"] = params
            best_result["per_year"] = per_year
            print(f"  Trial {trial.number}: Brier={mean_brier:.4f} *** NEW BEST ***")
        elif trial.number % 10 == 0:
            print(f"  Trial {trial.number}: Brier={mean_brier:.4f}")

        return mean_brier

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_params": study.best_params,
        "best_brier": study.best_value,
        "n_trials": n_trials,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }


# -- Grid search for HPC array jobs -------------------------------------------

GRID = []

def _build_grid():
    """Pre-defined grid of ~200 configs for HPC array jobs."""
    import itertools
    space = {
        "n_estimators": [200, 500, 1000, 2000],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "reg_lambda": [0.1, 1.0, 5.0],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [1, 5],
    }
    keys = list(space.keys())
    combos = list(itertools.product(*space.values()))
    return [dict(zip(keys, combo)) for combo in combos]


def run_grid_index(index, library, data, gender="M", data_dir="data"):
    """Run one grid config by index (for SLURM_ARRAY_TASK_ID)."""
    grid = _build_grid()
    if index >= len(grid):
        print(f"Grid index {index} out of range (max {len(grid)-1})")
        return None

    params = grid[index]
    print(f"Grid index {index}/{len(grid)-1}: {params}")
    mean_brier, per_year = evaluate_params(params, library, data, gender, data_dir)
    print(f"  Mean Brier: {mean_brier:.4f}")

    return {
        "grid_index": index,
        "params": params,
        "mean_brier": mean_brier,
        "per_year_brier": dict(zip(TOURNEY_YEARS, per_year)),
    }


def main():
    parser = argparse.ArgumentParser(description="GBM hyperparameter sweep")
    parser.add_argument("--library", choices=["xgboost", "lightgbm", "catboost"],
                        default="xgboost")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optuna trials (ignored in grid mode)")
    parser.add_argument("--grid-index", type=int, default=None,
                        help="Grid search index (for HPC array jobs)")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/gbm")
    args = parser.parse_args()

    print(f"Loading data...")
    data = prepare_data(args.data_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.grid_index is not None:
        result = run_grid_index(args.grid_index, args.library, data,
                                args.gender, args.data_dir)
        if result:
            path = output_dir / f"{args.library}_grid_{args.grid_index}.json"
            save_results(result, path, model_name=f"{args.library}_grid_{args.grid_index}")
            print(f"Saved → {path}")
    else:
        print(f"Running Optuna ({args.n_trials} trials, {args.library})...")
        result = run_optuna(args.library, data, args.n_trials, args.gender, args.data_dir)
        path = output_dir / f"{args.library}_optuna_{timestamp}.json"
        save_results(result, path, model_name=f"{args.library}_optuna")
        print(f"\nBest Brier: {result['best_brier']:.4f}")
        print(f"Best params: {json.dumps(result['best_params'], indent=2)}")
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
