"""Phase 3: Stacking ensemble with meta-learner.

Trains multiple base models, generates out-of-fold predictions,
then trains a meta-learner (logistic or isotonic regression) on the stacked predictions.

Usage:
  PYTHONPATH=src python -m train.train_ensemble
  PYTHONPATH=src python -m train.train_ensemble --meta isotonic
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from train.evaluate import (
    prepare_data, prepare_fold, brier_score, log_loss,
    TOURNEY_YEARS, save_results,
)
from train.features import FEATURE_COLS


# The 11 best features from autoresearch
BEST_FEATURES = [
    "elo_diff", "elo_pred", "home", "day_num",
    "sos_diff", "margin_mean_diff",
    "massey_avg_diff",
    "barthag_diff", "trank_adjO_diff", "trank_adjD_diff",
    "conf_elo_diff",
]
BEST_IDX = [FEATURE_COLS.index(c) for c in BEST_FEATURES]


def _get_base_models():
    """Base models for stacking. Each returns calibrated probabilities."""
    return {
        "xgb": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=2, learning_rate=0.03,
                reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
                random_state=42, eval_metric="logloss",
            )),
        ]),
        "xgb_30feat": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=2, learning_rate=0.03,
                reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
                random_state=42, eval_metric="logloss",
            )),
        ]),
        "lgb": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.03,
                reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7,
                min_child_weight=5, random_state=42, verbose=-1,
            )),
        ]),
        "cat": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CatBoostClassifier(
                iterations=500, depth=2, learning_rate=0.03,
                l2_leaf_reg=5.0, subsample=0.8,
                random_seed=42, verbose=0,
            )),
        ]),
    }


class StackingEnsemble:
    """Two-level stacking ensemble.

    Level 0: Base models generate out-of-fold predictions via 5-fold CV.
    Level 1: Meta-learner trained on stacked OOF predictions.
    """

    def __init__(self, meta_method="average", n_inner_folds=5):
        self.meta_method = meta_method
        self.n_inner_folds = n_inner_folds
        self.base_models = _get_base_models()
        self._fitted_bases = {}
        self._meta = None

    def fit(self, X, y):
        from sklearn.model_selection import KFold

        n_models = len(self.base_models)
        oof_preds = np.zeros((len(X), n_models))

        # Generate OOF predictions for each base model
        kf = KFold(n_splits=self.n_inner_folds, shuffle=True, random_state=42)

        for i, (name, factory) in enumerate(self.base_models.items()):
            # Select features: xgb_30feat uses all, others use best 11
            if name == "xgb_30feat":
                X_model = X
            else:
                X_model = X[:, BEST_IDX]

            for train_idx, val_idx in kf.split(X_model):
                model = factory()
                model.fit(X_model[train_idx], y[train_idx])
                oof_preds[val_idx, i] = model.predict_proba(X_model[val_idx])[:, 1]

        # Train meta-learner on OOF predictions
        if self.meta_method == "average":
            self._meta = None  # simple average at predict time
        elif self.meta_method == "logistic":
            self._meta = LogisticRegression(C=1.0, random_state=42)
            self._meta.fit(oof_preds, y)
        elif self.meta_method == "isotonic":
            # Isotonic on the average of base predictions
            avg_oof = oof_preds.mean(axis=1)
            self._meta = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._meta.fit(avg_oof, y)

        # Refit all base models on full training data
        for name, factory in self.base_models.items():
            model = factory()
            if name == "xgb_30feat":
                model.fit(X, y)
            else:
                model.fit(X[:, BEST_IDX], y)
            self._fitted_bases[name] = model

        return self

    def predict_proba(self, X):
        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self._fitted_bases.items()):
            if name == "xgb_30feat":
                base_preds[:, i] = model.predict_proba(X)[:, 1]
            else:
                base_preds[:, i] = model.predict_proba(X[:, BEST_IDX])[:, 1]

        if self.meta_method == "average":
            prob = base_preds.mean(axis=1)
        elif self.meta_method == "logistic":
            prob = self._meta.predict_proba(base_preds)[:, 1]
        elif self.meta_method == "isotonic":
            avg_pred = base_preds.mean(axis=1)
            prob = self._meta.predict(avg_pred)

        return np.column_stack([1 - prob, prob])


def run_ensemble_loyo(data, gender="M", data_dir="data",
                      meta_method="average", years=None):
    if years is None:
        years = TOURNEY_YEARS

    results = {}
    all_brier = []

    for year in years:
        t0 = time.time()
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)

        model = StackingEnsemble(meta_method=meta_method)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

        bs = brier_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        elapsed = time.time() - t0

        results[year] = {
            "brier": round(bs, 4),
            "log_loss": round(ll, 4),
            "n_games": len(y_test),
            "elapsed_s": round(elapsed, 1),
        }
        all_brier.append(bs)
        print(f"  {year}: Brier={bs:.4f}  ({elapsed:.1f}s)")

    results["overall"] = {
        "mean_brier": round(float(np.mean(all_brier)), 4),
        "std_brier": round(float(np.std(all_brier)), 4),
        "n_folds": len(years),
    }
    o = results["overall"]
    print(f"\n  LOYO CV: Brier = {o['mean_brier']:.4f} ± {o['std_brier']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stacking ensemble")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/ensemble")
    parser.add_argument("--meta", default="average",
                        choices=["average", "logistic", "isotonic"])
    args = parser.parse_args()

    data = prepare_data(args.data_dir)

    for meta in ([args.meta] if args.meta != "all" else ["average", "logistic", "isotonic"]):
        print(f"\n=== Meta-learner: {meta} ===")
        results = run_ensemble_loyo(data, args.gender, args.data_dir, meta)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"ensemble_{meta}_{timestamp}.json"
        save_results(results, path, model_name=f"ensemble_{meta}",
                     extra={"meta_method": meta})
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
