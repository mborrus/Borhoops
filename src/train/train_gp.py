"""Phase 2e: Gaussian Process classifier.

GP with RBF kernel — naturally outputs calibrated probabilities with uncertainty.
O(n³) training, so uses subset training for large datasets.

Usage:
  PYTHONPATH=src python src/train/train_gp.py
  PYTHONPATH=src python src/train/train_gp.py --max-train 10000
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from train.evaluate import (
    prepare_data, prepare_fold, brier_score, log_loss,
    TOURNEY_YEARS, save_results,
)


class GPClassifierWrapper:
    """Wrapper around sklearn GaussianProcessClassifier with subset training."""

    def __init__(self, max_train=5000, n_restarts=5, length_scale_bounds=(1e-2, 1e3)):
        self.max_train = max_train
        self.n_restarts = n_restarts
        self.length_scale_bounds = length_scale_bounds
        self._model = None
        self._mean = None
        self._std = None

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        # Standardize
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_std = (X - self._mean) / self._std

        # Subsample if too large (GP is O(n³))
        if len(X_std) > self.max_train:
            idx = np.random.RandomState(42).choice(
                len(X_std), self.max_train, replace=False)
            X_std = X_std[idx]
            y = y[idx]

        kernel = ConstantKernel() * RBF(length_scale_bounds=self.length_scale_bounds)
        self._model = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=42,
        )
        self._model.fit(X_std, y)
        return self

    def predict_proba(self, X):
        X_std = (X - self._mean) / self._std
        return self._model.predict_proba(X_std)


def run_gp_loyo(data, gender="M", data_dir="data", max_train=5000, years=None):
    if years is None:
        years = TOURNEY_YEARS

    results = {}
    all_brier = []

    for year in years:
        t0 = time.time()
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)

        model = GPClassifierWrapper(max_train=max_train)
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
    parser = argparse.ArgumentParser(description="Gaussian Process classifier")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/gp")
    parser.add_argument("--max-train", type=int, default=5000,
                        help="Max training samples (GP is O(n³))")
    args = parser.parse_args()

    data = prepare_data(args.data_dir)
    results = run_gp_loyo(data, args.gender, args.data_dir, args.max_train)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"gp_{args.gender}_{timestamp}.json"
    save_results(results, path, model_name="gaussian_process",
                 extra={"max_train": args.max_train})
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
