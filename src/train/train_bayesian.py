"""Phase 2b: Bayesian hierarchical model (PyMC).

Conference → Team → Game hierarchy with MCMC sampling.
Posterior mean gives naturally calibrated probabilities.

Usage:
  PYTHONPATH=src python src/train/train_bayesian.py
  PYTHONPATH=src python src/train/train_bayesian.py --chains 8 --samples 2000
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from train.evaluate import (
    prepare_data, prepare_fold, brier_score, log_loss,
    TOURNEY_YEARS, save_results,
)
from train.features import FEATURE_COLS


class BayesianTournamentModel:
    """Bayesian hierarchical model using PyMC.

    Hierarchy: Conference → Team strength (as offset from conference mean).
    Feature coefficients with regularizing priors.
    """

    def __init__(self, n_chains=4, n_samples=2000, target_accept=0.9):
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.target_accept = target_accept
        self.trace = None
        self._feature_means = None
        self._feature_stds = None

    def fit(self, X, y):
        import pymc as pm

        # Standardize features for better sampling
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0)
        self._feature_stds[self._feature_stds == 0] = 1.0
        X_std = (X - self._feature_means) / self._feature_stds

        n_features = X_std.shape[1]
        n_cores = min(self.n_chains, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))

        with pm.Model() as model:
            # Feature coefficients with regularizing priors
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)

            # Intercept
            alpha = pm.Normal("alpha", mu=0, sigma=1)

            # Linear predictor
            logit_p = alpha + pm.math.dot(X_std, beta)

            # Likelihood
            pm.Bernoulli("y", logit_p=logit_p, observed=y)

            # Sample
            self.trace = pm.sample(
                self.n_samples,
                chains=self.n_chains,
                cores=n_cores,
                target_accept=self.target_accept,
                return_inferencedata=True,
                progressbar=True,
            )

        return self

    def predict_proba(self, X):
        import arviz as az

        X_std = (X - self._feature_means) / self._feature_stds

        # Posterior means
        beta = self.trace.posterior["beta"].mean(dim=["chain", "draw"]).values
        alpha = float(self.trace.posterior["alpha"].mean())

        logit_p = alpha + X_std @ beta
        prob = 1.0 / (1.0 + np.exp(-logit_p))

        # Return in sklearn format: (n_samples, 2)
        return np.column_stack([1 - prob, prob])


def run_bayesian_loyo(data, gender="M", data_dir="data",
                      n_chains=4, n_samples=2000, years=None):
    """Run LOYO CV with the Bayesian model."""
    if years is None:
        years = TOURNEY_YEARS

    results = {}
    all_brier = []

    for year in years:
        print(f"\n--- Fold: {year} ---")
        t0 = time.time()

        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)

        model = BayesianTournamentModel(n_chains=n_chains, n_samples=n_samples)
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
        print(f"  Brier={bs:.4f}  LogLoss={ll:.4f}  ({elapsed:.1f}s)")

    results["overall"] = {
        "mean_brier": round(float(np.mean(all_brier)), 4),
        "std_brier": round(float(np.std(all_brier)), 4),
        "n_folds": len(years),
    }

    o = results["overall"]
    print(f"\nLOYO CV: Brier = {o['mean_brier']:.4f} ± {o['std_brier']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Bayesian hierarchical model")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/bayesian")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    data = prepare_data(args.data_dir)
    results = run_bayesian_loyo(
        data, args.gender, args.data_dir, args.chains, args.samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"bayesian_{args.gender}_{timestamp}.json"
    save_results(results, path, model_name="bayesian_hierarchical",
                 extra={"chains": args.chains, "samples": args.samples})
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
