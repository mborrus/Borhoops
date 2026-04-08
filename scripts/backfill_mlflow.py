"""Backfill MLflow from existing JSON results. Not committed to git.

Usage:
  PYTHONPATH=src python scripts/backfill_mlflow.py
"""

import json
import glob
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"file://{REPO_ROOT / 'mlruns'}"

mlflow.set_tracking_uri(TRACKING_URI)


def backfill_gbm():
    """Backfill all GBM grid search results."""
    files = sorted(glob.glob(str(REPO_ROOT / "results" / "gbm" / "*.json")))
    if not files:
        print("No GBM results found")
        return

    # Group by library
    by_lib = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        name = d.get("model", "")
        for lib in ("xgboost", "lightgbm", "catboost"):
            if lib in name:
                by_lib.setdefault(lib, []).append((f, d))
                break

    for lib, entries in by_lib.items():
        mlflow.set_experiment(f"gbm_{lib}")
        count = 0
        for filepath, d in entries:
            r = d.get("results", d)
            params = r.get("params", {})
            mean_brier = r.get("mean_brier", None)
            if mean_brier is None:
                continue

            metrics = {"mean_brier": mean_brier}
            for year, brier in r.get("per_year_brier", {}).items():
                metrics[f"brier_{year}"] = brier

            with mlflow.start_run(run_name=d.get("model", Path(filepath).stem)):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics(metrics)
                mlflow.set_tags({
                    "library": lib,
                    "source": "backfill",
                    "grid_index": str(r.get("grid_index", "")),
                })
            count += 1

        print(f"  {lib}: {count} runs logged")


def backfill_single_model(model_dir, experiment_name, model_tag):
    """Backfill results for a single-result model (bayesian, nn, lstm, gp)."""
    files = sorted(glob.glob(str(REPO_ROOT / "results" / model_dir / "*.json")))
    if not files:
        print(f"  {model_dir}: no results")
        return

    mlflow.set_experiment(experiment_name)
    count = 0
    for filepath in files:
        with open(filepath) as fh:
            d = json.load(fh)

        r = d.get("results", d)
        overall = r.get("overall", r)
        mean_brier = overall.get("mean_brier", None)
        if mean_brier is None:
            continue

        metrics = {"mean_brier": mean_brier}
        if "std_brier" in overall:
            metrics["std_brier"] = overall["std_brier"]
        for year, yr in r.items():
            if isinstance(yr, dict) and "brier" in yr and year != "overall":
                metrics[f"brier_{year}"] = yr["brier"]

        # Collect params from various locations in the JSON
        params = {}
        for key in ("chains", "samples", "max_train", "fold_index", "n_workers"):
            if key in d:
                params[key] = d[key]
        if "params" in r:
            params.update(r["params"])
        if "extra" in d:
            params.update(d["extra"])

        with mlflow.start_run(run_name=d.get("model", Path(filepath).stem)):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"model": model_tag, "source": "backfill"})
        count += 1

    print(f"  {model_dir}: {count} runs logged")


if __name__ == "__main__":
    print("Backfilling MLflow from existing results...")
    print(f"Tracking URI: {TRACKING_URI}")
    print()

    print("GBM:")
    backfill_gbm()

    print("\nOther models:")
    backfill_single_model("bayesian", "bayesian", "bayesian_hierarchical")
    backfill_single_model("nn", "neural_net", "neural_net")
    backfill_single_model("lstm", "lstm", "lstm_temporal")
    backfill_single_model("gp", "gaussian_process", "gaussian_process")

    print("\nDone. View with:")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI} --port 5001")
