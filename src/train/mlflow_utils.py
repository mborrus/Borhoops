"""MLflow logging helpers for Phase 2 training scripts.

Sets up file-based tracking at REPO_ROOT/mlruns/.
Each model type gets its own experiment name.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACKING_URI = f"file://{REPO_ROOT / 'mlruns'}"


def init_mlflow(experiment_name):
    """Initialize MLflow with file-based tracking. Returns True if available."""
    try:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        return True
    except Exception:
        return False


def log_run(model_name, params, metrics, tags=None):
    """Log a single run to MLflow. No-ops if MLflow unavailable."""
    try:
        import mlflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
    except Exception:
        pass


def log_loyo_results(model_name, params, results, tags=None):
    """Log LOYO CV results: overall metrics + per-year breakdown."""
    try:
        import mlflow
        overall = results.get("overall", results)
        metrics = {
            "mean_brier": overall.get("mean_brier", 0),
            "std_brier": overall.get("std_brier", 0),
        }
        # Per-year Brier scores
        for year, r in results.items():
            if year == "overall" or not isinstance(r, dict):
                continue
            metrics[f"brier_{year}"] = r.get("brier", 0)

        log_run(model_name, params, metrics, tags)
    except Exception:
        pass
