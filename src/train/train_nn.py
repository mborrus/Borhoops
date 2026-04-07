"""Phase 2c: Feed-forward neural network with temperature scaling (PyTorch).

Architecture: Input(40) → 128 → Dropout → 64 → Dropout → 32 → 1(sigmoid)
Calibration via temperature scaling on validation set.

Usage:
  PYTHONPATH=src python src/train/train_nn.py
  PYTHONPATH=src python src/train/train_nn.py --sweep  # hyperparameter sweep
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
from train.features import FEATURE_COLS


def _build_model(n_features, hidden1=128, hidden2=64, hidden3=32, dropout=0.3):
    import torch
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_features, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, hidden3),
        nn.ReLU(),
        nn.Linear(hidden3, 1),
    )


class NNClassifier:
    """PyTorch feed-forward net with temperature scaling for calibration."""

    def __init__(self, hidden1=128, hidden2=64, hidden3=32, dropout=0.3,
                 lr=0.001, weight_decay=1e-4, epochs=200, batch_size=512,
                 patience=20):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.temperature = 1.0
        self._model = None
        self._mean = None
        self._std = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        # Standardize
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_std = (X - self._mean) / self._std

        # Train/val split (last 15% for early stopping + temperature calibration)
        n_val = max(int(len(X_std) * 0.15), 100)
        idx = np.random.RandomState(42).permutation(len(X_std))
        train_idx, val_idx = idx[:-n_val], idx[-n_val:]

        X_train = torch.FloatTensor(X_std[train_idx])
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1)
        X_val = torch.FloatTensor(X_std[val_idx])
        y_val = torch.FloatTensor(y[val_idx]).unsqueeze(1)

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._model = _build_model(
            X_std.shape[1], self.hidden1, self.hidden2, self.hidden3, self.dropout)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self._model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_val)
                val_loss = criterion(val_logits, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self._model.load_state_dict(best_state)

        # Temperature scaling on validation set
        self._model.eval()
        with torch.no_grad():
            val_logits = self._model(X_val).numpy()
        self._calibrate_temperature(val_logits, y[val_idx])

        return self

    def _calibrate_temperature(self, logits, y_true):
        """Find temperature T that minimizes Brier score on validation set."""
        best_t = 1.0
        best_brier = float("inf")
        for t in np.arange(0.5, 3.0, 0.05):
            probs = 1.0 / (1.0 + np.exp(-logits.flatten() / t))
            bs = np.mean((y_true - probs) ** 2)
            if bs < best_brier:
                best_brier = bs
                best_t = t
        self.temperature = best_t

    def predict_proba(self, X):
        import torch
        X_std = (X - self._mean) / self._std
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.FloatTensor(X_std)).numpy().flatten()
        probs = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return np.column_stack([1 - probs, probs])


def run_nn_loyo(data, gender="M", data_dir="data", params=None, years=None):
    if years is None:
        years = TOURNEY_YEARS
    if params is None:
        params = {}

    results = {}
    all_brier = []

    for year in years:
        t0 = time.time()
        X_train, y_train, X_test, y_test = prepare_fold(data, gender, year, data_dir)

        model = NNClassifier(**params)
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


def run_sweep(data, gender="M", data_dir="data"):
    """Grid sweep over key NN hyperparameters."""
    configs = []
    for hidden1 in [64, 128, 256]:
        for dropout in [0.2, 0.3, 0.5]:
            for lr in [0.0005, 0.001, 0.003]:
                for wd in [1e-4, 1e-3]:
                    configs.append({
                        "hidden1": hidden1, "hidden2": hidden1 // 2,
                        "hidden3": hidden1 // 4, "dropout": dropout,
                        "lr": lr, "weight_decay": wd,
                    })

    print(f"Sweep: {len(configs)} configurations")
    best_brier = 1.0
    best_config = None

    for i, params in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {params}")
        results = run_nn_loyo(data, gender, data_dir, params)
        mb = results["overall"]["mean_brier"]
        if mb < best_brier:
            best_brier = mb
            best_config = params
            print(f"  *** NEW BEST: {mb:.4f} ***")

    return {"best_params": best_config, "best_brier": best_brier}


def main():
    parser = argparse.ArgumentParser(description="Neural network trainer")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/nn")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--grid-index", type=int, default=None,
                        help="Single config index for HPC array jobs")
    args = parser.parse_args()

    data = prepare_data(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.sweep:
        results = run_sweep(data, args.gender, args.data_dir)
        path = output_dir / f"nn_sweep_{timestamp}.json"
    else:
        print("Running default NN with LOYO CV...")
        results = run_nn_loyo(data, args.gender, args.data_dir)
        path = output_dir / f"nn_default_{timestamp}.json"

    save_results(results, path, model_name="neural_net")
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
