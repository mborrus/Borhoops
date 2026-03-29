"""Scoring utilities for model evaluation."""

import numpy as np
import pandas as pd


def brier_score(preds, outcomes):
    """Mean squared error between predicted probabilities and binary outcomes."""
    preds = np.asarray(preds, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((preds - outcomes) ** 2))


def calibration_table(preds, outcomes, bins=10):
    """Build a calibration table comparing predicted vs actual win rates.

    Args:
        preds:    array of predicted probabilities (0-1)
        outcomes: array of binary outcomes (0 or 1)
        bins:     number of equal-width bins from 0 to 1

    Returns:
        DataFrame with columns: bin_label, count, actual_win_pct, expected_win_pct
    """
    # TODO: Implement this function — see guidance below.
    #
    # Design choices to consider:
    #   - Since tournament predictions are oriented to the favorite (pred >= 0.5),
    #     you might only bin from 0.5 to 1.0 instead of 0 to 1.0.
    #   - Equal-width bins (e.g. 0.5-0.6, 0.6-0.7) are simple and readable.
    #     Equal-count bins adapt better to skewed distributions.
    #   - What label format? "0.50 - 0.60" is clear. So is "[50%, 60%)".
    #
    # Expected output: pd.DataFrame with one row per bin containing:
    #   bin_label        — string label for the bin
    #   count            — number of predictions in this bin
    #   actual_win_pct   — fraction of actual wins in this bin
    #   expected_win_pct — mean predicted probability in this bin
    raise NotImplementedError("Implement calibration_table()")
