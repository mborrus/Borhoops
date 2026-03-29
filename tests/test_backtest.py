import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate.metrics import brier_score


class TestBrierScore:
    def test_perfect_predictions(self):
        preds = [1.0, 0.0, 1.0]
        outcomes = [1, 0, 1]
        assert brier_score(preds, outcomes) == 0.0

    def test_worst_predictions(self):
        preds = [0.0, 1.0, 0.0]
        outcomes = [1, 0, 1]
        assert brier_score(preds, outcomes) == 1.0

    def test_coin_flip(self):
        preds = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1, 0, 1, 0]
        assert brier_score(preds, outcomes) == pytest.approx(0.25)

    def test_single_prediction(self):
        assert brier_score([0.7], [1]) == pytest.approx(0.09)

    def test_returns_float(self):
        assert isinstance(brier_score([0.5], [1]), float)
