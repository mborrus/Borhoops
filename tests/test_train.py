import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train.features import build_features, matchup_features, FEATURE_COLS


def _make_results(seasons=(2020, 2021)):
    """Minimal game results for two seasons."""
    rows = []
    for season in seasons:
        for day in range(1, 4):
            rows.append({
                "Season": season, "DayNum": day,
                "WTeamID": 1100, "WScore": 75,
                "LTeamID": 1200, "LScore": 65,
                "WLoc": "H", "NumOT": 0,
            })
            rows.append({
                "Season": season, "DayNum": day,
                "WTeamID": 1200, "WScore": 70,
                "LTeamID": 1300, "LScore": 60,
                "WLoc": "N", "NumOT": 0,
            })
    return pd.DataFrame(rows)


def _make_conferences(seasons=(2020, 2021)):
    rows = []
    for season in seasons:
        for team in (1100, 1200, 1300):
            rows.append({"Season": season, "TeamID": team, "ConfAbbrev": "big12"})
    return pd.DataFrame(rows)


class TestBuildFeatures:
    def test_shape(self):
        results = _make_results()
        conferences = _make_conferences()
        features_df, elo, gc, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons={2020, 2021},
        )
        # 2 seasons × 6 games = 12 rows
        assert len(features_df) == 12
        for col in FEATURE_COLS:
            assert col in features_df.columns, f"Missing column: {col}"
        assert "win" in features_df.columns
        assert "season" in features_df.columns
        assert set(features_df["season"]) == {2020, 2021}

    def test_feature_values(self):
        results = _make_results()
        conferences = _make_conferences()
        features_df, _, _, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons={2020, 2021},
        )
        # elo_pred should be between 0 and 1
        assert (features_df["elo_pred"] >= 0).all()
        assert (features_df["elo_pred"] <= 1).all()
        # home should be -1, 0, or 1
        assert set(features_df["home"].unique()).issubset({-1, 0, 1})
        # win should be 0 or 1
        assert set(features_df["win"].unique()).issubset({0, 1})

    def test_no_leakage(self):
        """Features for season 2020 should not differ based on whether 2021 data exists."""
        results_both = _make_results(seasons=(2020, 2021))
        results_one = _make_results(seasons=(2020,))
        conferences = _make_conferences(seasons=(2020, 2021))

        feat_both, _, _, _, _ = build_features(
            results_both, conferences, hfa_dict={}, location_dict={},
            seasons={2020},
        )
        feat_one, _, _, _, _ = build_features(
            results_one, conferences, hfa_dict={}, location_dict={},
            seasons={2020},
        )

        # Season 2020 features should be identical regardless of 2021 data
        np.testing.assert_array_almost_equal(
            feat_both[FEATURE_COLS].values,
            feat_one[FEATURE_COLS].values,
        )

    def test_returns_elo_and_game_counts(self):
        results = _make_results()
        conferences = _make_conferences()
        _, elo, gc, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons={2020},
        )
        assert isinstance(elo, dict)
        assert isinstance(gc, dict)
        assert 1100 in elo
        assert 1100 in gc
        assert gc[1100] > 0

    def test_empty_seasons_returns_empty_df(self):
        """Passing seasons=set() should return empty features but still run Elo."""
        results = _make_results()
        conferences = _make_conferences()
        features_df, elo, gc, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons=set(),
        )
        assert len(features_df) == 0
        assert len(elo) > 0  # Elo still computed


class TestMatchupFeatures:
    def test_neutral_site(self):
        gc = {1100: 30, 1200: 28}
        feat = matchup_features(1550, 1450, gc, 1100, 1200)

        assert feat["home"] == 0
        assert feat["elo_diff"] == 100
        assert 0 < feat["elo_pred"] < 1
        assert feat["game_count_avg"] == 29.0

    def test_sign_convention(self):
        """elo_diff should be elo_a - elo_b (lower team minus higher)."""
        gc = {1: 20, 2: 20}
        feat = matchup_features(1400, 1600, gc, 1, 2)
        assert feat["elo_diff"] == -200


class TestTrainAndPredictRoundtrip:
    def test_roundtrip(self, tmp_path):
        """Train on synthetic data, predict, verify probabilities in [0, 1]."""
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Synthetic training data
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, len(FEATURE_COLS))
        y = (X[:, 0] > 0).astype(int)  # simple decision boundary

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        path = tmp_path / "test_model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)

        probs = loaded.predict_proba(X)[:, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert len(probs) == n
