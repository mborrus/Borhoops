"""Tests for Phase 1 features: odds, polls, roster, seeds."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train.features import build_features, matchup_features, FEATURE_COLS, SEED_COLS
from train.extended_features import (
    ODDS_COLS, POLL_COLS, ROSTER_COLS,
    _spread_to_prob, odds_game_features, odds_matchup_features,
    poll_game_features, poll_matchup_features,
    roster_matchup_features, seed_matchup_features,
)


# -- Helpers -------------------------------------------------------------------

def _make_results(seasons=(2020, 2021)):
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


# -- Spread to probability conversion -----------------------------------------

class TestSpreadToProb:
    def test_even_game(self):
        assert _spread_to_prob(0) == pytest.approx(0.5)

    def test_favorite(self):
        # Negative spread = favored, should have high win prob
        assert _spread_to_prob(-10) > 0.7

    def test_underdog(self):
        # Positive spread = underdog, should have low win prob
        assert _spread_to_prob(10) < 0.3

    def test_symmetry(self):
        p1 = _spread_to_prob(-7.5)
        p2 = _spread_to_prob(7.5)
        assert p1 + p2 == pytest.approx(1.0)

    def test_nan(self):
        assert np.isnan(_spread_to_prob(np.nan))


# -- Odds features -------------------------------------------------------------

class TestOddsGameFeatures:
    def test_found(self):
        lookup = {(2020, 1, 1100, 1200): {"spread": -5.5, "over_under": 145.0}}
        feat = odds_game_features(lookup, 2020, 1, 1100, 1200)
        assert feat["spread"] == -5.5
        assert feat["spread_abs"] == 5.5
        assert feat["implied_prob"] > 0.5  # favorite
        assert feat["over_under"] == 145.0

    def test_missing(self):
        feat = odds_game_features({}, 2020, 1, 1100, 1200)
        assert np.isnan(feat["spread"])
        assert np.isnan(feat["spread_abs"])
        assert np.isnan(feat["implied_prob"])
        assert np.isnan(feat["over_under"])


class TestOddsMatchupFeatures:
    def test_stronger_team(self):
        avg = {
            (2020, 1100): {"avg_spread": -15.0, "avg_over_under": 140.0},
            (2020, 1200): {"avg_spread": -5.0, "avg_over_under": 150.0},
        }
        feat = odds_matchup_features(avg, 2020, 1100, 1200)
        # 1100 is typically bigger favorite → spread should be negative
        assert feat["spread"] == -10.0
        assert feat["spread_abs"] == 10.0
        assert feat["implied_prob"] > 0.5

    def test_missing_team(self):
        avg = {(2020, 1100): {"avg_spread": -10.0, "avg_over_under": 140.0}}
        feat = odds_matchup_features(avg, 2020, 1100, 1200)
        assert np.isnan(feat["spread"])


# -- Poll features -------------------------------------------------------------

class TestPollGameFeatures:
    def test_ranked_vs_unranked(self):
        lookup = {(2020, 1, 1100): {"rank": 5, "previous": 7, "points": 1200.0}}
        weeks = {(2020, 1100): 10}
        feat = poll_game_features(lookup, weeks, 2020, 7, 1100, 1200)
        assert feat["poll_rank_diff"] == 5 - 30  # ranked vs unranked
        assert feat["weeks_ranked_diff"] == 10 - 0

    def test_both_unranked(self):
        feat = poll_game_features({}, {}, 2020, 7, 1100, 1200)
        assert feat["poll_rank_diff"] == 0
        assert feat["poll_momentum_diff"] == 0
        assert feat["weeks_ranked_diff"] == 0


class TestPollMatchupFeatures:
    def test_end_of_season(self):
        lookup = {
            (2020, 18, 1100): {"rank": 3, "previous": 5, "points": 1300.0},
            (2020, 18, 1200): {"rank": 15, "previous": 12, "points": 600.0},
        }
        weeks = {(2020, 1100): 18, (2020, 1200): 10}
        feat = poll_matchup_features(lookup, weeks, 2020, 1100, 1200)
        assert feat["poll_rank_diff"] == 3 - 15
        assert feat["weeks_ranked_diff"] == 18 - 10

    def test_no_data(self):
        feat = poll_matchup_features({}, {}, 2020, 1100, 1200)
        assert feat["poll_rank_diff"] == 0


# -- Roster features -----------------------------------------------------------

class TestRosterFeatures:
    def test_diff(self):
        lookup = {
            (2020, 1100): {"avg_experience": 2.8, "senior_pct": 0.3},
            (2020, 1200): {"avg_experience": 2.0, "senior_pct": 0.1},
        }
        feat = roster_matchup_features(lookup, 2020, 1100, 1200)
        assert feat["avg_experience_diff"] == pytest.approx(0.8)
        assert feat["senior_pct_diff"] == pytest.approx(0.2)

    def test_missing(self):
        feat = roster_matchup_features({}, 2020, 1100, 1200)
        assert np.isnan(feat["avg_experience_diff"])
        assert np.isnan(feat["senior_pct_diff"])


# -- Seed features -------------------------------------------------------------

class TestSeedFeatures:
    def test_diff(self):
        seeds = {(2020, 1100): 1, (2020, 1200): 16}
        feat = seed_matchup_features(seeds, 2020, 1100, 1200)
        assert feat["seed_diff"] == -15  # 1 seed is better (lower number)

    def test_unseeded(self):
        seeds = {(2020, 1100): 4}
        feat = seed_matchup_features(seeds, 2020, 1100, 1200)
        assert feat["seed_diff"] == 4 - 17  # 1200 unseeded → 17


# -- Integration: new features in build_features and matchup_features ----------

class TestNewFeaturesInBuildFeatures:
    def test_all_new_cols_present(self):
        results = _make_results()
        conferences = _make_conferences()
        features_df, _, _, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons={2020, 2021},
        )
        for col in ODDS_COLS + POLL_COLS + ROSTER_COLS + SEED_COLS:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_feature_count(self):
        assert len(FEATURE_COLS) == 40

    def test_with_odds(self):
        results = _make_results()
        conferences = _make_conferences()
        odds_lookup = {
            (2020, 1, 1100, 1200): {"spread": -7.5, "over_under": 145.0},
        }
        features_df, _, _, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons={2020}, odds_lookup=odds_lookup,
        )
        # The game (1100 vs 1200, day 1) should have odds
        mask = (features_df["season"] == 2020) & (features_df["team_low"] == 1100) & (features_df["team_high"] == 1200)
        day1 = features_df[mask & (features_df["day_num"] == 1)]
        if len(day1) > 0:
            assert day1.iloc[0]["spread"] == -7.5
            assert day1.iloc[0]["spread_abs"] == 7.5

    def test_with_polls(self):
        results = _make_results()
        conferences = _make_conferences()
        poll_lookup = {
            (2020, 1, 1100): {"rank": 5, "previous": 7, "points": 1200.0},
        }
        weeks_ranked = {(2020, 1100): 10}
        features_df, _, _, _, _ = build_features(
            results, conferences, hfa_dict={}, location_dict={},
            seasons={2020}, poll_lookup=poll_lookup, weeks_ranked=weeks_ranked,
        )
        mask = (features_df["season"] == 2020) & (features_df["team_low"] == 1100)
        rows = features_df[mask]
        assert (rows["poll_rank_diff"] != 0).any()


class TestNewFeaturesInMatchupFeatures:
    def test_with_all_new_data(self):
        gc = {1100: 30, 1200: 28}
        seeds = {(2020, 1100): 3, (2020, 1200): 14}
        odds_avg = {
            (2020, 1100): {"avg_spread": -12.0, "avg_over_under": 145.0},
            (2020, 1200): {"avg_spread": -4.0, "avg_over_under": 150.0},
        }
        poll_lookup = {
            (2020, 18, 1100): {"rank": 8, "previous": 10, "points": 900.0},
        }
        weeks_ranked = {(2020, 1100): 15}
        roster_lookup = {
            (2020, 1100): {"avg_experience": 2.8, "senior_pct": 0.3},
            (2020, 1200): {"avg_experience": 2.2, "senior_pct": 0.15},
        }

        feat = matchup_features(
            1550, 1450, gc, 1100, 1200,
            season=2020, seeds=seeds,
            odds_team_avg=odds_avg,
            poll_lookup=poll_lookup, weeks_ranked=weeks_ranked,
            roster_lookup=roster_lookup,
        )

        assert feat["seed_diff"] == 3 - 14
        assert feat["spread"] == pytest.approx(-8.0)
        assert feat["poll_rank_diff"] == 8 - 30  # 1200 unranked
        assert feat["avg_experience_diff"] == pytest.approx(0.6)

        # All FEATURE_COLS should be present
        for col in FEATURE_COLS:
            assert col in feat, f"Missing column: {col}"
