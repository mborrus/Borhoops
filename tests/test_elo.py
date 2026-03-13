import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict.elo import (
    add_game_counts,
    calc_elo_win_tourney,
    distance_to_elo_impact,
    get_advantage,
    k_factor,
    point_differential_scaler,
    run_elo,
    update_elo,
)


class TestPointDifferentialScaler:
    def test_below_average_returns_one(self):
        assert point_differential_scaler(70, 65, mov_avg=12) == 1.0

    def test_at_average_returns_one(self):
        # MOV == 12 → log(0+1) = 0, so multiplier stays at 1.0
        assert point_differential_scaler(80, 68, mov_avg=12) == 1.0

    def test_above_average_scales_up(self):
        result = point_differential_scaler(90, 60, mov_avg=12)
        assert result > 1.0

    def test_double_average_roughly_two(self):
        # MOV of 24 (double the 12 avg) should be close to 2x
        result = point_differential_scaler(84, 60, mov_avg=12)
        assert 1.5 < result < 2.5


class TestKFactor:
    def test_first_game_is_k_start(self):
        assert k_factor(1, k_start=56, k_end=38) == 56

    def test_decays_linearly(self):
        # Game 2 should be 55 (56 - 1)
        assert k_factor(2, k_start=56, k_end=38) == 55

    def test_after_threshold_returns_k_end(self):
        # k_start - k_end = 18, so game 18+ → k_end
        assert k_factor(18, k_start=56, k_end=38) == 38
        assert k_factor(30, k_start=56, k_end=38) == 38


class TestDistanceToEloImpact:
    def test_zero_miles(self):
        assert distance_to_elo_impact(0) == 0.0

    def test_known_value(self):
        # 1000 miles → 8 * 10 = 80
        assert distance_to_elo_impact(1000) == pytest.approx(80.0)

    def test_positive(self):
        assert distance_to_elo_impact(500) > 0


class TestGetAdvantage:
    def test_known_team(self):
        hfa = {101: 3.0, 102: 1.5}
        assert get_advantage(101, hfa, scalar=26) == 3.0 * 26

    def test_unknown_team_uses_fallback(self):
        hfa = {101: 3.0}
        result = get_advantage(999, hfa, scalar=26)
        assert result == pytest.approx(2.12698 * 26)


class TestCalcEloWinTourney:
    def test_equal_elo_is_fifty_fifty(self):
        assert calc_elo_win_tourney(1500, 1500) == pytest.approx(0.5)

    def test_higher_elo_favored(self):
        assert calc_elo_win_tourney(1600, 1400) > 0.5

    def test_lower_elo_underdog(self):
        assert calc_elo_win_tourney(1400, 1600) < 0.5

    def test_boost_amplifies(self):
        no_boost = calc_elo_win_tourney(1600, 1400, boost=1.0)
        with_boost = calc_elo_win_tourney(1600, 1400, boost=1.07)
        assert with_boost > no_boost


class TestUpdateElo:
    def test_equal_elo_change(self):
        change = update_elo(1500, 1500, k=30)
        assert change == pytest.approx(15.0)

    def test_upset_gives_more_points(self):
        upset = update_elo(1400, 1600, k=30)
        expected = update_elo(1600, 1400, k=30)
        assert upset > expected

    def test_positive_change(self):
        assert update_elo(1500, 1500, k=30) > 0


class TestAddGameCounts:
    def test_sequential_counts(self):
        results = pd.DataFrame({
            "Season": [2025, 2025, 2025],
            "WTeamID": [1, 2, 1],
            "LTeamID": [2, 3, 3],
        })
        counted = add_game_counts(results)
        # Game 1: team 1 wins (count 1), team 2 loses (count 1)
        # Game 2: team 2 wins (count 2, played as loser before), team 3 loses (count 1)
        # Game 3: team 1 wins (count 2), team 3 loses (count 2)
        assert list(counted["WTeam_Game_Count"]) == [1, 2, 2]
        assert list(counted["LTeam_Game_Count"]) == [1, 1, 2]

    def test_resets_across_seasons(self):
        results = pd.DataFrame({
            "Season": [2024, 2025],
            "WTeamID": [1, 1],
            "LTeamID": [2, 2],
        })
        counted = add_game_counts(results)
        assert list(counted["WTeam_Game_Count"]) == [1, 1]


class TestRunElo:
    """Integration test with minimal synthetic data."""

    @pytest.fixture
    def minimal_data(self):
        results = pd.DataFrame({
            "Season": [2024, 2024, 2024, 2024],
            "DayNum": [1, 2, 3, 4],
            "WTeamID": [1, 2, 1, 3],
            "WScore": [80, 75, 70, 85],
            "LTeamID": [2, 3, 3, 1],
            "LScore": [60, 65, 65, 70],
            "WLoc": ["H", "A", "N", "H"],
            "NumOT": [0, 0, 0, 0],
            "WTeam_Game_Count": [1, 1, 2, 1],
            "LTeam_Game_Count": [1, 1, 1, 2],
        })
        conferences = pd.DataFrame({
            "Season": [2024, 2024, 2024],
            "TeamID": [1, 2, 3],
            "ConfAbbrev": ["big12", "big12", "sec"],
        })
        hfa_dict = {1: 2.5, 2: 2.0, 3: 1.5}
        location_dict = {
            1: (38.0, -97.0),   # Kansas area
            2: (40.0, -83.0),   # Ohio area
            3: (33.0, -84.0),   # Georgia area
        }
        return results, conferences, hfa_dict, location_dict

    def test_returns_dict(self, minimal_data):
        results, conferences, hfa_dict, location_dict = minimal_data
        elo = run_elo(results, conferences, hfa_dict, location_dict)
        assert isinstance(elo, dict)
        assert len(elo) == 3

    def test_winners_gain_elo(self, minimal_data):
        results, conferences, hfa_dict, location_dict = minimal_data
        elo = run_elo(results, conferences, hfa_dict, location_dict,
                      mean_reversion=0.0)
        # Team 1 won 2 games, lost 1. Team 3 won 1, lost 2.
        # Without mean reversion, team 1 should be above initial
        assert elo[1] > 1500

    def test_mean_reversion_pulls_toward_conf_mean(self, minimal_data):
        results, conferences, hfa_dict, location_dict = minimal_data
        elo_no_rev = run_elo(results, conferences, hfa_dict, location_dict,
                             mean_reversion=0.0)
        elo_with_rev = run_elo(results, conferences, hfa_dict, location_dict,
                               mean_reversion=0.3)
        # With mean reversion, spread should be tighter
        spread_no_rev = max(elo_no_rev.values()) - min(elo_no_rev.values())
        spread_with_rev = max(elo_with_rev.values()) - min(elo_with_rev.values())
        assert spread_with_rev < spread_no_rev
