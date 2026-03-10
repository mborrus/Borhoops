import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transform.transform_espn import (
    categorize_game,
    compute_day_num,
    load_day_zero,
    parse_season,
    transform,
    union_and_save,
)


@pytest.fixture
def data_dir(tmp_path):
    """Minimal data dir with crosswalk, seasons, Kaggle results, and ESPN game_info."""
    (tmp_path / "kaggle").mkdir()
    (tmp_path / "cbbpy").mkdir()
    (tmp_path / "derived").mkdir()

    crosswalk = pd.DataFrame({
        "espn_id": [100, 200, 300],
        "kaggle_m_id": [1100, 1200, 1300],
        "kaggle_w_id": [3100, 3200, 3300],
        "espn_location": ["Team A", "Team B", "Team C"],
        "kaggle_team_name": ["Team A", "Team B", "Team C"],
        "match_method": ["spelling", "spelling", "spelling"],
    })
    crosswalk.to_csv(tmp_path / "derived" / "espn_kaggle_crosswalk.csv", index=False)

    seasons = pd.DataFrame({
        "Season": [2025, 2026],
        "DayZero": ["11/04/2024", "11/03/2025"],
        "RegionW": ["W", "W"], "RegionX": ["X", "X"],
        "RegionY": ["Y", "Y"], "RegionZ": ["Z", "Z"],
    })
    seasons.to_csv(tmp_path / "kaggle" / "MSeasons.csv", index=False)

    kaggle_results = pd.DataFrame({
        "Season": [2025, 2025],
        "DayNum": [20, 30],
        "WTeamID": [1100, 1200],
        "WScore": [75, 80],
        "LTeamID": [1200, 1300],
        "LScore": [60, 70],
        "WLoc": ["H", "A"],
        "NumOT": [0, 1],
    })
    kaggle_results.to_csv(
        tmp_path / "kaggle" / "MRegularSeasonCompactResults.csv", index=False
    )

    espn_games = pd.DataFrame({
        "game_id": [1001, 1002, 1003],
        "home_id": [100, 200, 300],
        "away_id": [200, 300, 100],
        "home_score": [80, 60, 90],
        "away_score": [70, 75, 85],
        "home_win": [True, False, True],
        "num_ots": [0, 0, 1],
        "is_neutral": [False, False, True],
        "is_postseason": [False, False, False],
        "tournament": ["", "", ""],
        "game_day": ["January 15, 2026", "February 01, 2026", "February 10, 2026"],
    })
    espn_games.to_csv(tmp_path / "cbbpy" / "mens_game_info.csv", index=False)

    return tmp_path


class TestParseSeason:
    def test_jan_is_current_year(self):
        assert parse_season("January 15, 2026") == 2026

    def test_may_is_next_year(self):
        assert parse_season("May 01, 2025") == 2026

    def test_november(self):
        assert parse_season("November 10, 2025") == 2026


class TestComputeDayNum:
    def test_known_date(self):
        dz = {2026: pd.Timestamp("2025-11-03")}
        # Jan 15, 2026 = 73 days after Nov 3, 2025
        assert compute_day_num("January 15, 2026", dz) == 73

    def test_missing_season_returns_none(self):
        assert compute_day_num("January 15, 2030", {}) is None


class TestCategorizeGame:
    def test_regular_season(self):
        assert categorize_game(False, "") == "regular"

    def test_conference_tourney_is_regular(self):
        assert categorize_game(False, "Big East Tournament") == "regular"

    def test_ncaa_tourney(self):
        assert categorize_game(True, "NCAA Tournament") == "ncaa"

    def test_nit_is_secondary(self):
        assert categorize_game(True, "NIT") == "secondary"

    def test_cbi_is_secondary(self):
        assert categorize_game(True, "CBI") == "secondary"


class TestTransform:
    def test_basic_transform(self, data_dir):
        result = transform(data_dir, "mens")
        assert len(result) == 3
        assert list(result.columns[:8]) == [
            "Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"
        ]

    def test_winner_loser_flip(self, data_dir):
        result = transform(data_dir, "mens")
        # Game 1001: home_id=100 won → WTeamID=1100, LTeamID=1200
        g1 = result[result["WTeamID"] == 1100].iloc[0]
        assert g1["LTeamID"] == 1200
        assert g1["WScore"] == 80
        assert g1["LScore"] == 70

    def test_away_winner_flip(self, data_dir):
        result = transform(data_dir, "mens")
        # Game 1002: home_id=200 lost → WTeamID=1300 (away won), LTeamID=1200
        g2 = result[result["WTeamID"] == 1300].iloc[0]
        assert g2["LTeamID"] == 1200
        assert g2["WScore"] == 75
        assert g2["LScore"] == 60

    def test_wloc_home_win(self, data_dir):
        result = transform(data_dir, "mens")
        g1 = result[result["WTeamID"] == 1100].iloc[0]
        assert g1["WLoc"] == "H"

    def test_wloc_away_win(self, data_dir):
        result = transform(data_dir, "mens")
        g2 = result[result["WTeamID"] == 1300].iloc[0]
        # Game 1002: not neutral, away team won → WLoc = A
        assert g2["WLoc"] == "A"

    def test_wloc_neutral(self, data_dir):
        result = transform(data_dir, "mens")
        # Game 1003: neutral site
        g3 = result[result["NumOT"] == 1].iloc[0]
        assert g3["WLoc"] == "N"

    def test_missing_crosswalk_warns(self, data_dir):
        espn = pd.read_csv(data_dir / "cbbpy" / "mens_game_info.csv")
        espn.loc[0, "home_id"] = 9999  # unknown ESPN ID
        espn.to_csv(data_dir / "cbbpy" / "mens_game_info.csv", index=False)

        with pytest.warns(UserWarning, match="no crosswalk"):
            result = transform(data_dir, "mens")
        assert len(result) == 2  # one game dropped

    def test_no_espn_file_returns_empty(self, data_dir):
        (data_dir / "cbbpy" / "mens_game_info.csv").unlink()
        with pytest.warns(UserWarning):
            result = transform(data_dir, "mens")
        assert result.empty

    def test_all_regular_season(self, data_dir):
        result = transform(data_dir, "mens")
        assert (result["_category"] == "regular").all()


class TestUnionAndSave:
    def test_creates_combined_csv(self, data_dir):
        union_and_save(data_dir, "mens")
        out = data_dir / "derived" / "MRegularSeasonCompactResults_combined.csv"
        assert out.exists()
        df = pd.read_csv(out)
        # 2 kaggle + 3 espn (no overlap) = 5
        assert len(df) == 5

    def test_has_source_column(self, data_dir):
        union_and_save(data_dir, "mens")
        df = pd.read_csv(data_dir / "derived" / "MRegularSeasonCompactResults_combined.csv")
        assert "_source" in df.columns
        assert set(df["_source"].unique()) == {"kaggle", "espn"}

    def test_kaggle_wins_on_dedup(self, data_dir):
        # Add an ESPN game that overlaps with a Kaggle game on dedup key
        espn = pd.read_csv(data_dir / "cbbpy" / "mens_game_info.csv")
        overlap = pd.DataFrame({
            "game_id": [9999],
            "home_id": [100], "away_id": [200],
            "home_score": [99], "away_score": [50],  # different score than Kaggle's 75-60
            "home_win": [True], "num_ots": [0],
            "is_neutral": [False], "is_postseason": [False],
            "tournament": [""],
            # DayZero=11/04/2024, Season 2025, DayNum 20 → Dec 24, 2024 (matches Kaggle row)
            "game_day": ["November 24, 2024"],
        })
        espn = pd.concat([espn, overlap], ignore_index=True)
        espn.to_csv(data_dir / "cbbpy" / "mens_game_info.csv", index=False)

        union_and_save(data_dir, "mens")
        df = pd.read_csv(data_dir / "derived" / "MRegularSeasonCompactResults_combined.csv")

        # The overlap row (Season=2025, DayNum=20, WTeamID=1100, LTeamID=1200)
        # should keep Kaggle's score (75-60), not ESPN's (99-50)
        dup_row = df[(df["Season"] == 2025) & (df["DayNum"] == 20) &
                     (df["WTeamID"] == 1100) & (df["LTeamID"] == 1200)]
        assert len(dup_row) == 1
        assert dup_row.iloc[0]["WScore"] == 75
        assert dup_row.iloc[0]["_source"] == "kaggle"

    def test_sorted_by_season_daynum(self, data_dir):
        union_and_save(data_dir, "mens")
        df = pd.read_csv(data_dir / "derived" / "MRegularSeasonCompactResults_combined.csv")
        assert df["Season"].is_monotonic_increasing or (
            df.sort_values(["Season", "DayNum"]).reset_index(drop=True).equals(df)
        )

    def test_secondary_tourney_column(self, data_dir):
        # Add a secondary tourney game to ESPN
        espn = pd.read_csv(data_dir / "cbbpy" / "mens_game_info.csv")
        nit_game = pd.DataFrame({
            "game_id": [8888],
            "home_id": [100], "away_id": [200],
            "home_score": [70], "away_score": [65],
            "home_win": [True], "num_ots": [0],
            "is_neutral": [False], "is_postseason": [True],
            "tournament": ["NIT"],
            "game_day": ["March 20, 2026"],
        })
        espn = pd.concat([espn, nit_game], ignore_index=True)
        espn.to_csv(data_dir / "cbbpy" / "mens_game_info.csv", index=False)

        union_and_save(data_dir, "mens")
        out = data_dir / "derived" / "MSecondaryTourneyCompactResults_combined.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert "SecondaryTourney" in df.columns
        assert df.iloc[0]["SecondaryTourney"] == "NIT"

    def test_no_espn_data_skips(self, data_dir):
        (data_dir / "cbbpy" / "mens_game_info.csv").unlink()
        union_and_save(data_dir, "mens")
        # Should not create output (no data to combine)
        assert not (data_dir / "derived" / "MRegularSeasonCompactResults_combined.csv").exists()
