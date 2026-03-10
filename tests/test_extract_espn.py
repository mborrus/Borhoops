import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract_espn import get_season_gap, get_date_gap, save_to_cbbpy


@pytest.fixture
def data_dir(tmp_path):
    """Create a minimal data dir with Kaggle CSVs."""
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()

    results = pd.DataFrame({
        "Season": [2024, 2024, 2025, 2025],
        "DayNum": [10, 20, 50, 118],
        "WTeamID": [1101, 1102, 1101, 1102],
        "WScore": [70, 80, 75, 85],
        "LTeamID": [1103, 1104, 1103, 1104],
        "LScore": [60, 65, 70, 80],
        "WLoc": ["H", "A", "H", "N"],
        "NumOT": [0, 0, 0, 0],
    })
    results.to_csv(kaggle_dir / "MRegularSeasonCompactResults.csv", index=False)

    seasons = pd.DataFrame({
        "Season": [2024, 2025],
        "DayZero": ["11/14/2024", "11/14/2025"],
        "RegionW": ["W", "W"],
        "RegionX": ["X", "X"],
        "RegionY": ["Y", "Y"],
        "RegionZ": ["Z", "Z"],
    })
    seasons.to_csv(kaggle_dir / "MSeasons.csv", index=False)

    return tmp_path


class TestGetSeasonGap:
    def test_no_gap_when_current_season_covered(self, data_dir):
        """If Kaggle has the current season, gap is empty."""
        gap = get_season_gap(data_dir, _now=datetime(2025, 4, 15))
        assert gap == []

    def test_one_season_gap(self, data_dir):
        """If Kaggle ends at 2025 and we're in the 2026 season, gap is [2026]."""
        gap = get_season_gap(data_dir, _now=datetime(2026, 2, 1))
        assert gap == [2026]

    def test_multi_season_gap(self, data_dir):
        """If Kaggle is multiple years behind, returns all missing seasons."""
        gap = get_season_gap(data_dir, _now=datetime(2028, 1, 15))
        assert gap == [2026, 2027, 2028]

    def test_may_boundary(self, data_dir):
        """In May, the next season has started."""
        gap = get_season_gap(data_dir, _now=datetime(2026, 5, 1))
        assert gap == [2026, 2027]


class TestGetDateGap:
    def test_returns_correct_range(self, data_dir):
        """Gap starts day after last Kaggle game, ends yesterday."""
        start, end, season = get_date_gap(data_dir, _now=datetime(2026, 3, 20))
        # DayZero=11/14/2025, last DayNum=118 → 2026-03-12
        # So start = 2026-03-13, end = 2026-03-19
        assert start == "2026-03-13"
        assert end == "2026-03-19"
        assert season == 2025

    def test_no_gap_when_up_to_date(self, data_dir):
        """If Kaggle data is from yesterday, nothing to scrape."""
        # DayZero=11/14/2025 + 118 days = 2026-03-12
        # Set "today" to 2026-03-13 so yesterday = 2026-03-12 = last kaggle date
        start, end, season = get_date_gap(data_dir, _now=datetime(2026, 3, 13))
        assert start is None
        assert end is None


class TestSaveToCbbpy:
    def test_creates_new_files(self, tmp_path):
        info = pd.DataFrame({"game_id": [1, 2], "score": [70, 80]})
        box = pd.DataFrame({"game_id": [1, 1], "player_id": [10, 11], "pts": [20, 15]})

        save_to_cbbpy(info, box, "mens", tmp_path)

        assert (tmp_path / "mens_game_info.csv").exists()
        assert (tmp_path / "mens_game_boxscore.csv").exists()
        saved_info = pd.read_csv(tmp_path / "mens_game_info.csv")
        assert len(saved_info) == 2

    def test_deduplicates_on_append(self, tmp_path):
        existing = pd.DataFrame({"game_id": [1, 2], "score": [70, 80]})
        existing.to_csv(tmp_path / "mens_game_info.csv", index=False)

        existing_box = pd.DataFrame({"game_id": [1], "player_id": [10], "pts": [20]})
        existing_box.to_csv(tmp_path / "mens_game_boxscore.csv", index=False)

        new_info = pd.DataFrame({"game_id": [2, 3], "score": [82, 90]})
        new_box = pd.DataFrame({"game_id": [2, 3], "player_id": [10, 12], "pts": [25, 30]})

        save_to_cbbpy(new_info, new_box, "mens", tmp_path)

        saved = pd.read_csv(tmp_path / "mens_game_info.csv")
        assert len(saved) == 3
        # game_id 2 should have updated score
        assert saved[saved.game_id == 2].score.values[0] == 82
