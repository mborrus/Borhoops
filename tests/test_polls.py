"""Tests for polls extract and transform."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract.extract_polls import fetch_poll, _extract_espn_id, current_season, COLUMNS
from transform.transform_polls import transform_polls


class TestExtractPolls:
    def test_current_season(self):
        season = current_season()
        assert isinstance(season, int)
        assert 2020 <= season <= 2030

    def test_extract_espn_id(self):
        url = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/2509?lang=en"
        assert _extract_espn_id(url) == 2509

    def test_extract_espn_id_no_match(self):
        assert _extract_espn_id("http://example.com/no-team-here") is None

    @patch("extract.extract_polls.requests.get")
    def test_fetch_poll_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ranks": [
                {
                    "current": 1,
                    "previous": 2,
                    "points": 1500,
                    "firstPlaceVotes": 40,
                    "team": {"$ref": "http://example.com/teams/150?lang=en"},
                },
                {
                    "current": 2,
                    "previous": 1,
                    "points": 1400,
                    "firstPlaceVotes": 20,
                    "team": {"$ref": "http://example.com/teams/248?lang=en"},
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        rows = fetch_poll(2025, 10, poll_id=1)
        assert len(rows) == 2
        assert rows[0]["Rank"] == 1
        assert rows[0]["ESPN_TeamID"] == 150
        assert rows[0]["Poll"] == "AP"
        assert rows[0]["Season"] == 2025
        assert rows[0]["Week"] == 10

    @patch("extract.extract_polls.requests.get")
    def test_fetch_poll_coaches(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ranks": [{"current": 1, "team": {"$ref": "http://x.com/teams/99"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        rows = fetch_poll(2025, 5, poll_id=2)
        assert rows[0]["Poll"] == "Coaches"

    @patch("extract.extract_polls.requests.get")
    def test_fetch_poll_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ranks": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        rows = fetch_poll(2025, 99)
        assert rows == []

    @patch("extract.extract_polls.requests.get")
    def test_fetch_poll_http_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("404")
        rows = fetch_poll(2025, 1)
        assert rows == []


class TestTransformPolls:
    def test_transform_polls(self, tmp_path):
        (tmp_path / "polls").mkdir()
        (tmp_path / "derived").mkdir()

        polls = pd.DataFrame({
            "Season": [2025, 2025],
            "Week": [1, 1],
            "Poll": ["AP", "AP"],
            "Rank": [1, 2],
            "ESPN_TeamID": [150, 248],
            "Previous": [2, 1],
            "Points": [1500, 1400],
            "FirstPlaceVotes": [40, 20],
        })
        polls.to_csv(tmp_path / "polls" / "polls_2025.csv", index=False)

        crosswalk = pd.DataFrame({
            "espn_id": [150, 248],
            "kaggle_m_id": [1181, 1242],
        })
        crosswalk.to_csv(tmp_path / "derived" / "espn_kaggle_crosswalk.csv", index=False)

        ok = transform_polls(tmp_path)
        assert ok is True

        out = pd.read_csv(tmp_path / "derived" / "poll_rankings.csv")
        assert len(out) == 2
        assert "TeamID" in out.columns
        assert out["TeamID"].tolist() == [1181, 1242]

    def test_transform_polls_no_dir(self, tmp_path):
        assert transform_polls(tmp_path) is False

    def test_transform_polls_unmatched(self, tmp_path):
        (tmp_path / "polls").mkdir()
        (tmp_path / "derived").mkdir()

        polls = pd.DataFrame({
            "Season": [2025],
            "Week": [1],
            "Poll": ["AP"],
            "Rank": [1],
            "ESPN_TeamID": [99999],
            "Previous": [None],
            "Points": [1500],
            "FirstPlaceVotes": [40],
        })
        polls.to_csv(tmp_path / "polls" / "polls_2025.csv", index=False)

        crosswalk = pd.DataFrame({"espn_id": [150], "kaggle_m_id": [1181]})
        crosswalk.to_csv(tmp_path / "derived" / "espn_kaggle_crosswalk.csv", index=False)

        ok = transform_polls(tmp_path)
        assert ok is True
        out = pd.read_csv(tmp_path / "derived" / "poll_rankings.csv")
        assert len(out) == 0  # unmatched team filtered out
