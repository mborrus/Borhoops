"""Tests for Barttorvik extract and transform."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract.extract_barttorvik import fetch_season, current_season, KEEP_COLS
from transform.transform_barttorvik import map_teams, transform_barttorvik, MANUAL_MAP
from transform.team_utils import (
    load_kaggle_teams, build_name_to_id, fuzzy_match, match_team_name,
)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_trank_csv():
    """Barttorvik CSV with 25 rows (must exceed the 20-team threshold)."""
    n = 25
    return pd.DataFrame({
        "rank": list(range(1, n + 1)),
        "team": [f"Team{i}" for i in range(n)],
        "conf": ["ACC"] * n,
        "record": ["20-10"] * n,
        "adjoe": [110.0 + i for i in range(n)],
        "adjde": [90.0 + i for i in range(n)],
        "barthag": [0.90 - i * 0.01 for i in range(n)],
        "adjt": [68.0] * n,
        "oe Rank": list(range(1, n + 1)),
        "de Rank": list(range(1, n + 1)),
        "sos": [5.0] * n,
        "ncsos": [2.0] * n,
        "consos": [3.0] * n,
        "WAB": [4.0] * n,
        "WAB Rk": list(range(1, n + 1)),
    })


@pytest.fixture
def data_dir(tmp_path):
    """Minimal data dir with Kaggle MTeams and Barttorvik CSVs."""
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()

    teams = pd.DataFrame({
        "TeamID": [1181, 1163, 1211, 1242],
        "TeamName": ["Duke", "UConn", "Gonzaga", "Kentucky"],
        "FirstD1Season": [1985, 1985, 1985, 1985],
        "LastD1Season": [2026, 2026, 2026, 2026],
    })
    teams.to_csv(kaggle_dir / "MTeams.csv", index=False)

    barttorvik_dir = tmp_path / "barttorvik"
    barttorvik_dir.mkdir()

    df = pd.DataFrame({
        "Season": [2025, 2025, 2025],
        "Team": ["Duke", "Connecticut", "Gonzaga"],
        "Conf": ["ACC", "BE", "WCC"],
        "AdjO": [120.5, 118.2, 116.0],
        "AdjD": [88.1, 90.3, 92.5],
        "Barthag": [0.98, 0.95, 0.91],
        "AdjT": [70.0, 68.5, 71.2],
        "Rank": [1, 2, 3],
        "SOS": [8.5, 7.2, 5.1],
        "NonConSOS": [3.1, 2.8, 1.5],
        "ConSOS": [5.4, 4.4, 3.6],
        "WAB": [8.0, 6.5, 5.0],
    })
    df.to_csv(barttorvik_dir / "trank_2025.csv", index=False)

    return tmp_path


# ── team_utils tests ─────────────────────────────────────


class TestTeamUtils:
    def test_load_kaggle_teams(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        assert len(teams) == 4
        assert "TeamName" in teams.columns

    def test_build_name_to_id(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        name_to_id = build_name_to_id(teams)
        assert name_to_id["Duke"] == 1181
        assert name_to_id["UConn"] == 1163

    def test_fuzzy_match_exact(self):
        choices = ["Duke", "Kentucky", "Gonzaga"]
        assert fuzzy_match("Duke", choices) == "Duke"

    def test_fuzzy_match_close(self):
        choices = ["Duke", "Kentucky", "Gonzaga"]
        assert fuzzy_match("Gonzag", choices, threshold=70) == "Gonzaga"

    def test_fuzzy_match_no_match(self):
        choices = ["Duke", "Kentucky", "Gonzaga"]
        assert fuzzy_match("XYZXYZ", choices) is None

    def test_match_team_name_manual_override(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        name_to_id = build_name_to_id(teams)
        manual = {"Connecticut": "UConn"}
        tid = match_team_name("Connecticut", name_to_id, manual)
        assert tid == 1163

    def test_match_team_name_exact(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        name_to_id = build_name_to_id(teams)
        assert match_team_name("Duke", name_to_id) == 1181

    def test_match_team_name_no_match(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        name_to_id = build_name_to_id(teams)
        assert match_team_name("XYZXYZ", name_to_id) is None


# ── Extract tests ────────────────────────────────────────


class TestExtractBarttorvik:
    def test_current_season(self):
        season = current_season()
        assert isinstance(season, int)
        assert 2020 <= season <= 2030

    @patch("extract.extract_barttorvik.requests.get")
    def test_fetch_season_success(self, mock_get, sample_trank_csv):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_trank_csv.to_csv(index=False)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = fetch_season(2025)
        assert df is not None
        assert len(df) == 25
        assert "Season" in df.columns
        assert df["Season"].iloc[0] == 2025
        assert "Team" in df.columns
        assert "AdjO" in df.columns

    @patch("extract.extract_barttorvik.requests.get")
    def test_fetch_season_too_few_teams(self, mock_get):
        """Seasons with <20 teams are partial data and should be skipped."""
        small_df = pd.DataFrame({
            "rank": [1], "team": ["Duke"], "conf": ["ACC"],
            "adjoe": [120.0], "adjde": [88.0], "barthag": [0.98],
        })
        mock_resp = MagicMock()
        mock_resp.text = small_df.to_csv(index=False)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = fetch_season(2005)
        assert df is None

    @patch("extract.extract_barttorvik.requests.get")
    def test_fetch_season_http_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("404")
        df = fetch_season(1999)
        assert df is None


# ── Transform tests ──────────────────────────────────────


class TestTransformBarttorvik:
    def test_map_teams_manual_override(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        name_to_id = build_name_to_id(teams)

        df = pd.DataFrame({"Team": ["Connecticut", "Duke"]})
        result = map_teams(df, name_to_id)
        assert result["TeamID"].iloc[0] == 1163  # Connecticut → UConn
        assert result["TeamID"].iloc[1] == 1181   # Duke exact match

    def test_map_teams_unmatched(self, data_dir):
        teams = load_kaggle_teams(data_dir)
        name_to_id = build_name_to_id(teams)

        df = pd.DataFrame({"Team": ["Nonexistent University"]})
        result = map_teams(df, name_to_id)
        assert pd.isna(result["TeamID"].iloc[0])

    def test_transform_barttorvik_full(self, data_dir):
        ok = transform_barttorvik(data_dir)
        assert ok is True

        out = pd.read_csv(data_dir / "derived" / "barttorvik_ratings.csv")
        assert len(out) >= 2  # Duke and Gonzaga match exactly
        assert "TeamID" in out.columns
        assert "Season" in out.columns
        assert "AdjO" in out.columns
        assert "Barthag" in out.columns

    def test_transform_barttorvik_no_dir(self, tmp_path):
        ok = transform_barttorvik(tmp_path)
        assert ok is False

    def test_transform_barttorvik_empty_dir(self, tmp_path):
        (tmp_path / "kaggle").mkdir()
        (tmp_path / "barttorvik").mkdir()
        ok = transform_barttorvik(tmp_path)
        assert ok is False

    def test_output_columns(self, data_dir):
        transform_barttorvik(data_dir)
        out = pd.read_csv(data_dir / "derived" / "barttorvik_ratings.csv")
        expected = {"Season", "TeamID", "Team", "Conf", "AdjO", "AdjD", "AdjT",
                    "Barthag", "Rank", "SOS", "NonConSOS", "ConSOS", "WAB"}
        assert expected == set(out.columns)

    def test_manual_map_has_connecticut(self):
        """Key manual override for Barttorvik's most common mismatch."""
        assert MANUAL_MAP["Connecticut"] == "UConn"
        assert MANUAL_MAP["USC"] == "Southern California"
        assert MANUAL_MAP["Ole Miss"] == "Mississippi"
