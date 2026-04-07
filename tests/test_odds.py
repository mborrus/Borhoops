"""Tests for ESPN odds extract and transform."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract.extract_odds import (
    current_season,
    season_date_range,
    fetch_scoreboard,
    fetch_odds,
    scrape_date,
    load_existing_dates,
    save_season,
    extract_season,
    COLUMNS,
)
from transform.transform_odds import (
    load_crosswalk,
    load_day_zero,
    compute_day_num,
    transform_odds,
)


# ── Fixtures ──────────────────────────────────────────────


SCOREBOARD_RESPONSE = {
    "events": [
        {
            "id": "401234567",
            "competitions": [{
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"id": "150", "displayName": "Duke Blue Devils"},
                    },
                    {
                        "homeAway": "away",
                        "team": {"id": "2305", "displayName": "North Carolina Tar Heels"},
                    },
                ],
            }],
        },
        {
            "id": "401234568",
            "competitions": [{
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"id": "2250", "displayName": "Kansas Jayhawks"},
                    },
                    {
                        "homeAway": "away",
                        "team": {"id": "2132", "displayName": "Gonzaga Bulldogs"},
                    },
                ],
            }],
        },
    ]
}

ODDS_RESPONSE = {
    "items": [{
        "provider": {"name": "consensus"},
        "spread": -5.5,
        "overUnder": 148.5,
        "homeTeamOdds": {
            "moneyLine": -220,
            "spreadOdds": -110,
        },
        "awayTeamOdds": {
            "moneyLine": +185,
            "spreadOdds": -110,
        },
    }]
}

ODDS_RESPONSE_EMPTY = {"items": []}


@pytest.fixture
def data_dir(tmp_path):
    """Minimal data dir with crosswalk, seasons, and odds."""
    (tmp_path / "kaggle").mkdir()
    (tmp_path / "derived").mkdir()
    (tmp_path / "odds").mkdir()

    crosswalk = pd.DataFrame({
        "espn_id": [150, 2305, 2250, 2132],
        "kaggle_m_id": [1181, 1314, 1242, 1211],
        "kaggle_w_id": [3181, 3314, 3242, 3211],
        "espn_location": ["Duke", "North Carolina", "Kansas", "Gonzaga"],
        "kaggle_team_name": ["Duke", "North Carolina", "Kansas", "Gonzaga"],
        "match_method": ["spelling"] * 4,
    })
    crosswalk.to_csv(tmp_path / "derived" / "espn_kaggle_crosswalk.csv", index=False)

    seasons = pd.DataFrame({
        "Season": [2025, 2026],
        "DayZero": ["11/04/2024", "11/03/2025"],
        "RegionW": ["W", "W"], "RegionX": ["X", "X"],
        "RegionY": ["Y", "Y"], "RegionZ": ["Z", "Z"],
    })
    seasons.to_csv(tmp_path / "kaggle" / "MSeasons.csv", index=False)

    return tmp_path


@pytest.fixture
def sample_odds_csv(tmp_path):
    """A season CSV with one existing date."""
    odds_dir = tmp_path / "odds"
    odds_dir.mkdir(exist_ok=True)
    df = pd.DataFrame([{
        "Season": 2026,
        "Date": "2025-12-01",
        "ESPN_EventID": "401000001",
        "HomeTeam": "Duke Blue Devils",
        "AwayTeam": "North Carolina Tar Heels",
        "HomeESPN_ID": 150,
        "AwayESPN_ID": 2305,
        "Spread": -3.5,
        "OverUnder": 145.0,
        "HomeML": -180,
        "AwayML": 155,
        "HomeSpreadOdds": -110,
        "AwaySpreadOdds": -110,
        "Provider": "consensus",
    }])
    path = odds_dir / "ncaab_odds_2026.csv"
    df.to_csv(path, index=False)
    return path


# ── Extract tests ────────────────────────────────────────


class TestCurrentSeason:
    def test_returns_int(self):
        assert isinstance(current_season(), int)
        assert 2020 <= current_season() <= 2030

    def test_season_date_range(self):
        start, end = season_date_range(2025)
        assert start == datetime(2024, 11, 1)
        assert end.year == 2025
        assert end.month <= 4


class TestFetchScoreboard:
    @patch("extract.extract_odds.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SCOREBOARD_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        events = fetch_scoreboard(datetime(2025, 12, 1))
        assert len(events) == 2
        assert events[0]["event_id"] == "401234567"
        assert events[0]["home_team"] == "Duke Blue Devils"
        assert events[0]["home_espn_id"] == 150
        assert events[1]["away_team"] == "Gonzaga Bulldogs"

    @patch("extract.extract_odds.requests.get")
    def test_http_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("500")
        events = fetch_scoreboard(datetime(2025, 12, 1))
        assert events == []

    @patch("extract.extract_odds.requests.get")
    def test_empty_events(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        events = fetch_scoreboard(datetime(2025, 7, 15))
        assert events == []


class TestFetchOdds:
    @patch("extract.extract_odds.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = ODDS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        odds = fetch_odds("401234567")
        assert odds is not None
        assert odds["Spread"] == -5.5
        assert odds["OverUnder"] == 148.5
        assert odds["HomeML"] == -220
        assert odds["AwayML"] == 185
        assert odds["Provider"] == "consensus"

    @patch("extract.extract_odds.requests.get")
    def test_empty_items(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = ODDS_RESPONSE_EMPTY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        odds = fetch_odds("401234567")
        assert odds is None

    @patch("extract.extract_odds.requests.get")
    def test_http_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        odds = fetch_odds("401234567")
        assert odds is None


class TestScrapeDate:
    @patch("extract.extract_odds.fetch_odds")
    @patch("extract.extract_odds.fetch_scoreboard")
    @patch("extract.extract_odds.time.sleep")
    def test_scrapes_all_events(self, mock_sleep, mock_sb, mock_odds):
        mock_sb.return_value = [
            {"event_id": "401", "home_team": "Duke", "away_team": "UNC",
             "home_espn_id": 150, "away_espn_id": 2305},
        ]
        mock_odds.return_value = {
            "Spread": -5.5, "OverUnder": 148.5,
            "HomeML": -220, "AwayML": 185,
            "HomeSpreadOdds": -110, "AwaySpreadOdds": -110,
            "Provider": "consensus",
        }

        rows = scrape_date(datetime(2025, 12, 1), 2026)
        assert len(rows) == 1
        assert rows[0]["Season"] == 2026
        assert rows[0]["Date"] == "2025-12-01"
        assert rows[0]["Spread"] == -5.5
        mock_sleep.assert_called()

    @patch("extract.extract_odds.fetch_odds")
    @patch("extract.extract_odds.fetch_scoreboard")
    @patch("extract.extract_odds.time.sleep")
    def test_skips_missing_odds(self, mock_sleep, mock_sb, mock_odds):
        mock_sb.return_value = [
            {"event_id": "401", "home_team": "Duke", "away_team": "UNC",
             "home_espn_id": 150, "away_espn_id": 2305},
        ]
        mock_odds.return_value = None

        rows = scrape_date(datetime(2025, 12, 1), 2026)
        assert len(rows) == 0


class TestIdempotent:
    def test_load_existing_dates(self, sample_odds_csv):
        dates = load_existing_dates(sample_odds_csv)
        assert "2025-12-01" in dates

    def test_load_existing_dates_no_file(self, tmp_path):
        dates = load_existing_dates(tmp_path / "nonexistent.csv")
        assert dates == set()

    @patch("extract.extract_odds.scrape_date")
    def test_extract_season_skips_existing(self, mock_scrape, tmp_path):
        """Dates already in the CSV should not be re-scraped."""
        # Create a CSV with Dec 1 already scraped
        odds_dir = tmp_path / "odds"
        odds_dir.mkdir(exist_ok=True)
        existing = pd.DataFrame([{
            "Season": 2026, "Date": "2025-12-01", "ESPN_EventID": "401",
            "HomeTeam": "A", "AwayTeam": "B", "HomeESPN_ID": 1, "AwayESPN_ID": 2,
            "Spread": -3, "OverUnder": 140, "HomeML": -150, "AwayML": 130,
            "HomeSpreadOdds": -110, "AwaySpreadOdds": -110, "Provider": "test",
        }])
        existing.to_csv(odds_dir / "ncaab_odds_2026.csv", index=False)

        # Mock season_date_range to just cover Dec 1
        with patch("extract.extract_odds.season_date_range") as mock_range:
            mock_range.return_value = (datetime(2025, 12, 1), datetime(2025, 12, 1))
            extract_season(tmp_path, 2026, skip_existing=True)

        # scrape_date should NOT have been called (date already exists)
        mock_scrape.assert_not_called()


class TestSaveSeason:
    def test_creates_new_file(self, tmp_path):
        path = tmp_path / "test.csv"
        rows = [{
            "Season": 2026, "Date": "2025-12-02", "ESPN_EventID": "402",
            "HomeTeam": "A", "AwayTeam": "B", "HomeESPN_ID": 1, "AwayESPN_ID": 2,
            "Spread": -3, "OverUnder": 140, "HomeML": -150, "AwayML": 130,
            "HomeSpreadOdds": -110, "AwaySpreadOdds": -110, "Provider": "test",
        }]
        save_season(rows, path)
        df = pd.read_csv(path)
        assert len(df) == 1
        assert list(df.columns) == COLUMNS

    def test_appends_to_existing(self, sample_odds_csv):
        rows = [{
            "Season": 2026, "Date": "2025-12-02", "ESPN_EventID": "402",
            "HomeTeam": "Kansas Jayhawks", "AwayTeam": "Gonzaga Bulldogs",
            "HomeESPN_ID": 2250, "AwayESPN_ID": 2132,
            "Spread": -2, "OverUnder": 150, "HomeML": -140, "AwayML": 120,
            "HomeSpreadOdds": -110, "AwaySpreadOdds": -110, "Provider": "consensus",
        }]
        save_season(rows, sample_odds_csv)
        df = pd.read_csv(sample_odds_csv)
        assert len(df) == 2
        assert set(df["Date"]) == {"2025-12-01", "2025-12-02"}

    def test_deduplicates(self, sample_odds_csv):
        """Re-saving the same event should not create duplicates."""
        rows = [{
            "Season": 2026, "Date": "2025-12-01", "ESPN_EventID": "401000001",
            "HomeTeam": "Duke Blue Devils", "AwayTeam": "North Carolina Tar Heels",
            "HomeESPN_ID": 150, "AwayESPN_ID": 2305,
            "Spread": -4.0, "OverUnder": 146.0,
            "HomeML": -190, "AwayML": 165,
            "HomeSpreadOdds": -110, "AwaySpreadOdds": -110,
            "Provider": "consensus",
        }]
        save_season(rows, sample_odds_csv)
        df = pd.read_csv(sample_odds_csv)
        assert len(df) == 1
        # Should keep the newer version (keep="last")
        assert df["Spread"].iloc[0] == -4.0


# ── Transform tests ──────────────────────────────────────


class TestTransformOdds:
    def test_load_crosswalk(self, data_dir):
        lookup = load_crosswalk(data_dir)
        assert lookup[150] == 1181  # Duke
        assert lookup[2305] == 1314  # UNC

    def test_load_day_zero(self, data_dir):
        dz = load_day_zero(data_dir)
        assert 2025 in dz
        assert 2026 in dz
        assert dz[2026] == datetime(2025, 11, 3)

    def test_compute_day_num(self, data_dir):
        dz = load_day_zero(data_dir)
        # Dec 1, 2025 is 28 days after Nov 3, 2025 (DayZero for 2026 season)
        result = compute_day_num("2025-12-01", dz)
        assert result == 28

    def test_compute_day_num_missing_season(self, data_dir):
        dz = load_day_zero(data_dir)
        result = compute_day_num("2010-12-01", dz)
        assert result is None

    def test_transform_odds_full(self, data_dir):
        # Create an odds CSV
        odds_df = pd.DataFrame([{
            "Season": 2026, "Date": "2025-12-01", "ESPN_EventID": "401234567",
            "HomeTeam": "Duke Blue Devils", "AwayTeam": "North Carolina Tar Heels",
            "HomeESPN_ID": 150, "AwayESPN_ID": 2305,
            "Spread": -5.5, "OverUnder": 148.5,
            "HomeML": -220, "AwayML": 185,
            "HomeSpreadOdds": -110, "AwaySpreadOdds": -110,
            "Provider": "consensus",
        }])
        odds_df.to_csv(data_dir / "odds" / "ncaab_odds_2026.csv", index=False)

        ok = transform_odds(data_dir)
        assert ok is True

        out = pd.read_csv(data_dir / "derived" / "ncaab_odds.csv")
        assert len(out) == 1
        assert out["HomeTeamID"].iloc[0] == 1181  # Duke
        assert out["AwayTeamID"].iloc[0] == 1314  # UNC
        assert "DayNum" in out.columns
        assert out["DayNum"].iloc[0] == 28

    def test_transform_odds_no_dir(self, tmp_path):
        ok = transform_odds(tmp_path)
        assert ok is False

    def test_transform_odds_empty_dir(self, tmp_path):
        (tmp_path / "odds").mkdir()
        ok = transform_odds(tmp_path)
        assert ok is False

    def test_transform_drops_unmatched(self, data_dir):
        """Rows with ESPN IDs not in the crosswalk should be dropped."""
        odds_df = pd.DataFrame([
            {
                "Season": 2026, "Date": "2025-12-01", "ESPN_EventID": "401",
                "HomeTeam": "Duke", "AwayTeam": "UNC",
                "HomeESPN_ID": 150, "AwayESPN_ID": 2305,
                "Spread": -5.5, "OverUnder": 148.5,
                "HomeML": -220, "AwayML": 185,
                "HomeSpreadOdds": -110, "AwaySpreadOdds": -110,
                "Provider": "consensus",
            },
            {
                "Season": 2026, "Date": "2025-12-01", "ESPN_EventID": "402",
                "HomeTeam": "Unknown", "AwayTeam": "Also Unknown",
                "HomeESPN_ID": 99999, "AwayESPN_ID": 99998,
                "Spread": -1.0, "OverUnder": 130.0,
                "HomeML": -110, "AwayML": -110,
                "HomeSpreadOdds": -110, "AwaySpreadOdds": -110,
                "Provider": "consensus",
            },
        ])
        odds_df.to_csv(data_dir / "odds" / "ncaab_odds_2026.csv", index=False)

        transform_odds(data_dir)
        out = pd.read_csv(data_dir / "derived" / "ncaab_odds.csv")
        assert len(out) == 1  # only the matched row
