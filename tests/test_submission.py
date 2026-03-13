import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict.submission import extract_game_info, load_data, predict


class TestExtractGameInfo:
    def test_basic(self):
        assert extract_game_info("2025_1101_1102") == (2025, 1101, 1102)

    def test_womens(self):
        assert extract_game_info("2024_3101_3202") == (2024, 3101, 3202)


class TestLoadData:
    def test_falls_back_to_kaggle(self, tmp_path):
        """When combined files don't exist, loads from kaggle dir."""
        kaggle = tmp_path / "kaggle"
        kaggle.mkdir()
        derived = tmp_path / "derived"
        derived.mkdir()

        # Minimal CSVs
        pd.DataFrame({"Season": [2024], "DayNum": [1], "WTeamID": [1], "WScore": [70],
                       "LTeamID": [2], "LScore": [60], "WLoc": ["H"], "NumOT": [0]}
                      ).to_csv(kaggle / "MRegularSeasonCompactResults.csv", index=False)
        pd.DataFrame({"Season": [2024], "DayNum": [1], "WTeamID": [3], "WScore": [65],
                       "LTeamID": [4], "LScore": [55], "WLoc": ["H"], "NumOT": [0]}
                      ).to_csv(kaggle / "WRegularSeasonCompactResults.csv", index=False)
        pd.DataFrame({"TeamID": [1, 2]}).to_csv(kaggle / "MTeams.csv", index=False)
        pd.DataFrame({"TeamID": [3, 4]}).to_csv(kaggle / "WTeams.csv", index=False)
        pd.DataFrame({"Season": [2024], "TeamID": [1], "ConfAbbrev": ["big12"]}
                      ).to_csv(kaggle / "MTeamConferences.csv", index=False)
        pd.DataFrame({"Season": [2024], "TeamID": [3], "ConfAbbrev": ["sec"]}
                      ).to_csv(kaggle / "WTeamConferences.csv", index=False)
        pd.DataFrame({"ID": ["2024_1_2"], "Pred": [0.5]}
                      ).to_csv(kaggle / "SampleSubmissionStage2.csv", index=False)
        pd.DataFrame({"TeamID": [1], "HomeAdvantage": [2.0], "Gender": ["M"]}
                      ).to_csv(derived / "HomeFieldAdvantage.csv", index=False)
        pd.DataFrame({"TeamID": [1], "City": ["X"], "State": ["KS"],
                       "Latitude": [38.0], "Longitude": [-97.0], "gender": ["M"]}
                      ).to_csv(derived / "Home_Lookup.csv", index=False)

        data = load_data(tmp_path)
        assert "mens_results" in data
        assert len(data["mens_results"]) == 1

    def test_prefers_combined(self, tmp_path):
        """When combined files exist, uses them over kaggle."""
        kaggle = tmp_path / "kaggle"
        kaggle.mkdir()
        derived = tmp_path / "derived"
        derived.mkdir()

        # Combined has 2 rows, kaggle has 1
        pd.DataFrame({"Season": [2024, 2024], "DayNum": [1, 2], "WTeamID": [1, 1],
                       "WScore": [70, 75], "LTeamID": [2, 2], "LScore": [60, 65],
                       "WLoc": ["H", "A"], "NumOT": [0, 0]}
                      ).to_csv(derived / "M_combined.csv", index=False)
        pd.DataFrame({"Season": [2024], "DayNum": [1], "WTeamID": [1], "WScore": [70],
                       "LTeamID": [2], "LScore": [60], "WLoc": ["H"], "NumOT": [0]}
                      ).to_csv(kaggle / "MRegularSeasonCompactResults.csv", index=False)
        pd.DataFrame({"Season": [2024], "DayNum": [1], "WTeamID": [3], "WScore": [65],
                       "LTeamID": [4], "LScore": [55], "WLoc": ["H"], "NumOT": [0]}
                      ).to_csv(kaggle / "WRegularSeasonCompactResults.csv", index=False)
        pd.DataFrame({"TeamID": [1, 2]}).to_csv(kaggle / "MTeams.csv", index=False)
        pd.DataFrame({"TeamID": [3, 4]}).to_csv(kaggle / "WTeams.csv", index=False)
        pd.DataFrame({"Season": [2024], "TeamID": [1], "ConfAbbrev": ["big12"]}
                      ).to_csv(kaggle / "MTeamConferences.csv", index=False)
        pd.DataFrame({"Season": [2024], "TeamID": [3], "ConfAbbrev": ["sec"]}
                      ).to_csv(kaggle / "WTeamConferences.csv", index=False)
        pd.DataFrame({"ID": ["2024_1_2"]}).to_csv(kaggle / "SampleSubmissionStage2.csv", index=False)
        pd.DataFrame({"TeamID": [1], "HomeAdvantage": [2.0], "Gender": ["M"]}
                      ).to_csv(derived / "HomeFieldAdvantage.csv", index=False)
        pd.DataFrame({"TeamID": [1], "City": ["X"], "State": ["KS"],
                       "Latitude": [38.0], "Longitude": [-97.0], "gender": ["M"]}
                      ).to_csv(derived / "Home_Lookup.csv", index=False)

        data = load_data(tmp_path)
        assert len(data["mens_results"]) == 2


class TestPredict:
    def test_end_to_end(self, tmp_path):
        """Full predict pipeline with synthetic data."""
        kaggle = tmp_path / "kaggle"
        kaggle.mkdir()
        derived = tmp_path / "derived"
        derived.mkdir()
        output = tmp_path / "output"

        # Mens: 2 teams, 1 game
        pd.DataFrame({"Season": [2024], "DayNum": [1], "WTeamID": [1], "WScore": [80],
                       "LTeamID": [2], "LScore": [60], "WLoc": ["N"], "NumOT": [0]}
                      ).to_csv(kaggle / "MRegularSeasonCompactResults.csv", index=False)
        # Womens: 2 teams, 1 game
        pd.DataFrame({"Season": [2024], "DayNum": [1], "WTeamID": [3], "WScore": [75],
                       "LTeamID": [4], "LScore": [55], "WLoc": ["N"], "NumOT": [0]}
                      ).to_csv(kaggle / "WRegularSeasonCompactResults.csv", index=False)

        pd.DataFrame({"TeamID": [1, 2]}).to_csv(kaggle / "MTeams.csv", index=False)
        pd.DataFrame({"TeamID": [3, 4]}).to_csv(kaggle / "WTeams.csv", index=False)
        pd.DataFrame({"Season": [2024, 2024], "TeamID": [1, 2],
                       "ConfAbbrev": ["big12", "big12"]}
                      ).to_csv(kaggle / "MTeamConferences.csv", index=False)
        pd.DataFrame({"Season": [2024, 2024], "TeamID": [3, 4],
                       "ConfAbbrev": ["sec", "sec"]}
                      ).to_csv(kaggle / "WTeamConferences.csv", index=False)

        # Submission template with 2 matchups
        pd.DataFrame({"ID": ["2024_1_2", "2024_3_4"], "Pred": [0.5, 0.5]}
                      ).to_csv(kaggle / "SampleSubmissionStage2.csv", index=False)

        # HFA and locations (empty for simplicity — will use fallbacks)
        pd.DataFrame({"TeamID": [], "HomeAdvantage": [], "Gender": []}
                      ).to_csv(derived / "HomeFieldAdvantage.csv", index=False)
        pd.DataFrame({"TeamID": [], "City": [], "State": [],
                       "Latitude": [], "Longitude": [], "gender": []}
                      ).to_csv(derived / "Home_Lookup.csv", index=False)

        ok = predict(tmp_path, output)
        assert ok is True
        assert (output / "ComplexEloProbs.csv").exists()

        result = pd.read_csv(output / "ComplexEloProbs.csv")
        assert len(result) == 2
        assert list(result.columns) == ["ID", "Pred"]
        # Team 1 beat team 2, so should be favored
        assert result.loc[result["ID"] == "2024_1_2", "Pred"].values[0] > 0.5
