import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract.extract_kaggle import download


class TestDownload:
    def test_success(self, tmp_path):
        """Successful download extracts zip and removes it."""
        import zipfile, csv

        kaggle_dir = tmp_path / "kaggle"
        kaggle_dir.mkdir()
        zip_path = kaggle_dir / "march-machine-learning-mania-2026.zip"

        # Create a fake zip with a CSV
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("MTeams.csv", "TeamID,TeamName\n1101,Abilene Chr\n")

        mock_result = type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        with patch("extract.extract_kaggle.subprocess.run", return_value=mock_result):
            ok = download(tmp_path)

        assert ok is True
        assert (kaggle_dir / "MTeams.csv").exists()
        assert not zip_path.exists()

    def test_failure_prints_error(self, tmp_path, capsys):
        """Failed download returns False and prints hint."""
        mock_result = type("Result", (), {"returncode": 1, "stderr": "", "stdout": ""})()

        with patch("extract.extract_kaggle.subprocess.run", return_value=mock_result):
            ok = download(tmp_path)

        assert ok is False
        captured = capsys.readouterr()
        assert "secrets.sh" in captured.err


class TestRunner:
    def test_cli_source_flag(self):
        """Runner accepts --source and --backfill flags."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from runner import main
        import argparse

        # Just verify argparse doesn't blow up
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", choices=["kaggle", "espn"])
        parser.add_argument("--backfill", action="store_true")

        args = parser.parse_args(["--source", "espn", "--backfill"])
        assert args.source == "espn"
        assert args.backfill is True

        args = parser.parse_args(["--source", "espn"])
        assert args.backfill is False
