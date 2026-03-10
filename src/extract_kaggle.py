"""Download Kaggle competition data to data/kaggle/.

Requires:
  - KAGGLE_API_TOKEN env var (set via `source secrets.sh`)
  - kaggle CLI installed (`pip install kaggle`)
"""

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "march-machine-learning-mania-2026"


def download(data_dir: Path):
    kaggle_dir = data_dir / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    zip_path = kaggle_dir / f"{COMPETITION}.zip"

    print(f"Downloading {COMPETITION}...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", COMPETITION, "-p", str(kaggle_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "No error message"
        print(f"Kaggle download failed:\n{error}", file=sys.stderr)
        print("Hint: run `source secrets.sh` first", file=sys.stderr)
        return False

    if zip_path.exists():
        print(f"Extracting to {kaggle_dir}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(kaggle_dir)
        zip_path.unlink()

    csv_count = len(list(kaggle_dir.glob("*.csv")))
    print(f"Done — {csv_count} CSVs in {kaggle_dir}")
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    download(repo_root / "data")
