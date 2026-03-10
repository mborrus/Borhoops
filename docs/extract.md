# Data Extraction

## Setup

```bash
source ballenvy/bin/activate
source secrets.sh            # sets KAGGLE_API_TOKEN
pip install -e ./CBBpy       # local modified version
```

## Usage

```bash
python src/runner.py                    # all sources
python src/runner.py --source kaggle    # just kaggle
python src/runner.py --source espn      # just espn
```

## What each source does

| Source | Script | Output | Auth |
|--------|--------|--------|------|
| Kaggle | `src/extract_kaggle.py` | `data/kaggle/` (35 CSVs) | `KAGGLE_API_TOKEN` |
| ESPN | `src/extract_espn.py` | `data/cbbpy/` (game info + box scores) | None |

ESPN automatically detects the gap between Kaggle's latest season and today, then scrapes only what's missing.
