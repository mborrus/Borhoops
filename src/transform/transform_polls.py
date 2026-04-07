"""Transform poll rankings: map ESPN team IDs to Kaggle TeamIDs.

Reads per-season CSVs from data/polls/, joins with the ESPN-Kaggle crosswalk,
and writes a unified file with derived features (rank change, weeks ranked).
"""

from pathlib import Path

import pandas as pd


def load_crosswalk(data_dir: Path) -> dict[int, int]:
    """Load ESPN→Kaggle ID mapping. Returns {espn_id: kaggle_id}."""
    path = data_dir / "derived" / "espn_kaggle_crosswalk.csv"
    df = pd.read_csv(path)
    return dict(zip(df["espn_id"], df["kaggle_m_id"]))


def transform_polls(data_dir: Path) -> bool:
    polls_dir = data_dir / "polls"
    if not polls_dir.exists():
        print("No polls data directory found.")
        return False

    csv_files = sorted(polls_dir.glob("polls_*.csv"))
    if not csv_files:
        print("No poll CSV files found.")
        return False

    crosswalk = load_crosswalk(data_dir)
    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["TeamID"] = df["ESPN_TeamID"].map(crosswalk)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    matched = combined["TeamID"].notna()
    unmatched_ids = combined[~matched]["ESPN_TeamID"].unique()
    if len(unmatched_ids) > 0:
        print(f"  WARNING: {len(unmatched_ids)} ESPN IDs unmatched: {list(unmatched_ids)[:10]}")

    combined = combined[matched].copy()
    combined["TeamID"] = combined["TeamID"].astype(int)

    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "poll_rankings.csv"
    combined.to_csv(out_path, index=False)
    print(f"  {len(combined)} poll entries → {out_path.name}")
    return True


if __name__ == "__main__":
    import yaml
    repo_root = Path(__file__).resolve().parent.parent.parent
    with open(repo_root / "config.yaml") as f:
        config = yaml.safe_load(f)
    data_dir = repo_root / config["data_dir"]
    transform_polls(data_dir)
