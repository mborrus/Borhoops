"""Build HomeFieldAdvantage.csv: per-team home field advantage for M and W.

Computes each team's average scoring margin at home vs away (post-2000,
clipped 0-5 points), writes to data/derived/HomeFieldAdvantage.csv.

Usage:
    python scripts/build_hfa.py
"""

import os
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def home_field_advantage(results, season_cutoff=2000, clip_upper=5):
    """Compute per-team HFA from results.

    HFA = mean(HomePoints) - mean(AwayPoints), clipped to [0, clip_upper].
    Only uses seasons >= season_cutoff to avoid noisy early data.
    """
    filtered = results[results["Season"] >= season_cutoff]

    home_pts = (
        filtered[filtered["WLoc"] == "H"]
        .groupby("WTeamID")["WScore"]
        .mean()
        .reset_index()
        .rename(columns={"WTeamID": "TeamID", "WScore": "HomePoints"})
    )

    away_pts = (
        filtered[filtered["WLoc"].isin(["A", "N"])]
        .groupby("WTeamID")["WScore"]
        .mean()
        .reset_index()
        .rename(columns={"WTeamID": "TeamID", "WScore": "AwayPoints"})
    )

    merged = home_pts.merge(away_pts, on="TeamID", how="inner")
    merged["HomeAdvantage"] = (merged["HomePoints"] - merged["AwayPoints"]).clip(
        lower=0, upper=clip_upper
    )
    return merged[["TeamID", "HomeAdvantage"]]


def main():
    with open(REPO_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    data_dir = REPO_ROOT / config["data_dir"]
    output_path = data_dir / "derived" / "HomeFieldAdvantage.csv"

    mens = pd.read_csv(data_dir / "kaggle" / "MRegularSeasonCompactResults.csv")
    womens = pd.read_csv(data_dir / "kaggle" / "WRegularSeasonCompactResults.csv")

    mens_hfa = home_field_advantage(mens)
    mens_hfa["Gender"] = "M"

    womens_hfa = home_field_advantage(womens)
    womens_hfa["Gender"] = "W"

    combined = pd.concat([mens_hfa, womens_hfa], ignore_index=True)

    os.makedirs(output_path.parent, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} entries to {output_path}")


if __name__ == "__main__":
    main()
