"""Compute additional features from detailed results, seeds, and Massey ordinals."""

import numpy as np
import pandas as pd
from pathlib import Path


# -- Box score season stats (from DetailedResults) ----------------------------

POSSESSIONS_FTA_COEFF = 0.475  # standard approximation

def _possessions(fga, ora, to, fta):
    """Approximate possessions from box score stats."""
    return fga - ora + to + POSSESSIONS_FTA_COEFF * fta


def team_season_stats(detailed_results, season):
    """Compute per-team offensive and defensive season averages.

    Returns {TeamID: {stat_name: value}} for all teams in the given season.
    Uses the "Four Factors" framework plus efficiency metrics.
    """
    df = detailed_results[detailed_results["Season"] == season].copy()
    if df.empty:
        return {}

    stats = {}

    # Process each team's stats as both winner and loser
    for _, row in df.iterrows():
        for role, opp_role in [("W", "L"), ("L", "W")]:
            team = row[f"{role}TeamID"]
            if team not in stats:
                stats[team] = {k: [] for k in [
                    "off_eff", "def_eff", "efg_pct", "opp_efg_pct",
                    "to_rate", "opp_to_rate", "or_pct", "ft_rate",
                ]}

            # Offensive possessions
            off_poss = _possessions(
                row[f"{role}FGA"], row[f"{role}OR"],
                row[f"{role}TO"], row[f"{role}FTA"]
            )
            # Defensive possessions
            def_poss = _possessions(
                row[f"{opp_role}FGA"], row[f"{opp_role}OR"],
                row[f"{opp_role}TO"], row[f"{opp_role}FTA"]
            )

            if off_poss > 0:
                stats[team]["off_eff"].append(row[f"{role}Score"] / off_poss * 100)
                stats[team]["to_rate"].append(row[f"{role}TO"] / off_poss)
            if def_poss > 0:
                stats[team]["def_eff"].append(row[f"{opp_role}Score"] / def_poss * 100)
                stats[team]["opp_to_rate"].append(row[f"{opp_role}TO"] / def_poss)

            # Effective FG%: (FGM + 0.5 * FGM3) / FGA
            if row[f"{role}FGA"] > 0:
                stats[team]["efg_pct"].append(
                    (row[f"{role}FGM"] + 0.5 * row[f"{role}FGM3"]) / row[f"{role}FGA"]
                )
            if row[f"{opp_role}FGA"] > 0:
                stats[team]["opp_efg_pct"].append(
                    (row[f"{opp_role}FGM"] + 0.5 * row[f"{opp_role}FGM3"]) / row[f"{opp_role}FGA"]
                )

            # Offensive rebound %: OR / (OR + opp DR)
            total_or = row[f"{role}OR"] + row[f"{opp_role}DR"]
            if total_or > 0:
                stats[team]["or_pct"].append(row[f"{role}OR"] / total_or)

            # Free throw rate: FTA / FGA
            if row[f"{role}FGA"] > 0:
                stats[team]["ft_rate"].append(row[f"{role}FTA"] / row[f"{role}FGA"])

    # Average all lists
    return {
        team: {k: np.mean(v) if v else 0.0 for k, v in team_stats.items()}
        for team, team_stats in stats.items()
    }


STAT_COLS = ["off_eff", "def_eff", "efg_pct", "opp_efg_pct",
             "to_rate", "opp_to_rate", "or_pct", "ft_rate"]


def stat_matchup_features(stats_a, stats_b):
    """Difference features: team_a stats minus team_b stats."""
    feats = {}
    for col in STAT_COLS:
        a_val = stats_a.get(col, 0.0)
        b_val = stats_b.get(col, 0.0)
        feats[f"{col}_diff"] = a_val - b_val
    return feats


# -- Tournament seeds ---------------------------------------------------------

def load_seeds(data_dir, gender):
    """Load {(Season, TeamID): seed_num} dict."""
    prefix = "M" if gender == "M" else "W"
    path = Path(data_dir) / "kaggle" / f"{prefix}NCAATourneySeeds.csv"
    df = pd.read_csv(path)
    seeds = {}
    for _, row in df.iterrows():
        seed_str = row["Seed"]
        # Parse "W01" / "W01a" / "W01b" → 1
        seed_num = int(seed_str[1:3])
        seeds[(row["Season"], row["TeamID"])] = seed_num
    return seeds


def seed_matchup_features(seeds, season, team_a, team_b):
    """Seed difference feature. Lower seed = better. Unseeded teams get 17."""
    seed_a = seeds.get((season, team_a), 17)
    seed_b = seeds.get((season, team_b), 17)
    return {"seed_diff": seed_a - seed_b}


# -- Massey ordinals ----------------------------------------------------------

def load_massey_rankings(data_dir, systems=("POM", "MOR", "SAG", "DOK", "COL")):
    """Load end-of-season Massey ordinal ranks for selected systems.

    Returns {(Season, TeamID): avg_rank} — average rank across systems.
    """
    path = Path(data_dir) / "kaggle" / "MMasseyOrdinals.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    df = df[df["SystemName"].isin(systems)]

    # Use latest ranking day per season per system per team
    idx = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].idxmax()
    latest = df.loc[idx]

    # Average across systems
    avg = latest.groupby(["Season", "TeamID"])["OrdinalRank"].mean()
    return avg.to_dict()


def massey_matchup_features(massey_ranks, season, team_a, team_b):
    """Massey rank difference. Lower rank = better. Unknown teams get 200."""
    rank_a = massey_ranks.get((season, team_a), 200)
    rank_b = massey_ranks.get((season, team_b), 200)
    return {"massey_rank_diff": rank_a - rank_b}
