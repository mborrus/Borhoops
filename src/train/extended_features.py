"""Compute additional features from detailed results, seeds, Massey ordinals, and coaches."""

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

    for _, row in df.iterrows():
        for role, opp_role in [("W", "L"), ("L", "W")]:
            team = row[f"{role}TeamID"]
            if team not in stats:
                stats[team] = {k: [] for k in [
                    "off_eff", "def_eff", "efg_pct", "opp_efg_pct",
                    "to_rate", "opp_to_rate", "or_pct", "ft_rate",
                ]}

            off_poss = _possessions(
                row[f"{role}FGA"], row[f"{role}OR"],
                row[f"{role}TO"], row[f"{role}FTA"]
            )
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

            if row[f"{role}FGA"] > 0:
                stats[team]["efg_pct"].append(
                    (row[f"{role}FGM"] + 0.5 * row[f"{role}FGM3"]) / row[f"{role}FGA"]
                )
            if row[f"{opp_role}FGA"] > 0:
                stats[team]["opp_efg_pct"].append(
                    (row[f"{opp_role}FGM"] + 0.5 * row[f"{opp_role}FGM3"]) / row[f"{opp_role}FGA"]
                )

            total_or = row[f"{role}OR"] + row[f"{opp_role}DR"]
            if total_or > 0:
                stats[team]["or_pct"].append(row[f"{role}OR"] / total_or)

            if row[f"{role}FGA"] > 0:
                stats[team]["ft_rate"].append(row[f"{role}FTA"] / row[f"{role}FGA"])

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
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    seeds = {}
    for _, row in df.iterrows():
        seed_str = row["Seed"]
        seed_num = int(seed_str[1:3])
        seeds[(row["Season"], row["TeamID"])] = seed_num
    return seeds


def seed_matchup_features(seeds, season, team_a, team_b):
    """Seed difference feature. Lower seed = better. Unseeded teams get 17."""
    seed_a = seeds.get((season, team_a), 17)
    seed_b = seeds.get((season, team_b), 17)
    return {"seed_diff": seed_a - seed_b}


# -- Massey ordinals (per-system) ---------------------------------------------

MASSEY_SYSTEMS = ("POM", "SAG", "MOR", "DOK", "COL")
MASSEY_COLS = [f"massey_{s.lower()}_diff" for s in MASSEY_SYSTEMS] + ["massey_avg_diff"]


def load_massey_rankings(data_dir, systems=MASSEY_SYSTEMS):
    """Load end-of-season Massey ordinal ranks -- per-system and average.

    Returns:
        per_system: {system_name: {(Season, TeamID): rank}}
        avg:        {(Season, TeamID): avg_rank}
    """
    path = Path(data_dir) / "kaggle" / "MMasseyOrdinals.csv"
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    df = df[df["SystemName"].isin(systems)]

    idx = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].idxmax()
    latest = df.loc[idx]

    per_system = {}
    for sys_name in systems:
        sys_df = latest[latest["SystemName"] == sys_name]
        per_system[sys_name] = {
            (row.Season, row.TeamID): row.OrdinalRank
            for row in sys_df.itertuples()
        }

    avg = latest.groupby(["Season", "TeamID"])["OrdinalRank"].mean().to_dict()
    return per_system, avg


def massey_matchup_features(massey_per_system, massey_avg, season, team_a, team_b):
    """Per-system and average Massey rank differences. Unknown teams get NaN."""
    feats = {}
    for sys_name in MASSEY_SYSTEMS:
        ranks = massey_per_system.get(sys_name, {})
        rank_a = ranks.get((season, team_a), np.nan)
        rank_b = ranks.get((season, team_b), np.nan)
        feats[f"massey_{sys_name.lower()}_diff"] = rank_a - rank_b
    rank_a = massey_avg.get((season, team_a), np.nan)
    rank_b = massey_avg.get((season, team_b), np.nan)
    feats["massey_avg_diff"] = rank_a - rank_b
    return feats


# -- Coach tenure -------------------------------------------------------------

def load_coach_data(data_dir):
    """Load coach tenure and change indicators from MTeamCoaches.csv.

    Returns:
        tenure:  {(Season, TeamID): years_with_coach}
        changed: {(Season, TeamID): 1 if new coach this season, else 0}
    """
    path = Path(data_dir) / "kaggle" / "MTeamCoaches.csv"
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    idx = df.groupby(["Season", "TeamID"])["LastDayNum"].idxmax()
    df = df.loc[idx].sort_values(["TeamID", "Season"])

    tenure = {}
    changed = {}
    prev_coach = {}

    for _, row in df.iterrows():
        team, season, coach = row["TeamID"], row["Season"], row["CoachName"]
        prev = prev_coach.get(team)
        if prev and prev[0] == coach:
            years = prev[1] + 1
            is_new = 0
        else:
            years = 1
            is_new = 1 if prev else 0
        tenure[(season, team)] = years
        changed[(season, team)] = is_new
        prev_coach[team] = (coach, years)

    return tenure, changed


def coach_matchup_features(tenure, changed, season, team_a, team_b):
    """Coach tenure diff and coach change diff."""
    return {
        "coach_tenure_diff": tenure.get((season, team_a), 1) - tenure.get((season, team_b), 1),
        "coach_change_diff": changed.get((season, team_a), 0) - changed.get((season, team_b), 0),
    }
