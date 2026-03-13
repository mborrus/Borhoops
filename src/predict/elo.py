"""Elo rating engine with adjustments for HFA, travel, MOV, and variable K-factor."""

import numpy as np
import pandas as pd
from geopy.distance import distance as geo_distance


# -- Pure adjustment functions ------------------------------------------------

def point_differential_scaler(w_score, l_score, mov_avg=12):
    """K-factor multiplier based on margin of victory.

    Below league-average MOV → 1.0 (no bonus).
    Above → log-scaled multiplier (double the avg ≈ 2x).
    """
    mov = w_score - l_score
    if mov < mov_avg:
        return 1.0
    return 1 + (1 / np.log(mov_avg + 1)) * np.log(abs(mov - mov_avg) + 1)


def k_factor(game_number, k_start=56, k_end=38):
    """Linear decay from k_start to k_end over the first (k_start - k_end) games."""
    if game_number < k_start - k_end:
        return k_start - (game_number - 1)
    return k_end


def distance_to_elo_impact(miles):
    """Convert travel miles to Elo adjustment: 8 * miles^(1/3)."""
    return 8 * (miles ** (1 / 3))


_HFA_LEAGUE_AVG = 2.12698  # hardcoded from notebook; mean of HomeAdvantage column


def get_advantage(team_id, hfa_dict, scalar=26):
    """HFA in Elo points. Falls back to league average if team not found."""
    return hfa_dict.get(team_id, _HFA_LEAGUE_AVG) * scalar


def calc_elo_win_tourney(elo_a, elo_b, boost=1.07):
    """Tournament win probability for team A vs team B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) * boost / 400))


def update_elo(winner_elo, loser_elo, k=30):
    """Standard Elo change (positive = winner gained this many points)."""
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    return k * (1 - expected_win)


# -- DataFrame helpers --------------------------------------------------------

def add_game_counts(results):
    """Add per-team sequential game count columns (WTeam_Game_Count, LTeam_Game_Count)."""
    results = results.copy()
    game_counts = {}
    w_counts = []
    l_counts = []

    for _, row in results.iterrows():
        season = row["Season"]
        for team, counts_list in [(row["WTeamID"], w_counts), (row["LTeamID"], l_counts)]:
            key = (season, team)
            game_counts[key] = game_counts.get(key, 0) + 1
            counts_list.append(game_counts[key])

    results["WTeam_Game_Count"] = w_counts
    results["LTeam_Game_Count"] = l_counts
    return results


# -- Travel distance (fixed row1 bug from notebook) --------------------------

def _travel_miles(w_team, l_team, w_loc, location_dict):
    """Miles traveled by (winner, loser). Returns (0, 0) if coords missing."""
    w_coords = location_dict.get(w_team)
    l_coords = location_dict.get(l_team)
    if w_coords is None or l_coords is None:
        return 0.0, 0.0

    away_miles = geo_distance(w_coords, l_coords).miles

    if w_loc == "H":
        return 0.0, away_miles
    elif w_loc == "A":
        return away_miles, 0.0
    return 0.0, 0.0  # Neutral


# -- Main Elo loop -----------------------------------------------------------

def run_elo(results, conferences, hfa_dict, location_dict,
            initial_elo=1500, mean_reversion=0.3,
            k_start=56, k_end=38, hfa_scalar=26, mov_avg=12):
    """Run Elo ratings across all seasons.

    Args:
        results:       DataFrame with Season, DayNum, WTeamID, WScore, LTeamID,
                        LScore, WLoc, NumOT, WTeam_Game_Count, LTeam_Game_Count
        conferences:   DataFrame with Season, TeamID, ConfAbbrev
        hfa_dict:      {TeamID: HomeAdvantage} from HomeFieldAdvantage.csv
        location_dict: {TeamID: (lat, lon)} from Home_Lookup.csv

    Returns:
        {TeamID: final_elo} after processing all seasons.
    """
    seasons = sorted(results["Season"].unique())
    elo_ratings = {}

    for season in seasons:
        season_games = results[results["Season"] == season]
        active_teams = set(season_games["WTeamID"]).union(season_games["LTeamID"])

        prev_ratings = elo_ratings.copy()
        elo_ratings = {t: prev_ratings.get(t, initial_elo) for t in active_teams}

        print(f"Running season {season}")

        for _, row in season_games.iterrows():
            winner, loser = row["WTeamID"], row["LTeamID"]
            w_loc = row["WLoc"]

            # Travel adjustment
            w_miles, l_miles = _travel_miles(winner, loser, w_loc, location_dict)
            winner_elo = elo_ratings[winner] + distance_to_elo_impact(w_miles)
            loser_elo = elo_ratings[loser] + distance_to_elo_impact(l_miles)

            # Home field advantage (additive with travel)
            if w_loc == "H":
                winner_elo += get_advantage(winner, hfa_dict, hfa_scalar)
            elif w_loc == "A":
                loser_elo += get_advantage(loser, hfa_dict, hfa_scalar)

            # K-factor: game-number decay × point differential
            game_num = (row["WTeam_Game_Count"] + row["LTeam_Game_Count"]) / 2
            k = k_factor(game_num, k_start, k_end)
            k_coeff = point_differential_scaler(row["WScore"], row["LScore"], mov_avg)

            elo_change = update_elo(winner_elo, loser_elo, k * k_coeff)
            elo_ratings[winner] += elo_change
            elo_ratings[loser] -= elo_change

        # Conference mean reversion
        df = pd.DataFrame(list(elo_ratings.items()), columns=["TeamID", "Elo"])
        league_mean = df["Elo"].mean()
        divergence = initial_elo - league_mean

        season_conf = conferences[conferences["Season"] == season][["TeamID", "ConfAbbrev"]]
        df = df.merge(season_conf, on="TeamID", how="left")
        df["ConfAbbrev"] = df["ConfAbbrev"].fillna("unknown")

        conf_means = df.groupby("ConfAbbrev")["Elo"].mean().to_dict()

        elo_ratings = {
            row.TeamID: (
                (1 - mean_reversion) * row.Elo
                + mean_reversion * conf_means.get(row.ConfAbbrev, initial_elo)
            ) * (1 + divergence / initial_elo)
            for row in df.itertuples()
        }

        print(f"Mean Elo for season {season}: {league_mean}")
        print(f"divergence from mean: {divergence}")

    return elo_ratings
