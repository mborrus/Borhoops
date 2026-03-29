"""Extract ML features from the Elo rating loop."""

import numpy as np
import pandas as pd

from predict.elo import (
    _travel_miles, distance_to_elo_impact, get_advantage,
    k_factor, point_differential_scaler, update_elo, calc_elo_win_tourney,
)

FEATURE_COLS = ["elo_diff", "elo_pred", "home", "day_num", "game_count_avg"]


def build_features(results, conferences, hfa_dict, location_dict,
                   seasons, initial_elo=1500, mean_reversion=0.3,
                   k_start=56, k_end=38, hfa_scalar=26, mov_avg=12):
    """Run the Elo loop and collect per-game feature rows for specified seasons.

    Features are recorded *before* the Elo update for each game — no data leakage.

    Returns:
        (features_df, elo_ratings, game_counts)
        - features_df: DataFrame with FEATURE_COLS + win + season
        - elo_ratings: final {TeamID: elo} after all seasons
        - game_counts: final {TeamID: total_games} after all seasons
    """
    all_seasons = sorted(results["Season"].unique())
    elo_ratings = {}
    game_counts = {}
    rows = []

    for season in all_seasons:
        season_games = results[results["Season"] == season]
        active_teams = set(season_games["WTeamID"]).union(season_games["LTeamID"])

        prev_ratings = elo_ratings.copy()
        elo_ratings = {t: prev_ratings.get(t, initial_elo) for t in active_teams}

        # Reset per-season game counts for K-factor, but keep cumulative for features
        season_counts = {t: 0 for t in active_teams}
        collecting = season in seasons

        for _, row in season_games.iterrows():
            winner, loser = row["WTeamID"], row["LTeamID"]
            w_loc = row["WLoc"]

            # Increment game counts
            season_counts[winner] = season_counts.get(winner, 0) + 1
            season_counts[loser] = season_counts.get(loser, 0) + 1
            game_counts[winner] = game_counts.get(winner, 0) + 1
            game_counts[loser] = game_counts.get(loser, 0) + 1

            # Travel adjustment
            w_miles, l_miles = _travel_miles(winner, loser, w_loc, location_dict)
            winner_elo = elo_ratings[winner] + distance_to_elo_impact(w_miles)
            loser_elo = elo_ratings[loser] + distance_to_elo_impact(l_miles)

            # Home field advantage
            if w_loc == "H":
                winner_elo += get_advantage(winner, hfa_dict, hfa_scalar)
            elif w_loc == "A":
                loser_elo += get_advantage(loser, hfa_dict, hfa_scalar)

            # Record features before update
            if collecting:
                low = min(winner, loser)
                high = max(winner, loser)
                low_base = elo_ratings[low]
                high_base = elo_ratings[high]

                # Home indicator for lower-ID team
                if w_loc == "H":
                    home = 1 if winner == low else -1
                elif w_loc == "A":
                    home = -1 if winner == low else 1
                else:
                    home = 0

                rows.append({
                    "elo_diff": low_base - high_base,
                    "elo_pred": calc_elo_win_tourney(low_base, high_base, boost=1.0),
                    "home": home,
                    "day_num": row["DayNum"],
                    "game_count_avg": (game_counts[low] + game_counts[high]) / 2,
                    "win": 1 if winner == low else 0,
                    "season": season,
                })

            # K-factor and update
            game_num = (season_counts[winner] + season_counts[loser]) / 2
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
            r.TeamID: (
                (1 - mean_reversion) * r.Elo
                + mean_reversion * conf_means.get(r.ConfAbbrev, initial_elo)
            ) * (1 + divergence / initial_elo)
            for r in df.itertuples()
        }

    features_df = pd.DataFrame(rows, columns=FEATURE_COLS + ["win", "season"])
    return features_df, elo_ratings, game_counts


def matchup_features(elo_a, elo_b, game_counts, team_a, team_b):
    """Build a feature dict for a neutral-site tournament matchup.

    team_a should be the lower TeamID. Uses end-of-season Elo ratings.
    """
    return {
        "elo_diff": elo_a - elo_b,
        "elo_pred": calc_elo_win_tourney(elo_a, elo_b, boost=1.0),
        "home": 0,
        "day_num": 136,  # approximate tournament start day
        "game_count_avg": (game_counts.get(team_a, 30) + game_counts.get(team_b, 30)) / 2,
    }
