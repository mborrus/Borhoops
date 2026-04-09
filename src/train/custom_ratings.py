"""Custom rating systems inspired by top Kaggle solutions.

Colley Matrix: bias-free ranking from win/loss graph (3rd place)
SRS: Simple Rating System — iterative margin + opponent strength (3rd, 10th place)

Both compute per-season ratings that can be used as features.

Usage:
  from train.custom_ratings import compute_colley, compute_srs
  colley = compute_colley(results, season)   # {TeamID: rank}
  srs = compute_srs(results, season)         # {TeamID: rating}
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_colley(results, season):
    """Compute Colley Matrix rankings for one season.

    Solves (2I + C)r = 1 + (w - l)/2 where C is the connectivity matrix.
    Returns {TeamID: rank} (lower = better).
    """
    season_games = results[results["Season"] == season]
    teams = sorted(set(season_games["WTeamID"]).union(season_games["LTeamID"]))
    if len(teams) < 3:
        return {}

    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Build connectivity matrix and win/loss records
    C = np.zeros((n, n))
    wins = np.zeros(n)
    losses = np.zeros(n)

    for _, g in season_games.iterrows():
        wi = team_idx[g["WTeamID"]]
        li = team_idx[g["LTeamID"]]
        C[wi, li] += 1
        C[li, wi] += 1
        wins[wi] += 1
        losses[li] += 1

    # Colley equation: (2I + C)r = 1 + (w - l)/2
    total_games = wins + losses
    A = 2 * np.eye(n) + np.diag(total_games) - C  # Actually: 2I + diag(n_i) - C_ij
    # Corrected: A = (2 + n_i) * I - C where n_i = total games for team i
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2 + total_games[i]
        for j in range(n):
            if i != j:
                A[i, j] = -C[i, j]

    b = 1 + (wins - losses) / 2

    try:
        ratings = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return {}

    # Convert to ranks (higher rating = better = lower rank number)
    sorted_idx = np.argsort(-ratings)
    ranks = {}
    for rank, idx in enumerate(sorted_idx, 1):
        ranks[teams[idx]] = rank

    return ranks


def compute_srs(results, season, max_iter=100, tol=1e-6):
    """Compute Simple Rating System for one season.

    Iterative: rating_i = avg_margin_i + mean(opponent_ratings)
    Mean-centered each iteration. Returns {TeamID: rating}.
    """
    season_games = results[results["Season"] == season]
    teams = sorted(set(season_games["WTeamID"]).union(season_games["LTeamID"]))
    if len(teams) < 3:
        return {}

    # Build margin and opponent lists
    margins = defaultdict(list)  # {team: [margin1, margin2, ...]}
    opponents = defaultdict(list)  # {team: [opp1, opp2, ...]}

    for _, g in season_games.iterrows():
        margin = g["WScore"] - g["LScore"]
        margins[g["WTeamID"]].append(margin)
        margins[g["LTeamID"]].append(-margin)
        opponents[g["WTeamID"]].append(g["LTeamID"])
        opponents[g["LTeamID"]].append(g["WTeamID"])

    # Initialize ratings to average margin
    ratings = {}
    for t in teams:
        ratings[t] = np.mean(margins[t]) if margins[t] else 0.0

    # Iterate
    for iteration in range(max_iter):
        new_ratings = {}
        for t in teams:
            avg_margin = np.mean(margins[t]) if margins[t] else 0.0
            avg_opp_rating = np.mean([ratings[o] for o in opponents[t]]) if opponents[t] else 0.0
            new_ratings[t] = avg_margin + avg_opp_rating

        # Mean-center
        mean_rating = np.mean(list(new_ratings.values()))
        new_ratings = {t: r - mean_rating for t, r in new_ratings.items()}

        # Check convergence
        max_change = max(abs(new_ratings[t] - ratings[t]) for t in teams)
        ratings = new_ratings
        if max_change < tol:
            break

    return ratings


def build_colley_lookup(results, seasons=None):
    """Build {(Season, TeamID): colley_rank} for all specified seasons."""
    if seasons is None:
        seasons = sorted(results["Season"].unique())
    lookup = {}
    for season in seasons:
        ranks = compute_colley(results, season)
        for team, rank in ranks.items():
            lookup[(season, team)] = rank
    return lookup


def build_srs_lookup(results, seasons=None):
    """Build {(Season, TeamID): srs_rating} for all specified seasons."""
    if seasons is None:
        seasons = sorted(results["Season"].unique())
    lookup = {}
    for season in seasons:
        ratings = compute_srs(results, season)
        for team, rating in ratings.items():
            lookup[(season, team)] = rating
    return lookup


def colley_matchup_features(colley_lookup, season, team_a, team_b):
    """Colley rank difference for a matchup."""
    a = colley_lookup.get((season, team_a), 175)  # default: mid-pack
    b = colley_lookup.get((season, team_b), 175)
    return {"colley_rank_diff": a - b}


def srs_matchup_features(srs_lookup, season, team_a, team_b):
    """SRS rating difference for a matchup."""
    a = srs_lookup.get((season, team_a), 0.0)
    b = srs_lookup.get((season, team_b), 0.0)
    return {"srs_diff": a - b}


if __name__ == "__main__":
    # Quick test
    from predict.submission import load_data
    from predict.elo import add_game_counts

    data = load_data("data")
    results = add_game_counts(data["mens_results"])

    print("Computing Colley Matrix for 2025...")
    colley = compute_colley(results, 2025)
    top10 = sorted(colley.items(), key=lambda x: x[1])[:10]
    print("Top 10:")
    for team, rank in top10:
        print(f"  #{rank}: TeamID {team}")

    print("\nComputing SRS for 2025...")
    srs = compute_srs(results, 2025)
    top10 = sorted(srs.items(), key=lambda x: -x[1])[:10]
    print("Top 10:")
    for team, rating in top10:
        print(f"  {rating:+.1f}: TeamID {team}")
