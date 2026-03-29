"""Ablation study: measure impact of each Tier 1 feature group on tournament prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from evaluate.backtest import load_tourney_results
from evaluate.metrics import brier_score
from train.features import build_features, matchup_features, FEATURE_COLS
from train.extended_features import (
    team_season_stats, stat_matchup_features, STAT_COLS,
    load_seeds, seed_matchup_features,
    load_massey_rankings, massey_matchup_features,
)

DATA_DIR = "../data"


def build_extended_tourney_features(data, gender, year, elo_ratings, game_counts,
                                    detailed, seeds, massey_ranks):
    """Build feature dicts for each tournament game, with all feature groups."""
    tourney = load_tourney_results(DATA_DIR, gender, year)
    if tourney.empty:
        return [], []

    # Precompute season stats from detailed results
    season_stats = team_season_stats(detailed, year)

    feature_rows, outcomes = [], []
    for _, game in tourney.iterrows():
        low = min(game["WTeamID"], game["LTeamID"])
        high = max(game["WTeamID"], game["LTeamID"])

        # Base Elo features
        feat = matchup_features(
            elo_ratings.get(low, 1500), elo_ratings.get(high, 1500),
            game_counts, low, high,
        )

        # Detailed box score stats
        stats_low = season_stats.get(low, {})
        stats_high = season_stats.get(high, {})
        feat.update(stat_matchup_features(stats_low, stats_high))

        # Seeds
        feat.update(seed_matchup_features(seeds, year, low, high))

        # Massey
        feat.update(massey_matchup_features(massey_ranks, year, low, high))

        feature_rows.append(feat)
        outcomes.append(1 if game["WTeamID"] == low else 0)

    return feature_rows, outcomes


def build_extended_season_features(features_df, detailed, seeds, massey_ranks, gender):
    """Add extended features to existing season feature rows (for training)."""
    # Precompute season stats for all relevant seasons
    all_seasons = features_df["season"].unique()
    all_stats = {}
    for season in all_seasons:
        all_stats[season] = team_season_stats(detailed, season)

    # We need team IDs — reconstruct from Elo loop
    # Instead, we'll compute season-level averages and join
    # Simpler: compute per-team stats for each season, then for each game
    # we need to know which teams played. Problem: features_df doesn't have team IDs.
    #
    # For tournament prediction (our evaluation), we use build_extended_tourney_features.
    # For training, we need per-game extended features.
    # Solution: return season-level team stats and join during training.
    pass  # We'll handle this below in the ablation loop


def run_ablation():
    data = load_data(DATA_DIR)

    # Load extended data sources
    m_detailed = pd.read_csv(Path(DATA_DIR) / "kaggle" / "MRegularSeasonDetailedResults.csv")
    w_detailed = pd.read_csv(Path(DATA_DIR) / "kaggle" / "WRegularSeasonDetailedResults.csv")

    m_seeds = load_seeds(DATA_DIR, "M")
    w_seeds = load_seeds(DATA_DIR, "W")

    massey_ranks = load_massey_rankings(DATA_DIR)

    # Feature group definitions
    STAT_DIFF_COLS = [f"{c}_diff" for c in STAT_COLS]

    feature_groups = {
        "Elo only (baseline)": FEATURE_COLS,
        "+ Box scores":        FEATURE_COLS + STAT_DIFF_COLS,
        "+ Seeds":             FEATURE_COLS + ["seed_diff"],
        "+ Massey":            FEATURE_COLS + ["massey_rank_diff"],
        "+ Box + Seeds":       FEATURE_COLS + STAT_DIFF_COLS + ["seed_diff"],
        "+ Box + Massey":      FEATURE_COLS + STAT_DIFF_COLS + ["massey_rank_diff"],
        "All Tier 1":          FEATURE_COLS + STAT_DIFF_COLS + ["seed_diff", "massey_rank_diff"],
    }

    # For training: build game-level features with team IDs
    # We need to re-run the Elo loop but also track team IDs per game
    # so we can join box score stats. Let's build training data with all features.

    val_year = 2024  # tune on 2024
    test_year = 2025  # final eval on 2025

    for gender, label in [("M", "Men's"), ("W", "Women's")]:
        results_key = "mens_results" if gender == "M" else "womens_results"
        conf_key = "mens_conf" if gender == "M" else "womens_conf"

        detailed = m_detailed if gender == "M" else w_detailed
        seeds = m_seeds if gender == "M" else w_seeds

        results = data[results_key].copy()
        results = add_game_counts(results)
        hfa_dict = _build_hfa_dict(data["hfa"], gender)
        location_dict = _build_location_dict(data["home_lookup"], gender)

        data[results_key].attrs["data_dir"] = DATA_DIR

        for eval_year, eval_label in [(val_year, "Val"), (test_year, "Test")]:
            print(f"\n{'='*60}")
            print(f" {label} — {eval_year} Tournament ({eval_label})")
            print(f"{'='*60}")

            # Train on seasons < eval_year
            results_up_to = results[results["Season"] <= eval_year]
            conferences = data[conf_key][data[conf_key]["Season"] <= eval_year]

            train_seasons = sorted(results_up_to["Season"].unique())
            train_seasons = [s for s in train_seasons if s < eval_year]

            # Run Elo to get features + ratings
            from train.features import build_features as bf
            features_df, elo_ratings, game_counts = bf(
                results=results_up_to,
                conferences=conferences,
                hfa_dict=hfa_dict,
                location_dict=location_dict,
                seasons=set(train_seasons),
            )

            # Enrich training data with extended features
            # Need team IDs — rebuild from results
            # We'll build a parallel version that tracks teams
            train_rows = _build_training_with_extended(
                features_df, results_up_to, detailed, seeds,
                massey_ranks if gender == "M" else {}, train_seasons,
            )
            train_df = pd.DataFrame(train_rows)

            # Build test features (tournament games)
            test_rows, test_outcomes = build_extended_tourney_features(
                data, gender, eval_year, elo_ratings, game_counts,
                detailed, seeds, massey_ranks if gender == "M" else {},
            )
            if not test_rows:
                print(f"  No tournament data for {eval_year}")
                continue

            test_df = pd.DataFrame(test_rows)
            y_test = np.array(test_outcomes)

            for group_name, cols in feature_groups.items():
                # Skip Massey for women's
                if gender == "W" and "massey_rank_diff" in cols:
                    continue

                available_cols = [c for c in cols if c in train_df.columns]
                X_train = train_df[available_cols].values
                y_train = train_df["win"].values

                X_test = test_df[available_cols].values

                model = XGBClassifier(
                    n_estimators=200, max_depth=3,
                    random_state=42, eval_metric="logloss",
                )
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_test)[:, 1]
                score = brier_score(preds, y_test)
                print(f"  {group_name:<25} Brier {score:.4f}")


def _build_training_with_extended(features_df, results, detailed, seeds, massey_ranks,
                                  train_seasons):
    """Rebuild training rows with extended features by replaying game order."""
    # The features_df has one row per game in train_seasons, in game order.
    # We need to match each row to the actual game to get team IDs.
    # Strategy: iterate train_seasons games in same order as build_features,
    # and zip with features_df rows.

    train_games = results[results["Season"].isin(train_seasons)].copy()
    train_games = train_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    if len(train_games) != len(features_df):
        print(f"  WARNING: game count mismatch: {len(train_games)} vs {len(features_df)}")

    # Precompute season stats
    season_stats_cache = {}
    for season in train_seasons:
        season_stats_cache[season] = team_season_stats(detailed, season)

    rows = []
    for i, (_, game) in enumerate(train_games.iterrows()):
        if i >= len(features_df):
            break

        base = features_df.iloc[i].to_dict()
        season = game["Season"]
        low = min(game["WTeamID"], game["LTeamID"])
        high = max(game["WTeamID"], game["LTeamID"])

        # Box score stats (season-level averages — slight leakage within season
        # since we use full-season averages, but acceptable for training)
        stats = season_stats_cache.get(season, {})
        base.update(stat_matchup_features(stats.get(low, {}), stats.get(high, {})))

        # Seeds
        base.update(seed_matchup_features(seeds, season, low, high))

        # Massey
        base.update(massey_matchup_features(massey_ranks, season, low, high))

        rows.append(base)

    return rows


if __name__ == "__main__":
    run_ablation()
