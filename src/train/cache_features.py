"""Precompute and cache Elo ratings, team states, and per-game features.

Run once after new games are added. All model training/experimentation
then loads from cache instead of rebuilding the Elo loop.

Usage:
  PYTHONPATH=src python -m train.cache_features
  PYTHONPATH=src python -m train.cache_features --seasons 2015-2026

Output (in data/cache/):
  features_M.parquet     — per-game features for all men's regular season games
  features_W.parquet     — per-game features for all women's regular season games
  elo_M.json             — final Elo ratings, game counts for men's
  elo_W.json             — final Elo ratings, game counts for women's
  team_states_M.pkl      — end-of-season team rolling states
  team_states_W.pkl      — end-of-season team rolling states
  conf_elo_means_M.json  — conference Elo means per season
  conf_elo_means_W.json  — conference Elo means per season
  metadata.json          — timestamp, seasons, row counts
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from predict.elo import add_game_counts
from predict.submission import load_data, _build_hfa_dict, _build_location_dict
from train.features import build_features, FEATURE_COLS, _build_barttorvik_lookup


CACHE_DIR = Path("data/cache")


def cache_gender(data, gender, seasons=None):
    """Build and cache features for one gender."""
    results_key = "mens_results" if gender == "M" else "womens_results"
    conf_key = "mens_conf" if gender == "M" else "womens_conf"
    is_mens = gender == "M"

    results = add_game_counts(data[results_key].copy())
    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    detailed = data.get("mens_detailed" if is_mens else "womens_detailed")
    massey_per = data.get("massey_per_system", {}) if is_mens else {}
    massey_avg = data.get("massey_avg", {}) if is_mens else {}
    coach_tenure = data.get("coach_tenure", {}) if is_mens else {}
    coach_changed = data.get("coach_changed", {}) if is_mens else {}
    barttorvik = data.get("barttorvik") if is_mens else None
    odds_lookup = data.get("odds_lookup", {}) if is_mens else {}
    odds_team_avg = data.get("odds_team_avg", {}) if is_mens else {}
    poll_lookup = data.get("poll_lookup", {}) if is_mens else {}
    weeks_ranked = data.get("weeks_ranked", {}) if is_mens else {}
    roster_lookup = data.get("roster_lookup", {}) if is_mens else {}
    seeds = data.get("m_seeds", {}) if is_mens else data.get("w_seeds", {})

    all_seasons = sorted(results["Season"].unique())
    if seasons is None:
        seasons = set(all_seasons)

    print(f"  {gender}: building features for {len(seasons)} seasons ({min(seasons)}-{max(seasons)})...")
    t0 = time.time()

    features_df, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results,
        conferences=data[conf_key],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=seasons,
        detailed_results=detailed,
        massey_per_system=massey_per,
        massey_avg=massey_avg,
        coach_tenure=coach_tenure,
        coach_changed=coach_changed,
        barttorvik=barttorvik,
        odds_lookup=odds_lookup,
        odds_team_avg=odds_team_avg,
        poll_lookup=poll_lookup,
        weeks_ranked=weeks_ranked,
        roster_lookup=roster_lookup,
        seeds=seeds,
    )

    elapsed = time.time() - t0
    print(f"  {gender}: {len(features_df)} game features, {len(elo_ratings)} teams rated ({elapsed:.1f}s)")

    # Save features
    features_df.to_parquet(CACHE_DIR / f"features_{gender}.parquet", index=False)

    # Save Elo ratings and game counts (JSON-serializable)
    elo_json = {str(k): v for k, v in elo_ratings.items()}
    gc_json = {str(k): v for k, v in game_counts.items()}
    with open(CACHE_DIR / f"elo_{gender}.json", "w") as f:
        json.dump({"elo_ratings": elo_json, "game_counts": gc_json}, f)

    # Save team states (pickle — contains deques)
    with open(CACHE_DIR / f"team_states_{gender}.pkl", "wb") as f:
        pickle.dump(team_states, f)

    # Save conference Elo means
    cem_json = {str(k): v for k, v in conf_elo_means.items()}
    with open(CACHE_DIR / f"conf_elo_means_{gender}.json", "w") as f:
        json.dump(cem_json, f)

    # Save barttorvik lookup for matchup_features
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)
    with open(CACHE_DIR / f"barttorvik_lookup_{gender}.pkl", "wb") as f:
        pickle.dump(barttorvik_lookup, f)

    return len(features_df)


def load_cached_features(gender):
    """Load precomputed features from cache.

    Returns (features_df, elo_ratings, game_counts, team_states, conf_elo_means, barttorvik_lookup)
    """
    features_df = pd.read_parquet(CACHE_DIR / f"features_{gender}.parquet")

    with open(CACHE_DIR / f"elo_{gender}.json") as f:
        d = json.load(f)
        elo_ratings = {int(k): v for k, v in d["elo_ratings"].items()}
        game_counts = {int(k): v for k, v in d["game_counts"].items()}

    with open(CACHE_DIR / f"team_states_{gender}.pkl", "rb") as f:
        team_states = pickle.load(f)

    with open(CACHE_DIR / f"conf_elo_means_{gender}.json") as f:
        cem_raw = json.load(f)
        conf_elo_means = {int(k): v for k, v in cem_raw.items()}

    with open(CACHE_DIR / f"barttorvik_lookup_{gender}.pkl", "rb") as f:
        barttorvik_lookup = pickle.load(f)

    return features_df, elo_ratings, game_counts, team_states, conf_elo_means, barttorvik_lookup


def is_cache_fresh():
    """Check if cache exists and return metadata."""
    meta_path = CACHE_DIR / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Precompute and cache Elo features")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seasons", default=None,
                        help="Season range, e.g. '2015-2026'. Default: all available.")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(args.data_dir)

    if args.seasons:
        lo, hi = args.seasons.split("-")
        seasons = set(range(int(lo), int(hi) + 1))
    else:
        seasons = None  # all

    print("Building feature cache...")
    t0 = time.time()
    counts = {}
    for gender in ("M", "W"):
        counts[gender] = cache_gender(data, gender, seasons)

    # Save metadata
    meta = {
        "created": datetime.now().isoformat(),
        "seasons": args.seasons or "all",
        "mens_games": counts["M"],
        "womens_games": counts["W"],
        "feature_cols": FEATURE_COLS,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCache written to {CACHE_DIR}/")
    print(f"  M: {counts['M']} games")
    print(f"  W: {counts['W']} games")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"\nTo use in experiments:")
    print(f"  from train.cache_features import load_cached_features")
    print(f"  features_df, elo, gc, states, cem, bt = load_cached_features('M')")


if __name__ == "__main__":
    main()
