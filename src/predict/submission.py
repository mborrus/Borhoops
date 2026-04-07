"""Load data, run the Elo model for both genders, write Kaggle submission CSV."""

import pandas as pd
from pathlib import Path

from predict.elo import run_elo, add_game_counts, calc_elo_win_tourney


def extract_game_info(id_str):
    """Parse 'YEAR_ID1_ID2' → (year, team1, team2)."""
    parts = id_str.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_data(data_dir):
    """Load all input DataFrames needed for the Elo model.

    Prefers combined files from the v2 pipeline (data/derived/*_combined.csv).
    Falls back to raw Kaggle files if combined files don't exist.
    """
    data_dir = Path(data_dir)
    kaggle = data_dir / "kaggle"
    derived = data_dir / "derived"

    # Results — prefer v2 combined output, fall back to Kaggle
    combined_m = derived / "M_combined.csv"
    combined_w = derived / "W_combined.csv"

    if combined_m.exists():
        mens_results = pd.read_csv(combined_m)
    else:
        mens_results = pd.read_csv(kaggle / "MRegularSeasonCompactResults.csv")

    if combined_w.exists():
        womens_results = pd.read_csv(combined_w)
    else:
        womens_results = pd.read_csv(kaggle / "WRegularSeasonCompactResults.csv")

    from train.extended_features import (
        load_seeds, load_massey_rankings, load_coach_data,
        load_odds, load_polls, load_roster,
    )

    m_seeds = load_seeds(data_dir, "M")
    w_seeds = load_seeds(data_dir, "W")
    massey_per_system, massey_avg = load_massey_rankings(data_dir)
    coach_tenure, coach_changed = load_coach_data(data_dir)
    odds_lookup, odds_team_avg = load_odds(data_dir)
    poll_lookup, weeks_ranked = load_polls(data_dir)
    roster_lookup = load_roster(data_dir)

    # DetailedResults — prefer combined, fall back to Kaggle
    combined_m_det = derived / "MRegularSeasonDetailedResults_combined.csv"
    combined_w_det = derived / "WRegularSeasonDetailedResults_combined.csv"

    if combined_m_det.exists():
        mens_detailed = pd.read_csv(combined_m_det)
    elif (kaggle / "MRegularSeasonDetailedResults.csv").exists():
        mens_detailed = pd.read_csv(kaggle / "MRegularSeasonDetailedResults.csv")
    else:
        mens_detailed = pd.DataFrame()

    if combined_w_det.exists():
        womens_detailed = pd.read_csv(combined_w_det)
    elif (kaggle / "WRegularSeasonDetailedResults.csv").exists():
        womens_detailed = pd.read_csv(kaggle / "WRegularSeasonDetailedResults.csv")
    else:
        womens_detailed = pd.DataFrame()

    # Barttorvik T-Rank ratings
    barttorvik_path = derived / "barttorvik_ratings.csv"
    barttorvik = pd.read_csv(barttorvik_path) if barttorvik_path.exists() else pd.DataFrame()

    return {
        "mens_results": mens_results,
        "womens_results": womens_results,
        "mens_detailed": mens_detailed,
        "womens_detailed": womens_detailed,
        "mens_teams": pd.read_csv(kaggle / "MTeams.csv"),
        "womens_teams": pd.read_csv(kaggle / "WTeams.csv"),
        "mens_conf": pd.read_csv(kaggle / "MTeamConferences.csv"),
        "womens_conf": pd.read_csv(kaggle / "WTeamConferences.csv"),
        "submission": pd.read_csv(kaggle / "SampleSubmissionStage2.csv"),
        "hfa": pd.read_csv(derived / "HomeFieldAdvantage.csv"),
        "home_lookup": pd.read_csv(derived / "Home_Lookup.csv"),
        "m_seeds": m_seeds,
        "w_seeds": w_seeds,
        "massey_per_system": massey_per_system,
        "massey_avg": massey_avg,
        "coach_tenure": coach_tenure,
        "coach_changed": coach_changed,
        "barttorvik": barttorvik,
        "odds_lookup": odds_lookup,
        "odds_team_avg": odds_team_avg,
        "poll_lookup": poll_lookup,
        "weeks_ranked": weeks_ranked,
        "roster_lookup": roster_lookup,
    }


def _build_hfa_dict(hfa_df, gender):
    """Build {TeamID: HomeAdvantage} for one gender."""
    filtered = hfa_df[hfa_df["Gender"] == gender]
    return dict(zip(filtered["TeamID"], filtered["HomeAdvantage"]))


def _build_location_dict(home_lookup_df, gender):
    """Build {TeamID: (lat, lon)} for one gender."""
    filtered = home_lookup_df[home_lookup_df["gender"] == gender]
    return {
        row.TeamID: (row.Latitude, row.Longitude)
        for row in filtered.itertuples()
        if pd.notna(row.Latitude) and pd.notna(row.Longitude)
    }


def predict(data_dir, output_dir):
    """Full pipeline: load → run_elo for M+W → calc predictions → write CSV."""
    data = load_data(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_elo = {}

    for gender, results_key, conf_key in [
        ("M", "mens_results", "mens_conf"),
        ("W", "womens_results", "womens_conf"),
    ]:
        results = add_game_counts(data[results_key])
        hfa_dict = _build_hfa_dict(data["hfa"], gender)
        location_dict = _build_location_dict(data["home_lookup"], gender)

        elo = run_elo(
            results=results,
            conferences=data[conf_key],
            hfa_dict=hfa_dict,
            location_dict=location_dict,
        )
        all_elo.update(elo)
        print(f"{gender}: {len(elo)} teams rated")

    # Build submission
    submission = data["submission"].copy()
    submission[["Season", "TeamID1", "TeamID2"]] = (
        submission["ID"].apply(extract_game_info).tolist()
    )

    submission["Pred"] = submission.apply(
        lambda r: calc_elo_win_tourney(
            all_elo.get(r["TeamID1"], 1500),
            all_elo.get(r["TeamID2"], 1500),
        ),
        axis=1,
    )

    # Check for missing teams
    missing = submission[
        (~submission["TeamID1"].isin(all_elo)) | (~submission["TeamID2"].isin(all_elo))
    ]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} matchups have teams with no Elo rating (using 1500)")

    output = submission[["ID", "Pred"]]
    output.to_csv(output_dir / "ComplexEloProbs.csv", index=False)
    print(f"Wrote {len(output)} predictions to {output_dir / 'ComplexEloProbs.csv'}")
    return True
