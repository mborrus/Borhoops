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

    return {
        "mens_results": mens_results,
        "womens_results": womens_results,
        "mens_teams": pd.read_csv(kaggle / "MTeams.csv"),
        "womens_teams": pd.read_csv(kaggle / "WTeams.csv"),
        "mens_conf": pd.read_csv(kaggle / "MTeamConferences.csv"),
        "womens_conf": pd.read_csv(kaggle / "WTeamConferences.csv"),
        "submission": pd.read_csv(kaggle / "SampleSubmissionStage2.csv"),
        "hfa": pd.read_csv(derived / "HomeFieldAdvantage.csv"),
        "home_lookup": pd.read_csv(derived / "Home_Lookup.csv"),
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
