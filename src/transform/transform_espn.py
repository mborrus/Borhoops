"""Transform ESPN (CBBpy) game data into Kaggle compact results format.

Reads CBBpy game_info CSVs, maps ESPN IDs → Kaggle IDs via crosswalk,
flips home/away into winner/loser, computes DayNum from DayZero,
then unions with existing Kaggle files into combined outputs.

Usage:
  python src/transform_espn.py
"""

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENDER_MAP = {"mens": "M", "womens": "W"}

NCAA_TOURNEY_NAMES = {
    "NCAA Tournament", "NCAA", "ncaa tournament", "ncaa",
    "NCAA Men's Basketball Tournament", "NCAA Women's Basketball Tournament",
}


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_crosswalk(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "derived" / "espn_kaggle_crosswalk.csv")


def load_day_zero(data_dir: Path, gender: str) -> dict[int, datetime]:
    prefix = GENDER_MAP[gender]
    seasons = pd.read_csv(data_dir / "kaggle" / f"{prefix}Seasons.csv")
    return {
        row.Season: datetime.strptime(row.DayZero, "%m/%d/%Y")
        for row in seasons.itertuples()
    }


def parse_season(game_day: str) -> int:
    """NCAA season: games in May+ belong to next year's season."""
    dt = pd.to_datetime(game_day)
    return dt.year + 1 if dt.month >= 5 else dt.year


def compute_day_num(game_day: str, day_zero_map: dict[int, datetime]) -> int | None:
    dt = pd.to_datetime(game_day)
    season = dt.year + 1 if dt.month >= 5 else dt.year
    dz = day_zero_map.get(season)
    if dz is None:
        return None
    return (dt - dz).days


def categorize_game(is_postseason, tournament: str) -> str:
    tournament = str(tournament).strip() if pd.notna(tournament) else ""
    if not is_postseason:
        return "regular"
    if tournament in NCAA_TOURNEY_NAMES:
        return "ncaa"
    return "secondary"


def transform(data_dir: Path, gender: str) -> pd.DataFrame:
    prefix = GENDER_MAP[gender]
    espn_path = data_dir / "cbbpy" / f"{gender}_game_info.csv"

    if not espn_path.exists():
        warnings.warn(f"No ESPN data at {espn_path}")
        return pd.DataFrame()

    espn = pd.read_csv(espn_path)
    crosswalk = load_crosswalk(data_dir)
    day_zero_map = load_day_zero(data_dir, gender)

    kaggle_id_col = f"kaggle_{GENDER_MAP[gender].lower()}_id"
    id_lookup = crosswalk.set_index("espn_id")[kaggle_id_col].to_dict()

    espn["w_espn_id"] = espn.apply(
        lambda r: r["home_id"] if r["home_win"] else r["away_id"], axis=1
    )
    espn["l_espn_id"] = espn.apply(
        lambda r: r["away_id"] if r["home_win"] else r["home_id"], axis=1
    )

    espn["WTeamID"] = espn["w_espn_id"].map(id_lookup)
    espn["LTeamID"] = espn["l_espn_id"].map(id_lookup)

    missing_w = espn["WTeamID"].isna()
    missing_l = espn["LTeamID"].isna()
    missing = missing_w | missing_l
    if missing.any():
        bad_ids = set(
            espn.loc[missing_w, "w_espn_id"].tolist()
            + espn.loc[missing_l, "l_espn_id"].tolist()
        )
        warnings.warn(
            f"{gender}: {missing.sum()} games skipped — no crosswalk for ESPN IDs: {bad_ids}"
        )
        espn = espn[~missing].copy()

    espn["WTeamID"] = espn["WTeamID"].astype(int)
    espn["LTeamID"] = espn["LTeamID"].astype(int)

    espn["WScore"] = espn.apply(
        lambda r: r["home_score"] if r["home_win"] else r["away_score"], axis=1
    ).astype(int)
    espn["LScore"] = espn.apply(
        lambda r: r["away_score"] if r["home_win"] else r["home_score"], axis=1
    ).astype(int)

    def get_wloc(row):
        if row["is_neutral"]:
            return "N"
        return "H" if row["home_win"] else "A"

    espn["WLoc"] = espn.apply(get_wloc, axis=1)
    espn["Season"] = espn["game_day"].apply(parse_season)
    espn["DayNum"] = espn["game_day"].apply(lambda d: compute_day_num(d, day_zero_map))
    espn["NumOT"] = espn["num_ots"].astype(int)

    missing_daynum = espn["DayNum"].isna()
    if missing_daynum.any():
        bad_seasons = espn.loc[missing_daynum, "Season"].unique().tolist()
        warnings.warn(f"{gender}: {missing_daynum.sum()} games skipped — no DayZero for seasons: {bad_seasons}")
        espn = espn[~missing_daynum].copy()

    espn["DayNum"] = espn["DayNum"].astype(int)
    espn["_category"] = espn.apply(
        lambda r: categorize_game(r["is_postseason"], r.get("tournament", "")), axis=1
    )
    espn["_tournament"] = espn["tournament"].fillna("")

    cols = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT", "_category", "_tournament"]
    return espn[cols].copy()


def union_and_save(data_dir: Path, gender: str):
    """Split transformed ESPN by category, union with Kaggle, dedup, write combined CSVs."""
    prefix = GENDER_MAP[gender]
    transformed = transform(data_dir, gender)

    if transformed.empty:
        print(f"  {gender}: no ESPN data to transform")
        return

    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    dedup_cols = ["Season", "DayNum", "WTeamID", "LTeamID"]
    kaggle_cols = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]

    category_config = {
        "regular": f"{prefix}RegularSeasonCompactResults",
        "ncaa": f"{prefix}NCAATourneyCompactResults",
        "secondary": f"{prefix}SecondaryTourneyCompactResults",
    }

    for category, file_stem in category_config.items():
        espn_slice = transformed[transformed["_category"] == category].copy()
        kaggle_path = data_dir / "kaggle" / f"{file_stem}.csv"

        # Build the Kaggle side
        if kaggle_path.exists():
            kaggle_df = pd.read_csv(kaggle_path)
            kaggle_df["_source"] = "kaggle"
        else:
            kaggle_df = pd.DataFrame(columns=kaggle_cols + ["_source"])

        # Build the ESPN side
        out_cols = list(kaggle_cols)
        if category == "secondary":
            espn_slice["SecondaryTourney"] = espn_slice["_tournament"]
            if "SecondaryTourney" not in kaggle_df.columns:
                kaggle_df["SecondaryTourney"] = ""
            out_cols.append("SecondaryTourney")

        espn_slice = espn_slice[out_cols].copy()
        espn_slice["_source"] = "espn"

        # Stack: Kaggle first so it wins on dedup (keep="first")
        combined = pd.concat([kaggle_df, espn_slice], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedup_cols, keep="first")
        combined = combined.sort_values(["Season", "DayNum"]).reset_index(drop=True)

        if espn_slice.empty and kaggle_df.empty:
            continue

        out_path = derived_dir / f"{file_stem}_combined.csv"
        combined.to_csv(out_path, index=False)
        kaggle_count = (combined["_source"] == "kaggle").sum()
        espn_count = (combined["_source"] == "espn").sum()
        print(f"  {file_stem}: {len(combined)} rows ({kaggle_count} kaggle, {espn_count} espn) → {out_path.name}")


def transform_and_union(data_dir: Path) -> bool:
    for gender in ["mens", "womens"]:
        print(f"\nTransforming {gender}...")
        union_and_save(data_dir, gender)
    return True


if __name__ == "__main__":
    config = load_config()
    data_dir = REPO_ROOT / config["data_dir"]
    transform_and_union(data_dir)
