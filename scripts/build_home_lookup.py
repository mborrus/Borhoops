"""Build Home_Lookup.csv: each team's home city with lat/lon coordinates.

Finds each team's most frequent home-game city via DuckDB, geocodes with
Nominatim (rate-limited, ~15 min), and writes to data/derived/Home_Lookup.csv.
Includes manual entries for teams with no home game records.

Usage:
    python scripts/build_home_lookup.py
"""

import os
import time
from pathlib import Path

import duckdb as db
import pandas as pd
import yaml
from geopy.geocoders import Nominatim

REPO_ROOT = Path(__file__).resolve().parent.parent

# Manual entries for teams with no home game records in the data
MISSING_MENS = [
    {"TeamID": 1109, "City": "San Diego", "State": "CA"},
    {"TeamID": 1118, "City": "Savannah", "State": "GA"},
    {"TeamID": 1121, "City": "Augusta", "State": "GA"},
    {"TeamID": 1128, "City": "Birmingham", "State": "AL"},
    {"TeamID": 1134, "City": "Brooklyn", "State": "NY"},
    {"TeamID": 1215, "City": "Abilene", "State": "TX"},
    {"TeamID": 1289, "City": "Atlanta", "State": "GA"},
    {"TeamID": 1302, "City": "Chicago", "State": "IL"},
    {"TeamID": 1327, "City": "Oklahoma City", "State": "OK"},
    {"TeamID": 1432, "City": "Utica", "State": "NY"},
    {"TeamID": 1446, "City": "Canyon", "State": "TX"},
]

MISSING_WOMENS = [
    {"TeamID": 3128, "City": "Birmingham", "State": "AL"},
    {"TeamID": 3289, "City": "Atlanta", "State": "GA"},
    {"TeamID": 3445, "City": "Winston-Salem", "State": "NC"},
]


def get_home_locations(data_dir, gender):
    """Find each team's most frequent home-game city from GameCities + Results."""
    cities = db.sql(f"""
        SELECT * FROM '{data_dir}/kaggle/{gender}GameCities.csv'
        JOIN '{data_dir}/kaggle/Cities.csv' USING (CityID)
        JOIN '{data_dir}/kaggle/{gender}RegularSeasonCompactResults.csv'
            USING (Season, DayNum, WTeamID, LTeamID)
        WHERE City NOT IN ('MX', 'PR', 'VI', 'BA')
            AND Season != 2020
    """).to_df()

    home_games = cities.query('WLoc == "H"')
    counts = (
        home_games[["WTeamID", "City", "State"]]
        .drop_duplicates()
        .groupby(["WTeamID", "City", "State"])
        .size()
        .reset_index(name="counts")
    )
    most_frequent = (
        counts.sort_values(["WTeamID", "counts"], ascending=[True, False])
        .drop_duplicates("WTeamID")
        .drop(columns=["counts"])
        .rename(columns={"WTeamID": "TeamID"})
    )
    return most_frequent


def geocode_all(df):
    """Add Latitude/Longitude columns via Nominatim (1 req/sec rate limit)."""
    geolocator = Nominatim(user_agent="borhoops")
    lats, lons = [], []

    for _, row in df.iterrows():
        time.sleep(1)
        print(f"Geocoding {row['City']}, {row['State']}")
        location = geolocator.geocode(f"{row['City']}, {row['State']}", timeout=50)
        if location:
            lats.append(location.latitude)
            lons.append(location.longitude)
        else:
            lats.append(None)
            lons.append(None)

    df["Latitude"] = lats
    df["Longitude"] = lons
    return df


def main():
    with open(REPO_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    data_dir = REPO_ROOT / config["data_dir"]
    output_path = data_dir / "derived" / "Home_Lookup.csv"

    if output_path.exists():
        print(f"Already exists: {output_path}")
        print("Delete it to rebuild.")
        return

    # Build lookup for both genders
    m_lookup = get_home_locations(data_dir, "M")
    m_lookup = pd.concat([m_lookup, pd.DataFrame(MISSING_MENS)], ignore_index=True)
    m_lookup["gender"] = "M"

    w_lookup = get_home_locations(data_dir, "W")
    w_lookup = pd.concat([w_lookup, pd.DataFrame(MISSING_WOMENS)], ignore_index=True)
    w_lookup["gender"] = "W"

    combined = pd.concat([m_lookup, w_lookup], ignore_index=True)
    combined = geocode_all(combined)

    os.makedirs(output_path.parent, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} entries to {output_path}")


if __name__ == "__main__":
    main()
