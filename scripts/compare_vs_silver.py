"""Compare Borhoops Elo predictions vs Nate Silver's SBCB Elo for the 2026 tournament.

Generates head-to-head Brier scores, round-by-round breakdowns, and comparison charts.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "kaggle"
OUTPUT = ROOT / "Output"

# ── Load team lookups ────────────────────────────────────────────────────────

m_teams = pd.read_csv(DATA / "MTeams.csv")[["TeamID", "TeamName"]]
w_teams = pd.read_csv(DATA / "WTeams.csv")[["TeamID", "TeamName"]]

m_seeds = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
m_seeds_2026 = m_seeds[m_seeds.Season == 2026].copy()
m_seeds_2026["SeedNum"] = m_seeds_2026.Seed.str.extract(r"(\d+)").astype(int)
m_seeds_2026 = m_seeds_2026.merge(m_teams, on="TeamID")

w_seeds = pd.read_csv(DATA / "WNCAATourneySeeds.csv")
w_seeds_2026 = w_seeds[w_seeds.Season == 2026].copy()
w_seeds_2026["SeedNum"] = w_seeds_2026.Seed.str.extract(r"(\d+)").astype(int)
w_seeds_2026 = w_seeds_2026.merge(w_teams, on="TeamID")


# ── Load Nate Silver Elo ratings ─────────────────────────────────────────────

silver_men = pd.read_csv(Path.home() / "Downloads" / "nate_men.csv")
silver_women = pd.read_csv(Path.home() / "Downloads" / "nate_women.csv")

# Silver name → Kaggle name mapping (only mismatches)
silver_to_kaggle_men = {
    "UConn": "Connecticut",
    "U Miami (FL)": "Miami FL",
    "Miami University (OH)": "Miami OH",
    "St. John's": "St John's",
    "Saint Mary's (CA)": "St Mary's CA",
    "Saint Louis": "St Louis",
    "Iowa St.": "Iowa St",
    "Michigan St.": "Michigan St",
    "Oklahoma St.": "Oklahoma St",
    "Arkansas St.": "Arkansas St",
    "North Dakota St.": "N Dakota St",
    "Kennesaw St.": "Kennesaw",
    "Wright St.": "Wright St",
    "NC State": "NC State",
    "Wichita St.": "Wichita St",
    "Florida St.": "Florida St",
    "Arizona St.": "Arizona St",
    "Oregon St.": "Oregon St",
    "Montana St.": "Montana St",
    "Boise St.": "Boise St",
    "San Diego St.": "San Diego St",
    "McNeese": "McNeese St",
    "SMU": "SMU",
    "UCF": "UCF",
    "VCU": "VCU",
    "BYU": "BYU",
    "LIU": "LIU Brooklyn",
    "Northern Iowa": "Northern Iowa",
    "High Point": "High Point",
    "Queens": "Queens NC",
    "Penn": "Penn",
    "Troy": "Troy",
    "Cal Baptist": "Cal Baptist",
    "California Baptist": "Cal Baptist",
    "Utah St.": "Utah St",
    "Kennesaw St.": "Kennesaw",
    "Hofstra": "Hofstra",
    "Idaho": "Idaho",
    "Furman": "Furman",
    "Siena": "Siena",
    "Howard": "Howard",
    "UMBC": "UMBC",
    "Prairie View A&M": "Prairie View",
    "Tennessee St.": "Tennessee St",
    "Richmond": "Richmond",
    "Santa Clara": "Santa Clara",
    "Lehigh": "Lehigh",
    "Villanova": "Villanova",
    "Hawaii": "Hawaii",
    "Clemson": "Clemson",
    "South Florida": "South Florida",
}

silver_to_kaggle_women = {
    "UConn": "Connecticut",
    "U Miami (FL)": "Miami FL",
    "Miami University (OH)": "Miami OH",
    "Saint Mary's (CA)": "St Mary's CA",
    "Iowa St.": "Iowa St",
    "Michigan St.": "Michigan St",
    "Oklahoma St.": "Oklahoma St",
    "North Dakota St.": "N Dakota St",
    "NC State": "NC State",
    "Florida St.": "Florida St",
    "Arizona St.": "Arizona St",
    "Oregon St.": "Oregon St",
    "Montana St.": "Montana St",
    "Boise St.": "Boise St",
    "San Diego St.": "San Diego St",
    "McNeese": "McNeese St",
    "Arkansas St.": "Arkansas St",
    "Colorado St.": "Colorado St",
    "Southern": "Southern Univ",
    "Cal Baptist": "Cal Baptist",
    "California Baptist": "Cal Baptist",
    "Utah St.": "Utah St",
    "Missouri St.": "Missouri St",
    "Stephen F. Austin": "SF Austin",
    "S Dakota St.": "S Dakota St",
    "South Dakota St.": "S Dakota St",
    "UC San Diego": "UC San Diego",
    "Murray St.": "Murray St",
    "Jacksonville": "Jacksonville",
    "F Dickinson": "F Dickinson",
    "Fairleigh Dickinson": "F Dickinson",
    "Holy Cross": "Holy Cross",
    "WI Green Bay": "WI Green Bay",
    "Green Bay": "WI Green Bay",
    "Rhode Island": "Rhode Island",
    "Western Illinois": "W Illinois",
    "W Illinois": "W Illinois",
    "Col Charleston": "Col Charleston",
    "College of Charleston": "Col Charleston",
    "Virginia Tech": "Virginia Tech",
    "Samford": "Samford",
    "Howard": "Howard",
    "UT San Antonio": "UT San Antonio",
    "UTSA": "UT San Antonio",
    "Wichita St.": "Wichita St",
    "Kennesaw St.": "Kennesaw",
    "Sacramento St.": "Sacramento St",
    "Kansas St.": "Kansas St",
    "Wright St.": "Wright St",
    "Illinois St.": "Illinois St",
    "Fresno St.": "Fresno St",
    "Weber St.": "Weber St",
    "Penn St.": "Penn St",
    "Mississippi St.": "Mississippi St",
    "Murray St.": "Murray St",
    "Ball St.": "Ball St",
    "Kent St.": "Kent St",
    "UMass (Amherst)": "Massachusetts",
    "Tennessee St.": "Tennessee St",
    "SE Louisiana": "Southeastern LA",
    "UT Rio Grande Valley": "UT Rio Grande Vl",
    "Northern Kentucky": "N Kentucky",
    "East Tennessee St.": "E Tennessee St",
    "Central Arkansas": "Cent Arkansas",
    "Eastern Washington": "E Washington",
    "Eastern Kentucky": "E Kentucky",
    "Southern Indiana": "Southern Ind",
    "Jacksonville St.": "Jacksonville St",
    "South Carolina Upstate": "SC Upstate",
    "Eastern Michigan": "E Michigan",
    "Eastern Illinois": "E Illinois",
    "Southeast Missouri St.": "Southeastern MO",
    "North Carolina A&T": "NC A&T",
    "SIU Edwardsville": "SIU Edward",
    "MD Eastern Shore": "MD E Shore",
    "Northern Colorado": "N Colorado",
    "Northwestern St.": "Northwestern LA",
    "UC Irvine": "UC Irvine",
    "UC Santa Barbara": "UC Santa Barbara",
    "UC Riverside": "UC Riverside",
    "UC Davis": "UC Davis",
    "Northern Illinois": "N Illinois",
    "Southern Miss": "Southern Miss",
    "Southern Utah": "Southern Utah",
    "Central Michigan": "Cent Michigan",
    "St. John's": "St John's",
    "Old Dominion": "Old Dominion",
    "West Georgia": "West Georgia",
    "Idaho St.": "Idaho St",
    "Georgia Southern": "Ga Southern",
    "George Washington": "G Washington",
    "George Mason": "George Mason",
    "Saint Joseph's": "St Joseph's PA",
    "Mississippi": "Mississippi",
    "Ole Miss": "Mississippi",
}

def build_kaggle_name_map(teams_df):
    """TeamName (lowercase) → TeamID."""
    return {row.TeamName.lower(): row.TeamID for _, row in teams_df.iterrows()}

m_name_map = build_kaggle_name_map(m_teams)
w_name_map = build_kaggle_name_map(w_teams)


def resolve_silver_to_id(sb_name, silver_map, kaggle_map):
    """Resolve a Silver Bulletin team name to a Kaggle TeamID."""
    # Check explicit mapping
    if sb_name in silver_map:
        kaggle_name = silver_map[sb_name]
    else:
        kaggle_name = sb_name

    key = kaggle_name.lower()
    if key in kaggle_map:
        return kaggle_map[key]

    # Fuzzy: strip periods, apostrophes
    clean = key.replace(".", "").replace("'", "").replace("(", "").replace(")", "")
    for k, v in kaggle_map.items():
        kc = k.replace(".", "").replace("'", "").replace("(", "").replace(")", "")
        if clean == kc:
            return v

    return None


def build_silver_elo_map(silver_df, silver_aliases, kaggle_name_map):
    """Map Silver team names to (TeamID, Elo) pairs."""
    elo_map = {}
    unresolved = []
    for _, row in silver_df.iterrows():
        tid = resolve_silver_to_id(row.sb_name, silver_aliases, kaggle_name_map)
        if tid:
            elo_map[tid] = row.b_xelo_n
        else:
            unresolved.append(row.sb_name)
    if unresolved:
        print(f"  ⚠ Unresolved Silver teams ({len(unresolved)}):")
        for u in unresolved[:15]:
            print(f"    {u}")
        if len(unresolved) > 15:
            print(f"    ... and {len(unresolved) - 15} more")
    return elo_map


print("Mapping Silver team names to Kaggle IDs...")
m_silver_elo = build_silver_elo_map(silver_men, silver_to_kaggle_men, m_name_map)
w_silver_elo = build_silver_elo_map(silver_women, silver_to_kaggle_women, w_name_map)

print(f"  Men's: {len(m_silver_elo)} teams mapped")
print(f"  Women's: {len(w_silver_elo)} teams mapped")


# ── Win probability from Elo ─────────────────────────────────────────────────

def elo_win_prob(elo_a, elo_b, boost=1.0):
    """Standard Elo win probability."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) * boost / 400))


def generate_silver_preds(outcomes_df, silver_elo_map):
    """Generate predictions from Silver's Elo ratings for actual tournament games."""
    preds = []
    for _, row in outcomes_df.iterrows():
        low_id = int(row.ID.split("_")[1])
        high_id = int(row.ID.split("_")[2])
        if low_id in silver_elo_map and high_id in silver_elo_map:
            prob_low = elo_win_prob(silver_elo_map[low_id], silver_elo_map[high_id])
            preds.append(prob_low)
        else:
            preds.append(np.nan)
    return np.array(preds)


# ── Actual game outcomes (same as evaluate_2026.py) ──────────────────────────

m_aliases = {
    "uconn": "connecticut", "miami fl": "miami fl", "miami oh": "miami oh",
    "st. john's": "st john's", "st john's ny": "st john's",
    "north carolina": "north carolina", "unc": "north carolina",
    "cal baptist": "cal baptist", "california baptist": "cal baptist",
    "mcneese state": "mcneese st", "mcneese": "mcneese st",
    "north dakota state": "n dakota st",
    "prairie view a&m": "prairie view", "prairie view": "prairie view",
    "liu": "liu brooklyn", "long island": "liu brooklyn",
    "queens": "queens nc", "high point": "high point",
    "kennesaw state": "kennesaw", "kennesaw": "kennesaw",
    "saint louis": "st louis",
    "st. mary's": "st mary's ca", "south florida": "south florida",
    "miami (oh)": "miami oh",
}

w_aliases = {
    "uconn": "connecticut", "south carolina": "south carolina",
    "uc san diego": "uc san diego", "cal baptist": "cal baptist",
    "california baptist": "cal baptist",
    "college of charleston": "col charleston",
    "holy cross": "holy cross", "wi green bay": "wi green bay",
    "green bay": "wi green bay",
    "missouri st": "missouri st", "missouri state": "missouri st",
    "sf austin": "sf austin", "stephen f. austin": "sf austin",
    "southern": "southern univ", "samford": "samford",
    "s dakota st": "s dakota st", "south dakota state": "s dakota st",
    "f dickinson": "f dickinson", "fairleigh dickinson": "f dickinson",
    "fdu": "f dickinson",
    "virginia tech": "virginia tech", "nc state": "nc state",
    "oklahoma st": "oklahoma st", "oklahoma state": "oklahoma st",
    "w illinois": "w illinois", "western illinois": "w illinois",
    "colorado st": "colorado st", "colorado state": "colorado st",
    "michigan st": "michigan st", "michigan state": "michigan st",
}


def resolve_game_id(name, name_map, aliases):
    key = name.strip().lower()
    if key in aliases:
        key = aliases[key]
    if key in name_map:
        return name_map[key]
    for k, v in name_map.items():
        if key.replace(".", "").replace("'", "") == k.replace(".", "").replace("'", ""):
            return v
    return None


def build_game_outcomes(games_raw, name_map, aliases):
    outcomes = []
    for winner, loser, round_name in games_raw:
        w_id = resolve_game_id(winner, name_map, aliases)
        l_id = resolve_game_id(loser, name_map, aliases)
        if w_id and l_id:
            low, high = min(w_id, l_id), max(w_id, l_id)
            actual = 1.0 if w_id == low else 0.0
            outcomes.append({
                "ID": f"2026_{low}_{high}",
                "actual": actual,
                "round": round_name,
                "winner": winner,
                "loser": loser,
                "winner_id": w_id,
                "loser_id": l_id,
            })
    return pd.DataFrame(outcomes)


# Game results
men_games = [
    ("Howard", "UMBC", "First Four"), ("Miami OH", "SMU", "First Four"),
    ("Prairie View", "Lehigh", "First Four"), ("Texas", "NC State", "First Four"),
    ("Duke", "Siena", "Round of 64"), ("TCU", "Ohio St", "Round of 64"),
    ("St John's", "Northern Iowa", "Round of 64"), ("Kansas", "Cal Baptist", "Round of 64"),
    ("Louisville", "South Florida", "Round of 64"), ("Michigan St", "N Dakota St", "Round of 64"),
    ("UCLA", "UCF", "Round of 64"), ("UConn", "Furman", "Round of 64"),
    ("Florida", "Prairie View", "Round of 64"), ("Iowa", "Clemson", "Round of 64"),
    ("Vanderbilt", "McNeese", "Round of 64"), ("Nebraska", "Troy", "Round of 64"),
    ("VCU", "North Carolina", "Round of 64"), ("Illinois", "Penn", "Round of 64"),
    ("Texas A&M", "St Mary's CA", "Round of 64"), ("Houston", "Idaho", "Round of 64"),
    ("Arizona", "LIU", "Round of 64"), ("Utah St", "Villanova", "Round of 64"),
    ("High Point", "Wisconsin", "Round of 64"), ("Arkansas", "Hawaii", "Round of 64"),
    ("Texas", "BYU", "Round of 64"), ("Gonzaga", "Kennesaw", "Round of 64"),
    ("Miami FL", "Missouri", "Round of 64"), ("Purdue", "Queens", "Round of 64"),
    ("Michigan", "Howard", "Round of 64"), ("St Louis", "Georgia", "Round of 64"),
    ("Texas Tech", "Akron", "Round of 64"), ("Alabama", "Hofstra", "Round of 64"),
    ("Tennessee", "Miami OH", "Round of 64"), ("Virginia", "Wright St", "Round of 64"),
    ("Kentucky", "Santa Clara", "Round of 64"), ("Iowa St", "Tennessee St", "Round of 64"),
    ("Duke", "TCU", "Round of 32"), ("St John's", "Kansas", "Round of 32"),
    ("Michigan St", "Louisville", "Round of 32"), ("UConn", "UCLA", "Round of 32"),
    ("Iowa", "Florida", "Round of 32"), ("Nebraska", "Vanderbilt", "Round of 32"),
    ("Illinois", "VCU", "Round of 32"), ("Houston", "Texas A&M", "Round of 32"),
    ("Arizona", "Utah St", "Round of 32"), ("Arkansas", "High Point", "Round of 32"),
    ("Texas", "Gonzaga", "Round of 32"), ("Purdue", "Miami FL", "Round of 32"),
    ("Michigan", "St Louis", "Round of 32"), ("Alabama", "Texas Tech", "Round of 32"),
    ("Tennessee", "Virginia", "Round of 32"), ("Iowa St", "Kentucky", "Round of 32"),
    ("Duke", "St John's", "Sweet 16"), ("UConn", "Michigan St", "Sweet 16"),
    ("Iowa", "Nebraska", "Sweet 16"), ("Illinois", "Houston", "Sweet 16"),
    ("Arizona", "Arkansas", "Sweet 16"), ("Purdue", "Texas", "Sweet 16"),
    ("Michigan", "Alabama", "Sweet 16"), ("Tennessee", "Iowa St", "Sweet 16"),
    ("UConn", "Duke", "Elite 8"), ("Illinois", "Iowa", "Elite 8"),
    ("Arizona", "Purdue", "Elite 8"), ("Michigan", "Tennessee", "Elite 8"),
    ("UConn", "Illinois", "Final Four"), ("Michigan", "Arizona", "Final Four"),
    ("Michigan", "UConn", "Championship"),
]

women_games = [
    ("Missouri St", "SF Austin", "First Four"), ("Nebraska", "Richmond", "First Four"),
    ("Virginia", "Arizona St", "First Four"), ("Southern", "Samford", "First Four"),
    ("UConn", "UT San Antonio", "Round of 64"), ("Syracuse", "Iowa St", "Round of 64"),
    ("Maryland", "Murray St", "Round of 64"), ("North Carolina", "W Illinois", "Round of 64"),
    ("Notre Dame", "Fairfield", "Round of 64"), ("Ohio St", "Howard", "Round of 64"),
    ("Illinois", "Colorado", "Round of 64"), ("Vanderbilt", "High Point", "Round of 64"),
    ("UCLA", "Cal Baptist", "Round of 64"), ("Oklahoma St", "Princeton", "Round of 64"),
    ("Mississippi", "Gonzaga", "Round of 64"), ("Minnesota", "WI Green Bay", "Round of 64"),
    ("Baylor", "Nebraska", "Round of 64"), ("Duke", "Col Charleston", "Round of 64"),
    ("Texas Tech", "Villanova", "Round of 64"), ("LSU", "Jacksonville", "Round of 64"),
    ("Texas", "Missouri St", "Round of 64"), ("Oregon", "Virginia Tech", "Round of 64"),
    ("Kentucky", "James Madison", "Round of 64"), ("West Virginia", "Miami OH", "Round of 64"),
    ("Alabama", "Rhode Island", "Round of 64"), ("Louisville", "Vermont", "Round of 64"),
    ("NC State", "Tennessee", "Round of 64"), ("Michigan", "Holy Cross", "Round of 64"),
    ("South Carolina", "Southern", "Round of 64"), ("USC", "Clemson", "Round of 64"),
    ("Michigan St", "Colorado St", "Round of 64"), ("Oklahoma", "Idaho", "Round of 64"),
    ("Washington", "S Dakota St", "Round of 64"), ("TCU", "UC San Diego", "Round of 64"),
    ("Virginia", "Georgia", "Round of 64"), ("Iowa", "F Dickinson", "Round of 64"),
    ("UConn", "Syracuse", "Round of 32"), ("North Carolina", "Maryland", "Round of 32"),
    ("Notre Dame", "Ohio St", "Round of 32"), ("Vanderbilt", "Illinois", "Round of 32"),
    ("UCLA", "Oklahoma St", "Round of 32"), ("Minnesota", "Mississippi", "Round of 32"),
    ("Duke", "Baylor", "Round of 32"), ("LSU", "Texas Tech", "Round of 32"),
    ("Texas", "Oregon", "Round of 32"), ("Kentucky", "West Virginia", "Round of 32"),
    ("Louisville", "Alabama", "Round of 32"), ("Michigan", "NC State", "Round of 32"),
    ("South Carolina", "USC", "Round of 32"), ("Oklahoma", "Michigan St", "Round of 32"),
    ("TCU", "Washington", "Round of 32"), ("Virginia", "Iowa", "Round of 32"),
    ("UConn", "North Carolina", "Sweet 16"), ("Notre Dame", "Vanderbilt", "Sweet 16"),
    ("UCLA", "Minnesota", "Sweet 16"), ("Duke", "LSU", "Sweet 16"),
    ("Texas", "Kentucky", "Sweet 16"), ("Michigan", "Louisville", "Sweet 16"),
    ("South Carolina", "Oklahoma", "Sweet 16"), ("TCU", "Virginia", "Sweet 16"),
    ("UConn", "Notre Dame", "Elite 8"), ("UCLA", "Duke", "Elite 8"),
    ("Texas", "Michigan", "Elite 8"), ("South Carolina", "TCU", "Elite 8"),
    ("South Carolina", "UConn", "Final Four"), ("UCLA", "Texas", "Final Four"),
    ("UCLA", "South Carolina", "Championship"),
]

m_outcomes = build_game_outcomes(men_games, m_name_map, m_aliases)
w_outcomes = build_game_outcomes(women_games, w_name_map, w_aliases)

# ── Load Borhoops predictions ────────────────────────────────────────────────

elo_preds = pd.read_csv(OUTPUT / "ComplexEloProbs.csv")

# ── Generate Silver predictions ──────────────────────────────────────────────

m_outcomes["silver_pred"] = generate_silver_preds(m_outcomes, m_silver_elo)
w_outcomes["silver_pred"] = generate_silver_preds(w_outcomes, w_silver_elo)

# Merge Borhoops preds
m_outcomes = m_outcomes.merge(elo_preds.rename(columns={"Pred": "borhoops_pred"}), on="ID", how="left")
w_outcomes = w_outcomes.merge(elo_preds.rename(columns={"Pred": "borhoops_pred"}), on="ID", how="left")

# ── Scoring ──────────────────────────────────────────────────────────────────

def brier(preds, actuals):
    mask = ~np.isnan(preds)
    return np.mean((preds[mask] - actuals[mask]) ** 2)

def log_loss_fn(preds, actuals, eps=1e-15):
    mask = ~np.isnan(preds)
    p = np.clip(preds[mask], eps, 1 - eps)
    a = actuals[mask]
    return -np.mean(a * np.log(p) + (1 - a) * np.log(1 - p))

def correct_picks(preds, actuals):
    mask = ~np.isnan(preds)
    return ((preds[mask] > 0.5) == (actuals[mask] == 1.0)).sum(), mask.sum()


def compare(outcomes, label):
    """Print head-to-head comparison for a tournament."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    actuals = outcomes.actual.values
    bh = outcomes.borhoops_pred.values
    ns = outcomes.silver_pred.values

    bh_brier = brier(bh, actuals)
    ns_brier = brier(ns, actuals)
    bh_ll = log_loss_fn(bh, actuals)
    ns_ll = log_loss_fn(ns, actuals)
    bh_correct, bh_n = correct_picks(bh, actuals)
    ns_correct, ns_n = correct_picks(ns, actuals)

    silver_missing = np.isnan(ns).sum()
    if silver_missing > 0:
        print(f"  (Note: {silver_missing} games missing Silver Elo for one or both teams)")

    print(f"\n  {'Metric':<25} {'Borhoops':>12} {'Silver':>12} {'Diff':>12}")
    print(f"  {'-'*61}")
    print(f"  {'Brier Score':<25} {bh_brier:>12.4f} {ns_brier:>12.4f} {bh_brier - ns_brier:>+12.4f}")
    print(f"  {'Log Loss':<25} {bh_ll:>12.4f} {ns_ll:>12.4f} {bh_ll - ns_ll:>+12.4f}")
    print(f"  {'Correct Picks':<25} {bh_correct:>8}/{bh_n:<3} {ns_correct:>8}/{ns_n:<3} {bh_correct - ns_correct:>+12d}")
    print(f"  {'Accuracy':<25} {bh_correct/bh_n:>11.1%} {ns_correct/ns_n:>11.1%}")

    winner = "Borhoops" if bh_brier < ns_brier else "Silver"
    margin = abs(bh_brier - ns_brier) / max(bh_brier, ns_brier) * 100
    print(f"\n  >> {winner} wins by {margin:.1f}% on Brier score")

    # Round-by-round
    round_order = ["First Four", "Round of 64", "Round of 32", "Sweet 16",
                   "Elite 8", "Final Four", "Championship"]
    print(f"\n  {'Round':<20} {'BH Brier':>10} {'NS Brier':>10} {'BH Corr':>10} {'NS Corr':>10}")
    print(f"  {'-'*60}")
    round_data = []
    for r in round_order:
        rdf = outcomes[outcomes["round"] == r]
        if len(rdf) == 0:
            continue
        a = rdf.actual.values
        bh_r = brier(rdf.borhoops_pred.values, a)
        ns_r = brier(rdf.silver_pred.values, a)
        bh_c, bh_rn = correct_picks(rdf.borhoops_pred.values, a)
        ns_c, ns_rn = correct_picks(rdf.silver_pred.values, a)
        marker = " <" if bh_r < ns_r else " >" if bh_r > ns_r else ""
        print(f"  {r:<20} {bh_r:>10.4f} {ns_r:>10.4f} {bh_c:>6}/{bh_rn:<3} {ns_c:>6}/{ns_rn:<3}{marker}")
        round_data.append({
            "round": r, "bh_brier": bh_r, "ns_brier": ns_r,
            "bh_correct": bh_c, "bh_total": bh_rn,
            "ns_correct": ns_c, "ns_total": ns_rn,
        })

    # Games where models disagree on the favorite
    print(f"\n  Games where models picked different favorites:")
    disagree = outcomes[
        ((outcomes.borhoops_pred > 0.5) != (outcomes.silver_pred > 0.5)) &
        outcomes.silver_pred.notna()
    ].copy()
    if len(disagree) == 0:
        print(f"    (none)")
    else:
        for _, row in disagree.iterrows():
            bh_fav_id = int(row.ID.split("_")[1]) if row.borhoops_pred > 0.5 else int(row.ID.split("_")[2])
            ns_fav_id = int(row.ID.split("_")[1]) if row.silver_pred > 0.5 else int(row.ID.split("_")[2])
            bh_right = bh_fav_id == row.winner_id
            ns_right = ns_fav_id == row.winner_id
            bh_mark = "✓" if bh_right else "✗"
            ns_mark = "✓" if ns_right else "✗"
            # Resolve team names
            bh_fav = row.winner if bh_fav_id == row.winner_id else row.loser
            ns_fav = row.winner if ns_fav_id == row.winner_id else row.loser
            print(f"    {row['round']:20s}  BH: {bh_fav} ({row.borhoops_pred:.1%}) {bh_mark}  "
                  f"NS: {ns_fav} ({row.silver_pred:.1%}) {ns_mark}  "
                  f"→ {row.winner} won")

    return pd.DataFrame(round_data)


# ── Run comparisons ──────────────────────────────────────────────────────────

print("\n" + "█" * 65)
print("  BORHOOPS vs NATE SILVER — 2026 TOURNAMENT")
print("█" * 65)

m_rounds = compare(m_outcomes, "MEN'S TOURNAMENT")
w_rounds = compare(w_outcomes, "WOMEN'S TOURNAMENT")

# Combined
all_outcomes = pd.concat([m_outcomes, w_outcomes])
all_bh = brier(all_outcomes.borhoops_pred.values, all_outcomes.actual.values)
all_ns = brier(all_outcomes.silver_pred.values, all_outcomes.actual.values)
bh_c, bh_n = correct_picks(all_outcomes.borhoops_pred.values, all_outcomes.actual.values)
ns_c, ns_n = correct_picks(all_outcomes.silver_pred.values, all_outcomes.actual.values)

print(f"\n\n{'='*65}")
print(f"  COMBINED — {len(all_outcomes)} games")
print(f"{'='*65}")
print(f"  Borhoops Brier: {all_bh:.4f}   Silver Brier: {all_ns:.4f}   Diff: {all_bh - all_ns:+.4f}")
print(f"  Borhoops Correct: {bh_c}/{bh_n}   Silver Correct: {ns_c}/{ns_n}")

winner = "Borhoops" if all_bh < all_ns else "Silver"
print(f"\n  >> Overall winner: {winner}")


# ── Visualization ────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Borhoops vs Nate Silver — 2026 NCAA Tournament", fontsize=16, fontweight="bold")

round_order = ["First Four", "Round of 64", "Round of 32", "Sweet 16",
               "Elite 8", "Final Four", "Championship"]
short_labels = ["First 4", "R64", "R32", "S16", "E8", "F4", "Champ"]

# 1 & 2: Brier by round (side by side bars)
for i, (rdf, title) in enumerate([(m_rounds, "Men's"), (w_rounds, "Women's")]):
    ax = axes[0][i]
    rdf_ordered = rdf.set_index("round").reindex(round_order).dropna()
    x = np.arange(len(rdf_ordered))
    w = 0.35
    bars1 = ax.bar(x - w/2, rdf_ordered.bh_brier, w, label="Borhoops", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w/2, rdf_ordered.ns_brier, w, label="Silver", color="#DD8452", alpha=0.85)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.4, label="Coin flip")
    ax.set_xticks(x)
    labels = [short_labels[round_order.index(r)] for r in rdf_ordered.index]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title(f"{title} — Brier by Round")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.50)

# 3 & 4: Scatter — Borhoops pred vs Silver pred, colored by outcome
for i, (outcomes, title) in enumerate([(m_outcomes, "Men's"), (w_outcomes, "Women's")]):
    ax = axes[1][i]
    valid = outcomes.dropna(subset=["silver_pred", "borhoops_pred"])
    colors = valid.actual.map({1.0: "#4C72B0", 0.0: "#DD8452"})
    ax.scatter(valid.silver_pred, valid.borhoops_pred, c=colors, alpha=0.6,
              edgecolors="black", linewidth=0.3, s=40)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Silver Predicted Prob (lower ID)")
    ax.set_ylabel("Borhoops Predicted Prob (lower ID)")
    ax.set_title(f"{title} — Prediction Scatter")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0',
               markersize=8, label='Lower ID won'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DD8452',
               markersize=8, label='Higher ID won'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

plt.tight_layout()
outpath = OUTPUT / "borhoops_vs_silver_2026.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\n📊 Saved comparison chart to {outpath}")
plt.close()
