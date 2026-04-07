"""Evaluate 2026 tournament predictions against actual results.

Computes Brier score for the Elo model and baselines, and analyzes
bracket-level accuracy (game picks, Final Four, champion).
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "kaggle"
OUTPUT = ROOT / "Output"

# ── Load team name → ID lookups ──────────────────────────────────────────────

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

# Build name → ID lookup (use seeded teams first, then all teams as fallback)
def build_name_map(teams_df, seeds_df):
    """Map various team name forms to Kaggle TeamID."""
    name_map = {}
    for _, row in teams_df.iterrows():
        name_map[row.TeamName.lower()] = row.TeamID
    # Also map seeds for 2026 specifically
    for _, row in seeds_df.iterrows():
        name_map[row.TeamName.lower()] = row.TeamID
    return name_map

m_name_map = build_name_map(m_teams, m_seeds_2026)
w_name_map = build_name_map(w_teams, w_seeds_2026)

# Manual aliases for names that differ between sources
m_aliases = {
    "uconn": "connecticut", "miami fl": "miami fl", "miami oh": "miami oh",
    "st. john's": "st john's", "st john's ny": "st john's",
    "north carolina": "north carolina", "unc": "north carolina",
    "cal baptist": "cal baptist", "california baptist": "cal baptist",
    "mcneese state": "mcneese st", "mcneese": "mcneese st",
    "north dakota state": "n dakota st", "prairie view a&m": "prairie view",
    "prairie view": "prairie view", "liu": "liu brooklyn",
    "long island": "liu brooklyn", "queens": "queens nc",
    "high point": "high point", "kennesaw state": "kennesaw",
    "kennesaw": "kennesaw", "saint louis": "st louis",
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

def resolve_id(name, name_map, aliases):
    """Resolve a team name to Kaggle TeamID."""
    key = name.strip().lower()
    if key in aliases:
        key = aliases[key]
    if key in name_map:
        return name_map[key]
    # fuzzy: try without punctuation
    for k, v in name_map.items():
        if key.replace(".", "").replace("'", "") == k.replace(".", "").replace("'", ""):
            return v
    return None


# ── Actual 2026 tournament game results ──────────────────────────────────────
# Men's results (from On3 / sports-reference scrapes)

men_games_raw = [
    # First Four
    ("Howard", "UMBC", "First Four"),
    ("Miami OH", "SMU", "First Four"),  # Actually Miami OH lost... Let me recheck
    ("Prairie View", "Lehigh", "First Four"),
    ("Texas", "NC State", "First Four"),
    # Round of 64 - East
    ("Duke", "Siena", "Round of 64"),
    ("TCU", "Ohio St", "Round of 64"),
    ("St John's", "Northern Iowa", "Round of 64"),
    ("Kansas", "Cal Baptist", "Round of 64"),
    ("Louisville", "South Florida", "Round of 64"),
    ("Michigan St", "N Dakota St", "Round of 64"),
    ("UCLA", "UCF", "Round of 64"),
    ("UConn", "Furman", "Round of 64"),
    # Round of 64 - South
    ("Florida", "Prairie View", "Round of 64"),
    ("Iowa", "Clemson", "Round of 64"),
    ("Vanderbilt", "McNeese", "Round of 64"),
    ("Nebraska", "Troy", "Round of 64"),
    ("VCU", "North Carolina", "Round of 64"),
    ("Illinois", "Penn", "Round of 64"),
    ("Texas A&M", "St Mary's CA", "Round of 64"),
    ("Houston", "Idaho", "Round of 64"),
    # Round of 64 - West
    ("Arizona", "LIU", "Round of 64"),
    ("Utah St", "Villanova", "Round of 64"),
    ("High Point", "Wisconsin", "Round of 64"),
    ("Arkansas", "Hawaii", "Round of 64"),
    ("Texas", "BYU", "Round of 64"),
    ("Gonzaga", "Kennesaw", "Round of 64"),
    ("Miami FL", "Missouri", "Round of 64"),
    ("Purdue", "Queens", "Round of 64"),
    # Round of 64 - Midwest
    ("Michigan", "Howard", "Round of 64"),
    ("St Louis", "Georgia", "Round of 64"),
    ("Texas Tech", "Akron", "Round of 64"),
    ("Alabama", "Hofstra", "Round of 64"),
    ("Tennessee", "Miami OH", "Round of 64"),
    ("Virginia", "Wright St", "Round of 64"),
    ("Kentucky", "Santa Clara", "Round of 64"),
    ("Iowa St", "Tennessee St", "Round of 64"),
    # Round of 32
    ("Duke", "TCU", "Round of 32"),
    ("St John's", "Kansas", "Round of 32"),
    ("Michigan St", "Louisville", "Round of 32"),
    ("UConn", "UCLA", "Round of 32"),
    ("Iowa", "Florida", "Round of 32"),
    ("Nebraska", "Vanderbilt", "Round of 32"),
    ("Illinois", "VCU", "Round of 32"),
    ("Houston", "Texas A&M", "Round of 32"),
    ("Arizona", "Utah St", "Round of 32"),
    ("Arkansas", "High Point", "Round of 32"),
    ("Texas", "Gonzaga", "Round of 32"),
    ("Purdue", "Miami FL", "Round of 32"),
    ("Michigan", "St Louis", "Round of 32"),
    ("Alabama", "Texas Tech", "Round of 32"),
    ("Tennessee", "Virginia", "Round of 32"),
    ("Iowa St", "Kentucky", "Round of 32"),
    # Sweet 16
    ("Duke", "St John's", "Sweet 16"),
    ("UConn", "Michigan St", "Sweet 16"),
    ("Iowa", "Nebraska", "Sweet 16"),
    ("Illinois", "Houston", "Sweet 16"),
    ("Arizona", "Arkansas", "Sweet 16"),
    ("Purdue", "Texas", "Sweet 16"),
    ("Michigan", "Alabama", "Sweet 16"),
    ("Tennessee", "Iowa St", "Sweet 16"),
    # Elite 8
    ("UConn", "Duke", "Elite 8"),
    ("Illinois", "Iowa", "Elite 8"),
    ("Arizona", "Purdue", "Elite 8"),
    ("Michigan", "Tennessee", "Elite 8"),
    # Final Four
    ("UConn", "Illinois", "Final Four"),
    ("Michigan", "Arizona", "Final Four"),
    # Championship
    ("Michigan", "UConn", "Championship"),
]

# Women's results
women_games_raw = [
    # First Four
    ("Missouri St", "SF Austin", "First Four"),
    ("Nebraska", "Richmond", "First Four"),
    ("Virginia", "Arizona St", "First Four"),
    ("Southern", "Samford", "First Four"),
    # Round of 64 - Region 1 (Fort Worth)
    ("UConn", "UT San Antonio", "Round of 64"),
    ("Syracuse", "Iowa St", "Round of 64"),
    ("Maryland", "Murray St", "Round of 64"),
    ("North Carolina", "W Illinois", "Round of 64"),
    ("Notre Dame", "Fairfield", "Round of 64"),
    ("Ohio St", "Howard", "Round of 64"),
    ("Illinois", "Colorado", "Round of 64"),
    ("Vanderbilt", "High Point", "Round of 64"),
    # Round of 64 - Region 2 (Sacramento)
    ("UCLA", "Cal Baptist", "Round of 64"),
    ("Oklahoma St", "Princeton", "Round of 64"),
    ("Mississippi", "Gonzaga", "Round of 64"),
    ("Minnesota", "WI Green Bay", "Round of 64"),
    ("Baylor", "Nebraska", "Round of 64"),
    ("Duke", "Col Charleston", "Round of 64"),
    ("Texas Tech", "Villanova", "Round of 64"),
    ("LSU", "Jacksonville", "Round of 64"),
    # Round of 64 - Region 3 (Fort Worth)
    ("Texas", "Missouri St", "Round of 64"),
    ("Oregon", "Virginia Tech", "Round of 64"),
    ("Kentucky", "James Madison", "Round of 64"),
    ("West Virginia", "Miami OH", "Round of 64"),
    ("Alabama", "Rhode Island", "Round of 64"),
    ("Louisville", "Vermont", "Round of 64"),
    ("NC State", "Tennessee", "Round of 64"),
    ("Michigan", "Holy Cross", "Round of 64"),
    # Round of 64 - Region 4 (Sacramento)
    ("South Carolina", "Southern", "Round of 64"),
    ("USC", "Clemson", "Round of 64"),
    ("Michigan St", "Colorado St", "Round of 64"),
    ("Oklahoma", "Idaho", "Round of 64"),
    ("Washington", "S Dakota St", "Round of 64"),
    ("TCU", "UC San Diego", "Round of 64"),
    ("Virginia", "Georgia", "Round of 64"),
    ("Iowa", "F Dickinson", "Round of 64"),
    # Round of 32
    ("UConn", "Syracuse", "Round of 32"),
    ("North Carolina", "Maryland", "Round of 32"),
    ("Notre Dame", "Ohio St", "Round of 32"),
    ("Vanderbilt", "Illinois", "Round of 32"),
    ("UCLA", "Oklahoma St", "Round of 32"),
    ("Minnesota", "Mississippi", "Round of 32"),
    ("Duke", "Baylor", "Round of 32"),
    ("LSU", "Texas Tech", "Round of 32"),
    ("Texas", "Oregon", "Round of 32"),
    ("Kentucky", "West Virginia", "Round of 32"),
    ("Louisville", "Alabama", "Round of 32"),
    ("Michigan", "NC State", "Round of 32"),
    ("South Carolina", "USC", "Round of 32"),
    ("Oklahoma", "Michigan St", "Round of 32"),
    ("TCU", "Washington", "Round of 32"),
    ("Virginia", "Iowa", "Round of 32"),
    # Sweet 16
    ("UConn", "North Carolina", "Sweet 16"),
    ("Notre Dame", "Vanderbilt", "Sweet 16"),
    ("UCLA", "Minnesota", "Sweet 16"),
    ("Duke", "LSU", "Sweet 16"),
    ("Texas", "Kentucky", "Sweet 16"),
    ("Michigan", "Louisville", "Sweet 16"),  # actually Louisville lost
    ("South Carolina", "Oklahoma", "Sweet 16"),
    ("TCU", "Virginia", "Sweet 16"),
    # Elite 8
    ("UConn", "Notre Dame", "Elite 8"),
    ("UCLA", "Duke", "Elite 8"),
    ("Texas", "Michigan", "Elite 8"),
    ("South Carolina", "TCU", "Elite 8"),
    # Final Four
    ("South Carolina", "UConn", "Final Four"),
    ("UCLA", "Texas", "Final Four"),
    # Championship
    ("UCLA", "South Carolina", "Championship"),
]

# ── Resolve IDs ──────────────────────────────────────────────────────────────

def build_game_outcomes(games_raw, name_map, aliases, gender_prefix):
    """Convert (winner, loser, round) tuples to (ID_low_ID_high, actual_outcome) pairs."""
    outcomes = []
    unresolved = []
    for winner, loser, round_name in games_raw:
        w_id = resolve_id(winner, name_map, aliases)
        l_id = resolve_id(loser, name_map, aliases)
        if w_id is None:
            unresolved.append(f"  Winner: {winner}")
        if l_id is None:
            unresolved.append(f"  Loser: {loser}")
        if w_id and l_id:
            low, high = min(w_id, l_id), max(w_id, l_id)
            actual = 1.0 if w_id == low else 0.0
            game_id = f"2026_{low}_{high}"
            outcomes.append({
                "ID": game_id,
                "actual": actual,
                "round": round_name,
                "winner": winner,
                "loser": loser,
                "winner_id": w_id,
                "loser_id": l_id,
            })
    if unresolved:
        print(f"\n⚠ Unresolved {gender_prefix} teams:")
        for u in unresolved:
            print(u)
    return pd.DataFrame(outcomes)


m_outcomes = build_game_outcomes(men_games_raw, m_name_map, m_aliases, "Men's")
w_outcomes = build_game_outcomes(women_games_raw, w_name_map, w_aliases, "Women's")

# ── Load predictions ─────────────────────────────────────────────────────────

elo_preds = pd.read_csv(OUTPUT / "ComplexEloProbs.csv")

# ── Score function ───────────────────────────────────────────────────────────

def brier_score(preds, actuals):
    return np.mean((preds - actuals) ** 2)

def log_loss(preds, actuals, eps=1e-15):
    p = np.clip(preds, eps, 1 - eps)
    return -np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p))

def evaluate(outcomes_df, elo_df, seeds_df, label):
    """Score predictions for a set of actual outcomes."""
    merged = outcomes_df.merge(elo_df, on="ID", how="left")
    missing = merged.Pred.isna().sum()
    if missing > 0:
        print(f"  ⚠ {missing} games not found in predictions")
        merged = merged.dropna(subset=["Pred"])

    n = len(merged)

    # Elo model
    elo_brier = brier_score(merged.Pred.values, merged.actual.values)
    elo_logloss = log_loss(merged.Pred.values, merged.actual.values)
    elo_correct = ((merged.Pred > 0.5) == (merged.actual == 1.0)).sum()

    # Baseline: always 0.5
    coin_brier = brier_score(np.full(n, 0.5), merged.actual.values)

    # Baseline: seed-based (higher seed wins, use seed diff for prob)
    seed_map = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    seed_preds = []
    for _, row in merged.iterrows():
        low_id = int(row.ID.split("_")[1])
        high_id = int(row.ID.split("_")[2])
        s_low = seed_map.get(low_id, 8)
        s_high = seed_map.get(high_id, 8)
        # Lower seed number = better team → higher win prob
        # Simple logistic: P(low wins) = 1 / (1 + 10^((s_low - s_high)/5))
        diff = s_low - s_high  # negative if low_id is better seed
        seed_preds.append(1 / (1 + 10 ** (diff / 5)))
    seed_preds = np.array(seed_preds)
    seed_brier = brier_score(seed_preds, merged.actual.values)
    seed_correct = ((seed_preds > 0.5) == (merged.actual == 1.0)).sum()

    # Chalk baseline (always pick higher seed to win)
    chalk_correct = 0
    for _, row in merged.iterrows():
        low_id = int(row.ID.split("_")[1])
        high_id = int(row.ID.split("_")[2])
        s_low = seed_map.get(low_id, 8)
        s_high = seed_map.get(high_id, 8)
        # Pick the better-seeded team (lower seed number)
        if s_low < s_high:
            pred_winner = low_id
        elif s_high < s_low:
            pred_winner = high_id
        else:
            pred_winner = low_id  # tie → lower ID
        actual_winner = row.winner_id
        if pred_winner == actual_winner:
            chalk_correct += 1

    print(f"\n{'='*60}")
    print(f"  {label} — {n} tournament games")
    print(f"{'='*60}")
    print(f"\n  Brier Score (lower is better):")
    print(f"    Elo Model:      {elo_brier:.4f}")
    print(f"    Seed-Based:     {seed_brier:.4f}")
    print(f"    Coin Flip:      {coin_brier:.4f}")
    print(f"    Improvement over coin: {(coin_brier - elo_brier) / coin_brier * 100:.1f}%")
    print(f"    Improvement over seed: {(seed_brier - elo_brier) / seed_brier * 100:+.1f}%")
    print(f"\n  Log Loss: {elo_logloss:.4f}")
    print(f"\n  Correct Picks (favored team wins):")
    print(f"    Elo Model: {elo_correct}/{n} ({elo_correct/n*100:.1f}%)")
    print(f"    Seed-Based: {seed_correct}/{n} ({seed_correct/n*100:.1f}%)")
    print(f"    Chalk:      {chalk_correct}/{n} ({chalk_correct/n*100:.1f}%)")

    # Round-by-round breakdown
    round_order = ["First Four", "Round of 64", "Round of 32", "Sweet 16",
                   "Elite 8", "Final Four", "Championship"]
    print(f"\n  Round-by-Round Brier Score:")
    round_stats = []
    for r in round_order:
        rdf = merged[merged["round"] == r]
        if len(rdf) == 0:
            continue
        rb = brier_score(rdf.Pred.values, rdf.actual.values)
        rc = ((rdf.Pred > 0.5) == (rdf.actual == 1.0)).sum()
        print(f"    {r:20s}  Brier: {rb:.4f}  Correct: {rc}/{len(rdf)}")
        round_stats.append({"round": r, "brier": rb, "correct": rc,
                            "total": len(rdf), "pct": rc / len(rdf)})

    # Biggest upsets the model missed (confident but wrong)
    merged["error"] = (merged.Pred - merged.actual) ** 2
    merged["confidence"] = np.abs(merged.Pred - 0.5)

    print(f"\n  Biggest Misses (most confident wrong picks):")
    wrong = merged[((merged.Pred > 0.5) != (merged.actual == 1.0))].copy()
    wrong = wrong.sort_values("confidence", ascending=False)
    for _, row in wrong.head(10).iterrows():
        fav = row.winner if row.Pred > 0.5 else row.loser
        actual_w = row.winner
        print(f"    Pred: {fav} ({row.Pred:.1%}) → Actual: {actual_w} won  [{row['round']}]")

    print(f"\n  Best Calls (most confident correct picks on close games):")
    right = merged[((merged.Pred > 0.5) == (merged.actual == 1.0))].copy()
    # Sort by: close games (low confidence) that model still got right
    right_upsets = right[right.confidence < 0.15].sort_values("confidence")
    for _, row in right_upsets.head(10).iterrows():
        print(f"    Pred: {row.winner} ({row.Pred:.1%}) → Correct  [{row['round']}]")

    return merged, pd.DataFrame(round_stats)


# ── Bracket evaluation ───────────────────────────────────────────────────────

def evaluate_bracket(bracket_file, actual_games, label):
    """Compare bracket simulation picks vs actual outcomes."""
    bracket = pd.read_csv(bracket_file)
    print(f"\n{'='*60}")
    print(f"  {label} — Bracket Picks")
    print(f"{'='*60}")

    # Build actual winner per round+matchup
    round_order = ["Play-In", "Round of 64", "Round of 32", "Sweet 16",
                   "Elite 8", "Final Four", "Championship"]

    correct = 0
    wrong = 0
    details = []
    for _, brow in bracket.iterrows():
        pred_winner = brow["winner"]
        pred_id = brow["winner_id"]
        rnd = brow["round"]
        # Find matching actual game
        for _, arow in actual_games.iterrows():
            if arow["round"] == rnd:
                # Match by one of the teams
                teams_in_bracket = {brow["team_a"].lower(), brow["team_b"].lower()}
                if (arow["winner"].lower() in teams_in_bracket or
                    arow["loser"].lower() in teams_in_bracket):
                    # Check both teams match
                    actual_teams = {arow["winner"].lower(), arow["loser"].lower()}
                    if teams_in_bracket == actual_teams:
                        got_it = (pred_winner.lower() == arow["winner"].lower())
                        details.append({
                            "round": rnd,
                            "predicted_winner": pred_winner,
                            "actual_winner": arow["winner"],
                            "correct": got_it,
                            "win_prob": brow["win_prob"],
                        })
                        if got_it:
                            correct += 1
                        else:
                            wrong += 1
                        break

    print(f"\n  Overall: {correct}/{correct+wrong} correct ({correct/(correct+wrong)*100:.1f}%)")

    # By round
    ddf = pd.DataFrame(details)
    for r in round_order:
        rdf = ddf[ddf["round"] == r]
        if len(rdf) == 0:
            continue
        rc = rdf.correct.sum()
        print(f"    {r:20s}  {rc}/{len(rdf)} correct")

    # Show wrong picks
    wrong_picks = ddf[~ddf.correct]
    if len(wrong_picks) > 0:
        print(f"\n  Wrong bracket picks:")
        for _, row in wrong_picks.iterrows():
            print(f"    {row['round']:20s}  Picked: {row.predicted_winner} ({row.win_prob:.1%}) → "
                  f"Actual: {row.actual_winner}")

    # Final Four / Champion
    print(f"\n  Key picks:")
    for r in ["Final Four", "Championship"]:
        rdf = ddf[ddf["round"] == r]
        for _, row in rdf.iterrows():
            status = "✓" if row.correct else "✗"
            print(f"    {status} {row['round']}: Picked {row.predicted_winner}, "
                  f"Actual {row.actual_winner}")

    return ddf


# ── Run everything ───────────────────────────────────────────────────────────

print("\n" + "█" * 60)
print("  2026 NCAA TOURNAMENT MODEL EVALUATION")
print("█" * 60)

# Convert game tuples to DataFrames with actual outcomes for bracket comparison
m_actual = pd.DataFrame(men_games_raw, columns=["winner", "loser", "round"])
w_actual = pd.DataFrame(women_games_raw, columns=["winner", "loser", "round"])

print("\n\n▓▓▓ MEN'S TOURNAMENT ▓▓▓")
m_merged, m_round_stats = evaluate(m_outcomes, elo_preds, m_seeds_2026, "Men's Tournament")

print("\n\n▓▓▓ WOMEN'S TOURNAMENT ▓▓▓")
w_merged, w_round_stats = evaluate(w_outcomes, elo_preds, w_seeds_2026, "Women's Tournament")

# Combined
all_merged = pd.concat([m_merged, w_merged])
combined_brier = brier_score(all_merged.Pred.values, all_merged.actual.values)
combined_correct = ((all_merged.Pred > 0.5) == (all_merged.actual == 1.0)).sum()
n_total = len(all_merged)

print(f"\n\n{'='*60}")
print(f"  COMBINED — {n_total} games")
print(f"{'='*60}")
print(f"  Brier Score: {combined_brier:.4f}")
print(f"  Correct: {combined_correct}/{n_total} ({combined_correct/n_total*100:.1f}%)")
print(f"  vs Coin Flip: {(0.25 - combined_brier) / 0.25 * 100:.1f}% better")

# Bracket evaluations
print("\n\n▓▓▓ BRACKET SIMULATION EVALUATION ▓▓▓")
m_bracket_results = evaluate_bracket(OUTPUT / "M_bracket_2026.csv", m_actual, "Men's Bracket")
w_bracket_results = evaluate_bracket(OUTPUT / "W_bracket_2026.csv", w_actual, "Women's Bracket")


# ── Visualization ────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("2026 NCAA Tournament — Elo Model Evaluation", fontsize=16, fontweight="bold")

# 1. Brier score by round
round_order = ["First Four", "Round of 64", "Round of 32", "Sweet 16",
               "Elite 8", "Final Four", "Championship"]
for i, (stats, title) in enumerate([(m_round_stats, "Men's"), (w_round_stats, "Women's")]):
    ax = axes[0][i]
    stats_ordered = stats.set_index("round").reindex(round_order).dropna()
    colors = sns.color_palette("coolwarm", len(stats_ordered))
    bars = ax.bar(range(len(stats_ordered)), stats_ordered.brier, color=colors)
    ax.set_xticks(range(len(stats_ordered)))
    ax.set_xticklabels([r.replace("Round of ", "R") for r in stats_ordered.index],
                       rotation=45, ha="right", fontsize=9)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Coin flip")
    ax.set_ylabel("Brier Score")
    ax.set_title(f"{title} — Brier by Round")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.35)

# 2. Calibration plot
for i, (merged, title) in enumerate([(m_merged, "Men's"), (w_merged, "Women's")]):
    ax = axes[1][i]
    bins = np.linspace(0, 1, 11)
    merged["pred_bin"] = pd.cut(merged.Pred, bins=bins)
    cal = merged.groupby("pred_bin", observed=True).agg(
        mean_pred=("Pred", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count"),
    ).dropna()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.scatter(cal.mean_pred, cal.mean_actual, s=cal["count"] * 10,
              alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title(f"{title} — Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT / "evaluation_2026.png", dpi=150, bbox_inches="tight")
print(f"\n📊 Saved evaluation plot to {OUTPUT / 'evaluation_2026.png'}")
plt.close()
