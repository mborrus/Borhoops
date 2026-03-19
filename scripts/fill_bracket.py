"""Simulate the 2026 NCAA tournament bracket using Elo win probabilities.

Runs the full Elo model, then walks through the bracket slots round by round,
always advancing the team with the higher win probability.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from predict.elo import run_elo, add_game_counts, calc_elo_win_tourney, update_elo
from predict.submission import load_data, _build_hfa_dict, _build_location_dict

SEASON = 2026
ROUND_NAMES = {
    "R1": "Round of 64",
    "R2": "Round of 32",
    "R3": "Sweet 16",
    "R4": "Elite 8",
    "R5": "Final Four",
    "R6": "Championship",
}
REGION_NAMES = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}


def run_model(data_dir):
    """Run Elo for both genders, return {TeamID: elo}."""
    data = load_data(data_dir)
    all_elo = {}

    for gender, results_key, conf_key in [
        ("M", "mens_results", "mens_conf"),
        ("W", "womens_results", "womens_conf"),
    ]:
        results = add_game_counts(data[results_key])
        hfa_dict = _build_hfa_dict(data["hfa"], gender)
        location_dict = _build_location_dict(data["home_lookup"], gender)
        elo = run_elo(results, data[conf_key], hfa_dict, location_dict)
        all_elo.update(elo)

    return all_elo, data


def simulate_bracket(slots, seeds, teams, elo_ratings, k_tourney=38):
    """Walk through bracket slots, advancing higher-probability winner.

    Elo ratings are updated after each simulated game so that beating a strong
    opponent boosts a team's rating heading into the next round. Uses k_tourney
    (late-season K) with no MOV bonus since we don't simulate scores.
    """
    # Work on a copy so we don't mutate the caller's dict
    elo = dict(elo_ratings)

    seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))

    slot_winner = {}

    playin_slots = slots[~slots["Slot"].str.startswith("R")]
    regular_slots = slots[slots["Slot"].str.startswith("R")].sort_values("Slot")

    results = []

    for _, row in pd.concat([playin_slots, regular_slots]).iterrows():
        slot, strong, weak = row["Slot"], row["StrongSeed"], row["WeakSeed"]

        team_a = _resolve(strong, seed_to_team, slot_winner)
        team_b = _resolve(weak, seed_to_team, slot_winner)

        if team_a is None or team_b is None:
            continue

        elo_a = elo.get(team_a, 1500)
        elo_b = elo.get(team_b, 1500)
        win_prob = calc_elo_win_tourney(elo_a, elo_b)

        winner = team_a if win_prob >= 0.5 else team_b
        loser = team_b if winner == team_a else team_a
        slot_winner[slot] = winner

        # Update Elo ratings after the simulated game
        elo_change = update_elo(elo[winner], elo[loser], k=k_tourney)
        elo[winner] += elo_change
        elo[loser] -= elo_change

        round_key = slot[:2] if slot[:2].startswith("R") else "PI"
        round_name = ROUND_NAMES.get(round_key, "Play-In")

        results.append({
            "round": round_name,
            "slot": slot,
            "team_a": team_names.get(team_a, str(team_a)),
            "seed_a": strong,
            "elo_a": elo_a,
            "team_b": team_names.get(team_b, str(team_b)),
            "seed_b": weak,
            "elo_b": elo_b,
            "win_prob": win_prob if winner == team_a else 1 - win_prob,
            "winner": team_names.get(winner, str(winner)),
            "winner_id": winner,
        })

    return results


def _resolve(ref, seed_to_team, slot_winner):
    """Resolve a bracket reference to a TeamID."""
    if ref in seed_to_team:
        return seed_to_team[ref]
    if ref in slot_winner:
        return slot_winner[ref]
    return None


def format_seed(seed):
    """Turn 'W01' into '(1) West' style display."""
    region = seed[0]
    num = seed[1:].lstrip("0").rstrip("ab")
    region_name = REGION_NAMES.get(region, region)
    return f"({num}) {region_name}" if num else seed


def print_bracket(results, label):
    """Pretty-print the bracket results."""
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    current_round = None
    for r in results:
        if r["round"] != current_round:
            current_round = r["round"]
            print(f"\n  --- {current_round} ---")

        marker_a = ">>>" if r["winner"] == r["team_a"] else "   "
        marker_b = ">>>" if r["winner"] == r["team_b"] else "   "
        prob_display = f"{r['win_prob']:.0%}"

        print(f"  {marker_a} {r['team_a']:<22s} ({r['elo_a']:7.1f})")
        print(f"  {marker_b} {r['team_b']:<22s} ({r['elo_b']:7.1f})  → {r['winner']} wins ({prob_display})")
        print()

    # Print champion
    champ = results[-1]
    print(f"  {'*'*40}")
    print(f"  CHAMPION: {champ['winner']}")
    print(f"  {'*'*40}")


def save_bracket_csv(results, path, label):
    """Save bracket to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\nSaved {label} bracket to {path}")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    data_dir = repo / "data"
    output_dir = repo / "Output"
    output_dir.mkdir(exist_ok=True)

    print("Running Elo model...")
    elo_ratings, data = run_model(data_dir)

    kaggle = data_dir / "kaggle"

    for gender, label, seeds_file, slots_file, teams_key in [
        ("M", "MEN'S TOURNAMENT", "MNCAATourneySeeds.csv", "MNCAATourneySlots.csv", "mens_teams"),
        ("W", "WOMEN'S TOURNAMENT", "WNCAATourneySeeds.csv", "WNCAATourneySlots.csv", "womens_teams"),
    ]:
        seeds = pd.read_csv(kaggle / seeds_file)
        seeds = seeds[seeds["Season"] == SEASON]
        slots = pd.read_csv(kaggle / slots_file)
        slots = slots[slots["Season"] == SEASON]
        teams = data[teams_key]

        results = simulate_bracket(slots, seeds, teams, elo_ratings)
        print_bracket(results, label)
        save_bracket_csv(results, output_dir / f"{gender}_bracket_2026.csv", label)
