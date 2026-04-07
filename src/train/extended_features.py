"""Compute additional features from detailed results, seeds, Massey ordinals, coaches,
odds, polls, and roster data."""

import numpy as np
import pandas as pd
from pathlib import Path


# -- Box score season stats (from DetailedResults) ----------------------------

POSSESSIONS_FTA_COEFF = 0.475  # standard approximation

def _possessions(fga, ora, to, fta):
    """Approximate possessions from box score stats."""
    return fga - ora + to + POSSESSIONS_FTA_COEFF * fta


def team_season_stats(detailed_results, season):
    """Compute per-team offensive and defensive season averages.

    Returns {TeamID: {stat_name: value}} for all teams in the given season.
    Uses the "Four Factors" framework plus efficiency metrics.
    """
    df = detailed_results[detailed_results["Season"] == season].copy()
    if df.empty:
        return {}

    stats = {}

    for _, row in df.iterrows():
        for role, opp_role in [("W", "L"), ("L", "W")]:
            team = row[f"{role}TeamID"]
            if team not in stats:
                stats[team] = {k: [] for k in [
                    "off_eff", "def_eff", "efg_pct", "opp_efg_pct",
                    "to_rate", "opp_to_rate", "or_pct", "ft_rate",
                ]}

            off_poss = _possessions(
                row[f"{role}FGA"], row[f"{role}OR"],
                row[f"{role}TO"], row[f"{role}FTA"]
            )
            def_poss = _possessions(
                row[f"{opp_role}FGA"], row[f"{opp_role}OR"],
                row[f"{opp_role}TO"], row[f"{opp_role}FTA"]
            )

            if off_poss > 0:
                stats[team]["off_eff"].append(row[f"{role}Score"] / off_poss * 100)
                stats[team]["to_rate"].append(row[f"{role}TO"] / off_poss)
            if def_poss > 0:
                stats[team]["def_eff"].append(row[f"{opp_role}Score"] / def_poss * 100)
                stats[team]["opp_to_rate"].append(row[f"{opp_role}TO"] / def_poss)

            if row[f"{role}FGA"] > 0:
                stats[team]["efg_pct"].append(
                    (row[f"{role}FGM"] + 0.5 * row[f"{role}FGM3"]) / row[f"{role}FGA"]
                )
            if row[f"{opp_role}FGA"] > 0:
                stats[team]["opp_efg_pct"].append(
                    (row[f"{opp_role}FGM"] + 0.5 * row[f"{opp_role}FGM3"]) / row[f"{opp_role}FGA"]
                )

            total_or = row[f"{role}OR"] + row[f"{opp_role}DR"]
            if total_or > 0:
                stats[team]["or_pct"].append(row[f"{role}OR"] / total_or)

            if row[f"{role}FGA"] > 0:
                stats[team]["ft_rate"].append(row[f"{role}FTA"] / row[f"{role}FGA"])

    return {
        team: {k: np.mean(v) if v else 0.0 for k, v in team_stats.items()}
        for team, team_stats in stats.items()
    }


STAT_COLS = ["off_eff", "def_eff", "efg_pct", "opp_efg_pct",
             "to_rate", "opp_to_rate", "or_pct", "ft_rate"]


def stat_matchup_features(stats_a, stats_b):
    """Difference features: team_a stats minus team_b stats."""
    feats = {}
    for col in STAT_COLS:
        a_val = stats_a.get(col, 0.0)
        b_val = stats_b.get(col, 0.0)
        feats[f"{col}_diff"] = a_val - b_val
    return feats


# -- Tournament seeds ---------------------------------------------------------

def load_seeds(data_dir, gender):
    """Load {(Season, TeamID): seed_num} dict."""
    prefix = "M" if gender == "M" else "W"
    path = Path(data_dir) / "kaggle" / f"{prefix}NCAATourneySeeds.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    seeds = {}
    for _, row in df.iterrows():
        seed_str = row["Seed"]
        seed_num = int(seed_str[1:3])
        seeds[(row["Season"], row["TeamID"])] = seed_num
    return seeds


def seed_matchup_features(seeds, season, team_a, team_b):
    """Seed difference feature. Lower seed = better. Unseeded teams get 17."""
    seed_a = seeds.get((season, team_a), 17)
    seed_b = seeds.get((season, team_b), 17)
    return {"seed_diff": seed_a - seed_b}


# -- Massey ordinals (per-system) ---------------------------------------------

MASSEY_SYSTEMS = ("POM", "SAG", "MOR", "DOK", "COL")
MASSEY_COLS = [f"massey_{s.lower()}_diff" for s in MASSEY_SYSTEMS] + ["massey_avg_diff"]


def load_massey_rankings(data_dir, systems=MASSEY_SYSTEMS):
    """Load end-of-season Massey ordinal ranks -- per-system and average.

    Returns:
        per_system: {system_name: {(Season, TeamID): rank}}
        avg:        {(Season, TeamID): avg_rank}
    """
    path = Path(data_dir) / "kaggle" / "MMasseyOrdinals.csv"
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    df = df[df["SystemName"].isin(systems)]

    idx = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].idxmax()
    latest = df.loc[idx]

    per_system = {}
    for sys_name in systems:
        sys_df = latest[latest["SystemName"] == sys_name]
        per_system[sys_name] = {
            (row.Season, row.TeamID): row.OrdinalRank
            for row in sys_df.itertuples()
        }

    avg = latest.groupby(["Season", "TeamID"])["OrdinalRank"].mean().to_dict()
    return per_system, avg


def massey_matchup_features(massey_per_system, massey_avg, season, team_a, team_b):
    """Per-system and average Massey rank differences. Unknown teams get NaN."""
    feats = {}
    for sys_name in MASSEY_SYSTEMS:
        ranks = massey_per_system.get(sys_name, {})
        rank_a = ranks.get((season, team_a), np.nan)
        rank_b = ranks.get((season, team_b), np.nan)
        feats[f"massey_{sys_name.lower()}_diff"] = rank_a - rank_b
    rank_a = massey_avg.get((season, team_a), np.nan)
    rank_b = massey_avg.get((season, team_b), np.nan)
    feats["massey_avg_diff"] = rank_a - rank_b
    return feats


# -- Coach tenure -------------------------------------------------------------

def load_coach_data(data_dir):
    """Load coach tenure and change indicators from MTeamCoaches.csv.

    Returns:
        tenure:  {(Season, TeamID): years_with_coach}
        changed: {(Season, TeamID): 1 if new coach this season, else 0}
    """
    path = Path(data_dir) / "kaggle" / "MTeamCoaches.csv"
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    idx = df.groupby(["Season", "TeamID"])["LastDayNum"].idxmax()
    df = df.loc[idx].sort_values(["TeamID", "Season"])

    tenure = {}
    changed = {}
    prev_coach = {}

    for _, row in df.iterrows():
        team, season, coach = row["TeamID"], row["Season"], row["CoachName"]
        prev = prev_coach.get(team)
        if prev and prev[0] == coach:
            years = prev[1] + 1
            is_new = 0
        else:
            years = 1
            is_new = 1 if prev else 0
        tenure[(season, team)] = years
        changed[(season, team)] = is_new
        prev_coach[team] = (coach, years)

    return tenure, changed


def coach_matchup_features(tenure, changed, season, team_a, team_b):
    """Coach tenure diff and coach change diff."""
    return {
        "coach_tenure_diff": tenure.get((season, team_a), 1) - tenure.get((season, team_b), 1),
        "coach_change_diff": changed.get((season, team_a), 0) - changed.get((season, team_b), 0),
    }


# -- Odds (closing spreads + over/under) ----------------------------------------

SPREAD_TO_PROB_K = 15.0  # calibrated for NCAAB point spreads

ODDS_COLS = ["spread", "spread_abs", "implied_prob", "over_under"]


def _spread_to_prob(spread):
    """Convert point spread to win probability. Negative spread = favored."""
    if pd.isna(spread):
        return np.nan
    return 1.0 / (1.0 + 10.0 ** (spread / SPREAD_TO_PROB_K))


def load_odds(data_dir):
    """Load odds from derived/ncaab_odds.csv.

    Returns:
        game_lookup: {(Season, DayNum, low_team, high_team): {spread, over_under}}
            spread is from low-team perspective (negative = low team favored)
        team_season_avg: {(Season, TeamID): {avg_spread, avg_over_under}}
            avg_spread is from the team's own perspective
    """
    path = Path(data_dir) / "derived" / "ncaab_odds.csv"
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    game_lookup = {}
    team_totals = {}

    for _, row in df.iterrows():
        season = int(row["Season"])
        day_num = int(row["DayNum"])
        home = int(row["HomeTeamID"])
        away = int(row["AwayTeamID"])
        spread = row["Spread"]
        over_under = row.get("OverUnder", np.nan)

        low, high = min(home, away), max(home, away)
        # Spread is home team's line: negative = home favored.
        # Convert to low-team perspective.
        spread_low = spread if home == low else -spread

        key = (season, day_num, low, high)
        if key not in game_lookup:
            game_lookup[key] = {"spread": spread_low, "over_under": over_under}

        # Per-team season averages (each team's own-perspective spread)
        for team, ts in [(home, spread), (away, -spread)]:
            tkey = (season, team)
            if tkey not in team_totals:
                team_totals[tkey] = {"spread_sum": 0.0, "ou_sum": 0.0, "n_spread": 0, "n_ou": 0}
            if pd.notna(ts):
                team_totals[tkey]["spread_sum"] += ts
                team_totals[tkey]["n_spread"] += 1
            if pd.notna(over_under):
                team_totals[tkey]["ou_sum"] += over_under
                team_totals[tkey]["n_ou"] += 1

    team_season_avg = {}
    for tkey, t in team_totals.items():
        if t["n_spread"] > 0:
            team_season_avg[tkey] = {
                "avg_spread": t["spread_sum"] / t["n_spread"],
                "avg_over_under": t["ou_sum"] / t["n_ou"] if t["n_ou"] > 0 else np.nan,
            }

    return game_lookup, team_season_avg


def odds_game_features(odds_lookup, season, day_num, low, high):
    """Per-game odds features from low-team perspective."""
    entry = odds_lookup.get((season, day_num, low, high))
    if entry is None:
        return {"spread": np.nan, "spread_abs": np.nan,
                "implied_prob": np.nan, "over_under": np.nan}
    s = entry["spread"]
    return {
        "spread": s,
        "spread_abs": abs(s) if pd.notna(s) else np.nan,
        "implied_prob": _spread_to_prob(s),
        "over_under": entry["over_under"],
    }


def odds_matchup_features(team_season_avg, season, team_a, team_b):
    """Tournament matchup odds proxy: diff of season-average spreads."""
    a = team_season_avg.get((season, team_a))
    b = team_season_avg.get((season, team_b))
    if a is None or b is None:
        return {"spread": np.nan, "spread_abs": np.nan,
                "implied_prob": np.nan, "over_under": np.nan}
    # Each team's avg_spread is from their own perspective (negative = typically favored).
    # Diff: low_team avg minus high_team avg gives expected spread from low perspective.
    s = a["avg_spread"] - b["avg_spread"]
    ou = np.nanmean([a["avg_over_under"], b["avg_over_under"]])
    return {
        "spread": s,
        "spread_abs": abs(s),
        "implied_prob": _spread_to_prob(s),
        "over_under": ou,
    }


# -- Polls (AP Top 25) ----------------------------------------------------------

POLL_COLS = ["poll_rank_diff", "poll_momentum_diff", "weeks_ranked_diff"]

UNRANKED = 30  # default rank for unranked teams


def load_polls(data_dir):
    """Load poll rankings from derived/poll_rankings.csv.

    Returns:
        week_lookup: {(Season, Week, TeamID): {rank, previous, points}}
        weeks_ranked: {(Season, TeamID): int}  — cumulative weeks in Top 25
    """
    path = Path(data_dir) / "derived" / "poll_rankings.csv"
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    # Focus on AP poll
    df = df[df["Poll"] == "AP"].copy()

    week_lookup = {}
    weeks_ranked = {}

    for _, row in df.iterrows():
        season = int(row["Season"])
        week = int(row["Week"])
        team = int(row["TeamID"])
        rank = int(row["Rank"])
        previous = int(row["Previous"]) if pd.notna(row.get("Previous")) else UNRANKED
        points = row.get("Points", np.nan)

        week_lookup[(season, week, team)] = {
            "rank": rank,
            "previous": previous,
            "points": points if pd.notna(points) else 0.0,
        }

        wkey = (season, team)
        weeks_ranked[wkey] = weeks_ranked.get(wkey, 0) + 1

    return week_lookup, weeks_ranked


def _daynum_to_week(day_num):
    """Map DayNum to poll week (approximate). Season starts ~DayNum 0."""
    return max(1, day_num // 7)


def _get_poll_entry(week_lookup, season, week, team):
    """Find poll entry for a team at or before the given week."""
    for w in range(week, 0, -1):
        entry = week_lookup.get((season, w, team))
        if entry is not None:
            return entry
    return None


def poll_game_features(week_lookup, weeks_ranked, season, day_num, low, high):
    """Per-game poll features: rank diff, momentum diff, weeks ranked diff."""
    week = _daynum_to_week(day_num)

    low_entry = _get_poll_entry(week_lookup, season, week, low)
    high_entry = _get_poll_entry(week_lookup, season, week, high)

    low_rank = low_entry["rank"] if low_entry else UNRANKED
    high_rank = high_entry["rank"] if high_entry else UNRANKED

    low_prev = low_entry["previous"] if low_entry else UNRANKED
    high_prev = high_entry["previous"] if high_entry else UNRANKED

    # Momentum: positive = improving (rank went down numerically)
    low_momentum = low_prev - low_rank
    high_momentum = high_prev - high_rank

    low_weeks = weeks_ranked.get((season, low), 0)
    high_weeks = weeks_ranked.get((season, high), 0)

    return {
        "poll_rank_diff": low_rank - high_rank,
        "poll_momentum_diff": low_momentum - high_momentum,
        "weeks_ranked_diff": low_weeks - high_weeks,
    }


def poll_matchup_features(week_lookup, weeks_ranked, season, team_a, team_b):
    """Tournament matchup poll features: use latest available week."""
    # Find the latest week with any data for the season
    max_week = 0
    for key in week_lookup:
        if key[0] == season and key[1] > max_week:
            max_week = key[1]

    if max_week == 0:
        return {"poll_rank_diff": 0, "poll_momentum_diff": 0, "weeks_ranked_diff": 0}

    a_entry = _get_poll_entry(week_lookup, season, max_week, team_a)
    b_entry = _get_poll_entry(week_lookup, season, max_week, team_b)

    a_rank = a_entry["rank"] if a_entry else UNRANKED
    b_rank = b_entry["rank"] if b_entry else UNRANKED
    a_prev = a_entry["previous"] if a_entry else UNRANKED
    b_prev = b_entry["previous"] if b_entry else UNRANKED

    return {
        "poll_rank_diff": a_rank - b_rank,
        "poll_momentum_diff": (a_prev - a_rank) - (b_prev - b_rank),
        "weeks_ranked_diff": weeks_ranked.get((season, team_a), 0) - weeks_ranked.get((season, team_b), 0),
    }


# -- Roster experience -----------------------------------------------------------

ROSTER_COLS = ["avg_experience_diff", "senior_pct_diff"]


def load_roster(data_dir):
    """Load roster data from derived/roster_experience.csv.

    Returns:
        {(Season, TeamID): {avg_experience, senior_pct}}
    """
    path = Path(data_dir) / "derived" / "roster_experience.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(int(row["Season"]), int(row["TeamID"]))] = {
            "avg_experience": row["AvgExperience"],
            "senior_pct": row["SeniorPct"],
        }
    return lookup


def roster_matchup_features(roster_lookup, season, team_a, team_b):
    """Roster experience diff features."""
    a = roster_lookup.get((season, team_a), {})
    b = roster_lookup.get((season, team_b), {})
    return {
        "avg_experience_diff": a.get("avg_experience", np.nan) - b.get("avg_experience", np.nan),
        "senior_pct_diff": a.get("senior_pct", np.nan) - b.get("senior_pct", np.nan),
    }
