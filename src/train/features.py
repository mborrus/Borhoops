"""Extract ML features from the Elo rating loop.

Combines three feature sources:
  1. Elo-derived (base + compact rolling + detail rolling) — computed inside the loop
  2. Pre-computed lookups (Massey, coach, Barttorvik) — dict lookups by (Season, TeamID)
  3. Derived within-loop stats (close-game record, conference Elo)
"""

from collections import deque
from statistics import mean, stdev

import numpy as np
import pandas as pd

from predict.elo import (
    _travel_miles, distance_to_elo_impact, get_advantage,
    k_factor, point_differential_scaler, update_elo, calc_elo_win_tourney,
)
from train.extended_features import (
    _possessions, POSSESSIONS_FTA_COEFF,
    ODDS_COLS, POLL_COLS, ROSTER_COLS,
)

# Per-season Elo/game_count snapshots from the last build_features() call.
# Access via: from train.features import get_elo_snapshot
_last_snapshots = {}

def get_elo_snapshot(season):
    """Get Elo ratings at end of a specific season (from last build_features call)."""
    return _last_snapshots.get("elo", {}).get(season, {})

def get_gc_snapshot(season):
    """Get game counts at end of a specific season."""
    return _last_snapshots.get("gc", {}).get(season, {})

def get_all_snapshots():
    """Get all per-season Elo and game count snapshots."""
    return _last_snapshots.get("elo", {}), _last_snapshots.get("gc", {})


# -- Feature column groups (all are low_team - high_team diffs) ----------------

BASE_COLS = ["elo_diff", "elo_pred", "home", "day_num", "game_count_avg"]
COMPACT_COLS = ["sos_diff", "win_streak_diff", "rest_days_diff",
                "margin_mean_diff", "margin_std_diff"]
DETAIL_COLS = ["off_eff_r10_diff", "def_eff_r10_diff", "efg_pct_r10_diff",
               "opp_efg_pct_r10_diff", "to_rate_r10_diff", "or_pct_r10_diff",
               "pace_r10_diff"]
MASSEY_COLS = ["massey_pom_diff", "massey_sag_diff", "massey_mor_diff",
               "massey_dok_diff", "massey_col_diff", "massey_avg_diff"]
COACH_COLS = ["coach_tenure_diff", "coach_change_diff"]
DERIVED_COLS = ["close_win_pct_diff", "conf_elo_diff"]
BARTTORVIK_COLS = ["barthag_diff", "trank_adjO_diff", "trank_adjD_diff"]
SEED_COLS = ["seed_diff"]

FEATURE_COLS = (BASE_COLS + COMPACT_COLS + DETAIL_COLS +
                MASSEY_COLS + COACH_COLS + DERIVED_COLS + BARTTORVIK_COLS +
                ODDS_COLS + POLL_COLS + ROSTER_COLS + SEED_COLS)


# -- Rolling per-team state (compact + detail) ---------------------------------

def _init_team_state():
    return {
        "last_day_num": None,
        "win_streak": 0,
        "margins": deque(maxlen=10),
        "opp_elos": [],
        # Detail rolling (from box scores)
        "off_effs": deque(maxlen=10),
        "def_effs": deque(maxlen=10),
        "efg_pcts": deque(maxlen=10),
        "opp_efg_pcts": deque(maxlen=10),
        "to_rates": deque(maxlen=10),
        "or_pcts": deque(maxlen=10),
        "paces": deque(maxlen=10),
    }


def _extract_compact_features(state):
    sos = mean(state["opp_elos"]) if state["opp_elos"] else 1500.0
    margins = list(state["margins"])
    return {
        "sos": sos,
        "win_streak": state["win_streak"],
        "rest_days": None,
        "margin_mean": mean(margins) if margins else 0.0,
        "margin_std": stdev(margins) if len(margins) >= 2 else 0.0,
    }


def _extract_detail_features(state):
    def _mean_or_nan(d):
        return mean(d) if d else np.nan
    return {
        "off_eff_r10": _mean_or_nan(state["off_effs"]),
        "def_eff_r10": _mean_or_nan(state["def_effs"]),
        "efg_pct_r10": _mean_or_nan(state["efg_pcts"]),
        "opp_efg_pct_r10": _mean_or_nan(state["opp_efg_pcts"]),
        "to_rate_r10": _mean_or_nan(state["to_rates"]),
        "or_pct_r10": _mean_or_nan(state["or_pcts"]),
        "pace_r10": _mean_or_nan(state["paces"]),
    }


def _update_team_state(state, won, margin, opp_elo, day_num, detail_row=None, role=None, opp_role=None):
    state["last_day_num"] = day_num
    state["margins"].append(margin)
    state["opp_elos"].append(opp_elo)
    state["win_streak"] = state["win_streak"] + 1 if won else 0

    if detail_row is not None and role is not None:
        off_poss = _possessions(
            detail_row[f"{role}FGA"], detail_row[f"{role}OR"],
            detail_row[f"{role}TO"], detail_row[f"{role}FTA"],
        )
        def_poss = _possessions(
            detail_row[f"{opp_role}FGA"], detail_row[f"{opp_role}OR"],
            detail_row[f"{opp_role}TO"], detail_row[f"{opp_role}FTA"],
        )
        if off_poss > 0:
            state["off_effs"].append(detail_row[f"{role}Score"] / off_poss * 100)
            state["to_rates"].append(detail_row[f"{role}TO"] / off_poss)
        if def_poss > 0:
            state["def_effs"].append(detail_row[f"{opp_role}Score"] / def_poss * 100)
        if detail_row[f"{role}FGA"] > 0:
            state["efg_pcts"].append(
                (detail_row[f"{role}FGM"] + 0.5 * detail_row[f"{role}FGM3"]) / detail_row[f"{role}FGA"]
            )
        if detail_row[f"{opp_role}FGA"] > 0:
            state["opp_efg_pcts"].append(
                (detail_row[f"{opp_role}FGM"] + 0.5 * detail_row[f"{opp_role}FGM3"]) / detail_row[f"{opp_role}FGA"]
            )
        total_or = detail_row[f"{role}OR"] + detail_row[f"{opp_role}DR"]
        if total_or > 0:
            state["or_pcts"].append(detail_row[f"{role}OR"] / total_or)
        avg_poss = (off_poss + def_poss) / 2 if (off_poss > 0 and def_poss > 0) else max(off_poss, def_poss)
        if avg_poss > 0:
            state["paces"].append(avg_poss)


# -- Pre-computed lookup helpers -----------------------------------------------

def _build_team_conf_lookup(conferences):
    return {
        (row.Season, row.TeamID): row.ConfAbbrev
        for row in conferences.itertuples()
    }


def _get_team_conf_elo(team_id, season, team_conf_lookup, conf_elo_means):
    prev_season = season - 1
    if prev_season not in conf_elo_means:
        return 1500.0
    conf = team_conf_lookup.get((season, team_id))
    if conf is None:
        return 1500.0
    return conf_elo_means[prev_season].get(conf, 1500.0)


def _build_barttorvik_lookup(barttorvik_df):
    """Build {(Season, TeamID): {Barthag, AdjO, AdjD}} from barttorvik_ratings.csv."""
    if barttorvik_df is None or barttorvik_df.empty:
        return {}
    lookup = {}
    for _, row in barttorvik_df.iterrows():
        lookup[(int(row["Season"]), int(row["TeamID"]))] = {
            "Barthag": row.get("Barthag", np.nan),
            "AdjO": row.get("AdjO", np.nan),
            "AdjD": row.get("AdjD", np.nan),
        }
    return lookup


# -- Main feature builder -----------------------------------------------------

def build_features(results, conferences, hfa_dict, location_dict,
                   seasons, detailed_results=None, initial_elo=1500, mean_reversion=0.3,
                   k_start=56, k_end=38, hfa_scalar=26, mov_avg=12,
                   massey_per_system=None, massey_avg=None,
                   coach_tenure=None, coach_changed=None,
                   barttorvik=None,
                   odds_lookup=None, odds_team_avg=None,
                   poll_lookup=None, weeks_ranked=None,
                   roster_lookup=None, seeds=None):
    """Run the Elo loop and collect per-game feature rows for specified seasons.

    Features are recorded *before* the Elo update for each game — no data leakage.

    Returns:
        (features_df, elo_ratings, game_counts, team_states, conf_elo_means)
    """
    all_seasons = sorted(results["Season"].unique())
    elo_ratings = {}
    game_counts = {}
    team_states = {}
    conf_elo_means = {}
    elo_snapshots = {}
    gc_snapshots = {}
    team_conf_lookup = _build_team_conf_lookup(conferences)
    barttorvik_lookup = _build_barttorvik_lookup(barttorvik)
    rows = []

    # Build detail lookup: {(Season, DayNum, WTeamID, LTeamID): row_dict}
    detail_lookup = {}
    if detailed_results is not None and not detailed_results.empty:
        for _, dr in detailed_results.iterrows():
            key = (dr["Season"], dr["DayNum"], dr["WTeamID"], dr["LTeamID"])
            detail_lookup[key] = dr

    for season in all_seasons:
        season_games = results[results["Season"] == season]
        active_teams = set(season_games["WTeamID"]).union(season_games["LTeamID"])

        prev_ratings = elo_ratings.copy()
        elo_ratings = {t: prev_ratings.get(t, initial_elo) for t in active_teams}

        season_counts = {t: 0 for t in active_teams}
        close_results = {t: [] for t in active_teams}
        collecting = season in seasons

        # Reset per-season rolling state
        for t in active_teams:
            team_states[t] = _init_team_state()

        for _, row in season_games.iterrows():
            winner, loser = row["WTeamID"], row["LTeamID"]
            w_loc = row["WLoc"]
            day_num = row["DayNum"]
            margin = row["WScore"] - row["LScore"]

            season_counts[winner] = season_counts.get(winner, 0) + 1
            season_counts[loser] = season_counts.get(loser, 0) + 1
            game_counts[winner] = game_counts.get(winner, 0) + 1
            game_counts[loser] = game_counts.get(loser, 0) + 1

            w_miles, l_miles = _travel_miles(winner, loser, w_loc, location_dict)
            winner_elo = elo_ratings[winner] + distance_to_elo_impact(w_miles)
            loser_elo = elo_ratings[loser] + distance_to_elo_impact(l_miles)

            if w_loc == "H":
                winner_elo += get_advantage(winner, hfa_dict, hfa_scalar)
            elif w_loc == "A":
                loser_elo += get_advantage(loser, hfa_dict, hfa_scalar)

            if collecting:
                low = min(winner, loser)
                high = max(winner, loser)
                low_base = elo_ratings[low]
                high_base = elo_ratings[high]

                if w_loc == "H":
                    home = 1 if winner == low else -1
                elif w_loc == "A":
                    home = -1 if winner == low else 1
                else:
                    home = 0

                # Compact rolling features (before update)
                low_compact = _extract_compact_features(team_states[low])
                high_compact = _extract_compact_features(team_states[high])
                low_rest = (day_num - team_states[low]["last_day_num"]
                            if team_states[low]["last_day_num"] is not None else 5)
                high_rest = (day_num - team_states[high]["last_day_num"]
                             if team_states[high]["last_day_num"] is not None else 5)

                # Detail rolling features (before update)
                low_detail = _extract_detail_features(team_states[low])
                high_detail = _extract_detail_features(team_states[high])

                feat = {
                    # Base
                    "elo_diff": low_base - high_base,
                    "elo_pred": calc_elo_win_tourney(low_base, high_base, boost=1.0),
                    "home": home,
                    "day_num": day_num,
                    "game_count_avg": (game_counts[low] + game_counts[high]) / 2,
                    # Compact rolling
                    "sos_diff": low_compact["sos"] - high_compact["sos"],
                    "win_streak_diff": low_compact["win_streak"] - high_compact["win_streak"],
                    "rest_days_diff": low_rest - high_rest,
                    "margin_mean_diff": low_compact["margin_mean"] - high_compact["margin_mean"],
                    "margin_std_diff": low_compact["margin_std"] - high_compact["margin_std"],
                    # Detail rolling
                    "off_eff_r10_diff": low_detail["off_eff_r10"] - high_detail["off_eff_r10"],
                    "def_eff_r10_diff": low_detail["def_eff_r10"] - high_detail["def_eff_r10"],
                    "efg_pct_r10_diff": low_detail["efg_pct_r10"] - high_detail["efg_pct_r10"],
                    "opp_efg_pct_r10_diff": low_detail["opp_efg_pct_r10"] - high_detail["opp_efg_pct_r10"],
                    "to_rate_r10_diff": low_detail["to_rate_r10"] - high_detail["to_rate_r10"],
                    "or_pct_r10_diff": low_detail["or_pct_r10"] - high_detail["or_pct_r10"],
                    "pace_r10_diff": low_detail["pace_r10"] - high_detail["pace_r10"],
                    # Metadata (not in FEATURE_COLS)
                    "team_low": low,
                    "team_high": high,
                    "win": 1 if winner == low else 0,
                    "season": season,
                }

                # Massey ordinals
                if massey_per_system and massey_avg:
                    from train.extended_features import massey_matchup_features
                    feat.update(massey_matchup_features(
                        massey_per_system, massey_avg, season, low, high))
                else:
                    for col in MASSEY_COLS:
                        feat[col] = np.nan

                # Coach features
                if coach_tenure and coach_changed:
                    from train.extended_features import coach_matchup_features
                    feat.update(coach_matchup_features(
                        coach_tenure, coach_changed, season, low, high))
                else:
                    feat["coach_tenure_diff"] = 0
                    feat["coach_change_diff"] = 0

                # Close-game win pct
                low_close = close_results.get(low, [])
                high_close = close_results.get(high, [])
                low_pct = np.mean(low_close) if len(low_close) >= 3 else 0.5
                high_pct = np.mean(high_close) if len(high_close) >= 3 else 0.5
                feat["close_win_pct_diff"] = low_pct - high_pct

                # Conference Elo
                low_conf_elo = _get_team_conf_elo(low, season, team_conf_lookup, conf_elo_means)
                high_conf_elo = _get_team_conf_elo(high, season, team_conf_lookup, conf_elo_means)
                feat["conf_elo_diff"] = low_conf_elo - high_conf_elo

                # Barttorvik (use previous season's ratings for leak-free training)
                low_bt = barttorvik_lookup.get((season - 1, low), {})
                high_bt = barttorvik_lookup.get((season - 1, high), {})
                feat["barthag_diff"] = low_bt.get("Barthag", np.nan) - high_bt.get("Barthag", np.nan)
                feat["trank_adjO_diff"] = low_bt.get("AdjO", np.nan) - high_bt.get("AdjO", np.nan)
                feat["trank_adjD_diff"] = low_bt.get("AdjD", np.nan) - high_bt.get("AdjD", np.nan)

                # Odds
                if odds_lookup:
                    from train.extended_features import odds_game_features
                    feat.update(odds_game_features(odds_lookup, season, day_num, low, high))
                else:
                    for col in ODDS_COLS:
                        feat[col] = np.nan

                # Polls
                if poll_lookup and weeks_ranked is not None:
                    from train.extended_features import poll_game_features
                    feat.update(poll_game_features(poll_lookup, weeks_ranked, season, day_num, low, high))
                else:
                    for col in POLL_COLS:
                        feat[col] = 0

                # Roster
                if roster_lookup:
                    from train.extended_features import roster_matchup_features
                    feat.update(roster_matchup_features(roster_lookup, season, low, high))
                else:
                    for col in ROSTER_COLS:
                        feat[col] = np.nan

                # Seeds (NaN for regular season — only available at tournament time)
                if seeds:
                    from train.extended_features import seed_matchup_features
                    feat.update(seed_matchup_features(seeds, season, low, high))
                else:
                    feat["seed_diff"] = np.nan

                rows.append(feat)

            # Track close games
            if margin <= 5:
                close_results.setdefault(winner, []).append(1)
                close_results.setdefault(loser, []).append(0)

            # Update team states AFTER feature extraction
            detail_row = detail_lookup.get((season, day_num, winner, loser))
            _update_team_state(
                team_states[winner], won=True, margin=margin,
                opp_elo=elo_ratings[loser], day_num=day_num,
                detail_row=detail_row, role="W", opp_role="L",
            )
            _update_team_state(
                team_states[loser], won=False, margin=-margin,
                opp_elo=elo_ratings[winner], day_num=day_num,
                detail_row=detail_row, role="L", opp_role="W",
            )

            # K-factor and Elo update
            game_num = (season_counts[winner] + season_counts[loser]) / 2
            k = k_factor(game_num, k_start, k_end)
            k_coeff = point_differential_scaler(row["WScore"], row["LScore"], mov_avg)

            elo_change = update_elo(winner_elo, loser_elo, k * k_coeff)
            elo_ratings[winner] += elo_change
            elo_ratings[loser] -= elo_change

        # Conference mean reversion
        df = pd.DataFrame(list(elo_ratings.items()), columns=["TeamID", "Elo"])
        league_mean = df["Elo"].mean()
        divergence = initial_elo - league_mean

        season_conf = conferences[conferences["Season"] == season][["TeamID", "ConfAbbrev"]]
        df = df.merge(season_conf, on="TeamID", how="left")
        df["ConfAbbrev"] = df["ConfAbbrev"].fillna("unknown")
        conf_means = df.groupby("ConfAbbrev")["Elo"].mean().to_dict()
        conf_elo_means[season] = conf_means

        elo_ratings = {
            r.TeamID: (
                (1 - mean_reversion) * r.Elo
                + mean_reversion * conf_means.get(r.ConfAbbrev, initial_elo)
            ) * (1 + divergence / initial_elo)
            for r in df.itertuples()
        }

        # Save per-season snapshot (for caching / backtesting)
        elo_snapshots[season] = dict(elo_ratings)
        gc_snapshots[season] = dict(game_counts)

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        features_df = pd.DataFrame(columns=FEATURE_COLS + ["team_low", "team_high", "win", "season"])

    # Store snapshots in module-level dict for callers that need per-season Elo
    _last_snapshots["elo"] = elo_snapshots
    _last_snapshots["gc"] = gc_snapshots

    return features_df, elo_ratings, game_counts, team_states, conf_elo_means


def matchup_features(elo_a, elo_b, game_counts, team_a, team_b,
                     team_states=None, season=None, seeds=None,
                     massey_per_system=None, massey_avg=None,
                     coach_tenure=None, coach_changed=None,
                     conf_elo_means=None, conferences=None,
                     barttorvik_lookup=None,
                     odds_team_avg=None,
                     poll_lookup=None, weeks_ranked=None,
                     roster_lookup=None):
    """Build a feature dict for a neutral-site tournament matchup.

    team_a should be the lower TeamID. Uses current-season data for lookups.
    """
    feat = {
        "elo_diff": elo_a - elo_b,
        "elo_pred": calc_elo_win_tourney(elo_a, elo_b, boost=1.0),
        "home": 0,
        "day_num": 136,
        "game_count_avg": (game_counts.get(team_a, 30) + game_counts.get(team_b, 30)) / 2,
    }

    # Compact rolling
    if team_states and team_a in team_states and team_b in team_states:
        a_compact = _extract_compact_features(team_states[team_a])
        b_compact = _extract_compact_features(team_states[team_b])
        feat["sos_diff"] = a_compact["sos"] - b_compact["sos"]
        feat["win_streak_diff"] = a_compact["win_streak"] - b_compact["win_streak"]
        feat["rest_days_diff"] = 0
        feat["margin_mean_diff"] = a_compact["margin_mean"] - b_compact["margin_mean"]
        feat["margin_std_diff"] = a_compact["margin_std"] - b_compact["margin_std"]

        a_detail = _extract_detail_features(team_states[team_a])
        b_detail = _extract_detail_features(team_states[team_b])
        feat["off_eff_r10_diff"] = a_detail["off_eff_r10"] - b_detail["off_eff_r10"]
        feat["def_eff_r10_diff"] = a_detail["def_eff_r10"] - b_detail["def_eff_r10"]
        feat["efg_pct_r10_diff"] = a_detail["efg_pct_r10"] - b_detail["efg_pct_r10"]
        feat["opp_efg_pct_r10_diff"] = a_detail["opp_efg_pct_r10"] - b_detail["opp_efg_pct_r10"]
        feat["to_rate_r10_diff"] = a_detail["to_rate_r10"] - b_detail["to_rate_r10"]
        feat["or_pct_r10_diff"] = a_detail["or_pct_r10"] - b_detail["or_pct_r10"]
        feat["pace_r10_diff"] = a_detail["pace_r10"] - b_detail["pace_r10"]
    else:
        for col in COMPACT_COLS:
            feat[col] = 0.0
        for col in DETAIL_COLS:
            feat[col] = np.nan

    # Seeds
    if seeds and season:
        from train.extended_features import seed_matchup_features
        feat.update(seed_matchup_features(seeds, season, team_a, team_b))
    else:
        feat["seed_diff"] = np.nan

    # Massey
    if massey_per_system and massey_avg and season:
        from train.extended_features import massey_matchup_features
        feat.update(massey_matchup_features(
            massey_per_system, massey_avg, season, team_a, team_b))
    else:
        for col in MASSEY_COLS:
            feat[col] = np.nan

    # Coach
    if coach_tenure and coach_changed and season:
        from train.extended_features import coach_matchup_features
        feat.update(coach_matchup_features(
            coach_tenure, coach_changed, season, team_a, team_b))
    else:
        feat["coach_tenure_diff"] = 0
        feat["coach_change_diff"] = 0

    feat["close_win_pct_diff"] = 0.0

    # Conference Elo
    if conf_elo_means and conferences is not None and season:
        tcl = _build_team_conf_lookup(conferences)
        a_conf_elo = _get_team_conf_elo(team_a, season, tcl, conf_elo_means)
        b_conf_elo = _get_team_conf_elo(team_b, season, tcl, conf_elo_means)
        feat["conf_elo_diff"] = a_conf_elo - b_conf_elo
    else:
        feat["conf_elo_diff"] = 0.0

    # Barttorvik (use current season for tournament predictions)
    if barttorvik_lookup and season:
        a_bt = barttorvik_lookup.get((season, team_a), {})
        b_bt = barttorvik_lookup.get((season, team_b), {})
        feat["barthag_diff"] = a_bt.get("Barthag", np.nan) - b_bt.get("Barthag", np.nan)
        feat["trank_adjO_diff"] = a_bt.get("AdjO", np.nan) - b_bt.get("AdjO", np.nan)
        feat["trank_adjD_diff"] = a_bt.get("AdjD", np.nan) - b_bt.get("AdjD", np.nan)
    else:
        feat["barthag_diff"] = np.nan
        feat["trank_adjO_diff"] = np.nan
        feat["trank_adjD_diff"] = np.nan

    # Odds (season-average spread as tournament proxy)
    if odds_team_avg and season:
        from train.extended_features import odds_matchup_features
        feat.update(odds_matchup_features(odds_team_avg, season, team_a, team_b))
    else:
        for col in ODDS_COLS:
            feat[col] = np.nan

    # Polls (latest week)
    if poll_lookup and weeks_ranked is not None and season:
        from train.extended_features import poll_matchup_features
        feat.update(poll_matchup_features(poll_lookup, weeks_ranked, season, team_a, team_b))
    else:
        for col in POLL_COLS:
            feat[col] = 0

    # Roster
    if roster_lookup and season:
        from train.extended_features import roster_matchup_features
        feat.update(roster_matchup_features(roster_lookup, season, team_a, team_b))
    else:
        for col in ROSTER_COLS:
            feat[col] = np.nan

    return feat
