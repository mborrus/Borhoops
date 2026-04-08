"""Phase 2d: LSTM temporal model (PyTorch).

Represents each team's season as a sequence of per-game feature vectors.
Siamese LSTM encodes both teams, then an MLP predicts from the embedding diff.

  Team A: [game1, game2, ..., gameN] → LSTM → embedding_A
  Team B: [game1, game2, ..., gameN] → LSTM → embedding_B
  P(A wins) = sigmoid(MLP(embedding_A - embedding_B))

Usage:
  PYTHONPATH=src python src/train/train_lstm.py
  PYTHONPATH=src python src/train/train_lstm.py --hidden-size 64 --epochs 100
"""

import argparse
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from train.evaluate import (
    prepare_data, brier_score, log_loss,
    TOURNEY_YEARS, save_results, _load_tourney_results, _get_gender_data,
)
from train.features import build_features, FEATURE_COLS, _build_barttorvik_lookup
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict


# Per-game stats tracked for each team (subset of features that are per-game, not diffs)
GAME_STATS = [
    "margin", "rest_days", "off_eff", "def_eff", "efg_pct",
    "opp_efg_pct", "to_rate", "or_pct", "pace", "opp_elo", "won",
]


def _build_team_sequences(features_df, season):
    """Build per-team game sequences from the features DataFrame.

    Returns {TeamID: np.array of shape (n_games, n_stats)} for one season.
    """
    season_df = features_df[features_df["season"] == season].copy()
    if season_df.empty:
        return {}

    sequences = defaultdict(list)

    for _, row in season_df.iterrows():
        low, high = row["team_low"], row["team_high"]
        day = row["day_num"]

        # For each team, record their perspective of this game
        for team, is_low in [(low, True), (high, False)]:
            sign = 1 if is_low else -1
            game_vec = [
                sign * row.get("margin_mean_diff", 0),
                sign * row.get("rest_days_diff", 0),
                sign * row.get("off_eff_r10_diff", 0),
                sign * row.get("def_eff_r10_diff", 0),
                sign * row.get("efg_pct_r10_diff", 0),
                sign * row.get("opp_efg_pct_r10_diff", 0),
                sign * row.get("to_rate_r10_diff", 0),
                sign * row.get("or_pct_r10_diff", 0),
                sign * row.get("pace_r10_diff", 0),
                sign * row.get("sos_diff", 0),
                row["win"] if is_low else 1 - row["win"],
            ]
            sequences[team].append((day, game_vec))

    # Sort by day and convert to arrays
    result = {}
    for team, games in sequences.items():
        games.sort(key=lambda x: x[0])
        result[team] = np.array([g[1] for g in games], dtype=np.float32)

    return result


class SiameseLSTM:
    """Siamese LSTM for matchup prediction."""

    def __init__(self, input_size=11, hidden_size=32, mlp_hidden=16,
                 lr=0.001, epochs=50, batch_size=64, max_seq_len=35):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mlp_hidden = mlp_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self._lstm = None
        self._mlp = None

    def _build(self):
        import torch
        import torch.nn as nn

        self._lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self._mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, 1),
        )

    def _encode_sequences(self, sequences, team_ids):
        """Encode a batch of team sequences into embeddings."""
        import torch

        batch = []
        lengths = []
        for tid in team_ids:
            seq = sequences.get(tid)
            if seq is None:
                seq = np.zeros((1, self.input_size), dtype=np.float32)
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            batch.append(seq)
            lengths.append(len(seq))

        # Pad to max length in batch
        max_len = max(lengths)
        padded = np.zeros((len(batch), max_len, self.input_size), dtype=np.float32)
        for i, seq in enumerate(batch):
            padded[i, :len(seq)] = seq

        x = torch.FloatTensor(padded)
        # Pack padded sequence for efficiency
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self._lstm(packed)
        return h_n.squeeze(0)  # (batch, hidden_size)

    def fit(self, matchups, sequences_by_season, outcomes):
        """Train the LSTM on matchup data.

        Args:
            matchups: list of (season, team_a, team_b) tuples
            sequences_by_season: {season: {team_id: np.array}}
            outcomes: np.array of 0/1 (did team_a win?)
        """
        import torch
        import torch.nn as nn

        self._build()
        optimizer = torch.optim.Adam(
            list(self._lstm.parameters()) + list(self._mlp.parameters()),
            lr=self.lr,
        )
        criterion = nn.BCEWithLogitsLoss()
        y = torch.FloatTensor(outcomes)

        for epoch in range(self.epochs):
            idx = np.random.permutation(len(matchups))
            total_loss = 0
            n_batches = 0

            for start in range(0, len(idx), self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                batch_matchups = [matchups[i] for i in batch_idx]

                # Get sequences for this batch
                a_ids = [m[1] for m in batch_matchups]
                b_ids = [m[2] for m in batch_matchups]
                seasons = [m[0] for m in batch_matchups]

                # Use correct season's sequences for each matchup
                a_seqs = {}
                b_seqs = {}
                for s, a, b in batch_matchups:
                    s_seqs = sequences_by_season.get(s, {})
                    a_seqs[a] = s_seqs.get(a, np.zeros((1, self.input_size), dtype=np.float32))
                    b_seqs[b] = s_seqs.get(b, np.zeros((1, self.input_size), dtype=np.float32))

                emb_a = self._encode_sequences(a_seqs, a_ids)
                emb_b = self._encode_sequences(b_seqs, b_ids)

                logits = self._mlp(emb_a - emb_b).squeeze(1)
                loss = criterion(logits, y[batch_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return self

    def predict_proba_matchups(self, matchups, sequences_by_season):
        """Predict win probabilities for a list of matchups."""
        import torch

        self._lstm.eval()
        self._mlp.eval()

        all_probs = []
        with torch.no_grad():
            for start in range(0, len(matchups), self.batch_size):
                batch = matchups[start:start + self.batch_size]
                a_ids = [m[1] for m in batch]
                b_ids = [m[2] for m in batch]

                a_seqs = {}
                b_seqs = {}
                for s, a, b in batch:
                    s_seqs = sequences_by_season.get(s, {})
                    a_seqs[a] = s_seqs.get(a, np.zeros((1, self.input_size), dtype=np.float32))
                    b_seqs[b] = s_seqs.get(b, np.zeros((1, self.input_size), dtype=np.float32))

                emb_a = self._encode_sequences(a_seqs, a_ids)
                emb_b = self._encode_sequences(b_seqs, b_ids)
                logits = self._mlp(emb_a - emb_b).squeeze(1)
                probs = torch.sigmoid(logits).numpy()
                all_probs.extend(probs)

        return np.array(all_probs)

    def predict_proba(self, X):
        raise NotImplementedError("Use predict_proba_matchups for LSTM")


def run_lstm_loyo(data, gender="M", data_dir="data", params=None, years=None):
    """LOYO CV for the LSTM model. Precomputes features once, then splits by fold."""
    if years is None:
        years = TOURNEY_YEARS
    if params is None:
        params = {}

    gd = _get_gender_data(data, gender)
    results_df = add_game_counts(data[gd["results_key"]].copy())
    hfa_dict = _build_hfa_dict(data["hfa"], gender)
    location_dict = _build_location_dict(data["home_lookup"], gender)

    # Precompute features ONCE for all seasons
    print("  Precomputing features for all seasons...")
    all_seasons = sorted(results_df["Season"].unique())
    features_df, elo_ratings, game_counts, team_states, conf_elo_means = build_features(
        results=results_df,
        conferences=data[gd["conf_key"]],
        hfa_dict=hfa_dict,
        location_dict=location_dict,
        seasons=set(all_seasons),
        detailed_results=gd["detailed"],
        massey_per_system=gd["massey_per"],
        massey_avg=gd["massey_avg"],
        coach_tenure=gd["coach_tenure"],
        coach_changed=gd["coach_changed"],
        barttorvik=gd["barttorvik"],
        odds_lookup=gd["odds_lookup"],
        odds_team_avg=gd["odds_team_avg"],
        poll_lookup=gd["poll_lookup"],
        weeks_ranked=gd["weeks_ranked"],
        roster_lookup=gd["roster_lookup"],
        seeds=gd["seeds"],
    )

    # Precompute all per-season team sequences
    print("  Building team sequences...")
    sequences_by_season = {}
    for s in all_seasons:
        sequences_by_season[s] = _build_team_sequences(features_df, s)

    tourney = _load_tourney_results(data_dir, gender)
    results = {}
    all_brier = []

    for year in years:
        t0 = time.time()

        # Training matchups (regular season, excluding target year)
        train_df = features_df[features_df["season"] != year]
        train_matchups = list(zip(
            train_df["season"].values,
            train_df["team_low"].values.astype(int),
            train_df["team_high"].values.astype(int),
        ))
        train_outcomes = train_df["win"].values

        # Tournament test matchups
        tourney_year = tourney[tourney["Season"] == year]
        test_matchups = list(zip(
            tourney_year["Season"].values,
            tourney_year["LowTeam"].values.astype(int),
            tourney_year["HighTeam"].values.astype(int),
        ))
        y_test = tourney_year["LowTeamWon"].values

        model = SiameseLSTM(**params)
        model.fit(train_matchups, sequences_by_season, train_outcomes)
        y_pred = model.predict_proba_matchups(test_matchups, sequences_by_season)

        bs = brier_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        elapsed = time.time() - t0

        results[year] = {
            "brier": round(bs, 4),
            "log_loss": round(ll, 4),
            "n_games": len(y_test),
            "elapsed_s": round(elapsed, 1),
        }
        all_brier.append(bs)
        print(f"  {year}: Brier={bs:.4f}  ({elapsed:.1f}s)")

    results["overall"] = {
        "mean_brier": round(float(np.mean(all_brier)), 4),
        "std_brier": round(float(np.std(all_brier)), 4),
        "n_folds": len(years),
    }
    o = results["overall"]
    print(f"\n  LOYO CV: Brier = {o['mean_brier']:.4f} ± {o['std_brier']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="LSTM temporal model")
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/lstm")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    data = prepare_data(args.data_dir)
    params = {"hidden_size": args.hidden_size, "epochs": args.epochs}
    results = run_lstm_loyo(data, args.gender, args.data_dir, params)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"lstm_{args.gender}_{timestamp}.json"
    save_results(results, path, model_name="lstm_temporal", extra={"params": params})
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
