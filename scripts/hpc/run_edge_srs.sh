#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/run_esrs_%j.out
#SBATCH --error=logs/run_esrs_%j.err

set -eo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -u -c "
import sys, numpy as np, pandas as pd
from itertools import combinations
sys.path.insert(0, '.')
from train.evaluate import prepare_data
from train.features import FEATURE_COLS, build_features
from train.extended_features import _spread_to_prob
from train.custom_ratings import compute_colley, compute_srs
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = prepare_data('$LOCAL/data')
results = add_game_counts(data['mens_results'].copy())

# Precompute Colley + SRS
print('Computing Colley + SRS...')
colley_all, srs_all = {}, {}
for season in range(2003, 2027):
    if season in data['mens_results']['Season'].values:
        for t, r in compute_colley(data['mens_results'], season).items(): colley_all[(season, t)] = r
        for t, r in compute_srs(data['mens_results'], season).items(): srs_all[(season, t)] = r

print('Building features...')
features_df, _, _, _, _ = build_features(
    results=results, conferences=data['mens_conf'],
    hfa_dict=_build_hfa_dict(data['hfa'], 'M'),
    location_dict=_build_location_dict(data['home_lookup'], 'M'),
    seasons=set(range(2013, 2026)),
    detailed_results=data.get('mens_detailed'),
    massey_per_system=data.get('massey_per_system', {}),
    massey_avg=data.get('massey_avg', {}),
    coach_tenure=data.get('coach_tenure', {}),
    coach_changed=data.get('coach_changed', {}),
    barttorvik=data.get('barttorvik'),
    odds_lookup=data.get('odds_lookup', {}),
    odds_team_avg=data.get('odds_team_avg', {}),
    poll_lookup=data.get('poll_lookup', {}),
    weeks_ranked=data.get('weeks_ranked', {}),
    roster_lookup=data.get('roster_lookup', {}),
    seeds=data.get('m_seeds', {}),
)

X = features_df[FEATURE_COLS].values
y = features_df['win'].values
spreads = features_df['spread'].values
has_odds = ~np.isnan(spreads)
seasons = features_df['season'].values
daynums = features_df['day_num'].values
team_lows = features_df['team_low'].values
team_highs = features_df['team_high'].values

# Add colley_diff and srs_diff as extra columns
colley_diffs = np.array([colley_all.get((int(s), int(tl)), 175) - colley_all.get((int(s), int(th)), 175)
                         for s, tl, th in zip(seasons, team_lows, team_highs)])
srs_diffs = np.array([srs_all.get((int(s), int(tl)), 0) - srs_all.get((int(s), int(th)), 0)
                       for s, tl, th in zip(seasons, team_lows, team_highs)])
X_aug = np.column_stack([X, colley_diffs, srs_diffs])
N_BASE = len(FEATURE_COLS)
COLLEY_IDX = N_BASE
SRS_IDX = N_BASE + 1

# Load ML odds
odds = pd.read_csv('$LOCAL/data/derived/ncaab_odds.csv')
odds = odds[(odds['HomeML'] != 0) & (odds['AwayML'] != 0) & odds['HomeML'].notna() & odds['AwayML'].notna()]
ml_lookup = {}
for _, row in odds.iterrows():
    home, away = int(row['HomeTeamID']), int(row['AwayTeamID'])
    low, high = min(home, away), max(home, away)
    key = (int(row['Season']), int(row['DayNum']), low, high)
    if home == low:
        ml_lookup[key] = {'low_ml': row['HomeML'], 'high_ml': row['AwayML']}
    else:
        ml_lookup[key] = {'low_ml': row['AwayML'], 'high_ml': row['HomeML']}

def ml_to_payout(ml):
    if ml > 0: return ml / 100
    else: return 100 / abs(ml)

def evaluate_edge(feat_indices, C=1.0, vegas_lo=0.45, vegas_hi=0.55, edge_thresh=0.15):
    all_bets = []
    for year in range(2016, 2026):
        train_mask = (seasons < year) & has_odds
        test_mask = (seasons == year) & has_odds
        if test_mask.sum() == 0: continue
        m = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(C=C, max_iter=1000))])
        m.fit(np.nan_to_num(X_aug[train_mask][:, feat_indices], nan=0.0), y[train_mask])
        mp = m.predict_proba(np.nan_to_num(X_aug[test_mask][:, feat_indices], nan=0.0))[:, 1]
        vp = np.array([_spread_to_prob(s) for s in spreads[test_mask]])
        yt = y[test_mask]
        dn = daynums[test_mask]; tl = team_lows[test_mask]; th = team_highs[test_mask]
        sweet = (vp >= vegas_lo) & (vp <= vegas_hi) & (np.abs(mp - vp) >= edge_thresh)
        for i in np.where(sweet)[0]:
            key = (year, int(dn[i]), int(tl[i]), int(th[i]))
            ml = ml_lookup.get(key)
            if ml is None: continue
            bet_on_low = mp[i] > vp[i]
            won = (yt[i] == 1) == bet_on_low
            payout = ml_to_payout(ml['low_ml'] if bet_on_low else ml['high_ml'])
            all_bets.append({'won': won, 'profit': payout if won else -1.0})
    if not all_bets: return 0, 0, 0, 0
    n = len(all_bets)
    wins = sum(b['won'] for b in all_bets)
    profit = sum(b['profit'] for b in all_bets)
    return n, wins/n, profit/n, profit

# Feature name mapping
FEAT_NAMES = list(FEATURE_COLS) + ['colley_diff', 'srs_diff']

# Test the previous best + SRS and Colley
print('=== Edge Model with SRS + Colley ===')
print()

BASE_BEST = [FEATURE_COLS.index('home'), FEATURE_COLS.index('massey_avg_diff'),
             FEATURE_COLS.index('margin_mean_diff'), FEATURE_COLS.index('conf_elo_diff')]

configs = [
    ('Previous best (4 feat)', BASE_BEST),
    ('+ srs_diff', BASE_BEST + [SRS_IDX]),
    ('+ colley_diff', BASE_BEST + [COLLEY_IDX]),
    ('+ srs + colley', BASE_BEST + [SRS_IDX, COLLEY_IDX]),
    ('srs replacing conf_elo', [FEATURE_COLS.index('home'), FEATURE_COLS.index('massey_avg_diff'),
                                 FEATURE_COLS.index('margin_mean_diff'), SRS_IDX]),
    ('home + massey + srs', [FEATURE_COLS.index('home'), FEATURE_COLS.index('massey_avg_diff'), SRS_IDX]),
    ('home + massey + srs + conf_elo', [FEATURE_COLS.index('home'), FEATURE_COLS.index('massey_avg_diff'),
                                         SRS_IDX, FEATURE_COLS.index('conf_elo_diff')]),
]

for label, idx in configs:
    for vlo, vhi, thresh in [(0.45, 0.55, 0.15), (0.45, 0.55, 0.20), (0.43, 0.57, 0.15), (0.47, 0.53, 0.20)]:
        n, wr, roi, profit = evaluate_edge(idx, C=1.0, vegas_lo=vlo, vegas_hi=vhi, edge_thresh=thresh)
        if n > 20:
            marker = ' ***' if wr > 0.85 else ''
            print(f'  win={wr:.1%} bets={n:3d} ROI={roi:+.1%} profit={profit:+.0f}u  v=[{vlo},{vhi}] e>{thresh}  {label}{marker}')

# Also test all 3-4 feature combos that include SRS
print()
print('=== Best combos including SRS ===')
CANDIDATES = [FEATURE_COLS.index('home'), FEATURE_COLS.index('massey_avg_diff'),
              FEATURE_COLS.index('margin_mean_diff'), FEATURE_COLS.index('conf_elo_diff'),
              FEATURE_COLS.index('elo_diff'), FEATURE_COLS.index('barthag_diff'),
              FEATURE_COLS.index('sos_diff'), SRS_IDX, COLLEY_IDX]

best = []
for size in [3, 4, 5]:
    for combo in combinations(CANDIDATES, size):
        if SRS_IDX not in combo: continue
        idx = list(combo)
        n, wr, roi, profit = evaluate_edge(idx, C=1.0, vegas_lo=0.45, vegas_hi=0.55, edge_thresh=0.15)
        if n > 30:
            feat_names = [FEAT_NAMES[i] for i in idx]
            best.append((wr, n, roi, profit, feat_names))

best.sort(key=lambda x: -x[0])
for wr, n, roi, profit, feats in best[:10]:
    print(f'  win={wr:.1%} bets={n:3d} ROI={roi:+.1%} profit={profit:+.0f}u  {feats}')
" 2>&1