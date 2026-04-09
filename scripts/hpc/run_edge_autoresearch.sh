#!/bin/bash
#SBATCH --job-name=opt
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/opt_edge_%j.out
#SBATCH --error=logs/opt_edge_%j.err

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
import sys, time, numpy as np, pandas as pd
from itertools import combinations
sys.path.insert(0, '.')
from train.evaluate import prepare_data
from train.features import FEATURE_COLS, build_features
from train.extended_features import _spread_to_prob
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = prepare_data('$LOCAL/data')
results = add_game_counts(data['mens_results'].copy())
hfa = _build_hfa_dict(data['hfa'], 'M')
loc = _build_location_dict(data['home_lookup'], 'M')

features_df, _, _, _, _ = build_features(
    results=results, conferences=data['mens_conf'],
    hfa_dict=hfa, location_dict=loc,
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

def evaluate_edge(feat_names, model_type='lr', C=1.0, depth=2, vegas_lo=0.45, vegas_hi=0.55, edge_thresh=0.15):
    feat_idx = [FEATURE_COLS.index(c) for c in feat_names]
    all_bets = []
    for year in range(2016, 2026):
        train_mask = (seasons < year) & has_odds
        test_mask = (seasons == year) & has_odds
        if test_mask.sum() == 0: continue
        if model_type == 'lr':
            m = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(C=C, max_iter=1000))])
        else:
            m = XGBClassifier(n_estimators=500, max_depth=depth, learning_rate=0.03,
                reg_lambda=5.0, subsample=0.8, colsample_bytree=0.7, random_state=42, eval_metric='logloss', verbosity=0)
        m.fit(np.nan_to_num(X[train_mask][:, feat_idx], nan=0.0), y[train_mask])
        mp = m.predict_proba(np.nan_to_num(X[test_mask][:, feat_idx], nan=0.0))[:, 1]
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

print('=== Edge Model Autoresearch ===')
print()

# Phase 1: Feature subset search around minimal5
ALL_CANDIDATES = ['elo_diff', 'elo_pred', 'home', 'day_num', 'sos_diff', 'margin_mean_diff',
    'massey_avg_diff', 'barthag_diff', 'trank_adjO_diff', 'trank_adjD_diff', 'conf_elo_diff',
    'off_eff_r10_diff', 'def_eff_r10_diff', 'efg_pct_r10_diff', 'pace_r10_diff',
    'margin_std_diff', 'win_streak_diff', 'rest_days_diff']

MINIMAL5 = ['elo_diff', 'elo_pred', 'home', 'massey_avg_diff', 'margin_mean_diff']

# Test minimal5 baseline
n, wr, roi, profit = evaluate_edge(MINIMAL5, 'lr', C=1.0)
print(f'  LR minimal5 C=1:     bets={n:3d} win={wr:.1%} ROI={roi:+.1%} profit={profit:+.0f}u')

# Test adding each candidate to minimal5
print()
print('=== Add one feature to minimal5 ===')
for feat in ALL_CANDIDATES:
    if feat in MINIMAL5: continue
    feats = MINIMAL5 + [feat]
    n, wr, roi, profit = evaluate_edge(feats, 'lr', C=1.0)
    marker = ' ***' if wr > 0.78 else ''
    print(f'  +{feat:25s}: bets={n:3d} win={wr:.1%} ROI={roi:+.1%} profit={profit:+.0f}u{marker}')

# Test dropping each from minimal5
print()
print('=== Drop one from minimal5 ===')
for feat in MINIMAL5:
    feats = [f for f in MINIMAL5 if f != feat]
    n, wr, roi, profit = evaluate_edge(feats, 'lr', C=1.0)
    print(f'  -{feat:25s}: bets={n:3d} win={wr:.1%} ROI={roi:+.1%} profit={profit:+.0f}u')

# Phase 2: C-value sweep on minimal5
print()
print('=== C-value sweep (LR minimal5) ===')
for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
    n, wr, roi, profit = evaluate_edge(MINIMAL5, 'lr', C=C)
    print(f'  C={C:6.2f}: bets={n:3d} win={wr:.1%} ROI={roi:+.1%} profit={profit:+.0f}u')

# Phase 3: Vegas range sweep
print()
print('=== Vegas range sweep (LR minimal5 C=1) ===')
for lo, hi in [(0.45, 0.55), (0.43, 0.57), (0.40, 0.60), (0.47, 0.53)]:
    for thresh in [0.10, 0.15, 0.20]:
        n, wr, roi, profit = evaluate_edge(MINIMAL5, 'lr', C=1.0, vegas_lo=lo, vegas_hi=hi, edge_thresh=thresh)
        if n > 20:
            print(f'  vegas=[{lo},{hi}] edge>{thresh}: bets={n:4d} win={wr:.1%} ROI={roi:+.1%} profit={profit:+.0f}u')

# Phase 4: Best combos of features (3-7 feature subsets)
print()
print('=== Best 4-feature subsets ===')
best_4 = []
TOP_FEATS = ['elo_diff', 'elo_pred', 'home', 'massey_avg_diff', 'margin_mean_diff',
             'barthag_diff', 'sos_diff', 'conf_elo_diff']
for combo in combinations(TOP_FEATS, 4):
    feats = list(combo)
    n, wr, roi, profit = evaluate_edge(feats, 'lr', C=1.0)
    if n > 50:
        best_4.append((wr, n, roi, profit, feats))
best_4.sort(key=lambda x: -x[0])
for wr, n, roi, profit, feats in best_4[:10]:
    print(f'  win={wr:.1%} bets={n:3d} ROI={roi:+.1%}  {feats}')
" 2>&1