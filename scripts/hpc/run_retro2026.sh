#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/eval_retro_%j.out
#SBATCH --error=logs/eval_retro_%j.err

set -eo pipefail

LOCAL=/scratch/$SLURM_JOB_ID
mkdir -p $LOCAL
trap "rm -rf $LOCAL" EXIT

cp -r /home/mborrus/Borhoops/data $LOCAL/data
cp -r /home/mborrus/Borhoops/src $LOCAL/src
cp /tmp/tourney_2026_espn.csv $LOCAL/tourney_2026_espn.csv
cp /tmp/original_2026_submission.csv $LOCAL/original_2026_submission.csv

source ~/miniconda3/etc/profile.d/conda.sh
conda activate borhoops

cd $LOCAL/src
PYTHONPATH=$LOCAL/src python -u -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from train.custom_ratings import compute_colley, compute_srs
from predict.submission import load_data
from predict.elo import add_game_counts
from predict.submission import _build_hfa_dict, _build_location_dict
from train.features import FEATURE_COLS, build_features, matchup_features, _build_barttorvik_lookup
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from evaluate.metrics import brier_score

DATA_DIR = '$LOCAL/data'
FEAT_BASE = ['elo_diff', 'elo_pred', 'home', 'day_num', 'sos_diff', 'margin_mean_diff',
             'massey_avg_diff', 'barthag_diff', 'trank_adjO_diff', 'trank_adjD_diff', 'conf_elo_diff']
FEAT_IDX = [FEATURE_COLS.index(c) for c in FEAT_BASE]
data = load_data(DATA_DIR)

colley_all, srs_all = {}, {}
for gk, rk in [('M', 'mens_results'), ('W', 'womens_results')]:
    for season in range(2003, 2027):
        if season in data[rk]['Season'].values:
            for t, r in compute_colley(data[rk], season).items(): colley_all[(gk, season, t)] = r
            for t, r in compute_srs(data[rk], season).items(): srs_all[(gk, season, t)] = r

tourney_espn = pd.read_csv('$LOCAL/tourney_2026_espn.csv')
tourney_espn = tourney_espn[~tourney_espn['notes'].str.contains('First Four|Play-In', case=False, na=False)]
cw = pd.read_csv(f'{DATA_DIR}/derived/espn_kaggle_crosswalk.csv')
espn_to_m = dict(zip(cw['espn_id'], cw['kaggle_m_id']))
espn_to_w = dict(zip(cw['espn_id'], cw['kaggle_w_id']))
orig_sub = pd.read_csv('$LOCAL/original_2026_submission.csv')
orig_lookup = dict(zip(orig_sub['ID'], orig_sub['Pred']))

all_preds = {}
for gender in ['M', 'W']:
    is_mens = gender == 'M'
    C = 100 if is_mens else 0.15
    rk = 'mens_results' if is_mens else 'womens_results'
    ck = 'mens_conf' if is_mens else 'womens_conf'
    results = add_game_counts(data[rk].copy())
    _, elo, gc, ts, cem = build_features(
        results=results, conferences=data[ck],
        hfa_dict=_build_hfa_dict(data['hfa'], gender),
        location_dict=_build_location_dict(data['home_lookup'], gender),
        seasons=set(),
        detailed_results=data.get('mens_detailed' if is_mens else 'womens_detailed'),
        massey_per_system=data.get('massey_per_system',{}) if is_mens else {},
        massey_avg=data.get('massey_avg',{}) if is_mens else {},
        coach_tenure=data.get('coach_tenure',{}) if is_mens else {},
        coach_changed=data.get('coach_changed',{}) if is_mens else {},
        barttorvik=data.get('barttorvik') if is_mens else None,
    )
    bt_lk = _build_barttorvik_lookup(data.get('barttorvik') if is_mens else None)
    seeds = data.get('m_seeds',{}) if is_mens else data.get('w_seeds',{})
    prefix = 'M' if is_mens else 'W'
    tourney = pd.read_csv(f'{DATA_DIR}/kaggle/{prefix}NCAATourneyCompactResults.csv')
    train_rows, train_y = [], []
    for ty in [y for y in range(2003, 2026) if y != 2020]:
        for _, g in tourney[tourney['Season']==ty].iterrows():
            low = min(g['WTeamID'], g['LTeamID'])
            high = max(g['WTeamID'], g['LTeamID'])
            feat = matchup_features(elo.get(low,1500), elo.get(high,1500), gc, low, high,
                team_states=ts, season=ty, seeds=seeds,
                massey_per_system=data.get('massey_per_system',{}) if is_mens else {},
                massey_avg=data.get('massey_avg',{}) if is_mens else {},
                coach_tenure=data.get('coach_tenure',{}) if is_mens else {},
                coach_changed=data.get('coach_changed',{}) if is_mens else {},
                conf_elo_means=cem, conferences=data[ck], barttorvik_lookup=bt_lk,)
            rb = [feat.get(c,0) for c in FEATURE_COLS]
            sd = seeds.get((ty,low),17) - seeds.get((ty,high),17)
            cd = colley_all.get((gender,ty,low),175) - colley_all.get((gender,ty,high),175)
            sr = srs_all.get((gender,ty,low),0) - srs_all.get((gender,ty,high),0)
            row = [rb[i] for i in FEAT_IDX] + [sd, cd, sr]
            outcome = 1 if g['WTeamID']==low else 0
            train_rows.append(row); train_y.append(outcome)
            train_rows.append([-v for v in row]); train_y.append(1-outcome)
    model = Pipeline([('imp',SimpleImputer(strategy='median')),('scl',StandardScaler()),
                      ('clf',LogisticRegression(C=C,solver='lbfgs',max_iter=1000))])
    model.fit(np.nan_to_num(np.array(train_rows),nan=0.0), np.array(train_y))

    sub_template = pd.read_csv(f'{DATA_DIR}/kaggle/SampleSubmissionStage2.csv')
    for _, row in sub_template.iterrows():
        parts = row['ID'].split('_')
        year, a, b = int(parts[0]), int(parts[1]), int(parts[2])
        if year != 2026: continue
        if is_mens and a >= 3000: continue
        if not is_mens and a < 3000: continue
        feat = matchup_features(elo.get(a,1500), elo.get(b,1500), gc, a, b,
            team_states=ts, season=2026, seeds=seeds,
            massey_per_system=data.get('massey_per_system',{}) if is_mens else {},
            massey_avg=data.get('massey_avg',{}) if is_mens else {},
            coach_tenure=data.get('coach_tenure',{}) if is_mens else {},
            coach_changed=data.get('coach_changed',{}) if is_mens else {},
            conf_elo_means=cem, conferences=data[ck], barttorvik_lookup=bt_lk,)
        rb = [feat.get(c,0) for c in FEATURE_COLS]
        sd = seeds.get((2026,a),17) - seeds.get((2026,b),17)
        cd = colley_all.get((gender,2026,a),175) - colley_all.get((gender,2026,b),175)
        sr = srs_all.get((gender,2026,a),0) - srs_all.get((gender,2026,b),0)
        x = np.nan_to_num(np.array([[rb[i] for i in FEAT_IDX] + [sd, cd, sr]]), nan=0.0)
        all_preds[row['ID']] = model.predict_proba(x)[0,1]

new_preds, orig_preds, outcomes = [], [], []
for _, g in tourney_espn.iterrows():
    mapper = espn_to_m if g['gender'] == 'M' else espn_to_w
    hk = mapper.get(g['home_espn_id']); ak = mapper.get(g['away_espn_id']); wk = mapper.get(g['winner_espn_id'])
    if hk is None or ak is None or wk is None: continue
    low, high = min(int(hk), int(ak)), max(int(hk), int(ak))
    key = f'2026_{low}_{high}'
    np_ = all_preds.get(key); op_ = orig_lookup.get(key)
    if np_ is None or op_ is None: continue
    outcome = 1 if int(wk) == low else 0
    new_preds.append(np_); orig_preds.append(op_); outcomes.append(outcome)

print(f'=== 2026 Tournament Retroactive ===')
print(f'Original Elo:               {brier_score(outcomes, orig_preds):.7f}')
print(f'LR Tournament (Colley+SRS): {brier_score(outcomes, new_preds):.7f}')
print(f'Games: {len(outcomes)}')
for gender in ['M', 'W']:
    mapper = espn_to_m if gender == 'M' else espn_to_w
    gp, go, gn = [], [], []
    for _, g in tourney_espn[tourney_espn['gender']==gender].iterrows():
        hk = mapper.get(g['home_espn_id']); ak = mapper.get(g['away_espn_id']); wk = mapper.get(g['winner_espn_id'])
        if hk is None or ak is None or wk is None: continue
        low, high = min(int(hk), int(ak)), max(int(hk), int(ak))
        key = f'2026_{low}_{high}'
        np_, op_ = all_preds.get(key), orig_lookup.get(key)
        if np_ is None or op_ is None: continue
        gp.append(op_); gn.append(np_); go.append(1 if int(wk) == low else 0)
    if gp: print(f'  {gender}: Elo={brier_score(go,gp):.4f} LR={brier_score(go,gn):.4f}')
"
