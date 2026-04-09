#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/eval_ablation_%j.out
#SBATCH --error=logs/eval_ablation_%j.err

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
EVAL_YEARS = [y for y in range(2015, 2026) if y != 2020]
ALL_FEATS = ['elo_diff', 'elo_pred', 'home', 'day_num', 'sos_diff', 'margin_mean_diff',
             'massey_avg_diff', 'barthag_diff', 'trank_adjO_diff', 'trank_adjD_diff', 'conf_elo_diff']

data = load_data(DATA_DIR)
colley_all, srs_all = {}, {}
for gk, rk in [('M', 'mens_results'), ('W', 'womens_results')]:
    for season in range(2003, 2027):
        if season in data[rk]['Season'].values:
            for t, r in compute_colley(data[rk], season).items(): colley_all[(gk, season, t)] = r
            for t, r in compute_srs(data[rk], season).items(): srs_all[(gk, season, t)] = r

def run_ablation(drop_feature=None):
    feat_list = [f for f in ALL_FEATS if f != drop_feature]
    feat_idx = [FEATURE_COLS.index(c) for c in feat_list]
    all_preds, all_outcomes = [], []
    for year in EVAL_YEARS:
        for gender in ['M', 'W']:
            is_mens = gender == 'M'
            C = 100 if is_mens else 0.15
            rk = 'mens_results' if is_mens else 'womens_results'
            ck = 'mens_conf' if is_mens else 'womens_conf'
            results = add_game_counts(data[rk].copy())
            results = results[results['Season'] <= year]
            _, elo, gc, ts, cem = build_features(
                results=results, conferences=data[ck][data[ck]['Season'] <= year],
                hfa_dict=_build_hfa_dict(data['hfa'], gender),
                location_dict=_build_location_dict(data['home_lookup'], gender),
                seasons=set(), detailed_results=data.get('mens_detailed' if is_mens else 'womens_detailed'),
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
            train_rows, test_rows, train_y, test_y = [], [], [], []
            for ty in [y for y in range(2003, year+1) if y != 2020]:
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
                    row = [rb[i] for i in feat_idx]
                    if drop_feature != 'seed_diff': row.append(sd)
                    if drop_feature != 'colley_rank_diff': row.append(cd)
                    if drop_feature != 'srs_diff': row.append(sr)
                    outcome = 1 if g['WTeamID']==low else 0
                    if ty==year: test_rows.append(row); test_y.append(outcome)
                    else: train_rows.append(row); train_y.append(outcome); train_rows.append([-v for v in row]); train_y.append(1-outcome)
            if not test_rows or not train_rows: continue
            model = Pipeline([('imp',SimpleImputer(strategy='median')),('scl',StandardScaler()),
                              ('clf',LogisticRegression(C=C,solver='lbfgs',max_iter=1000))])
            model.fit(np.nan_to_num(np.array(train_rows),nan=0.0), np.array(train_y))
            preds = model.predict_proba(np.nan_to_num(np.array(test_rows),nan=0.0))[:,1]
            all_preds.extend(preds); all_outcomes.extend(test_y)
    return brier_score(all_preds, all_outcomes)

base = run_ablation(None)
print(f'{base:.6f}  ALL 14 features (baseline)')
for feat in ALL_FEATS + ['seed_diff', 'colley_rank_diff', 'srs_diff']:
    b = run_ablation(feat)
    delta = b - base
    print(f'{b:.6f}  drop {feat:25s} delta={delta:+.6f}')
"
