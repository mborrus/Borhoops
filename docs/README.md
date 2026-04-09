# Documentation Index

## Project Rules
- [Golden Rules](golden_rules.md) — Non-negotiable coding standards

## Data Pipeline
- [Extract Overview](extract.md) — Data extraction pipeline
- [Extract: Barttorvik](extract_barttorvik.md) — T-Rank ratings scraper
- [Extract: Odds](extract_odds.md) — ESPN odds scraper
- [Extract: Polls](extract_polls.md) — AP/Coaches poll scraper
- [Extract: Vegas](extract_vegas.md) — The Odds API scraper
- [Transform: Team Utils](transform_team_utils.md) — Fuzzy matching for team names
- [Features](features.md) — Feature pipeline reference (40 features)

## Model Development
- [HPC Plan](hpc_plan.md) — Original plan for model zoo on Gaia cluster
- [Phase 1: Feature Integration](phase1_implementation.md) — Expanding features from 30 → 40
- [Phase 2: Overnight Log](phase2_overnight_log.md) — Model zoo results, autoresearch log, all experiments

## Analysis & Findings
- [Competition Analysis](competition_analysis.md) — Top Kaggle 2026 solutions analyzed (3rd, 10th, 19th, 21st, 49th place)
- [Edge Model Analysis](edge_model_analysis.md) — Model vs Vegas: better calibration but no directional edge
- [Sweet Spot Discovery](sweet_spot_discovery.md) — Regular season betting edge (75% win rate on pick'em games)
- [Seed Override Analysis](seed_override_analysis.md) — Override top seeds in early rounds: dead neutral over 10 years

## Session Logs
- [Session Summary: Apr 7-8](session_summary_apr7_8.md) — Full summary of 2-day model development session

## Other
- [Data Mining Proposal](Data%20Mining%20Proposal.docx) — Original project proposal
- [journal/](journal/) — Development journal entries
- [learning_notes/](learning_notes/) — Research notes

## Current Best Models

### Game Model (Kaggle/Brier)
**LR + 14 features + Colley + SRS, separate M/W regularization**
- LOSO CV Brier: **0.1616**
- 2026 retroactive: **0.1276** (beats Elo submission 0.1283)
- Features: elo_diff, elo_pred, home, day_num, sos_diff, margin_mean_diff, massey_avg_diff, barthag_diff, trank_adjO_diff, trank_adjD_diff, conf_elo_diff, seed_diff, colley_rank_diff, srs_diff
- Config: LR C=100 (men's), C=0.15 (women's), trained on tournament games 2003-2025

### Edge Model (Betting)
**LR + 4 features, regular season pick'em games**
- Win rate: **78.6%** on 429 bets (2016-2025)
- ROI: **+52%** with real moneyline odds
- Features: home, massey_avg_diff, sos_diff, conf_elo_diff
- Filter: Vegas 45-55% implied, model edge ≥15%

### Regular Season Model
**XGBoost + 11 features**
- Regular season Brier: **0.1656** (beats Vegas 0.1741)
- No directional edge over Vegas on disagreement games
