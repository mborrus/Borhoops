# 2025 Kaggle Competition (Feb 26 – Mar 18, 2025)

Pre-Claude sessions reconstructed from git history.

## What happened
- Built the entire v1 notebook-based pipeline from scratch
- Started with EDA and data wrangling, ended with multiple Kaggle submissions
- Progression: Random baseline → Basic Elo → Nate Silver's Elo → Complicated Elo (custom)

## Key milestones
- **Feb 26**: First commit — repo setup, EDA notebook, boilerplate
- **Mar 11**: First working model using Nate Silver's SBCB Elo ratings
- **Mar 12**: Extended to women's bracket
- **Mar 15**: Built Basic Elo, Random Choice baselines. Started Complicated Elo
- **Mar 16**: Finished Complicated Elo with all features (HFA, travel distance, MOV, conference mean reversion, tourney boost)
- **Mar 18**: Final data update and Kaggle submission

## Complicated Elo features
- Per-team home field advantage from historical scoring margins
- Geocoded travel distance impact: `8 * miles^(1/3)`
- MOV-scaled K-factor via log function (above league-average MOV of 12)
- Variable K: starts at 56, decays linearly to 38 by game 20
- 30% conference mean reversion between seasons
- 1.07x tourney boost on Elo differences

## Submissions created
- Random values, Equal values (baselines)
- BasicEloProbs
- NateEloProbs
- ComplexEloProbs (with and without conference reversion)

## Outcome
- Submitted to Kaggle March Machine Learning Mania 2025
- Scored on Brier score (MSE of probabilities vs outcomes)
