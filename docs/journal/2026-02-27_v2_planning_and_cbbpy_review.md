# Session: 2026-02-27 — v2 Planning & CBBpy Review

## Goal
Start planning the v2 automated pipeline and evaluate CBBpy as a live data source.

## What was done
- Created `v1` branch to preserve working notebooks
- Wrote `CLAUDE.md` with full project context, model documentation, and v2 pipeline plan
- Reviewed the entire CBBpy codebase (ESPN scraper library)
- Documented 29 issues in `learning_notes/cbbpy_code_review.md`
- Created `data_exploration/` with test notebook and crosswalk builder script

## Decisions made
- **v2 stack**: Airflow (orchestration) + dbt (transforms) + Snowflake (warehouse)
- **Live data source**: CBBpy for daily ESPN game results + locations
- **Key blocker identified**: ESPN→Kaggle TeamID crosswalk needs to be built

## CBBpy review findings
- 8 runtime crash bugs that would hit during normal usage
- Major design issues: module-level mutable state, 1700-line monolith, copy-pasted code
- No mock tests, so test suite breaks with any ESPN format change
- Conclusion: usable but needs patching before relying on it in production

## Next steps identified
- Build the ESPN→Kaggle team ID crosswalk lookup table
- Start building the ELT pipeline
- 2026 Kaggle competition deadline: March 19, 2026
