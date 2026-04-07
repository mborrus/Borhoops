#!/bin/bash
# Run a single autoresearch experiment.
# Usage: ./autoresearch/run.sh
#
# This is the command the agent runs each iteration.
# Output goes to run.log, then grep for results.

set -e
cd "$(dirname "$0")/.."

python autoresearch/prepare.py > autoresearch/run.log 2>&1

echo ""
echo "=== Results ==="
grep "^mean_brier:\|^n_games:\|^elapsed_seconds:" autoresearch/run.log
