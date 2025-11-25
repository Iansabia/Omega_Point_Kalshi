#!/bin/bash
# Helper script to run backtest with proper PYTHONPATH

cd "$(dirname "$0")"
PYTHONPATH=. python scripts/run_backtest.py "$@"
